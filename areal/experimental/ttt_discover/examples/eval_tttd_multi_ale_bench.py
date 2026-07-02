#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
TTT-Discover Multi-Model ALE-Bench Evaluation.

Evaluates multiple LoRA adapters on the full ALE-Bench corpus using the same
Self-refine x1 protocol as training:

1. For each model (baseline / teacher / student / ...), load its LoRA adapter.
2. Generate ``ale_bench_eval_n_candidates`` responses for every ALE-Bench problem.
3. Run public evaluation (50 local test cases) for each candidate inside the
   rollout reward function so GPU generation overlaps with CPU/Docker scoring.
4. Pick the candidate with the highest median public-case score per problem.
5. Run private evaluation on the selected candidate.
6. Save per-model, per-problem results plus overall averages.

Usage:
    python areal/experimental/ttt_discover/examples/eval_tttd_multi_ale_bench.py \
        --config areal/experimental/ttt_discover/examples/conf/fsdp_lora_vllm_ale_bench_qwen3_8b_distill_multi.yaml
"""

import functools
import json
import os
import sys
from typing import Any

import torch.distributed as dist

from areal import PPOTrainer
from areal.api.alloc_mode import ModelAllocation
from areal.api.alloc_mode import _AllocationMode as AllocationMode
from areal.api.cli_args import load_expr_config
from areal.api.engine_api import WeightUpdateMeta
from areal.api.io_struct import FinetuneSpec
from areal.experimental.ttt_discover.actor import TTTDActor
from areal.experimental.ttt_discover.ale_bench_eval import (
    ale_bench_public_reward_fn,
    combine_ale_bench_results,
    evaluate_problem_subset_with_public_scores,
    list_ale_bench_problem_ids,
)
from areal.experimental.ttt_discover.config import (
    TTTDDistillConfig,
    create_env_from_config,
)
from areal.experimental.ttt_discover.envs.ale_bench import (
    create_initial_state_ale_bench,
)
from areal.experimental.ttt_discover.sampler import create_sampler
from areal.experimental.ttt_discover.workflow_v2 import (
    MultiProblemTTTDiscoverWorkflowV2,
)
from areal.infra import current_platform
from areal.utils import logging, seeding
from areal.utils.environ import is_single_controller
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("eval_tttd_multi_ale_bench")


class TTTDAleBenchMultiEvalTrainer(PPOTrainer):
    """Multi-model ALE-Bench evaluation trainer.

    Mirrors the initialization of ``TTTDMultiEvalTrainer`` but replaces the
    single-problem rollout loop with full-corpus ALE-Bench evaluation.
    """

    def __init__(self, config: TTTDDistillConfig):
        self.config = config
        rank = int(os.getenv("RANK", "0"))
        if is_single_controller():
            logging.setup_file_logging(StatsLogger.get_log_path(config.stats_logger))

        # Load tokenizer and processor
        self.processor, self.tokenizer = load_hf_processor_and_tokenizer(
            config.tokenizer_path
        )

        # Scheduler (skip if null)
        self.scheduler = None
        if is_single_controller():
            sched_type = getattr(config.scheduler, "type", None)
            if sched_type is not None and sched_type != "null":
                self.scheduler = self._init_scheduler()

        # Set seed
        seeding.set_random_seed(config.seed, key=f"eval{rank}")

        # Parse allocation mode
        self.allocation_mode = AllocationMode.from_str(config.allocation_mode)
        self.actor_alloc = ModelAllocation.from_str(config.actor.backend, name="actor")
        self.rollout_alloc = ModelAllocation.from_str(
            config.rollout.backend, name="rollout"
        )
        self._amend_xccl_weight_update_envvar()

        # Create a minimal fresh sampler (not used for ALE-Bench corpus eval, but
        # required by the actor/connect_sampler API).
        max_head_offpolicyness = getattr(config.rollout, "max_head_offpolicyness", 2)
        max_version_history = max_head_offpolicyness + 1
        self.is_sync_mode = max_head_offpolicyness == 0
        if self.is_sync_mode and config.sampler.lazy_puct_sampling:
            config.sampler.lazy_puct_sampling = False

        import tempfile

        fresh_log_path = tempfile.mkdtemp(prefix="eval_fresh_sampler_")
        self.sampler = create_sampler(
            sampler_type=getattr(config.sampler, "type", "puct"),
            log_path=fresh_log_path,
            env_type=getattr(config.sampler, "env_type", "ale_bench"),
            budget_s=getattr(config.sampler, "save_freq", 100),
            initial_exp_type=getattr(
                config.sampler, "initial_exp_type", "best_available"
            ),
            batch_size=getattr(config.sampler, "batch_size", 8),
            resume_step=None,
            c_puct=getattr(config.sampler, "c_puct", 1.5),
            gamma=getattr(config.sampler, "gamma", 0.95),
            max_children=getattr(config.sampler, "max_children", 100),
            max_states=getattr(config.sampler, "max_states", 10000),
            top_k=getattr(config.sampler, "top_k", 1000),
            temperature=getattr(config.sampler, "temperature", 1.0),
            max_version_history=max_version_history,
            sampling_strategy="puct",
            problem_id=getattr(config.sampler, "problem_id", ""),
        )
        logger.info(
            f"[EvalSampler] Fresh sampler created with {len(self.sampler._states)} states"
        )

        # Create actor (no critic/ref - eval only)
        self.actor = self._create_tttd_actor(config.actor)
        self.ref = None

        # ALE-Bench full-corpus evaluation state
        self._ale_bench_eval_enabled = getattr(config, "ale_bench_eval_enabled", True)
        self._ale_bench_eval_output_dir = ""
        self._ale_bench_eval_problem_ids: list[str] = []
        self._ale_bench_eval_envs: dict[str, Any] = {}
        self._setup_ale_bench_eval(config)

        # Keep a fallback env for workflow construction
        self.env = None
        if self._ale_bench_eval_envs:
            self.env = next(iter(self._ale_bench_eval_envs.values()))

        # Determine which models to evaluate
        eval_models = getattr(config, "eval_models", None)
        if eval_models is None:
            eval_ckpt_dir = os.path.join(
                config.saver.fileroot,
                config.experiment_name,
                config.trial_name,
                "eval_checkpoints",
            )
            teacher_path = getattr(config, "teacher_lora_path", None)
            if teacher_path is None and config.teacher is not None:
                teacher_path = config.teacher.path
            student_path = getattr(config, "student_lora_path", None)
            if student_path is None and os.path.isdir(
                os.path.join(eval_ckpt_dir, "student")
            ):
                student_path = os.path.join(eval_ckpt_dir, "student")
            current_exp = config.experiment_name
            eval_models = {
                "baseline": None,
                "teacher": teacher_path,
                f"{current_exp}-student": student_path,
            }

            # Auto-discover students from other experiments under the same fileroot
            fileroot = config.saver.fileroot
            trial_name = config.trial_name
            if os.path.isdir(fileroot):
                for exp_name in sorted(os.listdir(fileroot)):
                    exp_dir = os.path.join(fileroot, exp_name)
                    if not os.path.isdir(exp_dir):
                        continue
                    other_student = os.path.join(
                        exp_dir, trial_name, "eval_checkpoints", "student"
                    )
                    if os.path.isdir(other_student):
                        label = f"{exp_name}-student"
                        if exp_name == current_exp:
                            continue
                        if label not in eval_models:
                            eval_models[label] = other_student
                            logger.info(
                                f"[MultiEval] Auto-discovered student from other experiment: "
                                f"{label} -> {other_student}"
                            )

        eval_models = {
            k: v for k, v in eval_models.items() if v is not None or k == "baseline"
        }
        if not eval_models:
            raise ValueError(
                "No evaluation models found. Set teacher_path / student_lora_path or eval_models."
            )
        self._eval_models = eval_models

        # Initialize models
        self._initialize_engines()

        # Connect sampler to actor for distributed synchronization
        self.actor.connect_sampler(self.sampler)

        # Initialize inference engines (training + eval rollouts)
        self.rollout = self._init_rollout(config.rollout, is_eval=False)
        self.eval_rollout = self._init_rollout(config.rollout, is_eval=True)

        # Setup weight update meta and connect to inference engine
        self._setup_weight_update_meta()

        # Setup stats logger only
        self._setup_stats_logger()

        # Store workflow kwargs for later use
        self._workflow_kwargs = {}

    def _create_tttd_actor(self, actor_config):
        """Create TTTDActor."""
        actor = TTTDActor(config=actor_config)
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor

    def _initialize_engines(self):
        """Initialize training engines."""
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=self.config.eval_batch_size,
            train_batch_size=self.config.eval_batch_size,
        )
        self.actor.initialize(
            addr=None, ft_spec=ft_spec, alloc_mode=self.allocation_mode, role="actor"
        )

    def _setup_weight_update_meta(self):
        """Setup weight update meta and connect to inference engine."""
        config = self.config
        if config.actor.weight_update_mode == "disk":
            disk_kwargs = {
                "experiment_name": config.experiment_name,
                "trial_name": config.trial_name,
                "file_root": config.cluster.fileroot,
                "name": "default",
                "clear_checkpoint_after_load": True,
            }
            if config.actor.use_lora:
                disk_kwargs.update(
                    {
                        "use_lora": config.actor.use_lora,
                        "lora_name": config.gconfig.lora_name,
                        "lora_int_id": 1,
                        "base_model_name": config.actor.path,
                    }
                )
            self.weight_update_meta = WeightUpdateMeta.from_disk(**disk_kwargs)
        elif config.actor.weight_update_mode == "xccl":
            if self.allocation_mode.train_backend == "megatron":
                self.weight_update_meta = WeightUpdateMeta.from_megatron_xccl(
                    self.allocation_mode
                )
            else:
                xccl_kwargs = {"gen_allocation": self.rollout_alloc}
                if config.actor.use_lora:
                    xccl_kwargs.update(
                        {
                            "use_lora": config.actor.use_lora,
                            "lora_name": config.gconfig.lora_name,
                            "lora_int_id": 1,
                            "base_model_name": config.actor.path,
                        }
                    )
                self.weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(**xccl_kwargs)
        else:
            raise ValueError(
                f"Invalid weight update mode: {config.actor.weight_update_mode}"
            )

        self.actor.connect_engine(self.rollout, self.weight_update_meta)
        logger.info(f"[Rank {dist.get_rank()}] connect_engine done")

    def _setup_stats_logger(self):
        """Setup stats logger only (no saver/recover/evaluator for eval)."""
        config = self.config
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=config.eval_batch_size,
            train_batch_size=config.eval_batch_size,
        )
        self.stats_logger = StatsLogger(config, ft_spec)

    def _setup_ale_bench_eval(self, config: TTTDDistillConfig):
        """Discover all ALE-Bench problems and create per-problem envs for eval."""
        if not self._ale_bench_eval_enabled:
            return

        try:
            self._ale_bench_eval_problem_ids = list_ale_bench_problem_ids(
                lite_version=config.ale_bench_eval_lite_version
            )
        except Exception as e:
            logger.warning(
                f"[AleBenchEval] Failed to list ALE-Bench problems: {e}. Disabling eval."
            )
            self._ale_bench_eval_enabled = False
            return

        logger.info(
            f"[AleBenchEval] Discovered {len(self._ale_bench_eval_problem_ids)} problems "
            f"(lite={config.ale_bench_eval_lite_version})"
        )

        output_dir = config.ale_bench_eval_output_dir
        if not output_dir:
            output_dir = os.path.join(
                config.saver.fileroot,
                config.experiment_name,
                config.trial_name,
                "ale_bench_eval",
            )
        self._ale_bench_eval_output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Only global rank 0 creates envs; candidate generation runs on DP head.
        if self.actor.rank != 0:
            logger.info("[AleBenchEval] Skipping env creation on non-zero ranks")
            return

        original_problem_id = getattr(config.sampler, "problem_id", "")
        original_num_cpus = getattr(config.sampler, "num_cpus", 2)
        # Use the eval-specific worker count for ALE-Bench sessions; the training
        # sampler may use a much smaller num_cpus which would make public eval slow.
        config.sampler.num_cpus = config.ale_bench_eval_num_workers
        cpu_count = os.cpu_count() or 1
        total_workers = (
            config.ale_bench_eval_num_workers
            * config.ale_bench_eval_n_parallel_problems
        )
        if total_workers > cpu_count:
            logger.warning(
                f"[AleBenchEval] Eval may oversubscribe CPU: "
                f"num_workers({config.ale_bench_eval_num_workers}) × "
                f"n_parallel_problems({config.ale_bench_eval_n_parallel_problems}) = "
                f"{total_workers} > cpu_count({cpu_count}). "
                f"Consider reducing ale_bench_eval_num_workers or "
                f"ale_bench_eval_n_parallel_problems."
            )
        logger.info(
            f"[AleBenchEval] Creating {len(self._ale_bench_eval_problem_ids)} envs "
            f"with num_workers={config.ale_bench_eval_num_workers}"
        )
        for problem_id in self._ale_bench_eval_problem_ids:
            config.sampler.problem_id = problem_id
            config.sampler.env_type = "ale_bench"
            try:
                env = create_env_from_config(config)
                self._ale_bench_eval_envs[problem_id] = env
            except Exception as e:
                logger.warning(
                    f"[AleBenchEval] Failed to create env for {problem_id}: {e}"
                )
        config.sampler.problem_id = original_problem_id
        config.sampler.num_cpus = original_num_cpus

    def _zero_lora_weights(self, engine):
        """Zero out all LoRA parameters so the model behaves like the pure base model."""
        logger.info("[LoadAdapter] Zeroing out LoRA weights for base model evaluation")
        n_zeroed = 0
        for name, param in engine.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                param.data.zero_()
                n_zeroed += 1
        logger.info(f"[LoadAdapter] Zeroed {n_zeroed} LoRA parameters")

    def _load_peft_lora_adapter(self, engine, path: str):
        """Load a PEFT LoRA adapter checkpoint into the FSDP-wrapped actor."""
        from safetensors.torch import load_file
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        adapter_path = os.path.join(path, "adapter_model.safetensors")
        if not os.path.isfile(adapter_path):
            raise ValueError(
                f"LoRA adapter not found at {adapter_path}. "
                f"Expected a PEFT checkpoint with adapter_model.safetensors."
            )

        logger.info(f"[LoadAdapter] Loading LoRA adapter from {path}")

        if dist.get_rank() == 0:
            raw_state = load_file(adapter_path)
            fixed_state = {}
            for k, v in raw_state.items():
                if not k.startswith("base_model.model."):
                    k = f"base_model.model.{k}"
                if ".lora_A.weight" in k:
                    k = k.replace(".lora_A.weight", ".lora_A.default.weight")
                elif ".lora_B.weight" in k:
                    k = k.replace(".lora_B.weight", ".lora_B.default.weight")
                fixed_state[k] = v
        else:
            fixed_state = {}

        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=False,
            broadcast_from_rank0=True,
            strict=False,
        )
        set_model_state_dict(engine.model, fixed_state, options=options)
        logger.info("[LoadAdapter] LoRA adapter loaded into actor successfully")

    def _switch_model(self, label: str, version: int):
        """Load the requested adapter and push weights to the inference engine."""
        lora_path = self._eval_models[label]

        if not lora_path:
            logger.info(
                f"[MultiEval-{label}] Evaluating pure base model (no LoRA adapter)"
            )
            self._zero_lora_weights(self.actor)
        else:
            logger.info(f"[MultiEval-{label}] Loading LoRA from {lora_path}")
            self._load_peft_lora_adapter(self.actor, lora_path)

        logger.info(
            f"[MultiEval-{label}] Pushing weights to vLLM via update_weights()..."
        )
        self.rollout.pause()
        self.eval_rollout.pause()
        versioned_meta = self.weight_update_meta.with_version(version)
        self.actor.update_weights(versioned_meta)
        self.actor.set_version(version)
        self.rollout.set_version(version)
        self.eval_rollout.set_version(version)
        if dist.is_initialized():
            dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()
        self.rollout.resume()
        self.eval_rollout.resume()
        logger.info(f"[MultiEval-{label}] Weights pushed and rollouts resumed.")

    def _normalize_rollout_batch(self, rollout_batch) -> dict[str, Any]:
        """Ensure rollout_batch is a single dict for downstream processing.

        Mirrors ``TTTDDistillTrainer._normalize_rollout_batch`` but only keeps
        the fields needed for ALE-Bench evaluation.
        """
        if isinstance(rollout_batch, dict):
            return rollout_batch
        if not isinstance(rollout_batch, list):
            raise TypeError(f"Unexpected rollout_batch type: {type(rollout_batch)}")
        if len(rollout_batch) == 1:
            return rollout_batch[0]

        from areal.utils.data import concat_batch

        problem_ids: list[str] = []
        metadata_list: list[dict[str, Any]] = []
        for d in rollout_batch:
            pids = d.pop("_problem_ids", None)
            if isinstance(pids, (list, tuple)):
                problem_ids.extend(pids)
            elif pids is not None:
                problem_ids.append(pids)

            md = d.pop("_metadata", None)
            if isinstance(md, (list, tuple)):
                metadata_list.extend(md)
            elif md is not None:
                metadata_list.append(md)

        batched, _ = concat_batch(rollout_batch)
        batched["_problem_ids"] = problem_ids
        batched["_metadata"] = metadata_list
        return batched

    def _generate_ale_bench_eval_candidates(
        self, label: str, global_step: int
    ) -> tuple[dict[str, list[str]], dict[str, list[dict[str, Any]]]]:
        """Generate candidate responses for ALE-Bench evaluation for the current model."""
        config = self.config
        n_candidates = config.ale_bench_eval_n_candidates

        if self.actor.rank != 0:
            return {}, {}

        eval_problem_ids = self._ale_bench_eval_problem_ids
        local_candidates: dict[str, list[str]] = {}
        public_results_by_problem: dict[str, list[dict[str, Any]]] = {}

        if not eval_problem_ids:
            return local_candidates, public_results_by_problem

        base_eval_gconfig = config.eval_gconfig or config.gconfig
        eval_gconfig = base_eval_gconfig.new(n_samples=n_candidates)

        public_reward_fn = functools.partial(
            ale_bench_public_reward_fn,
            lite_version=config.ale_bench_eval_lite_version,
            session_duration_hours=4.0,
            ale_bench_num_workers=config.ale_bench_eval_num_workers,
        )

        # Public eval can be CPU/Docker heavy; allow concurrent reward workers so
        # multiple problems' public evaluations overlap.  Cap at n_parallel_problems
        # to avoid oversubscribing the CPU pool used by each ALE-Bench session.
        max_reward_workers = max(1, config.ale_bench_eval_n_parallel_problems)
        eval_workflow_kwargs = dict(
            env=self.env,
            problem_envs=self._ale_bench_eval_envs,
            gconfig=eval_gconfig,
            tokenizer=self.tokenizer,
            enable_thinking=config.enable_thinking,
            max_prompt_thinking_tokens=config.max_prompt_thinking_tokens,
            batch_size=1,
            group_size=n_candidates,
            lazy_sampling=False,
            reward_fn=public_reward_fn,
            max_reward_workers=max_reward_workers,
            distill_mode=True,
        )
        eval_workflow_cls = MultiProblemTTTDiscoverWorkflowV2

        data_list: list[dict[str, Any]] = []
        for problem_id in eval_problem_ids:
            env = self._ale_bench_eval_envs.get(problem_id)
            if env is None:
                continue
            state = create_initial_state_ale_bench(problem_id=problem_id)
            prompt = (
                env.get_prompt_distill(state)
                if hasattr(env, "get_prompt_distill")
                else env.get_prompt(state)
            )
            data_list.append(
                {
                    "prompt": prompt,
                    "state_id": state.id,
                    "_state_obj": state,
                    "_problem_id": problem_id,
                }
            )

        if not data_list:
            return local_candidates, public_results_by_problem

        logger.info(
            f"[AleBenchEval-{label}] Generating {n_candidates} candidates "
            f"for {len(data_list)} problems"
        )

        for data in data_list:
            self.eval_rollout.submit(
                data,
                eval_workflow_cls,
                workflow_kwargs=eval_workflow_kwargs,
                group_size=n_candidates,
                is_eval=True,
            )

        logger.info(
            f"[AleBenchEval-{label}] Submitted {len(data_list)} rollout requests"
        )
        results = self.eval_rollout.wait(len(data_list), timeout=None)
        logger.info(f"[AleBenchEval-{label}] Got {len(results)} rollout results")

        batch = self._normalize_rollout_batch(results)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]
        problem_ids = batch.get("_problem_ids", [])
        metadata_list = batch.get("_metadata", [])

        prompt_lens = (attention_mask & (loss_mask == 0)).sum(dim=1).cpu().numpy()
        comp_lens = loss_mask.sum(dim=1).cpu().numpy()

        for i in range(input_ids.shape[0]):
            if i > 0 and i % 10 == 0:
                logger.info(
                    f"[AleBenchEval-{label}] Processed {i}/{input_ids.shape[0]} rollout results"
                )
            problem_id = problem_ids[i] if i < len(problem_ids) else ""
            env = self._ale_bench_eval_envs.get(problem_id)
            if env is None:
                continue
            prompt_len = int(prompt_lens[i])
            comp_len = int(comp_lens[i])
            completion_ids = (
                input_ids[i, prompt_len : prompt_len + comp_len].cpu().tolist()
            )
            completion_text = self.tokenizer.decode(completion_ids)
            code = env.extract_code(completion_text)
            local_candidates.setdefault(problem_id, []).append(
                code if code is not None else ""
            )

            md = metadata_list[i] if i < len(metadata_list) else {}
            public_info = md.get("metadata", md) if isinstance(md, dict) else {}
            public_result = {
                "idx": len(local_candidates.get(problem_id, [])) - 1,
                "code": code if code is not None else "",
                "public": {
                    "median_case_score": float(public_info.get("public_median", 0.0)),
                    "overall_absolute_score": float(
                        public_info.get("public_overall_absolute", 0.0)
                    ),
                    "overall_relative_score": float(
                        public_info.get("public_overall_relative", 0.0) or 0.0
                    ),
                    "judge_result": str(
                        public_info.get("public_judge_result", "UNKNOWN")
                    ),
                    "num_cases": int(public_info.get("public_num_cases", 0)),
                    "rank": int(public_info.get("public_rank", -1)),
                    "performance": int(public_info.get("public_performance", -1)),
                },
            }
            public_results_by_problem.setdefault(problem_id, []).append(public_result)

        total_count = sum(len(v) for v in local_candidates.values())
        logger.info(
            f"[AleBenchEval-{label}] Generated {total_count} candidate codes "
            f"across {len(local_candidates)} problems"
        )
        return local_candidates, public_results_by_problem

    def _run_ale_bench_eval_for_model(
        self, label: str, global_step: int
    ) -> dict[str, Any]:
        """Run full ALE-Bench public->private evaluation for one model."""
        config = self.config

        if self.actor.rank != 0:
            logger.info(
                f"[AleBenchEval-{label}] Rank {self.actor.data_parallel_rank} skipping, "
                f"only rank 0 runs"
            )
            return {"model": label, "skipped": True}

        logger.info(f"[AleBenchEval-{label}] Starting candidate generation")
        candidates_by_problem, public_results_by_problem = (
            self._generate_ale_bench_eval_candidates(label, global_step)
        )
        logger.info(f"[AleBenchEval-{label}] Candidate generation done")

        eval_problem_ids = self._ale_bench_eval_problem_ids
        logger.info(
            f"[AleBenchEval-{label}] Running private eval for {len(eval_problem_ids)} problems"
        )

        # Reuse the env sessions created in _setup_ale_bench_eval so private
        # eval does not rebuild Rust tools in each worker.
        problem_sessions = {
            problem_id: env.session
            for problem_id, env in self._ale_bench_eval_envs.items()
            if hasattr(env, "session") and env.session is not None
        }
        selection_method = getattr(config, "ale_bench_eval_selection_method", "median")
        local_results = evaluate_problem_subset_with_public_scores(
            eval_problem_ids,
            public_results_by_problem,
            lite_version=config.ale_bench_eval_lite_version,
            session_duration_hours=4.0,
            ale_bench_num_workers=config.ale_bench_eval_num_workers,
            n_parallel_problems=config.ale_bench_eval_n_parallel_problems,
            problem_sessions=problem_sessions,
            selection_method=selection_method,
        )
        logger.info(
            f"[AleBenchEval-{label}] Private eval done, {len(local_results)} results, "
            f"selection={selection_method}"
        )

        training_problem_ids = {mt.problem_id for mt in config.multi_teacher}
        output = combine_ale_bench_results(
            local_results,
            self._ale_bench_eval_problem_ids,
            training_problem_ids,
        )
        output.update(
            {
                "model": label,
                "lora_path": self._eval_models[label],
                "n_candidates": config.ale_bench_eval_n_candidates,
                "lite_version": config.ale_bench_eval_lite_version,
                "session_duration_hours": 4.0,
                "ale_bench_num_workers": config.ale_bench_eval_num_workers,
                "n_parallel_problems": config.ale_bench_eval_n_parallel_problems,
                "selection_method": selection_method,
            }
        )

        output_path = os.path.join(
            self._ale_bench_eval_output_dir,
            f"ale_bench_eval_results_{label}_step_{global_step:06d}.json",
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        avg_all = output.get("average_all", {})
        avg_oot = output.get("average_out_of_training", {})
        logger.info(f"[AleBenchEval-{label}] Saved results to {output_path}")
        logger.info(
            f"[AleBenchEval-{label}] "
            f"all_abs={avg_all.get('absolute_score', 0.0):.2f} "
            f"all_perf={avg_all.get('performance', 0.0):.2f} "
            f"oot_abs={avg_oot.get('absolute_score', 0.0):.2f} "
            f"oot_perf={avg_oot.get('performance', 0.0):.2f} "
            f"success={avg_all.get('count', 0)}/{len(self._ale_bench_eval_problem_ids)}"
        )
        return output

    def run_eval(self, workflow_class=None, workflow_kwargs=None):
        """Run ALE-Bench evaluation on all configured models."""
        is_dp_head = self.actor.rank == 0

        logger.info(
            f"[MultiEval] Starting ALE-Bench evaluation of {len(self._eval_models)} models: "
            f"{list(self._eval_models.keys())}"
        )

        all_results: dict[str, dict[str, Any]] = {}
        for idx, label in enumerate(self._eval_models):
            if dist.is_initialized():
                dist.barrier()
            self._switch_model(label, version=idx)
            all_results[label] = self._run_ale_bench_eval_for_model(
                label, global_step=idx
            )

        # Save combined comparison summary on DP head
        if is_dp_head:
            comparison_path = os.path.join(
                self._ale_bench_eval_output_dir,
                "ale_bench_eval_comparison.json",
            )
            with open(comparison_path, "w") as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"[MultiEval] Comparison results saved to {comparison_path}")

            logger.info("\n" + "=" * 90)
            logger.info("ALE-BENCH EVALUATION COMPARISON")
            logger.info("=" * 90)
            for label in self._eval_models:
                r = all_results[label]
                if r.get("skipped"):
                    logger.info(f"{label:25s} | skipped")
                    continue
                avg_all = r.get("average_all", {})
                avg_oot = r.get("average_out_of_training", {})
                logger.info(
                    f"{label:25s} | all_abs={avg_all.get('absolute_score', 0.0):.2f} "
                    f"oot_abs={avg_oot.get('absolute_score', 0.0):.2f} "
                    f"perf={avg_all.get('performance', 0.0):.2f} "
                    f"success={avg_all.get('count', 0)}/{len(self._ale_bench_eval_problem_ids)}"
                )
            logger.info("=" * 90)

        if dist.is_initialized():
            dist.barrier()

    def close(self):
        """Cleanup resources."""
        self.stats_logger.close()
        if hasattr(self, "eval_rollout") and self.eval_rollout is not None:
            self.eval_rollout.destroy()
        if hasattr(self, "rollout") and self.rollout is not None:
            self.rollout.destroy()
        if hasattr(self, "actor") and self.actor is not None:
            self.actor.destroy()
        from areal.utils import perf_tracer

        perf_tracer.save(force=True)


def main(args):
    """Main evaluation function."""
    config, _ = load_expr_config(args, TTTDDistillConfig)

    if config.tokenizer_path:
        from areal.utils.hf_utils import load_hf_tokenizer

        tokenizer = load_hf_tokenizer(config.tokenizer_path)
        if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    # Ensure eval-related ALE-Bench settings have sensible defaults if missing.
    if not getattr(config, "ale_bench_eval_enabled", False):
        config.ale_bench_eval_enabled = True
    if getattr(config, "ale_bench_eval_n_candidates", 0) <= 0:
        config.ale_bench_eval_n_candidates = 15
    if getattr(config, "ale_bench_eval_num_workers", 0) <= 0:
        config.ale_bench_eval_num_workers = 1
    if getattr(config, "ale_bench_eval_n_parallel_problems", 0) <= 0:
        config.ale_bench_eval_n_parallel_problems = 1
    if not getattr(config, "ale_bench_eval_selection_method", ""):
        config.ale_bench_eval_selection_method = "median"

    with TTTDAleBenchMultiEvalTrainer(config) as trainer:
        trainer.run_eval()


if __name__ == "__main__":
    main(sys.argv[1:])
