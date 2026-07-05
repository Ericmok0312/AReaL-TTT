#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
TTT-Discover Teacher Hint-Sampling Ablation on ALE-Bench.

Evaluates all teachers listed in ``config.multi_teacher``. Each teacher is
evaluated on its own ALE-Bench problem (``multi_teacher[i].problem_id``) while
ablating *how privileged hints are sampled* from its trained sampler. The
evaluation follows the same public→private ALE-Bench protocol as
``eval_tttd_multi_ale_bench.py`` but is restricted to each teacher's training
problem. Additionally, the pure base model (all LoRA weights zeroed out) is
evaluated on the union of all teacher problems once per hint mode, using each
problem's teacher sampler to provide the privileged hints. This gives a
per-mode baseline comparable to the teacher results.

Each mode uses the same distill eval prompt and the same hint format
(code + score); only the selection strategy for the privileged state(s)
changes. Modes can be specified either as plain strings or as dicts with
``mode`` and ``k`` (and an optional ``label``). The default ablation runs:

- ``best_worst_combined`` with ``k=1`` (label ``best_worst_k1``)
- ``best`` with ``k=1`` (label ``best_k1``)
- ``best`` with ``k=2`` (label ``best_k2``)
- ``worst_nonzero`` with ``k=1`` (label ``worst_k1``)
- ``diverse_best`` with ``k=2`` (label ``diverse_best_k2``)
- ``milestone`` (label ``milestone``)

Supported modes:

- ``none``                  : no privileged hint.
- ``best``                  : highest-value state(s) in the sampler.
- ``worst_nonzero``         : worst positive-reward state(s).
- ``breakthrough``          : largest parent->child improvement transition.
- ``milestone``             : whole-path milestone hints from pre-extracted JSON.
- ``best_worst_combined``   : strong + weak reference states.
- ``diverse_best``          : best state(s) from different PUCT branches.
- ``diverse_worst_nonzero`` : worst non-zero state(s) from different branches.
- ``diverse_best_worst``    : combined diverse best + diverse weak references.

ALE-Bench sessions are globally cached per problem and live for 24 hours so
that baseline + all teachers share exactly one session per problem. Sessions
are closed together at shutdown.

Usage:
    python areal/experimental/ttt_discover/examples/eval_teacher_hint_ablation_ale_bench.py \
        --config areal/experimental/ttt_discover/examples/conf/fsdp_lora_vllm_ale_bench_qwen3_8b_distill_multi.yaml \
        ++eval_hint_modes=[{mode:best_worst_combined,k:1},{mode:best,k:2}]

Because this is an eval-only run, pass ``++recover.mode=off`` (or set it in the
config) so the deprecated local launcher does not restart the job after the
trainer exits cleanly.
"""

import copy
import functools
import json
import os
import signal
import sys
from collections.abc import Callable
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
    average_all_candidates_stats,
    close_all_cached_ale_bench_sessions,
    combine_ale_bench_results,
    derive_selection_outputs_from_all_candidates,
    evaluate_all_candidates_private,
    evaluate_problem_subset_with_public_scores,
)
from areal.experimental.ttt_discover.config import TTTDDistillConfig
from areal.experimental.ttt_discover.envs.ale_bench import (
    AleBenchEnv,
    close_all_ale_bench_sessions,
    create_initial_state_ale_bench,
)
from areal.experimental.ttt_discover.sampler import (
    _find_latest_sampler_step,
    create_sampler,
    create_sampler_from_config,
)
from areal.experimental.ttt_discover.workflow_v2 import (
    MultiProblemTTTDiscoverWorkflowV2,
)
from areal.infra import current_platform
from areal.utils import logging, seeding
from areal.utils.environ import is_single_controller
from areal.utils.hf_utils import load_hf_processor_and_tokenizer
from areal.utils.stats_logger import StatsLogger

logger = logging.getLogger("eval_teacher_hint_ablation_ale_bench")


# Hint modes supported by this ablation.
# These mirror ``PUCTSampler.get_hint_states`` plus a no-hint baseline, a
# whole-path milestone mode, and a warm-start improvement mode.
HINT_MODES = [
    "none",
    "best",
    "worst_nonzero",
    "breakthrough",
    "milestone",
    "best_worst_combined",
    "diverse_best",
    "diverse_worst_nonzero",
    "diverse_best_worst",
    "warm_start",
]

# Default hint combinations for the ablation. Each entry can be a plain mode
# string (k=1) or a dict with ``mode``, ``k``, and optional ``combine`` /
# ``label``. When ``combine=True`` and k>1, all k states are shown together
# in a single prompt instead of rotating across candidates.
DEFAULT_EVAL_HINT_MODES: list[dict[str, Any] | str] = [
    {"mode": "none", "label": "none"},
    {"mode": "best_worst_combined", "k": 1, "label": "best_worst_k1"},
    {"mode": "best", "k": 1, "label": "best_k1"},
    {"mode": "best", "k": 2, "label": "best_k2"},
    {"mode": "best", "k": 2, "label": "best_k2_combined", "combine": True},
    {"mode": "worst_nonzero", "k": 1, "label": "worst_k1"},
    {"mode": "diverse_best", "k": 2, "label": "diverse_best_k2"},
    {
        "mode": "diverse_best",
        "k": 2,
        "label": "diverse_best_k2_combined",
        "combine": True,
    },
    {"mode": "milestone", "label": "milestone"},
    {"mode": "warm_start", "label": "warm_start"},
]

# Global ALE-Bench session duration for evaluation (1 day).
_EVAL_SESSION_DURATION_HOURS = 24.0


def _mode_spec_label(spec: dict[str, Any]) -> str:
    """Return the display label for a mode spec."""
    if "label" in spec:
        return str(spec["label"])
    mode = spec["mode"]
    if mode == "none" or mode == "milestone":
        return mode
    return f"{mode}_k{spec.get('k', 1)}"


def _normalize_mode_spec(item: Any) -> dict[str, Any] | None:
    """Normalize a mode config item into a spec dict.

    Accepts either a mode string or a dict with ``mode`` and optional ``k`` /
    ``label`` / ``combine``. Returns ``None`` if the mode is not supported.
    """
    if isinstance(item, str):
        item = {"mode": item.strip()}
    if not isinstance(item, dict):
        return None
    mode = item.get("mode")
    if mode not in HINT_MODES:
        return None
    spec = {"mode": mode, "k": int(item.get("k", 1))}
    if item.get("combine"):
        spec["combine"] = True
    if "label" in item:
        spec["label"] = str(item["label"])
    else:
        spec["label"] = _mode_spec_label(spec)
    return spec


class TeacherHintAblationAleBenchTrainer(PPOTrainer):
    """Evaluate a trained teacher on ALE-Bench with multiple hint types."""

    def __init__(self, config: TTTDDistillConfig):
        self.config = config
        self._closed = False
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

        # Environment is created per-teacher so each teacher is evaluated on its
        # own ALE-Bench problem (taken from multi_teacher[i].problem_id).
        self.env: Any | None = None
        self._score_type = "minimize"

        # Sampler / hint-sampler setup
        max_head_offpolicyness = getattr(config.rollout, "max_head_offpolicyness", 2)
        self._max_version_history = max_head_offpolicyness + 1
        self.is_sync_mode = max_head_offpolicyness == 0
        if self.is_sync_mode and config.sampler.lazy_puct_sampling:
            config.sampler.lazy_puct_sampling = False

        # Build evaluation specs for one or multiple teachers.
        self._eval_teacher_specs = self._build_teacher_specs(
            config, self._max_version_history
        )
        self._teacher_path = self._eval_teacher_specs[0]["lora_path"]
        self.hint_sampler = self._eval_teacher_specs[0]["hint_sampler"]
        logger.info(
            f"[Eval] Will evaluate {len(self._eval_teacher_specs)} teacher(s): "
            f"{[s['label'] for s in self._eval_teacher_specs]}"
        )

        # Create a single fresh sampler for actor version sync. The actual
        # per-problem envs are created lazily in run().
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
            max_version_history=self._max_version_history,
            sampling_strategy="puct",
            problem_id=getattr(config.sampler, "problem_id", ""),
        )
        logger.info(
            f"[EvalSampler] Fresh sampler created with {len(self.sampler._states)} states"
        )

        # Create actor
        self.actor = self._create_tttd_actor(config.actor)
        self.ref = None

        # Initialize engines and rollouts
        self._initialize_engines()
        self.actor.connect_sampler(self.sampler)
        self.rollout = self._init_rollout(config.rollout, is_eval=False)
        self.eval_rollout = self._init_rollout(config.rollout, is_eval=True)
        self._setup_weight_update_meta()
        self._setup_stats_logger()
        self._workflow_kwargs = {}

    def _get_score_type(self, env: Any) -> str:
        """Return 'minimize' or 'maximize' for the ALE-Bench env."""
        if hasattr(env, "problem") and hasattr(env.problem, "metadata"):
            return str(env.problem.metadata.score_type.value)
        if hasattr(env, "maximize"):
            return "maximize" if env.maximize else "minimize"
        return "minimize"

    def _create_hint_sampler(
        self,
        config: TTTDDistillConfig,
        problem_id: str,
        sampler_checkpoint: str,
        max_version_history: int,
    ) -> Any:
        """Create and load a hint sampler for a specific problem."""
        teacher_sampler_config = copy.deepcopy(config.sampler)
        teacher_sampler_config.problem_id = problem_id
        if sampler_checkpoint:
            teacher_sampler_config.checkpoint_dir = sampler_checkpoint

        hint_sampler = create_sampler_from_config(
            config=teacher_sampler_config,
            env_type=getattr(config.sampler, "env_type", "ale_bench"),
            max_version_history=max_version_history,
        )

        if sampler_checkpoint:
            latest_step = _find_latest_sampler_step(
                sampler_checkpoint,
                getattr(config.sampler, "type", "puct"),
            )
            if latest_step is not None:
                logger.info(
                    f"[HintSampler] Loading latest checkpoint at step {latest_step} "
                    f"from {sampler_checkpoint}"
                )
                hint_sampler._load(latest_step)
                hint_sampler._current_step = 0
            else:
                logger.warning(
                    f"[HintSampler] No checkpoint found in {sampler_checkpoint}. "
                    f"Using fresh sampler state."
                )

        logger.info(
            f"[HintSampler] Loaded {len(hint_sampler._states)} states, "
            f"T={hint_sampler._T}"
        )
        return hint_sampler

    def _create_env_for_problem(self, problem_id: str) -> Any:
        """Create an ALE-Bench environment for the given problem ID.

        The env creates/reuses a globally cached ALE-Bench session via
        ``AleBenchEnv``. We force a 24-hour session duration so that all eval
        envs for the same problem share exactly one session.
        """
        config = self.config
        env_kwargs = dict(
            problem_id=problem_id,
            lite_version=getattr(config, "ale_bench_eval_lite_version", False),
            eval_timeout=getattr(config.sampler, "eval_timeout", 600),
            log_dir=config.saver.fileroot,
            num_cpus=getattr(
                config,
                "ale_bench_eval_num_workers",
                getattr(config.sampler, "num_cpus", 2),
            ),
            reward_scale=getattr(config.sampler, "reward_scale", None),
            session_duration_seconds=int(_EVAL_SESSION_DURATION_HOURS * 3600),
        )
        return AleBenchEnv(**env_kwargs)

    def _load_milestone_hints(
        self, lora_path: str, sampler_checkpoint: str
    ) -> dict | None:
        """Try to load milestone hints JSON for a teacher.

        Searches, in order:
        1. ``<sampler_checkpoint>/milestone_hints.json``
        2. ``<lora_path>/milestone_hints.json``
        3. ``self.config.milestone_hints``
        """
        config = self.config
        candidates = [
            os.path.join(sampler_checkpoint, "milestone_hints.json"),
            os.path.join(lora_path, "milestone_hints.json"),
        ]
        if getattr(config, "milestone_hints", None):
            candidates.append(config.milestone_hints)

        for path in candidates:
            if path and os.path.isfile(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    logger.info(
                        f"[MilestoneHints] Loaded {len(data.get('paths', []))} paths "
                        f"from {path}"
                    )
                    return data
                except Exception as e:
                    logger.warning(f"[MilestoneHints] Failed to load {path}: {e}")
        return None

    def _build_teacher_specs(
        self, config: TTTDDistillConfig, max_version_history: int
    ) -> list[dict[str, Any]]:
        """Build a list of teacher evaluation specs from ``config.multi_teacher``.

        Falls back to the legacy single-teacher fields if ``multi_teacher`` is empty.
        """
        multi_teacher = list(config.multi_teacher) if config.multi_teacher else []
        if not multi_teacher:
            teacher_path = getattr(config, "teacher_lora_path", None)
            if teacher_path is None and config.teacher is not None:
                teacher_path = config.teacher.path
            sampler_checkpoint = config.teacher_sampler_checkpoint
            problem_id = getattr(config.sampler, "problem_id", "")
            if not teacher_path:
                raise ValueError(
                    "config.multi_teacher or teacher_lora_path/teacher.path must be provided."
                )
            multi_teacher = [
                type(
                    "TeacherSpec",
                    (),
                    {
                        "problem_id": problem_id,
                        "lora_path": teacher_path,
                        "sampler_checkpoint": sampler_checkpoint,
                    },
                )()
            ]

        specs: list[dict[str, Any]] = []
        for i, entry in enumerate(multi_teacher):
            problem_id = getattr(entry, "problem_id", "") or getattr(
                config.sampler, "problem_id", ""
            )
            lora_path = getattr(entry, "lora_path", "")
            sampler_ckpt = getattr(entry, "sampler_checkpoint", "")
            if not lora_path:
                raise ValueError(f"multi_teacher[{i}] missing lora_path")
            label = (
                getattr(entry, "label", None)
                or problem_id
                or os.path.basename(os.path.normpath(lora_path))
                or f"teacher_{i}"
            )
            hint_sampler = self._create_hint_sampler(
                config, problem_id, sampler_ckpt, max_version_history
            )
            milestone_hints = self._load_milestone_hints(lora_path, sampler_ckpt)
            specs.append(
                {
                    "label": label,
                    "problem_id": problem_id,
                    "lora_path": lora_path,
                    "sampler_checkpoint": sampler_ckpt,
                    "hint_sampler": hint_sampler,
                    "milestone_hints": milestone_hints,
                }
            )
        return specs

    def _hint_code_fence(self) -> str:
        """Return the Markdown code-fence language for hint code blocks."""
        env = getattr(self, "env", None)
        language = (
            getattr(env, "code_language", "cpp20").lower()
            if env is not None
            else "cpp20"
        )
        if language.startswith("cpp") or language.startswith("c++"):
            return "cpp"
        if language.startswith("py"):
            return "python"
        return language

    def _display_score(self, state: Any) -> tuple[float | None, str]:
        """Return (raw_score, label) for a state, handling AC1 vs ALE-Bench."""
        raw = getattr(state, "raw_score", None)
        if raw is None and state.value is not None:
            # AC1 uses value = -raw_score; ALE-Bench stores raw_score directly.
            raw = -state.value
        label = (
            "average raw score per public case"
            if hasattr(state, "parent_raw_scores")
            else "score"
        )
        return (raw, label) if raw is not None else (None, label)

    def _log_hint_state_info(self, problem_id: str, label: str, state: Any) -> None:
        """Log a concise summary of a sampled hint state."""
        code = getattr(state, "code", None) or ""
        score, score_label = self._display_score(state)
        preview = code[:200].replace("\n", " ") if code else ""
        logger.info(
            f"[Hint][{problem_id}] {label}: code_len={len(code)}, "
            f"{score_label}={score if score is not None else 'n/a'}, "
            f"preview={preview!r}"
        )

    def _select_warm_start_state(self, sampler: Any) -> Any | None:
        """Select a mediocre accepted state to use as a warm-start hint.

        We sort by ``state.value`` (PUCT's normalized score, higher = better)
        and pick the state at ``config.eval_warm_start_percentile``.  This
        gives a working but clearly suboptimal starting solution, regardless
        of whether the underlying problem is a minimization or maximization.
        """
        states = getattr(sampler, "_states", [])
        candidates = [
            s
            for s in states
            if getattr(s, "code", None)
            and s.code.strip()
            and getattr(s, "value", None) is not None
            and s.value > 0
        ]
        if not candidates:
            # Fallback: states with a positive raw score (ALE-Bench specific).
            candidates = [
                s
                for s in states
                if getattr(s, "code", None)
                and s.code.strip()
                and getattr(s, "raw_score", None) is not None
                and s.raw_score > 0
            ]
        if not candidates:
            # Last resort: any state that has code.
            candidates = [
                s for s in states if getattr(s, "code", None) and s.code.strip()
            ]
        if not candidates:
            logger.warning("[WarmStart] No eligible warm-start state found")
            return None

        candidates.sort(key=lambda s: float(s.value))
        percentile = float(getattr(self.config, "eval_warm_start_percentile", 0.25))
        idx = min(int(len(candidates) * percentile), len(candidates) - 1)
        selected = candidates[idx]
        logger.info(
            f"[WarmStart] Selected state at percentile {percentile:.2f} "
            f"(idx={idx}/{len(candidates)}, value={selected.value:.6f}, "
            f"raw_score={getattr(selected, 'raw_score', None)})"
        )
        return selected

    def _build_warm_start_hint(self, state: Any) -> str:
        """Build a neutral hint that asks the model to improve an existing solution."""
        fence = self._hint_code_fence()
        code = state.code.strip()
        score, score_label = self._display_score(state)

        lines = [
            "[Existing Solution]",
            "Here is an existing C++20 solution for this problem.",
            f"```{fence}\n{code}\n```",
        ]
        if score is not None:
            lines.append(f"This solution achieves a {score_label} of {score:.6f}.")
        lines.append("Improve it to achieve a better score.\n")
        return "\n\n".join([""] + lines)

    def _extract_milestones_for_spec(self, spec: dict[str, Any]) -> list[dict]:
        """Extract milestone paths from a teacher spec's hint sampler.

        Caches the result in the spec so repeated modes do not recompute.
        """
        cached = spec.get("_extracted_milestone_paths")
        if cached is not None:
            return cached

        config = self.config
        paths = spec["hint_sampler"].extract_milestone_paths(
            n_paths=getattr(config, "eval_milestone_n_paths", 10),
            n_milestones=getattr(config, "eval_milestone_n_milestones", 4),
            min_improvement=getattr(config, "eval_milestone_min_improvement", 0.001),
        )
        spec["_extracted_milestone_paths"] = paths
        return paths

    def _extract_milestones_for_current_teacher(self) -> list[dict]:
        """Extract milestone paths from the current teacher's hint sampler."""
        return self._extract_milestones_for_spec(self._current_teacher_spec)

    def _get_prompt_framing_text(self, study_object: str = "reference approach") -> str:
        """Return the final instruction for a hint based on config framing.

        ``config.eval_hint_prompt_framing`` controls how strongly the model is
        pushed to generalize the reference:

        - ``generalize`` (default): identify principles and write fresh code.
        - ``minimal``: just ask for a solution without referencing the hints.
        """
        framing = getattr(self.config, "eval_hint_prompt_framing", "generalize")
        if framing == "minimal":
            return (
                "Now generate your own independent C++20 solution for this problem.\n"
            )
        # Default: generalize
        return (
            f"Study the {study_object} above. Identify the underlying principles, "
            f"and try to generalize them. Then generate your own independent C++20 "
            f"solution. Do not copy it verbatim; write a fresh implementation based "
            f"on your generalized understanding.\n"
        )

    def _build_single_hint(self, privileged_state: Any) -> str:
        """Build a short hint from a single privileged state."""
        fence = self._hint_code_fence()
        hint_parts = []
        if privileged_state.code and privileged_state.code.strip():
            hint_parts.append(f"```{fence}\n{privileged_state.code.strip()}\n```")
        score, score_label = self._display_score(privileged_state)
        if score is not None:
            hint_parts.append(
                f"This reference approach achieves a {score_label} of {score:.6f}."
            )
        hint_text = "\n".join(hint_parts)
        return (
            f"\n\n[Reference Approach]\n{hint_text}\n\n"
            + self._get_prompt_framing_text("reference approach")
        )

    def _build_best_worst_hint(self, best_states: Any, worst_states: Any) -> str:
        """Build a combined hint showing strong and weak reference approaches."""
        fence = self._hint_code_fence()
        hint_parts: list[str] = []

        if not isinstance(best_states, (list, tuple)):
            best_states = [best_states]
        if not isinstance(worst_states, (list, tuple)):
            worst_states = [worst_states]

        if best_states:
            hint_parts.append(
                f"Below are {len(best_states)} strong reference approach(es):"
            )
            for idx, state in enumerate(best_states, start=1):
                if state.code and state.code.strip():
                    hint_parts.append(f"Strong example {idx}:")
                    hint_parts.append(f"```{fence}\n{state.code.strip()}\n```")
                score, score_label = self._display_score(state)
                if score is not None:
                    hint_parts.append(
                        f"This approach achieves a {score_label} of {score:.6f}."
                    )
                hint_parts.append("")

        if worst_states:
            hint_parts.append(
                f"\nBelow are {len(worst_states)} weaker but still valid reference approach(es):"
            )
            for idx, state in enumerate(worst_states, start=1):
                if state.code and state.code.strip():
                    hint_parts.append(f"Weak example {idx}:")
                    hint_parts.append(f"```{fence}\n{state.code.strip()}\n```")
                score, score_label = self._display_score(state)
                if score is not None:
                    hint_parts.append(
                        f"This approach achieves a {score_label} of {score:.6f}."
                    )
                hint_parts.append("")

        hint_text = "\n".join(hint_parts).strip()
        return (
            f"\n\n[Reference Approaches]\n{hint_text}\n\n"
            + self._get_prompt_framing_text("approaches")
        )

    def _build_multi_state_hint(
        self,
        states: list[Any],
        title: str = "Reference Approaches",
        preamble: str = "Below are reference approaches:",
        example_label: str = "Reference example",
    ) -> str:
        """Build a hint that shows multiple states of the same type together."""
        fence = self._hint_code_fence()
        hint_parts: list[str] = []
        hint_parts.append(preamble)
        for idx, state in enumerate(states, start=1):
            if state.code and state.code.strip():
                hint_parts.append(f"{example_label} {idx}:")
                hint_parts.append(f"```{fence}\n{state.code.strip()}\n```")
            score, score_label = self._display_score(state)
            if score is not None:
                hint_parts.append(
                    f"This approach achieves a {score_label} of {score:.6f}."
                )
            hint_parts.append("")

        hint_text = "\n".join(hint_parts).strip()
        return f"\n\n[{title}]\n{hint_text}\n\n" + self._get_prompt_framing_text(
            "approaches"
        )

    def _build_breakthrough_pair_hint(self, parent_state: Any, child_state: Any) -> str:
        """Build a simple breakthrough hint showing a single parent -> child pair."""
        fence = self._hint_code_fence()
        hint_parts: list[str] = []

        parent_score, parent_label = self._display_score(parent_state)
        if parent_score is not None:
            hint_parts.append(
                f"A previous approach achieved {parent_label} {parent_score:.6f}."
            )

        child_score, child_label = self._display_score(child_state)
        if child_score is not None:
            improvement = child_score - (
                parent_score if parent_score is not None else 0
            )
            hint_parts.append(
                f"A breakthrough approach achieves {child_label} {child_score:.6f} "
                f"(improvement: +{improvement:.6f})."
            )

        if child_state.code and child_state.code.strip():
            code = child_state.code.strip()
            max_code_len = 4000
            if len(code) > max_code_len:
                code = code[:max_code_len] + "\n... (truncated)"
            hint_parts.append(f"```{fence}\n{code}\n```")

        hint_text = "\n".join(hint_parts)
        return (
            f"\n\n[Breakthrough Transition]\n{hint_text}\n\n"
            + self._get_prompt_framing_text("transition")
        )

    def _build_milestone_hint(self, milestones: list[dict]) -> str:
        """Build a whole-path milestone hint from a list of milestone dicts."""
        if not milestones:
            return ""
        fence = self._hint_code_fence()
        lines = ["\n=== Strategy Evolution Hints ==="]
        lines.append("Below are key phases discovered during search.\n")
        lines.append(self._get_prompt_framing_text("phases"))
        for i, ms in enumerate(milestones):
            phase_label = (
                ["Baseline", "Phase 1", "Phase 2", "Phase 3"][i]
                if i < 4
                else f"Phase {i}"
            )
            value = ms.get("value")
            raw_score = ms.get("raw_score")
            score_str = ""
            if raw_score is not None:
                score_str = f" (raw_score={raw_score:.6f})"
            elif value is not None:
                score_str = f" (value={value:.6f})"
            lines.append(f"--- {phase_label}{score_str} ---")
            code = ms.get("code", "")
            if code:
                code = code.strip()
                if code.startswith("```"):
                    code = code.strip("`").strip()
                    if code.startswith("python"):
                        code = code[6:].strip()
                lines.append(f"```{fence}\n{code}\n```")
            lines.append("")
        lines.append("=== End Hints ===\n")
        return "\n".join(lines)

    def _build_hint_text(self, label: str, payload: Any) -> str:
        """Build hint text from a (label, payload) tuple as returned by get_hint_states."""
        if (
            label == "breakthrough"
            and isinstance(payload, (tuple, list))
            and len(payload) == 2
        ):
            return self._build_breakthrough_pair_hint(payload[0], payload[1])
        if (
            label in ("best_worst_combined", "diverse_best_worst")
            and isinstance(payload, (tuple, list))
            and len(payload) == 2
        ):
            return self._build_best_worst_hint(payload[0], payload[1])
        return self._build_single_hint(payload)

    def _build_multi_problem_hint_fn(
        self,
        mode_spec: dict[str, Any],
        state_to_spec: dict[str, dict[str, Any]],
    ) -> Callable:
        """Build a per-state hint function for ``mode_spec`` across problems.

        Each initial state (keyed by ``state.id``) is mapped to its teacher spec
        so that the correct hint sampler / milestone hints are used for that
        problem. The hint cycles through the available pool for the state on
        each call, matching the behavior of the single-problem eval.
        """
        config = self.config
        mode = mode_spec["mode"]

        if mode == "none":

            def hint_fn(state):
                return ""

            return hint_fn

        state_hint_data: dict[str, Any] = {}
        counters: dict[str, int] = {}

        for sid, spec in state_to_spec.items():
            counters[sid] = 0

            if mode == "warm_start":
                # Cache the selected warm-start state in the spec so the base
                # model baseline and the teacher see exactly the same starting
                # solution for each problem.
                warm_state = spec.get("_warm_start_state")
                if warm_state is None:
                    warm_state = self._select_warm_start_state(spec["hint_sampler"])
                    spec["_warm_start_state"] = warm_state
                if warm_state is None:
                    state_hint_data[sid] = {"type": "fixed", "hint_text": ""}
                else:
                    state_hint_data[sid] = {
                        "type": "fixed",
                        "hint_text": self._build_warm_start_hint(warm_state),
                    }
                continue

            if mode == "milestone" or (
                mode == "breakthrough" and spec.get("milestone_hints")
            ):
                milestone_hints = spec.get("milestone_hints") or {}
                milestone_paths = milestone_hints.get("paths", [])
                if not milestone_paths:
                    milestone_paths = self._extract_milestones_for_spec(spec)
                if self.actor.rank == 0:
                    logger.info(
                        f"[Hint][{mode}] problem={spec['problem_id']} "
                        f"using {len(milestone_paths)} milestone path(s)"
                    )
                state_hint_data[sid] = {
                    "type": "milestone",
                    "paths": milestone_paths,
                }
            else:
                k = mode_spec.get("k", getattr(config, "eval_hint_k", 1))
                min_improvement = getattr(config, "eval_hint_min_improvement", 0.001)
                deterministic = getattr(config, "eval_hint_deterministic", False)
                hint_pool = spec["hint_sampler"].get_hint_states(
                    mode=mode,
                    k=k,
                    min_improvement=min_improvement,
                    deterministic=deterministic,
                )
                if self.actor.rank == 0:
                    logger.info(
                        f"[Hint][{mode}] problem={spec['problem_id']} "
                        f"sampled {len(hint_pool)} hint state(s) "
                        f"(k={k}, min_improvement={min_improvement}, "
                        f"deterministic={deterministic})"
                    )
                    for hint_label, payload in hint_pool:
                        if isinstance(payload, (list, tuple)):
                            for i, st in enumerate(payload):
                                self._log_hint_state_info(
                                    spec["problem_id"], f"{hint_label}[{i}]", st
                                )
                        else:
                            self._log_hint_state_info(
                                spec["problem_id"], hint_label, payload
                            )

                # When ``combine=True`` and the pool has multiple states of the
                # same type, show them all together in one prompt instead of
                # rotating across candidates.
                if mode_spec.get("combine") and len(hint_pool) > 1:
                    states = [payload for _label, payload in hint_pool]
                    if mode == "best":
                        combined = self._build_multi_state_hint(
                            states,
                            title="Strong Reference Approaches",
                            preamble=f"Below are {len(states)} strong reference approaches:",
                            example_label="Strong example",
                        )
                    elif mode == "worst_nonzero":
                        combined = self._build_multi_state_hint(
                            states,
                            title="Weaker Reference Approaches",
                            preamble=(
                                f"Below are {len(states)} weaker but still valid "
                                f"reference approaches:"
                            ),
                            example_label="Weak example",
                        )
                    elif mode == "diverse_best":
                        combined = self._build_multi_state_hint(
                            states,
                            title="Diverse Strong Reference Approaches",
                            preamble=(
                                f"Below are {len(states)} strong reference approaches "
                                f"from different search branches:"
                            ),
                            example_label="Diverse strong example",
                        )
                    elif mode == "diverse_worst_nonzero":
                        combined = self._build_multi_state_hint(
                            states,
                            title="Diverse Weak Reference Approaches",
                            preamble=(
                                f"Below are {len(states)} weak but still valid "
                                f"reference approaches from different search branches:"
                            ),
                            example_label="Diverse weak example",
                        )
                    else:
                        # Fallback for any other mode with combine=True.
                        combined = self._build_multi_state_hint(states)
                    state_hint_data[sid] = {"type": "fixed", "hint_text": combined}
                else:
                    state_hint_data[sid] = {"type": "pool", "pool": hint_pool}

        def hint_fn(state):
            sid = getattr(state, "id", "")
            data = state_hint_data.get(sid)
            if data is None:
                return ""

            counter = counters[sid]
            counters[sid] = counter + 1

            if data["type"] == "fixed":
                return data["hint_text"]

            if data["type"] == "milestone":
                paths = data["paths"]
                if not paths:
                    return ""
                path = paths[counter % len(paths)]
                return self._build_milestone_hint(path.get("milestones", []))

            pool = data["pool"]
            if not pool:
                return ""
            label, payload = pool[counter % len(pool)]
            return self._build_hint_text(label, payload)

        return hint_fn

    def _create_tttd_actor(self, actor_config):
        actor = TTTDActor(config=actor_config)
        actor.create_process_group(parallel_strategy=self.allocation_mode.train)
        return actor

    def _initialize_engines(self):
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=self.config.eval_batch_size,
            train_batch_size=self.config.eval_batch_size,
        )
        self.actor.initialize(
            addr=None, ft_spec=ft_spec, alloc_mode=self.allocation_mode, role="actor"
        )

    def _setup_weight_update_meta(self):
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
        config = self.config
        ft_spec = FinetuneSpec(
            total_train_epochs=1,
            dataset_size=config.eval_batch_size,
            train_batch_size=config.eval_batch_size,
        )
        self.stats_logger = StatsLogger(config, ft_spec)

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
        from safetensors.torch import load_file
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        adapter_path = os.path.join(path, "adapter_model.safetensors")
        if not os.path.isfile(adapter_path):
            raise ValueError(f"LoRA adapter not found at {adapter_path}")
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

    def _normalize_rollout_batch(self, rollout_batch) -> dict[str, Any]:
        """Ensure rollout_batch is a single dict for downstream processing.

        Mirrors ``TTTDDistillTrainer._normalize_rollout_batch`` but preserves
        ``_problem_ids`` and ``_metadata`` needed for multi-problem evaluation.
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
        if problem_ids:
            batched["_problem_ids"] = problem_ids
        if metadata_list:
            batched["_metadata"] = metadata_list
        return batched

    def _run_single_mode(self, mode_spec: dict[str, Any]) -> dict[str, Any]:
        """Run ALE-Bench public→private evaluation for one hint mode."""
        config = self.config
        mode = mode_spec["mode"]
        mode_label = mode_spec["label"]
        problem_id = self._current_teacher_spec["problem_id"]
        logger.info("=" * 60)
        logger.info(f"[Eval] Starting mode: {mode_label} | problem: {problem_id}")
        logger.info("=" * 60)

        if self.actor.rank != 0:
            logger.info(
                f"[Eval][{mode_label}] Rank {self.actor.rank} skipping candidate generation"
            )
            return {
                "mode": mode,
                "mode_label": mode_label,
                "problem_id": problem_id,
                "skipped": True,
            }

        n_candidates = getattr(config, "ale_bench_eval_n_candidates", 15)

        # Create initial state early so we can bind the hint function to it.
        state = create_initial_state_ale_bench(problem_id=problem_id)
        state_to_spec = {state.id: self._current_teacher_spec}
        hint_fn = self._build_multi_problem_hint_fn(mode_spec, state_to_spec)

        # Public reward function runs inside the rollout worker.
        public_reward_fn = functools.partial(
            ale_bench_public_reward_fn,
            lite_version=getattr(config, "ale_bench_eval_lite_version", False),
            session_duration_hours=_EVAL_SESSION_DURATION_HOURS,
            ale_bench_num_workers=config.ale_bench_eval_num_workers,
        )

        base_eval_gconfig = config.eval_gconfig or config.gconfig
        eval_gconfig = base_eval_gconfig.new(n_samples=n_candidates)

        # Public eval can be CPU/Docker heavy; allow concurrent reward workers so
        # multiple problems' public evaluations overlap. Cap at n_parallel_problems
        # to avoid oversubscribing the CPU pool used by each ALE-Bench session.
        max_reward_workers = max(1, config.ale_bench_eval_n_parallel_problems)
        workflow_kwargs = dict(
            env=self.env,
            problem_envs={problem_id: self.env},
            gconfig=eval_gconfig,
            tokenizer=self.tokenizer,
            enable_thinking=getattr(config, "enable_thinking", False),
            max_prompt_thinking_tokens=getattr(
                config, "max_prompt_thinking_tokens", 26000
            ),
            sampler=self.sampler,
            batch_size=1,
            group_size=n_candidates,
            lazy_sampling=False,
            reward_fn=public_reward_fn,
            max_reward_workers=max_reward_workers,
            hint_fn=hint_fn,
            hint_placement="append",
            distill_mode=True,
        )

        prompt = (
            self.env.get_prompt_distill(state)
            if hasattr(self.env, "get_prompt_distill")
            else self.env.get_prompt(state)
        )
        data = {
            "prompt": prompt,
            "state_id": state.id,
            "_state_obj": state,
            "_problem_id": problem_id,
        }

        logger.info(
            f"[AleBenchEval-{mode_label}] Generating {n_candidates} candidates for {problem_id}"
        )
        task_id = self.eval_rollout.submit(
            data,
            MultiProblemTTTDiscoverWorkflowV2,
            workflow_kwargs=workflow_kwargs,
            group_size=n_candidates,
            is_eval=True,
        )
        task_metadata = {
            task_id: {
                "problem_id": problem_id,
                "state_id": state.id,
                "model": self._current_teacher_spec["label"],
                "mode": mode,
                "mode_label": mode_label,
                "k": mode_spec.get("k", 1),
            }
        }

        results = self.eval_rollout.wait(1, timeout=None)
        logger.info(f"[AleBenchEval-{mode_label}] Got {len(results)} rollout results")
        batch = self._normalize_rollout_batch(results)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]
        metadata_list = batch.get("_metadata", [])

        prompt_lens = (attention_mask & (loss_mask == 0)).sum(dim=1).cpu().numpy()
        comp_lens = loss_mask.sum(dim=1).cpu().numpy()

        candidates: list[str] = []
        public_results_by_problem: dict[str, list[dict[str, Any]]] = {problem_id: []}
        for i in range(input_ids.shape[0]):
            prompt_len = int(prompt_lens[i])
            comp_len = int(comp_lens[i])
            completion_ids = (
                input_ids[i, prompt_len : prompt_len + comp_len].cpu().tolist()
            )
            completion_text = self.tokenizer.decode(completion_ids)
            code = self.env.extract_code(completion_text)
            candidates.append(code if code is not None else "")

            md = metadata_list[i] if i < len(metadata_list) else {}
            public_info = md.get("metadata", md) if isinstance(md, dict) else {}
            public_result = {
                "idx": len(candidates) - 1,
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
            public_results_by_problem[problem_id].append(public_result)

        total_count = len(candidates)
        logger.info(
            f"[AleBenchEval-{mode_label}] Generated {total_count} candidate codes for {problem_id}"
        )

        # Run private evaluation for both median and best-public selection.
        problem_sessions = {}
        if hasattr(self.env, "session") and self.env.session is not None:
            problem_sessions[problem_id] = self.env.session

        problem_score_types = {problem_id: self._score_type}
        training_problem_ids = {problem_id}

        def _run_private_for_selection(sel_method: str) -> dict[str, Any]:
            logger.info(
                f"[AleBenchEval-{mode_label}] Running private eval for {problem_id} "
                f"with selection={sel_method}"
            )
            local_results = evaluate_problem_subset_with_public_scores(
                [problem_id],
                public_results_by_problem,
                lite_version=getattr(config, "ale_bench_eval_lite_version", False),
                session_duration_hours=_EVAL_SESSION_DURATION_HOURS,
                ale_bench_num_workers=config.ale_bench_eval_num_workers,
                n_parallel_problems=config.ale_bench_eval_n_parallel_problems,
                problem_sessions=problem_sessions,
                selection_method=sel_method,
                problem_score_types=problem_score_types,
            )
            combined = combine_ale_bench_results(
                local_results,
                [problem_id],
                training_problem_ids,
            )
            combined["selection_method"] = sel_method
            return combined

        # If we are evaluating all candidates privately anyway, derive median and
        # best_public from the same pool instead of running two extra private evals.
        run_all_cands = getattr(config, "eval_private_eval_all_candidates", False)
        all_candidates_output: dict[str, Any] | None = None
        if run_all_cands:
            logger.info(
                f"[AleBenchEval-{mode_label}] Running private eval for all "
                f"{n_candidates} candidates on {problem_id}"
            )
            local_results = evaluate_all_candidates_private(
                [problem_id],
                public_results_by_problem,
                lite_version=getattr(config, "ale_bench_eval_lite_version", False),
                session_duration_hours=_EVAL_SESSION_DURATION_HOURS,
                ale_bench_num_workers=config.ale_bench_eval_num_workers,
                n_parallel_problems=config.ale_bench_eval_n_parallel_problems,
                problem_sessions=problem_sessions,
                problem_score_types=problem_score_types,
            )
            result = local_results[0]
            all_candidates_output = {
                "problem_id": problem_id,
                "score_type": self._score_type,
                "private_results": result.get("private_results", []),
                "private_stats": result.get("private_stats", {}),
            }
            derived = derive_selection_outputs_from_all_candidates(
                public_results_by_problem,
                local_results,
                [problem_id],
                training_problem_ids,
                ["median", "best_public"],
                problem_score_types,
            )
            median_output = derived["median"]
            best_public_output = derived["best_public"]
        else:
            median_output = _run_private_for_selection("median")
            best_public_output = _run_private_for_selection("best_public")

        output = {
            "mode": mode,
            "mode_label": mode_label,
            "k": mode_spec.get("k", 1),
            "problem_id": problem_id,
            "lora_path": self._teacher_path,
            "n_candidates": n_candidates,
            "lite_version": getattr(config, "ale_bench_eval_lite_version", False),
            "session_duration_hours": _EVAL_SESSION_DURATION_HOURS,
            "ale_bench_num_workers": config.ale_bench_eval_num_workers,
            "n_parallel_problems": config.ale_bench_eval_n_parallel_problems,
            "task_metadata": task_metadata,
            "median": median_output,
            "best_public": best_public_output,
            "all_candidates": all_candidates_output,
        }

        median_avg = median_output.get("average_all", {})
        best_public_avg = best_public_output.get("average_all", {})
        logger.info("=" * 60)
        logger.info(f"[Eval][{mode_label}] RESULTS for {problem_id}")
        logger.info(
            f"[Eval][{mode_label}] median       abs={median_avg.get('absolute_score', 0.0):.4f} "
            f"perf={median_avg.get('performance', 0.0):.4f}"
        )
        logger.info(
            f"[Eval][{mode_label}] best_public  abs={best_public_avg.get('absolute_score', 0.0):.4f} "
            f"perf={best_public_avg.get('performance', 0.0):.4f}"
        )
        if all_candidates_output is not None:
            stats = all_candidates_output["private_stats"]
            abs_stats = stats.get("absolute_score", {})
            logger.info(
                f"[Eval][{mode_label}] all_cands    "
                f"mean={abs_stats.get('mean', 0.0):.4f} "
                f"median={abs_stats.get('median', 0.0):.4f} "
                f"std={abs_stats.get('std', 0.0):.4f} "
                f"min={abs_stats.get('min', 0.0):.4f} "
                f"max={abs_stats.get('max', 0.0):.4f} "
                f"accepted={stats.get('count_accepted', 0)}/{stats.get('count', 0)}"
            )
        logger.info("=" * 60)

        return output

    def _run_multi_problem_mode_eval(
        self,
        mode_spec: dict[str, Any],
        problem_envs: dict[str, Any],
        spec_by_problem: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Run public→private ALE-Bench eval for one hint mode across problems."""
        config = self.config
        mode = mode_spec["mode"]
        mode_label = mode_spec["label"]
        problem_ids = sorted(problem_envs.keys())
        n_candidates = getattr(config, "ale_bench_eval_n_candidates", 15)
        lite_version = getattr(config, "ale_bench_eval_lite_version", False)
        fallback_env = next(iter(problem_envs.values()))

        # Fresh states for this mode so prior mode mutations don't leak through.
        states: list[Any] = []
        state_to_problem: dict[str, str] = {}
        state_to_spec: dict[str, dict[str, Any]] = {}
        for problem_id in problem_ids:
            state = create_initial_state_ale_bench(problem_id=problem_id)
            states.append(state)
            state_to_problem[state.id] = problem_id
            state_to_spec[state.id] = spec_by_problem[problem_id]
        hint_fn = self._build_multi_problem_hint_fn(mode_spec, state_to_spec)

        public_reward_fn = functools.partial(
            ale_bench_public_reward_fn,
            lite_version=lite_version,
            session_duration_hours=_EVAL_SESSION_DURATION_HOURS,
            ale_bench_num_workers=config.ale_bench_eval_num_workers,
        )

        base_eval_gconfig = config.eval_gconfig or config.gconfig
        eval_gconfig = base_eval_gconfig.new(n_samples=n_candidates)
        max_reward_workers = max(1, config.ale_bench_eval_n_parallel_problems)

        workflow_kwargs = dict(
            env=fallback_env,
            problem_envs=problem_envs,
            gconfig=eval_gconfig,
            tokenizer=self.tokenizer,
            enable_thinking=getattr(config, "enable_thinking", False),
            max_prompt_thinking_tokens=getattr(
                config, "max_prompt_thinking_tokens", 26000
            ),
            sampler=self.sampler,
            batch_size=1,
            group_size=n_candidates,
            lazy_sampling=False,
            reward_fn=public_reward_fn,
            max_reward_workers=max_reward_workers,
            hint_fn=hint_fn,
            hint_placement="append",
            distill_mode=True,
        )

        data_list: list[dict[str, Any]] = []
        for state in states:
            problem_id = state_to_problem[state.id]
            env = problem_envs[problem_id]
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

        logger.info(
            f"[Baseline-{mode_label}] Generating {n_candidates} candidates for "
            f"{len(data_list)} problems"
        )
        task_metadata: dict[int, dict[str, Any]] = {}
        for data in data_list:
            task_id = self.eval_rollout.submit(
                data,
                MultiProblemTTTDiscoverWorkflowV2,
                workflow_kwargs=workflow_kwargs,
                group_size=n_candidates,
                is_eval=True,
            )
            task_metadata[task_id] = {
                "problem_id": data.get("_problem_id", ""),
                "state_id": data.get("state_id", ""),
                "model": "baseline",
                "mode": mode,
                "mode_label": mode_label,
                "k": mode_spec.get("k", 1),
            }

        results = self.eval_rollout.wait(len(data_list), timeout=None)
        logger.info(f"[Baseline-{mode_label}] Got {len(results)} rollout results")
        batch = self._normalize_rollout_batch(results)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]
        problem_ids_from_batch = batch.get("_problem_ids", [])
        metadata_list = batch.get("_metadata", [])

        prompt_lens = (attention_mask & (loss_mask == 0)).sum(dim=1).cpu().numpy()
        comp_lens = loss_mask.sum(dim=1).cpu().numpy()

        public_results_by_problem: dict[str, list[dict[str, Any]]] = {
            pid: [] for pid in problem_ids
        }
        for i in range(input_ids.shape[0]):
            problem_id = (
                problem_ids_from_batch[i] if i < len(problem_ids_from_batch) else ""
            )
            env = problem_envs.get(problem_id)
            if env is None:
                continue
            prompt_len = int(prompt_lens[i])
            comp_len = int(comp_lens[i])
            completion_ids = (
                input_ids[i, prompt_len : prompt_len + comp_len].cpu().tolist()
            )
            completion_text = self.tokenizer.decode(completion_ids)
            code = env.extract_code(completion_text)

            md = metadata_list[i] if i < len(metadata_list) else {}
            public_info = md.get("metadata", md) if isinstance(md, dict) else {}
            public_result = {
                "idx": len(public_results_by_problem.get(problem_id, [])),
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

        total_count = sum(len(v) for v in public_results_by_problem.values())
        logger.info(
            f"[Baseline-{mode_label}] Generated {total_count} candidate codes across "
            f"{len(public_results_by_problem)} problems"
        )

        problem_sessions = {
            pid: env.session
            for pid, env in problem_envs.items()
            if hasattr(env, "session") and env.session is not None
        }
        problem_score_types = {
            pid: self._get_score_type(env) for pid, env in problem_envs.items()
        }
        training_problem_ids = set(problem_ids)

        def _run_private_for_selection(sel_method: str) -> dict[str, Any]:
            logger.info(
                f"[Baseline-{mode_label}] Running private eval with selection={sel_method}"
            )
            local_results = evaluate_problem_subset_with_public_scores(
                problem_ids,
                public_results_by_problem,
                lite_version=lite_version,
                session_duration_hours=_EVAL_SESSION_DURATION_HOURS,
                ale_bench_num_workers=config.ale_bench_eval_num_workers,
                n_parallel_problems=config.ale_bench_eval_n_parallel_problems,
                problem_sessions=problem_sessions,
                selection_method=sel_method,
                problem_score_types=problem_score_types,
            )
            combined = combine_ale_bench_results(
                local_results, problem_ids, training_problem_ids
            )
            combined["selection_method"] = sel_method
            return combined

        # If we are evaluating all candidates privately anyway, derive median and
        # best_public from the same pool instead of running two extra private evals.
        run_all_cands = getattr(config, "eval_private_eval_all_candidates", False)
        all_candidates_output: dict[str, Any] | None = None
        if run_all_cands:
            logger.info(
                f"[Baseline-{mode_label}] Running private eval for all "
                f"{n_candidates} candidates on {len(problem_ids)} problems"
            )
            local_results = evaluate_all_candidates_private(
                problem_ids,
                public_results_by_problem,
                lite_version=lite_version,
                session_duration_hours=_EVAL_SESSION_DURATION_HOURS,
                ale_bench_num_workers=config.ale_bench_eval_num_workers,
                n_parallel_problems=config.ale_bench_eval_n_parallel_problems,
                problem_sessions=problem_sessions,
                problem_score_types=problem_score_types,
            )
            all_candidates_output = {
                "problem_ids": problem_ids,
                "per_problem": local_results,
                "average_stats": average_all_candidates_stats(
                    [r.get("private_stats", {}) for r in local_results]
                ),
            }
            derived = derive_selection_outputs_from_all_candidates(
                public_results_by_problem,
                local_results,
                problem_ids,
                training_problem_ids,
                ["median", "best_public"],
                problem_score_types,
            )
            median_output = derived["median"]
            best_public_output = derived["best_public"]
        else:
            median_output = _run_private_for_selection("median")
            best_public_output = _run_private_for_selection("best_public")

        median_avg = median_output.get("average_all", {})
        best_public_avg = best_public_output.get("average_all", {})
        logger.info("=" * 60)
        logger.info(f"[Baseline-{mode_label}] RESULTS")
        logger.info(
            f"[Baseline-{mode_label}] median       abs={median_avg.get('absolute_score', 0.0):.4f} "
            f"perf={median_avg.get('performance', 0.0):.4f} "
            f"success={median_avg.get('count', 0)}/{len(problem_ids)}"
        )
        logger.info(
            f"[Baseline-{mode_label}] best_public  abs={best_public_avg.get('absolute_score', 0.0):.4f} "
            f"perf={best_public_avg.get('performance', 0.0):.4f} "
            f"success={best_public_avg.get('count', 0)}/{len(problem_ids)}"
        )
        if all_candidates_output is not None:
            avg_stats = all_candidates_output.get("average_stats", {})
            abs_stats = avg_stats.get("absolute_score", {})
            logger.info(
                f"[Baseline-{mode_label}] all_cands    "
                f"mean={abs_stats.get('mean', 0.0):.4f} "
                f"median={abs_stats.get('median', 0.0):.4f} "
                f"std={abs_stats.get('std', 0.0):.4f} "
                f"min={abs_stats.get('min', 0.0):.4f} "
                f"max={abs_stats.get('max', 0.0):.4f} "
                f"accepted={avg_stats.get('count_accepted', 0):.1f}/{avg_stats.get('count', 0):.1f}"
            )
        logger.info("=" * 60)

        return {
            "mode": mode,
            "mode_label": mode_label,
            "k": mode_spec.get("k", 1),
            "problem_ids": problem_ids,
            "n_candidates": n_candidates,
            "lite_version": lite_version,
            "session_duration_hours": _EVAL_SESSION_DURATION_HOURS,
            "ale_bench_num_workers": config.ale_bench_eval_num_workers,
            "n_parallel_problems": config.ale_bench_eval_n_parallel_problems,
            "task_metadata": task_metadata,
            "median": median_output,
            "best_public": best_public_output,
            "all_candidates": all_candidates_output,
        }

    def _run_baseline_eval(
        self, mode_specs: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Run the pure base model on all unique teacher problems with each mode.

        For every mode spec in ``mode_specs``, the base model (all LoRA weights
        zeroed out) generates candidates for every teacher problem using the
        hints from that problem's teacher sampler. Public and private
        evaluations are run for both median and best-public candidate selection.
        """
        config = self.config
        problem_ids = sorted({spec["problem_id"] for spec in self._eval_teacher_specs})
        if not problem_ids:
            logger.warning("[Baseline] No teacher problems found, skipping.")
            return None

        if self.actor.rank != 0:
            logger.info(f"[Baseline] Rank {self.actor.rank} skipping baseline eval")
            return {"skipped": True}

        logger.info(
            f"[Baseline] Evaluating base model on {len(problem_ids)} "
            f"teacher problem(s): {problem_ids}"
        )

        problem_envs: dict[str, Any] = {}
        for problem_id in problem_ids:
            problem_envs[problem_id] = self._create_env_for_problem(problem_id)
        self._baseline_envs = problem_envs

        spec_by_problem = {
            spec["problem_id"]: spec for spec in self._eval_teacher_specs
        }

        # Baseline measures the pure base model *with* hints.  The no-hint
        # ``none`` mode is skipped here because the base model's performance on
        # ALE-Bench without hints has already been evaluated separately; we only
        # need to compare how much hints help the base model vs. the teacher.
        baseline_mode_specs = [m for m in mode_specs if m["mode"] != "none"]
        results_by_mode: dict[str, dict[str, Any]] = {}
        for mode_spec in baseline_mode_specs:
            mode_label = mode_spec["label"]
            logger.info(f"[Baseline] Running mode: {mode_label}")
            results_by_mode[mode_label] = self._run_multi_problem_mode_eval(
                mode_spec,
                problem_envs,
                spec_by_problem,
            )

        return {
            "baseline_type": "pure_base_model",
            "lora_path": None,
            "problem_ids": problem_ids,
            "n_candidates": getattr(config, "ale_bench_eval_n_candidates", 15),
            "lite_version": getattr(config, "ale_bench_eval_lite_version", False),
            "session_duration_hours": _EVAL_SESSION_DURATION_HOURS,
            "ale_bench_num_workers": config.ale_bench_eval_num_workers,
            "n_parallel_problems": config.ale_bench_eval_n_parallel_problems,
            "results_by_mode": results_by_mode,
        }

    def _push_teacher_weights(self) -> None:
        """Push the currently loaded teacher weights to the inference engine."""
        logger.info("[Eval] Pushing weights to vLLM...")
        self.rollout.pause()
        self.eval_rollout.pause()
        versioned_meta = self.weight_update_meta.with_version(0)
        self.actor.update_weights(versioned_meta)
        self.actor.set_version(0)
        self.rollout.set_version(0)
        self.eval_rollout.set_version(0)
        if dist.is_initialized():
            dist.barrier(group=self.actor.cpu_group)
        current_platform.synchronize()
        self.rollout.resume()
        self.eval_rollout.resume()
        logger.info("[Eval] Weights pushed and rollouts resumed.")

    def run(self):
        """Run the full hint ablation across all teachers and save combined results.

        If ``config.eval_hint_framing_ablation`` is non-empty, the same set of
        modes is evaluated once per listed prompt framing (e.g.
        ``[generalize, minimal]``) and a separate result file is written for each
        framing, plus a combined comparison file.
        """
        config = self.config

        # Determine modes to run. Allow either plain strings or dicts with
        # ``mode``/``k``/``label``. Normalize to a list of spec dicts.
        raw_modes = getattr(config, "eval_hint_modes", DEFAULT_EVAL_HINT_MODES)
        if isinstance(raw_modes, str):
            raw_modes = [m.strip() for m in raw_modes.split(",")]
        mode_specs = [_normalize_mode_spec(m) for m in raw_modes]
        mode_specs = [m for m in mode_specs if m is not None]
        if not mode_specs:
            raise ValueError(f"No valid eval_hint_modes. Supported modes: {HINT_MODES}")
        logger.info(
            f"[Eval] Running hint ablation over modes: "
            f"{[m['label'] for m in mode_specs]}"
        )

        # Prompt framings to test. If an explicit ablation list is given, run all
        # of them in one shot; otherwise fall back to the single configured framing.
        framing_list = list(getattr(config, "eval_hint_framing_ablation", []))
        if not framing_list:
            framing_list = [getattr(config, "eval_hint_prompt_framing", "generalize")]
        run_baseline = getattr(config, "eval_hint_ablation_run_baseline", True)
        logger.info(
            f"[Eval] Running over prompt framings: {framing_list} "
            f"(baseline={'on' if run_baseline else 'off'})"
        )

        if self.actor.rank == 0:
            output_dir = os.path.join(
                config.saver.fileroot, config.experiment_name, config.trial_name
            )
            unique_problem_ids = sorted(
                {spec["problem_id"] for spec in self._eval_teacher_specs}
            )
            logger.info("=" * 60)
            logger.info("[Eval] Run summary")
            logger.info("=" * 60)
            logger.info(f"  Experiment:  {config.experiment_name}")
            logger.info(f"  Trial:       {config.trial_name}")
            logger.info(f"  Output dir:  {output_dir}")
            logger.info(f"  Model:       {config.actor.path}")
            logger.info(f"  Allocation:  {config.allocation_mode}")
            logger.info(f"  #Teachers:   {len(self._eval_teacher_specs)}")
            logger.info(f"  #Problems:   {len(unique_problem_ids)}")
            logger.info(f"  Problem IDs: {unique_problem_ids}")
            logger.info(f"  Modes:       {[m['label'] for m in mode_specs]}")
            logger.info(f"  Framings:    {framing_list}")
            logger.info(f"  Baseline:    {'on' if run_baseline else 'off'}")
            logger.info("[Eval] ALE-Bench eval config:")
            logger.info(
                f"    n_candidates:           "
                f"{getattr(config, 'ale_bench_eval_n_candidates', 15)}"
            )
            logger.info(
                f"    lite_version:           "
                f"{getattr(config, 'ale_bench_eval_lite_version', False)}"
            )
            logger.info(
                f"    num_workers:            {config.ale_bench_eval_num_workers}"
            )
            logger.info(
                f"    n_parallel_problems:    "
                f"{config.ale_bench_eval_n_parallel_problems}"
            )
            logger.info(f"    session_duration_hours: {_EVAL_SESSION_DURATION_HOURS}")
            logger.info(
                f"    eval_all_candidates:    "
                f"{getattr(config, 'eval_private_eval_all_candidates', False)}"
            )
            logger.info("[Eval] Teacher specs:")
            for spec in self._eval_teacher_specs:
                logger.info(
                    f"    - {spec['label']:20s} "
                    f"problem={spec['problem_id']} "
                    f"lora={spec['lora_path']}"
                )
            logger.info("=" * 60)

        all_framing_summaries: dict[str, dict[str, Any]] = {}

        for framing in framing_list:
            config.eval_hint_prompt_framing = framing
            logger.info("=" * 60)
            logger.info(f"[Eval] Prompt framing: {framing}")
            logger.info("=" * 60)

            all_teacher_results: dict[str, dict[str, Any]] = {}
            combined_task_metadata: dict[int, dict[str, Any]] = {}

            # Baseline evaluation on all unique teacher problems.
            baseline_results = None
            if run_baseline:
                logger.info("[Baseline] Using pure base model (zero LoRA weights)")
                self._zero_lora_weights(self.actor)
                self._push_teacher_weights()
                baseline_results = self._run_baseline_eval(mode_specs)
                if baseline_results and not baseline_results.get("skipped"):
                    for mode_result in baseline_results.get(
                        "results_by_mode", {}
                    ).values():
                        combined_task_metadata.update(
                            mode_result.get("task_metadata", {})
                        )
                if dist.is_initialized():
                    dist.barrier()

            # Keep one env object per problem across framings so we do not
            # recreate ALE-Bench env wrappers and so all sessions can be closed.
            if not hasattr(self, "_teacher_envs"):
                self._teacher_envs: dict[str, Any] = {}

            for spec in self._eval_teacher_specs:
                teacher_label = spec["label"]
                self._current_teacher_spec = spec
                self._teacher_path = spec["lora_path"]
                self.hint_sampler = spec["hint_sampler"]

                # NOTE: ALE-Bench sessions are globally cached per problem, so we
                # do NOT close the previous teacher's session here. All sessions are
                # closed together in ``close()``.
                problem_id = spec["problem_id"]
                if self.actor.rank == 0:
                    # Prefer a previously created teacher env, then a baseline env.
                    if problem_id in self._teacher_envs:
                        self.env = self._teacher_envs[problem_id]
                        logger.info(
                            f"[Eval] Reusing cached teacher env for problem {problem_id}"
                        )
                    elif (
                        hasattr(self, "_baseline_envs")
                        and problem_id in self._baseline_envs
                    ):
                        self.env = self._baseline_envs[problem_id]
                        self._teacher_envs[problem_id] = self.env
                        logger.info(
                            f"[Eval] Reusing baseline env for problem {problem_id}"
                        )
                    else:
                        self.env = self._create_env_for_problem(problem_id)
                        self._teacher_envs[problem_id] = self.env
                    self._score_type = self._get_score_type(self.env)
                    logger.info("=" * 60)
                    logger.info(f"[Eval] Evaluating teacher: {teacher_label}")
                    logger.info(f"[Eval]   Problem: {problem_id}")
                    logger.info(f"[Eval]   LoRA path: {self._teacher_path}")
                    logger.info(
                        f"[Eval]   Sampler checkpoint: {spec['sampler_checkpoint']}"
                    )
                    logger.info(f"[Eval]   Score type: {self._score_type}")
                    logger.info("=" * 60)
                else:
                    self.env = None

                # Load this teacher's LoRA and push to vLLM
                self._load_peft_lora_adapter(self.actor, self._teacher_path)
                self._push_teacher_weights()

                results_by_mode: dict[str, dict[str, Any]] = {}
                for mode_spec in mode_specs:
                    mode_result = self._run_single_mode(mode_spec)
                    results_by_mode[mode_spec["label"]] = mode_result
                    combined_task_metadata.update(mode_result.get("task_metadata", {}))
                    if dist.is_initialized():
                        dist.barrier()

                # Aggregate only on DP head.
                if self.actor.rank == 0:

                    def _sort_key(r: dict[str, Any], sel: str) -> tuple[float, float]:
                        avg = r.get(sel, {}).get("average_all", {})
                        return (
                            avg.get("absolute_score", 0.0),
                            avg.get("performance", 0.0),
                        )

                    all_teacher_results[teacher_label] = {
                        "problem_id": spec["problem_id"],
                        "lora_path": spec["lora_path"],
                        "sampler_checkpoint": spec["sampler_checkpoint"],
                        "results_by_mode": results_by_mode,
                        "ranking_by_median": [
                            {
                                "rank": i + 1,
                                "mode": r["mode"],
                                "mode_label": r.get("mode_label", r["mode"]),
                                "private_absolute_score": r.get("median", {})
                                .get("average_all", {})
                                .get("absolute_score", 0.0),
                                "private_performance": r.get("median", {})
                                .get("average_all", {})
                                .get("performance", 0.0),
                            }
                            for i, r in enumerate(
                                sorted(
                                    results_by_mode.values(),
                                    key=lambda r: _sort_key(r, "median"),
                                    reverse=True,
                                )
                            )
                        ],
                        "ranking_by_best_public": [
                            {
                                "rank": i + 1,
                                "mode": r["mode"],
                                "mode_label": r.get("mode_label", r["mode"]),
                                "private_absolute_score": r.get("best_public", {})
                                .get("average_all", {})
                                .get("absolute_score", 0.0),
                                "private_performance": r.get("best_public", {})
                                .get("average_all", {})
                                .get("performance", 0.0),
                            }
                            for i, r in enumerate(
                                sorted(
                                    results_by_mode.values(),
                                    key=lambda r: _sort_key(r, "best_public"),
                                    reverse=True,
                                )
                            )
                        ],
                    }

            # Build combined summary across teachers for this framing.
            summary = {
                "prompt_framing": framing,
                "modes": [m["label"] for m in mode_specs],
                "mode_specs": mode_specs,
                "baseline_results": baseline_results
                if baseline_results is not None
                else {},
                "teachers": all_teacher_results,
            }
            all_framing_summaries[framing] = summary

            if self.actor.rank == 0:
                output_dir = os.path.join(
                    config.saver.fileroot,
                    config.experiment_name,
                    config.trial_name,
                )
                os.makedirs(output_dir, exist_ok=True)

                # When testing multiple framings, keep separate files so later
                # iterations do not overwrite earlier ones.
                suffix = "" if len(framing_list) == 1 else f"_{framing}"
                output_path = os.path.join(
                    output_dir, f"eval_teacher_hint_ablation{suffix}.json"
                )
                with open(output_path, "w") as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"[Eval] Combined ablation results saved to {output_path}")

                # Save task-id -> (model, hint mode, problem) mapping so rollout
                # trajectories dumped by AReaL can be matched back to their eval
                # context. AReaL writes eval rollouts to
                # ``<log_path>/eval-rollout/<version>/<task_id>.jsonl``.
                #
                # To inspect the actual generated code for a task_id:
                #   1. Load this JSON to get ``task_metadata[str(task_id)]``.
                #   2. Read ``<log_path>/eval-rollout/<version>/<task_id>.jsonl``.
                #   3. Each line is a rollout record with ``prompt``, completion
                #      text, and public-eval metadata.
                mapping_path = os.path.join(
                    output_dir,
                    f"eval_teacher_hint_ablation_task_metadata{suffix}.json",
                )
                with open(mapping_path, "w") as f:
                    json.dump(
                        {
                            "task_metadata": {
                                str(k): v for k, v in combined_task_metadata.items()
                            },
                            "rollout_dump_subdir": "eval-rollout",
                        },
                        f,
                        indent=2,
                    )
                logger.info(f"[Eval] Task metadata mapping saved to {mapping_path}")

                # Print baseline + per-teacher ranking for this framing.
                logger.info("=" * 60)
                logger.info(f"TEACHER HINT ABLATION - FINAL RESULTS [{framing}]")
                logger.info("=" * 60)
                if baseline_results and not baseline_results.get("skipped"):
                    logger.info("Baseline (pure base model):")
                    for mode, mode_result in baseline_results.get(
                        "results_by_mode", {}
                    ).items():
                        logger.info(f"  Mode: {mode}")
                        median_avg = mode_result.get("median", {}).get(
                            "average_all", {}
                        )
                        best_public_avg = mode_result.get("best_public", {}).get(
                            "average_all", {}
                        )
                        logger.info(
                            f"    median       abs={median_avg.get('absolute_score', 0.0):.4f} "
                            f"perf={median_avg.get('performance', 0.0):.4f} "
                            f"success={median_avg.get('count', 0)}/"
                            f"{len(baseline_results.get('problem_ids', []))}"
                        )
                        logger.info(
                            f"    best_public  abs={best_public_avg.get('absolute_score', 0.0):.4f} "
                            f"perf={best_public_avg.get('performance', 0.0):.4f} "
                            f"success={best_public_avg.get('count', 0)}/"
                            f"{len(baseline_results.get('problem_ids', []))}"
                        )
                for teacher_label, teacher_summary in all_teacher_results.items():
                    logger.info(f"Teacher: {teacher_label}")
                    logger.info("  Ranking by median selection:")
                    for entry in teacher_summary["ranking_by_median"]:
                        label = entry.get("mode_label", entry["mode"])
                        logger.info(
                            f"    #{entry['rank']} {label:18s} "
                            f"abs={entry['private_absolute_score']:.4f} "
                            f"perf={entry['private_performance']:.4f}"
                        )
                    logger.info("  Ranking by best-public selection:")
                    for entry in teacher_summary["ranking_by_best_public"]:
                        label = entry.get("mode_label", entry["mode"])
                        logger.info(
                            f"    #{entry['rank']} {label:18s} "
                            f"abs={entry['private_absolute_score']:.4f} "
                            f"perf={entry['private_performance']:.4f}"
                        )
                logger.info("=" * 60)

            if dist.is_initialized():
                dist.barrier()

        # After all framings, write a combined comparison file if applicable.
        if self.actor.rank == 0 and len(framing_list) > 1:
            output_dir = os.path.join(
                config.saver.fileroot,
                config.experiment_name,
                config.trial_name,
            )
            os.makedirs(output_dir, exist_ok=True)
            comparison_path = os.path.join(
                output_dir, "eval_teacher_hint_ablation_framing_comparison.json"
            )
            with open(comparison_path, "w") as f:
                json.dump(all_framing_summaries, f, indent=2)
            logger.info(f"[Eval] Framing comparison saved to {comparison_path}")

    def close(self):
        """Close all globally-shared ALE-Bench sessions and cleanup resources.

        Safe to call multiple times.
        """
        if self._closed:
            return
        self._closed = True
        # Collect every env that may hold a session reference. Because sessions
        # are globally cached per problem, deduplicate by ``id(session)`` before
        # closing to avoid double-close.
        all_envs: list[Any] = []
        env = getattr(self, "env", None)
        if env is not None:
            all_envs.append(env)
        if hasattr(self, "_teacher_envs"):
            all_envs.extend(self._teacher_envs.values())
        if hasattr(self, "_baseline_envs"):
            all_envs.extend(self._baseline_envs.values())

        closed_ids: set[int] = set()
        for env in all_envs:
            session = getattr(env, "session", None)
            if session is None:
                continue
            sid = id(session)
            if sid in closed_ids:
                env.session = None
                continue
            closed_ids.add(sid)
            try:
                session.close()
                logger.info("[AleBenchEval] Closed ALE-Bench env session")
            except Exception as e:
                logger.warning(f"[AleBenchEval] Failed to close env session: {e}")
            finally:
                env.session = None

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
    config, _ = load_expr_config(args, TTTDDistillConfig)

    if config.tokenizer_path:
        from areal.utils.hf_utils import load_hf_tokenizer

        tokenizer = load_hf_tokenizer(config.tokenizer_path)
        if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
        if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
            config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)

    if not hasattr(config, "eval_batch_size"):
        config.eval_batch_size = getattr(config.sampler, "batch_size", 8)

    # Default ALE-Bench eval settings
    if getattr(config, "ale_bench_eval_n_candidates", 0) <= 0:
        config.ale_bench_eval_n_candidates = 15
    if getattr(config, "ale_bench_eval_num_workers", 0) <= 0:
        config.ale_bench_eval_num_workers = 1
    if getattr(config, "ale_bench_eval_n_parallel_problems", 0) <= 0:
        config.ale_bench_eval_n_parallel_problems = 1
    if not hasattr(config, "ale_bench_eval_lite_version"):
        config.ale_bench_eval_lite_version = False

    trainer = TeacherHintAblationAleBenchTrainer(config)

    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.warning(
            f"[Main] Received {sig_name} (signal {signum}), "
            "closing ALE-Bench sessions and exiting."
        )
        # Close module-level cached sessions first to release Docker containers
        # and temp tool_dirs as quickly as possible.
        try:
            close_all_ale_bench_sessions()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Main] Failed to close env sessions: {e}")
        try:
            close_all_cached_ale_bench_sessions()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Main] Failed to close eval sessions: {e}")
        # Run the full trainer cleanup (rollout/actor destruction, etc.).
        try:
            trainer.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[Main] trainer.close() failed: {e}")
        # Restore the default handler and re-raise the signal so the process
        # actually terminates.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main(sys.argv[1:])
