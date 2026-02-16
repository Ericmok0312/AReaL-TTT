#!/usr/bin/env python3
"""
TTT-Discover Training with FSDP + LoRA + vLLM + Group Rollout

Uses vLLM's distributed rollout (each DP rank handles its own parents).
No single-rank bottleneck, no broadcast needed.

Usage:
    torchrun --nproc_per_node=8 train_fsdp_lora_vllm.py \
        --config-path conf/fsdp_lora_vllm.yaml
"""

import os
import sys
import warnings
from copy import deepcopy
from typing import Any

import torch
import torch.distributed as dist

from areal import current_platform
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.utils import seeding, stats_tracker
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

from areal.experimental.ttt_discover.config import TTTDPPOActorConfig
from areal.experimental.ttt_discover.actor import TTTDActor
from areal.experimental.ttt_discover.dataloader import create_tttd_dataloader
from areal.experimental.ttt_discover.sampler import create_sampler_from_config
from areal.experimental.ttt_discover.envs.env import BaseEnv, EnvResult
from areal.experimental.ttt_discover.workflow import TTTDiscoverWorkflow

warnings.filterwarnings("ignore", category=DeprecationWarning)


def ensure_initial_lora_adapter(
    base_model_path: str,
    lora_output_path: str,
    lora_rank: int,
    lora_alpha: int,
    target_modules: list[str],
    tokenizer_path: str | None = None,
) -> str:
    """Ensure initial LoRA adapter exists for vLLM to load at startup.
    
    If the LoRA adapter does not exist, create one from the base model.
    This is required for vLLM LoRA mode - vLLM needs a valid LoRA at startup.
    
    Parameters
    ----------
    base_model_path : str
        Path to the base model (HuggingFace format)
    lora_output_path : str
        Path to save the LoRA adapter
    lora_rank : int
        LoRA rank
    lora_alpha : int
        LoRA alpha
    target_modules : list[str]
        Target modules for LoRA (e.g., ["all-linear"])
    tokenizer_path : str | None
        Path to tokenizer (defaults to base_model_path)
    
    Returns
    -------
    str
        Path to the LoRA adapter
    """
    import os
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Ensure absolute path
    lora_output_path = os.path.abspath(lora_output_path)
    
    # Check if LoRA already exists
    adapter_config_path = os.path.join(lora_output_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        print(f"Initial LoRA adapter already exists at {lora_output_path}")
        return lora_output_path
    
    print(f"Creating initial LoRA adapter at {lora_output_path}...")
    print(f"  Base model: {base_model_path}")
    print(f"  LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"  Target modules: {target_modules}")
    
    # Create parent directory if needed
    parent_dir = os.path.dirname(lora_output_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
        print(f"  Created parent directory: {parent_dir}")
    
    # Load base model
    print("Loading base model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            device_map="cpu",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load base model from {base_model_path}: {e}")
    
    # Load tokenizer
    tok_path = tokenizer_path or base_model_path
    print(f"Loading tokenizer from {tok_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer from {tok_path}: {e}")
    
    # Configure LoRA
    target_mods = "all-linear" if target_modules == ["all-linear"] else target_modules
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_mods,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    
    # Save LoRA adapter
    print(f"Saving LoRA adapter to {lora_output_path}...")
    os.makedirs(lora_output_path, exist_ok=True)
    model.save_pretrained(lora_output_path)
    tokenizer.save_pretrained(lora_output_path)
    
    # Verify files were created
    if not os.path.exists(adapter_config_path):
        raise RuntimeError(f"Failed to create adapter_config.json at {lora_output_path}")
    
    print(f"Initial LoRA adapter created successfully at {lora_output_path}")
    print(f"  Contents: {os.listdir(lora_output_path)}")
    return lora_output_path


def process_batch_and_update_sampler(
    batch: dict,
    metadata: list[dict],
    sampler,
    env: BaseEnv,
    group_size: int,
) -> dict:
    """
    Process batch and update PUCTSampler.
    
    Assumes AReaL returns ordered: [parent_0_x64, parent_1_x64, ...]
    Each DP rank processes its local batch.
    
    Args:
        batch: The trajectory batch from prepare_batch
        metadata: List of metadata dicts from workflow._batch_metadata
        sampler: The state sampler (PUCTSampler)
        env: The environment
        group_size: Number of rollouts per parent
    """
    batch_size = batch["rewards"].shape[0]
    num_parents = batch_size // group_size
    
    # Metadata and batch should now be aligned (both include failed rollouts)
    if len(metadata) != batch_size:
        print(f"[WARNING] Metadata length mismatch: metadata={len(metadata)}, "
              f"batch_size={batch_size}. This should not happen with proper alignment.")
    
    # Create child states for valid rollouts only
    children = []
    parents = []
    
    for i in range(num_parents):
        start = i * group_size
        parent_state = metadata[start]["parent_state"]
        
        for idx in range(start, start + group_size):
            m = metadata[idx]
            if not m["is_valid"]:
                continue
            
            child = env.create_state(
                parent_state=parent_state,
                code=m["code"],
                reward=batch["rewards"][idx].item(),
                result=EnvResult(
                    reward=batch["rewards"][idx].item(),
                    observation=m["observation"],
                    is_valid=m["is_valid"],
                ),
                timestep=parent_state.timestep + 1,
            )
            children.append(child)
            parents.append(parent_state)
    
    # Update sampler (PUCTSampler handles top-k internally)
    if children:
        sampler.update_states(children, parents, save=False)
    
    return batch


def main(args):
    config, _ = load_expr_config(args, TTTDPPOActorConfig)
    config: TTTDPPOActorConfig
    
    rank = int(os.getenv("RANK", 0))
    
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode["actor"].parallel
    assert parallel_strategy is not None
    
    # TTTDPPOActorConfig extends PPOActorConfig, so we use config directly
    actor = TTTDActor(config=config)
    actor.create_process_group(parallel_strategy=parallel_strategy)
    
    # TTT-Discover does not require a traditional dataset.
    # PUCTSampler manages states internally and creates initial states
    # automatically based on initial_exp_type and env_type configuration.
    sampler = create_sampler_from_config(
        config=config.sampler,
        log_path=config.saver.fileroot,
        env_type="custom",
    )
    
    # batch_size is the number of parent states per step per DP rank
    # Each parent is expanded to group_size rollouts (config.gconfig.n_samples)
    batch_size = config.sampler.batch_size
    
    # vLLM: each DP rank has its own dataloader shard
    train_dataloader = create_tttd_dataloader(
        state_sampler=sampler,
        rank=actor.data_parallel_rank,
        world_size=actor.data_parallel_world_size,
        batch_size=batch_size,
    )
    
    # For TTT-Discover, dataset_size is conceptual - represents total rollouts
    # __len__ returns a large number for training loop compatibility
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * batch_size,
        train_batch_size=batch_size,
    )
    
    # Initialize FSDP actor first
    actor.initialize(None, ft_spec)
    
    # Check that initial LoRA adapter exists (vLLM requires it at startup)
    if config.use_lora:
        import json
        lora_output_path = "./lora_init"
        
        if hasattr(config, 'vllm') and isinstance(config.vllm, dict):
            lora_modules_str = config.vllm.get('lora_modules', '')
            if lora_modules_str:
                try:
                    lora_modules = json.loads(lora_modules_str)
                    if isinstance(lora_modules, dict):
                        lora_output_path = lora_modules.get('path', lora_output_path)
                except json.JSONDecodeError:
                    pass
        
        lora_output_path = os.path.abspath(lora_output_path)
        adapter_config_path = os.path.join(lora_output_path, "adapter_config.json")
        
        if not os.path.exists(adapter_config_path):
            raise RuntimeError(
                f"\n"
                f"Initial LoRA adapter not found at {lora_output_path}\n"
                f"\n"
                f"For LoRA training with vLLM, you must first create the initial LoRA adapter\n"
                f"BEFORE starting training, because vLLM loads it at startup.\n"
                f"\n"
                f"Please run the preparation script first:\n"
                f"  python prepare_lora_init.py --config-path conf/fsdp_lora_vllm.yaml\n"
                f"\n"
                f"Then start training:\n"
                f"  torchrun --nproc_per_node=8 train_fsdp_lora_vllm.py --config-path conf/fsdp_lora_vllm.yaml"
            )
        
        if actor.is_data_parallel_head():
            print(f"[Rank {rank}] LoRA adapter verified at {lora_output_path}")
    
    # Setup weight update meta for LoRA
    if config.weight_update_mode == "disk":
        weight_update_meta = WeightUpdateMeta.from_disk(
            config.saver.experiment_name,
            config.saver.trial_name,
            config.saver.fileroot,
            use_lora=config.use_lora,
            lora_name=config.gconfig.lora_name,
            lora_int_id=1,
            base_model_name=config.path,
        )
    elif config.weight_update_mode == "xccl":
        weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(
            allocation_mode,
            use_lora=config.use_lora,
            lora_name=config.gconfig.lora_name,
            lora_int_id=1,
            base_model_name=config.path,
        )
    else:
        raise ValueError(
            f"Invalid weight_update_mode: {config.weight_update_mode}. "
            "Expected 'disk' or 'xccl'."
        )
    
    # vLLM: distributed rollout (initialized after LoRA weights are saved)
    rollout = RemotevLLMEngine(config.rollout)
    eval_rollout = RemotevLLMEngine(deepcopy(config.rollout))
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()
    
    actor.connect_engine(rollout, weight_update_meta)
    
    ref = None
    if config.kl_ctl > 0 and config.ref is not None:
        # ref is now a PPOActorConfig object (properly typed in TTTDPPOActorConfig)
        ref = TTTDActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)
    
    # ======================================================================
    # Environment Setup (Auto-selected based on config.sampler.env_type)
    # ======================================================================
    # NOTE: Initial states are created internally by PUCTSampler based on
    # config.sampler.initial_exp_type ('best_available', 'none', 'random', etc.)
    # The env_type parameter determines which initial state to create.
    
    env_type = getattr(config.sampler, 'env_type', 'cp')
    
    if env_type == 'cp':
        # Circle Packing: maximize sum of radii for packing n circles
        from areal.experimental.ttt_discover.envs import CirclePackingEnv
        env = CirclePackingEnv(
            n_item=getattr(config.sampler, 'n_item', 26),
            eval_timeout=60,
            log_dir=config.saver.fileroot,
        )
    elif env_type in ['trimul', 'mla_decode_nvidia', 'mla_decode']:
        # GPU Mode: optimize GPU kernels (requires Modal for actual GPU execution)
        from areal.experimental.ttt_discover.envs import GpuModeEnv
        task_name = 'trimul' if env_type == 'trimul' else 'mla_decode'
        gpu_type = getattr(config.sampler, 'gpu_type', 'H100')
        env = GpuModeEnv(
            task_name=task_name,
            gpu_type=gpu_type,
            eval_timeout=getattr(config.sampler, 'eval_timeout', 300),
            log_dir=config.saver.fileroot,
        )
    elif env_type == 'erdos':
        # Erdos Min Overlap: minimize C5 overlap bound
        from areal.experimental.ttt_discover.envs import ErdosEnv
        env = ErdosEnv(
            n=getattr(config.sampler, 'n', 100),
            eval_timeout=getattr(config.sampler, 'eval_timeout', 60),
            log_dir=config.saver.fileroot,
        )
    elif env_type in ['ac1', 'inequalities']:
        # AlphaEvolve AC1: optimize height sequences for inequalities
        from areal.experimental.ttt_discover.envs import InequalitiesEnv
        env = InequalitiesEnv(
            budget_s=getattr(config.sampler, 'budget_s', 1000),
            eval_timeout=getattr(config.sampler, 'eval_timeout', 60),
            log_dir=config.saver.fileroot,
        )
    else:
        raise ValueError(
            f"Unknown env_type: {env_type}. Supported: 'cp', 'trimul', 'mla_decode_nvidia', "
            f"'mla_decode', 'erdos', 'ac1', 'inequalities'. "
            f"Please set config.sampler.env_type to a supported value."
        )
    
    # Ensure stop_token_ids are set
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    
    workflow = TTTDiscoverWorkflow(
        env=env,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
    )
    
    group_size = config.gconfig.n_samples
    
    saver = Saver(config.saver, ft_spec)
    stats_logger = StatsLogger(config, ft_spec)
    evaluator = Evaluator(config.evaluator, ft_spec)
    recover_handler = RecoverHandler(config.recover, ft_spec)
    
    recover_info = recover_handler.load(
        actor, saver, evaluator, stats_logger,
        train_dataloader, inference_engine=rollout,
        weight_update_meta=weight_update_meta,
    )
    start_step = recover_info.last_step_info.next().global_step if recover_info else 0
    
    # TTT-Discover uses fixed number of steps (paper uses 50 steps)
    # NOT epochs, since PUCTSampler manages states internally
    max_steps = getattr(config, 'max_steps', 50)
    
    # Track best reward for logging
    best_reward = float('-inf')
    
    for global_step in range(start_step, max_steps):
        step_info = StepInfo(
            global_step=global_step,
            epoch=global_step,  # Each step is its own "epoch" for logging
            epoch_step=global_step,
            steps_per_epoch=max_steps,
        )
        
        # Initialize metadata context for this batch.
        # This must be done before prepare_batch so that metadata can be collected
        # even when workflow is wrapped by GroupedRolloutWorkflow.
        batch_metadata = workflow.init_batch_metadata()
        
        try:
            # vLLM: each rank does its own rollout (no broadcast needed)
            with stats_tracker.record_timing("rollout"):
                batch = actor.prepare_batch(
                    train_dataloader,
                    workflow=workflow,
                    group_size=group_size,
                    should_accept_fn=lambda sample: True,
                )
            
            with stats_tracker.record_timing("group_processing"):
                # Metadata was collected via contextvars during rollout,
                # so it's available here even with GroupedRolloutWorkflow.
                training_batch = process_batch_and_update_sampler(
                    batch=batch,
                    metadata=batch_metadata,
                    sampler=sampler,
                    env=env,
                    group_size=group_size,
                )
            # Only rank 0 flushes to avoid conflicts
            if actor.is_data_parallel_head():
                sampler.flush(step=global_step)
            
            # Track and log best reward
            step_rewards = training_batch["rewards"].cpu().numpy()
            step_max_reward = float(step_rewards.max())
            step_mean_reward = float(step_rewards.mean())
            best_reward = max(best_reward, step_max_reward)
            
            if actor.is_data_parallel_head():
                print(f"[Step {global_step}] "
                      f"Max Reward: {step_max_reward:.4f} | "
                      f"Mean Reward: {step_mean_reward:.4f} | "
                      f"Best Overall: {best_reward:.4f}")
        
        finally:
            # Clean up metadata context after batch processing
            workflow.reset_batch_metadata()
        
        dist.barrier(group=actor.cpu_group)
        
        if config.should_compute_prox_logp():
            with stats_tracker.record_timing("recompute_logp"):
                training_batch["prox_logp"] = actor.compute_logp(training_batch)
        
        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                training_batch["ref_logp"] = ref.compute_logp(training_batch)
        
        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(training_batch)
        
        with stats_tracker.record_timing("train_step"):
            actor.ppo_update(training_batch)
            actor.step_lr_scheduler()
        
        rollout.pause()
        
        with stats_tracker.record_timing("update_weights"):
            actor.update_weights(weight_update_meta)
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)
        
        with stats_tracker.record_timing("save"):
            saver.save(actor, step_info.epoch, step_info.epoch_step, global_step, tokenizer=tokenizer)
        
        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor, step_info, saver, evaluator, stats_logger,
                train_dataloader, tokenizer=tokenizer,
            )
        
        dist.barrier(group=actor.cpu_group)
        current_platform.synchronize()
        
        with stats_tracker.record_timing("eval"):
            def evaluate_fn():
                pass
            evaluator.evaluate(evaluate_fn, step_info.epoch, step_info.epoch_step, global_step)
        
        stats = actor.export_stats()
        stats_logger.commit(step_info.epoch, step_info.epoch_step, global_step, stats)
        
        dist.barrier(group=actor.cpu_group)
        current_platform.synchronize()
        rollout.resume()
    
    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
