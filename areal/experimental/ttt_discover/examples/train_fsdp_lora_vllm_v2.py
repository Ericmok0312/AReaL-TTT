#!/usr/bin/env python3
"""
TTT-Discover Training V2 - 使用 Workflow 内部 Sampler 更新

Key differences from V1:
- Workflow holds sampler reference, updates internally
- No metadata list, no indexing issues
- Simplified training loop with distributed state gathering

Usage:
    torchrun --nproc_per_node=8 train_fsdp_lora_vllm_v2.py \
        --config-path conf/fsdp_lora_vllm.yaml
"""

import os
import sys
import warnings
from copy import deepcopy

import torch
import torch.distributed as dist

from areal import current_platform
from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import load_expr_config
from areal.api.io_struct import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.engine.vllm_remote import RemotevLLMEngine
from areal.utils import logging, seeding, stats_tracker

logger = logging.getLogger("train_fsdp_lora_vllm_v2")
from areal.utils.evaluator import Evaluator
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.recover import RecoverHandler
from areal.utils.saver import Saver
from areal.utils.stats_logger import StatsLogger

from areal.experimental.ttt_discover.config import TTTDPPOActorConfig
from areal.experimental.ttt_discover.actor import TTTDActor
from areal.experimental.ttt_discover.dataloader import create_tttd_dataloader
from areal.experimental.ttt_discover.sampler import create_sampler_from_config
from areal.experimental.ttt_discover.workflow_v2 import TTTDiscoverWorkflowV2

warnings.filterwarnings("ignore", category=DeprecationWarning)


def gather_states_across_ranks(actor, local_children, local_parents):
    """
    Gather states from all data parallel ranks using all_gather_object.
    
    This uses PyTorch's all_gather_object which is designed for Python objects
    and handles device placement automatically (works with both NCCL and Gloo).
    
    Args:
        actor: TTTDActor with data parallel info
        local_children: List of child states from this rank
        local_parents: List of parent states from this rank
        
    Returns:
        (all_children, all_parents) on rank 0, (None, None) on other ranks
    """
    if not dist.is_initialized() or actor.data_parallel_world_size <= 1:
        return local_children, local_parents
    
    # Serialize states to dicts for communication
    children_dicts = [s.to_dict() for s in local_children]
    parents_dicts = [s.to_dict() for s in local_parents]
    
    # Gather from all ranks using all_gather_object (handles NCCL automatically)
    all_children_dicts = [None] * actor.data_parallel_world_size
    all_parents_dicts = [None] * actor.data_parallel_world_size
    
    dist.all_gather_object(all_children_dicts, children_dicts, group=actor.data_parallel_group)
    dist.all_gather_object(all_parents_dicts, parents_dicts, group=actor.data_parallel_group)
    
    # On rank 0, deserialize and combine all states
    if int(os.getenv("RANK", 0)) == 0:
        from areal.experimental.ttt_discover.state import state_from_dict
        
        all_children = []
        all_parents = []
        
        for rank_children, rank_parents in zip(all_children_dicts, all_parents_dicts):
            if rank_children:  # Skip empty lists
                all_children.extend([state_from_dict(d) for d in rank_children])
            if rank_parents:
                all_parents.extend([state_from_dict(d) for d in rank_parents])
        
        return all_children, all_parents
    else:
        return None, None


def main(args):
    config, _ = load_expr_config(args, TTTDPPOActorConfig)
    config: TTTDPPOActorConfig
    
    rank = int(os.getenv("RANK", 0))
    
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode["actor"].parallel
    assert parallel_strategy is not None
    
    actor = TTTDActor(config=config)
    actor.create_process_group(parallel_strategy=parallel_strategy)
    
    # Create sampler first (before workflow)
    sampler = create_sampler_from_config(
        config=config.sampler,
        log_path=config.saver.fileroot,
        env_type="custom",
    )
    
    batch_size = config.sampler.batch_size  # Total parents per step (e.g., 8)
    group_size = config.gconfig.n_samples   # Rollouts per parent (e.g., 64)
    
    # Use only_dp_head=True because AReaL's prepare_batch only runs rollout on DP head
    # and broadcasts results to all ranks. This ensures correct total rollout count.
    # world_size=1: Treat as single-process dataset (rank 0 produces all data)
    train_dataloader = create_tttd_dataloader(
        state_sampler=sampler,
        rank=0 if actor.dp_rank == 0 else -1,
        world_size=1,
        batch_size=batch_size,  # Total batch size (8 parents total, not per rank)
        only_dp_head=True,      # Only DP head produces data
    )
    
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * batch_size,
        train_batch_size=batch_size,
    )
    
    actor.initialize(None, ft_spec)
    
    # Check LoRA adapter exists (same as V1)
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
            raise RuntimeError(f"Initial LoRA adapter not found at {lora_output_path}")
    
    # Weight update meta
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
    else:
        weight_update_meta = WeightUpdateMeta.from_fsdp_xccl(
            allocation_mode,
            use_lora=config.use_lora,
            lora_name=config.gconfig.lora_name,
            lora_int_id=1,
            base_model_name=config.path,
        )
    
    rollout = RemotevLLMEngine(config.rollout)
    eval_rollout = RemotevLLMEngine(deepcopy(config.rollout))
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()
    
    actor.connect_engine(rollout, weight_update_meta)
    
    ref = None
    if config.kl_ctl > 0 and config.ref is not None:
        ref = TTTDActor(config=config.ref)
        ref.create_process_group(parallel_strategy=parallel_strategy)
        ref.initialize(None, ft_spec)
    
    # Environment setup (same as V1)
    env_type = getattr(config.sampler, 'env_type', 'cp')
    if env_type == 'cp':
        from areal.experimental.ttt_discover.envs import CirclePackingEnv
        env = CirclePackingEnv(
            n_item=getattr(config.sampler, 'n_item', 26),
            eval_timeout=60,
            log_dir=config.saver.fileroot,
        )
    # ... other env types ...
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
    
    # Ensure stop tokens
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    
    # Create V2 workflow - inject sampler directly!
    workflow = TTTDiscoverWorkflowV2(
        env=env,
        sampler=sampler,  # 直接注入
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=config.enable_thinking,
    )
    
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
    
    max_steps = getattr(config, 'max_steps', 50)
    best_reward = float('-inf')
    
    for global_step in range(start_step, max_steps):
        step_info = StepInfo(
            global_step=global_step,
            epoch=global_step,
            epoch_step=global_step,
            steps_per_epoch=max_steps,
        )
        
        # Reset workflow buffers (clears any stale pending updates)
        workflow.reset_sync()
        
        # Rollout - workflow updates sampler internally!
        with stats_tracker.record_timing("rollout"):
            batch = actor.prepare_batch(
                train_dataloader,
                workflow=workflow,
                group_size=group_size,
                should_accept_fn=lambda sample: True,
            )
        
        # Flush sampler updates (commits all buffered updates from this batch)
        # Should we wait for all codes executed?
        with stats_tracker.record_timing("sampler_update"):
            # Get local updates from this rank
            local_children, local_parents = workflow.get_pending_updates_sync(clear=True)
            
            # Aggregate updates from all ranks using all_gather_object
            all_children, all_parents = gather_states_across_ranks(
                actor, local_children, local_parents
            )
            
            # Only rank 0 updates sampler and saves
            if actor.dp_rank == 0:
                if all_children:
                    sampler.update_states(all_children, all_parents, save=False)
                    sampler.flush(step=global_step)
                    logger.info(f"[Step {global_step}] Aggregated {len(all_children)} updates "
                                f"from {actor.data_parallel_world_size} ranks, "
                                f"{len(set(p.id for p in all_parents))} unique parents")
        
        # Log rewards and actual rollout count
        local_rollouts = batch["rewards"].shape[0]  # Rollouts on this rank
        step_rewards = batch["rewards"].cpu().numpy()
        step_max_reward = float(step_rewards.max())
        step_mean_reward = float(step_rewards.mean())
        best_reward = max(best_reward, step_max_reward)
        
        # Compute total rollouts across all ranks
        total_rollouts_tensor = torch.tensor(local_rollouts, dtype=torch.int64, device='cuda')
        dist.all_reduce(total_rollouts_tensor, op=dist.ReduceOp.SUM, group=actor.data_parallel_group)
        total_rollouts = int(total_rollouts_tensor.item())
        
        if actor.dp_rank == 0:
            logger.info(f"[Step {global_step}] Total rollouts: {total_rollouts} (local: {local_rollouts}, expected: {batch_size * group_size})")
        
        # Collect metrics for stats_logger
        metrics = {
            "rollout/total": total_rollouts,
            "reward/max": step_max_reward,
            "reward/mean": step_mean_reward,
            "reward/best_overall": best_reward,
        }
        
        # Training (same as V1)
        if config.should_compute_prox_logp():
            with stats_tracker.record_timing("recompute_logp"):
                batch["prox_logp"] = actor.compute_logp(batch)
        
        if ref is not None:
            with stats_tracker.record_timing("ref_logp"):
                batch["ref_logp"] = ref.compute_logp(batch)
        
        with stats_tracker.record_timing("compute_advantage"):
            actor.compute_advantages(batch)
        
        # Add advantage statistics
        if "advantages" in batch:
            adv = batch["advantages"].cpu().numpy()
            metrics.update({
                "advantage/mean": float(adv.mean()),
                "advantage/std": float(adv.std()),
                "advantage/min": float(adv.min()),
                "advantage/max": float(adv.max()),
            })
        
        with stats_tracker.record_timing("train_step"):
            actor.ppo_update(batch)
            actor.step_lr_scheduler()
        
        # Add training stats
        stats = actor.export_stats()
        metrics.update({
            "train/entropy": stats.get('entropy/avg', 0.0),
            "train/actor_loss": stats.get('actor_loss/avg', 0.0),
            "train/approx_kl": stats.get('approx_kl/avg', 0.0),
        })
        
        # Log to stats_logger (wandb/swanlab/tensorboard)
        if actor.dp_rank == 0:
            stats_logger.commit(
                epoch=step_info.epoch,
                step=step_info.epoch_step,
                global_step=global_step,
                data=metrics,
            )
            # Also print concise summary
            logger.info(f"[Step {global_step}] "
                       f"Reward: max={step_max_reward:.4f}, mean={step_mean_reward:.4f}, best={best_reward:.4f} | "
                       f"Loss: {metrics['train/actor_loss']:.4f}, KL: {metrics['train/approx_kl']:.4f}")
                
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
        rollout.resume()
    
    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
