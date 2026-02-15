#!/usr/bin/env python3
"""
TTT-Discover Training with FSDP + LoRA + Group Rollout

Usage:
    torchrun --nproc_per_node=8 train_fsdp_lora.py \
        --config-path conf/fsdp_lora.yaml
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
from areal.engine.sglang_remote import RemoteSGLangEngine
from areal.utils import seeding, stats_tracker
from areal.utils.data import broadcast_tensor_container, tensor_container_to
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
from areal.infra.dist_rollout import redistribute_trajectories

warnings.filterwarnings("ignore", category=DeprecationWarning)


def bcast_and_split_from_rank0(batch: dict) -> dict:
    """Broadcast batch from rank 0 and split across all ranks."""
    batch = broadcast_tensor_container(batch, src_rank=0)
    
    if len(batch) == 0:
        return batch
    
    bs = batch["input_ids"].shape[0]
    world_size = dist.get_world_size()
    bs_per_rank = bs // world_size
    rank = dist.get_rank()
    
    local_slice = slice(rank * bs_per_rank, (rank + 1) * bs_per_rank)
    
    local_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            local_batch[k] = v[local_slice]
        elif k == "_tttd_metadata":
            local_batch[k] = v[local_slice]
        else:
            local_batch[k] = v
    
    return redistribute_trajectories([local_batch]).data


def process_batch_and_update_sampler(
    batch: dict,
    sampler,
    env: BaseEnv,
    group_size: int,
) -> dict:
    """
    Process batch and update PUCTSampler.
    
    Assumes AReaL returns ordered: [parent_0_x64, parent_1_x64, ...]
    """
    metadata = batch["_tttd_metadata"]
    batch_size = batch["rewards"].shape[0]
    num_parents = batch_size // group_size
    
    # Create child states for ALL rollouts
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
    parallel_strategy = allocation_mode.train
    assert parallel_strategy is not None
    
    actor = TTTDActor(config=config.actor)
    actor.create_process_group(parallel_strategy=parallel_strategy)
    
    # TTT-Discover does not require a traditional dataset.
    # PUCTSampler manages states internally and creates initial states
    # automatically based on initial_exp_type and env_type configuration.
    sampler = create_sampler_from_config(
        config=config.sampler,
        log_path=config.saver.fileroot,
        env_type="custom",
    )
    
    # batch_size is the number of parent states per step
    # Each parent is expanded to group_size rollouts (config.gconfig.n_samples)
    batch_size = config.sampler.batch_size
    
    train_dataloader = create_tttd_dataloader(
        sampler=sampler,
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
    
    rollout = RemoteSGLangEngine(config.rollout)
    eval_rollout = RemoteSGLangEngine(deepcopy(config.rollout))
    rollout.initialize(train_data_parallel_size=1)
    eval_rollout.config.max_head_offpolicyness = int(1e12)
    eval_rollout.initialize()
    
    weight_update_meta = WeightUpdateMeta.from_disk(
        config.saver.experiment_name,
        config.saver.trial_name,
        config.saver.fileroot,
        use_lora=True,
    )
    
    actor.initialize(None, ft_spec)
    actor.connect_engine(rollout, weight_update_meta)
    
    ref = None
    if config.actor.kl_ctl > 0 and config.ref is not None:
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
    
    for global_step in range(start_step, max_steps):
        step_info = StepInfo(
            global_step=global_step,
            epoch=global_step,  # Each step is its own "epoch" for logging
            epoch_step=global_step,
            steps_per_epoch=max_steps,
        )
        
        with stats_tracker.record_timing("rollout"):
            batch = None
            if rank == 0:
                batch = actor.prepare_batch(
                    train_dataloader,
                    workflow=workflow,
                    group_size=group_size,
                    should_accept_fn=lambda sample: True,
                )
                batch = tensor_container_to(batch, actor.device)
            batch = bcast_and_split_from_rank0(batch)
        
        current_platform.synchronize()
        dist.barrier(group=actor.cpu_group)
        
        with stats_tracker.record_timing("group_processing"):
            training_batch = process_batch_and_update_sampler(
                batch=batch,
                sampler=sampler,
                env=env,
                group_size=group_size,
            )
            sampler.flush(step=global_step)
        
        if config.actor.should_compute_prox_logp():
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
