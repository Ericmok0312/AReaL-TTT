#!/usr/bin/env python3
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

from areal.experimental.ttt_discover.config import (
    TTTDPPOActorConfig,
    create_env_from_config,
)
from areal.experimental.ttt_discover.actor import TTTDActor
from areal.experimental.ttt_discover.dataloader import create_tttd_dataloader
from areal.experimental.ttt_discover.sampler import create_sampler_from_config
from areal.experimental.ttt_discover.workflow_v2 import TTTDiscoverWorkflowV2
from areal.experimental.ttt_discover.ttt_logger import TTTTrainingLogger

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _format_state_table(states: list, title: str = "States") -> str:
    """Format a list of states as a readable table."""
    if not states:
        return f"{title}: <empty>"
    
    lines = [f"\n{'='*80}", f"{title} (count={len(states)})", f"{'='*80}"]
    lines.append(f"{'Index':<6} {'ID':<36} {'Timestep':<10} {'Value':<12} {'Parent ID':<36}")
    lines.append("-" * 100)
    
    for i, state in enumerate(states):
        parent_id = state.parents[0].get("id", "N/A") if state.parents else "root"
        value_str = f"{state.value:.4f}" if state.value is not None else "N/A"
        lines.append(
            f"{i:<6} {state.id:<36} {state.timestep:<10} {value_str:<12} {parent_id:<36}"
        )
    lines.append(f"{'='*80}\n")
    return "\n".join(lines)


def _format_sampler_summary(sampler, title: str = "Sampler State") -> str:
    """Format sampler internal state as a readable table."""
    lines = [f"\n{'#'*80}", f"{title}", f"{'#'*80}"]
    
    # Basic sampler info
    if hasattr(sampler, '_states'):
        all_states = sampler._states
        lines.append(f"Total states in buffer: {len(all_states)}")
        lines.append(f"Current step: {getattr(sampler, '_current_step', 'N/A')}")
        
        if hasattr(sampler, '_T'):
            lines.append(f"PUCT T (total expansions): {sampler._T}")
        if hasattr(sampler, '_n'):
            lines.append(f"PUCT n (visit counts): {len(sampler._n)} entries")
        if hasattr(sampler, '_m'):
            lines.append(f"PUCT m (max rewards): {len(sampler._m)} entries")
        
        # States table
        if all_states:
            lines.append(f"\n{'-'*100}")
            lines.append(f"{'Idx':<5} {'State ID':<36} {'TS':<5} {'Value':<10} {'Parent':<36} {'n':<6} {'m':<10}")
            lines.append(f"{'-'*100}")
            
            for idx, state in enumerate(all_states):
                parent_id = state.parents[0].get("id", "root") if state.parents else "root"
                value_str = f"{state.value:.4f}" if state.value is not None else "N/A"
                
                # Get PUCT stats if available
                n_val = sampler._n.get(state.id, 0) if hasattr(sampler, '_n') else 0
                m_val = sampler._m.get(state.id, 0.0) if hasattr(sampler, '_m') else 0.0
                m_str = f"{m_val:.4f}" if m_val != 0.0 else "N/A"
                
                lines.append(
                    f"{idx:<5} {state.id:<36} {state.timestep:<5} {value_str:<10} "
                    f"{parent_id:<36} {n_val:<6} {m_str:<10}"
                )
        else:
            lines.append("No states in buffer")
    else:
        lines.append("Sampler has no _states attribute")
    
    lines.append(f"{'#'*80}\n")
    return "\n".join(lines)


def main(args):
    config, _ = load_expr_config(args, TTTDPPOActorConfig)
    config: TTTDPPOActorConfig
    
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    
    # Create actor first to get correct dp_rank
    actor = TTTDActor(config=config)
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode["actor"].parallel
    assert parallel_strategy is not None
    actor.create_process_group(parallel_strategy=parallel_strategy)
    
    rank = actor.dp_rank
    world_size = actor.data_parallel_world_size
    
    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    
    # Create sampler
    sampler = create_sampler_from_config(
        config=config.sampler,
        log_path=config.saver.fileroot,
        env_type=config.sampler.env_type if hasattr(config.sampler, 'env_type') else 'cp',
    )
    
    batch_size = config.sampler.batch_size  # Total group per step (e.g., 8)
    group_size = config.gconfig.n_samples   # Rollouts per parent (e.g., 64)
    
    # dp_rank == 0 is the DP head
    is_dp_head = rank == 0
    
    # DEBUG: Log distributed configuration
    if is_dp_head:
        logger.info(f"=== Distributed Configuration ===")
        logger.info(f"DP world size: {world_size}")
        logger.info(f"Sampler batch_size: {batch_size}")
        logger.info(f"Group size: {group_size}")
        logger.info(f"Total rollouts: {batch_size * group_size}")
        logger.info(f"Is DP head: {is_dp_head}")
        logger.info(f"================================")
    
    # Distributed Sampler Mode: Each rank samples its own shard
    use_dp_shard = world_size > 1
    if is_dp_head:
        logger.info(f"=== DataLoader Configuration ===")
        logger.info(f"use_dp_head_only: {not use_dp_shard}")
        logger.info(f"world_size: {world_size}")
        logger.info(f"batch_size (global): {batch_size}")
        logger.info(f"local_batch_size: {batch_size // world_size if use_dp_shard else batch_size}")
        logger.info(f"================================")
    
    train_dataloader = create_tttd_dataloader(
        state_sampler=sampler,
        rank=rank,                    # DP rank
        world_size=world_size,        # DP world size
        batch_size=batch_size,        # Global batch size
    )
    
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * batch_size,
        train_batch_size=batch_size,
    )
    
    actor.initialize(None, ft_spec)
    
    # ============================================================
    # CONFIGURATION VERIFICATION for Scheme A (DP Head Only)
    # ============================================================
    if is_dp_head:
        logger.info("=== Scheme A Configuration Verification ===")
        logger.info(f"Allocation mode: {config.allocation_mode}")
        logger.info(f"Parallel Strategy: {parallel_strategy}")
        logger.info(f"  - DP size: {parallel_strategy.dp_size}")
        logger.info(f"  - TP size: {parallel_strategy.tp_size}")
        logger.info(f"  - PP size: {parallel_strategy.pp_size}")
        logger.info(f"Actor DP rank: {actor.dp_rank}")
        logger.info(f"Actor DP world size: {actor.data_parallel_world_size}")
        logger.info(f"Is DP head: {is_dp_head}")
        
        # Verify Scheme A requirements
        if parallel_strategy.dp_size != 1:
            logger.warning(
                f"WARNING: For Scheme A (DP head only), dp_size should be 1, "
                f"got {parallel_strategy.dp_size}. This means multiple ranks "
                f"will act as DP heads and generate data independently!"
            )
        if parallel_strategy.tp_size <= 1:
            logger.warning(
                f"WARNING: tp_size={parallel_strategy.tp_size}. With dp=1, tp>1 "
                f"is expected to distribute work across GPUs."
            )
        logger.info("==========================================")
    
    # Barrier to ensure all ranks see the verification
    if dist.is_initialized():
        dist.barrier()
    
    # ============================================================
    # Verify LoRA adapter exists
    # ============================================================
    if config.use_lora and not config.skip_lora_check:
        import json
        lora_output_path = "./lora_init"
        # Support both dict (legacy) and vLLMConfig dataclass
        if hasattr(config, 'vllm'):
            if isinstance(config.vllm, dict):
                lora_modules_str = config.vllm.get('lora_modules', '')
            else:
                # vLLMConfig dataclass
                lora_modules_str = getattr(config.vllm, 'lora_modules', '') or ''
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
            error_msg = (
                f"\n{'='*80}\n"
                f"ERROR: Initial LoRA adapter not found at {lora_output_path}\n"
                f"{'='*80}\n\n"
                f"The LoRA adapter must be created BEFORE starting training because\n"
                f"vLLM loads it at startup (before this training script runs).\n\n"
                f"To fix this, run the preparation script first:\n"
                f"  python areal/experimental/ttt_discover/examples/prepare_lora_init.py \\\n"
                f"    --config-path <your_config.yaml>\n\n"
                f"If you have already initialized the LoRA adapter elsewhere, you can\n"
                f"skip this check by adding '+skip_lora_check=true' to your command.\n"
                f"{'='*80}\n"
            )
            raise RuntimeError(error_msg)
        
        if is_dp_head:
            logger.info(f"[LoRA Check] ✓ LoRA adapter verified at {lora_output_path}")
    
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
    rollout.initialize(train_data_parallel_size=parallel_strategy.dp_size)
    
    actor.connect_engine(rollout, weight_update_meta)
    actor.connect_sampler(sampler)  # Connect sampler to actor for internal use and synchronization

    ref = None
    if config.kl_ctl > 0 and config.ref is not None:
        # Reference model uses DP-only strategy to avoid TP + torch.compile issues
        from areal.api.alloc_mode import ParallelStrategy
        ref_parallel_strategy = ParallelStrategy(
            tensor_parallel_size=1,      # No TP for ref
            pipeline_parallel_size=1,
            data_parallel_size=parallel_strategy.world_size,  # Use all GPUs as DP
            context_parallel_size=1,
        )
        ref = TTTDActor(config=config.ref)
        ref.create_process_group(parallel_strategy=ref_parallel_strategy)
        ref.initialize(None, ft_spec)
        logger.info(f"Reference model using DP-only strategy: {ref_parallel_strategy}")
    
    # Environment setup (AC1 or CP)
    env = create_env_from_config(config)
    logger.info(f"[Env] Created {env.__class__.__name__} with eval_timeout={env.eval_timeout}s")
    
    # Ensure stop tokens
    if tokenizer.pad_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.pad_token_id)
    if tokenizer.eos_token_id not in config.gconfig.stop_token_ids:
        config.gconfig.stop_token_ids.append(tokenizer.eos_token_id)
    
    # Create V2 workflow - inject sampler directly!
    # Paper: "limit the total length of the prompt and the thinking tokens to 26000"
    # NOTE: batch_size here is LOCAL batch_size (per-rank)
    local_batch_size = batch_size // world_size if world_size > 1 else batch_size
    workflow = TTTDiscoverWorkflowV2(
        env=env,
        gconfig=config.gconfig,
        tokenizer=tokenizer,
        enable_thinking=config.enable_thinking,
        max_prompt_thinking_tokens=config.max_prompt_thinking_tokens,
        batch_size=local_batch_size,  # Local batch_size per rank
        group_size=group_size,  # Number of rollouts per parent
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
    
    # ============================================================
    # FIX: Clear stale state after recovery (CORRECTED VERSION)
    # ============================================================
    if recover_info:
        import queue as queue_module
        
        if is_dp_head:
            logger.info(f"[Resume] Cleaning up stale state for step {start_step}")
        
        dispatcher = rollout.workflow_executor._dispatcher
        
        # 1. Clear _pending_inputs - CRITICAL: prevents _commit_loop from submitting stale tasks
        with dispatcher._input_lock:
            stale_inputs = len(dispatcher._pending_inputs)
            if stale_inputs > 0:
                dispatcher._pending_inputs.clear()
                if is_dp_head:
                    logger.info(f"[Resume] Cleared {stale_inputs} stale pending inputs")
        
        # 2. Clear AsyncTaskRunner queues - CRITICAL: tasks may have been submitted before cleanup
        runner = dispatcher.runner
        queue_cleared = 0
        for q in [runner.input_queue, runner.output_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                    queue_cleared += 1
                except queue_module.Empty:
                    break
        if is_dp_head and queue_cleared > 0:
            logger.info(f"[Resume] Cleared {queue_cleared} items from async task queues")
        
        # 3. CRITICAL FIX: Set correct 'accepted' count for capacity calculation
        # Normal training: accepted = step * consumer_batch_size at each step start
        # Recovery: version = start_step, so accepted should be start_step * consumer_batch_size
        # This ensures staleness_capacity = (0 + 37 + 1) * 2 - 74 = 2 (same as normal training)
        sm = dispatcher.staleness_manager
        with sm.lock:
            consumer_bs = sm.consumer_batch_size
            # Calculate what accepted should be: completed_steps * consumer_batch_size
            completed_steps = start_step  # Steps 0..start_step-1 are completed
            expected_accepted = completed_steps * consumer_bs
            
            old_accepted = sm.rollout_stat.accepted
            old_running = sm.rollout_stat.running
            old_enqueued = sm.rollout_stat.enqueued
            
            # Reset running and enqueued (these are transient)
            sm.rollout_stat.running = 0
            sm.rollout_stat.enqueued = 0
            
            # Set accepted to expected value for correct capacity calculation
            if old_accepted != expected_accepted:
                sm.rollout_stat.accepted = expected_accepted
                if is_dp_head:
                    logger.info(f"[Resume] Fixed accepted count: {old_accepted} -> {expected_accepted} "
                               f"(expected after {completed_steps} completed steps, consumer_batch_size={consumer_bs})")
            elif is_dp_head:
                logger.info(f"[Resume] accepted count is correct: {old_accepted}")
            
            # Verify capacity will be correct
            version = rollout.get_version()  # Should be start_step
            max_staleness = sm.max_staleness
            capacity = (max_staleness + version + 1) * consumer_bs - sm.rollout_stat.accepted
            if is_dp_head:
                logger.info(f"[Resume] staleness_capacity = ({max_staleness} + {version} + 1) * {consumer_bs} - {sm.rollout_stat.accepted} = {capacity}")
        
        # 4. Clear data_generator cache to force fresh iteration
        if hasattr(rollout.workflow_executor, 'data_generator'):
            delattr(rollout.workflow_executor, 'data_generator')
            if is_dp_head:
                logger.info("[Resume] Cleared data generator cache")
        
        # 5. Clear _pending_results to prevent stale results from being collected
        with dispatcher._result_lock:
            stale_results = len(dispatcher._pending_results)
            if stale_results > 0:
                dispatcher._pending_results.clear()
                dispatcher._active_task_ids.clear()
                if is_dp_head:
                    logger.info(f"[Resume] Cleared {stale_results} stale pending results")
        
        if is_dp_head:
            logger.info(f"[Resume] Ready to start from step {start_step}")
    
    # ============================================================
    # Print PUCTSampler state after recovery (for verification)
    # ============================================================
    if recover_info and is_dp_head and hasattr(sampler, '_states'):
        logger.info("="*80)
        logger.info(f"[PUCTSampler State] Loaded {len(sampler._states)} states at step {start_step}")
        logger.info(f"[PUCTSampler State] _T={sampler._T}, _n entries={len(sampler._n)}, _m entries={len(sampler._m)}")
        
        # 打印前10个 states
        display_count = min(10, len(sampler._states))
        if display_count > 0:
            logger.info(f"[PUCTSampler State] Top {display_count} states by value:")
            # 按 value 排序，取前10
            sorted_states = sorted(
                sampler._states, 
                key=lambda s: s.value if s.value is not None else float('-inf'), 
                reverse=True
            )[:display_count]
            
            for i, state in enumerate(sorted_states):
                parent_info = ""
                if state.parents:
                    parent_id = state.parents[0].get('id', 'N/A')[:8] if state.parents else 'N/A'
                    parent_info = f" (parent={parent_id}...)"
                logger.info(f"  [{i+1}] id={state.id[:8]}... timestep={state.timestep} "
                           f"value={state.value:.4f}{parent_info}")
        logger.info("="*80)
    
    max_steps = getattr(config, 'max_steps', 50)
    best_reward = float('-inf')
    
    # ============================================================
    # Initialize Training History Logger
    # ============================================================
    # 生成完整的 save_steps 列表（包含所有可能需要记录的 step）
    all_save_steps = getattr(config, 'save_steps', list(range(0, max_steps)))
    
    # 对于续训：只记录 >= start_step 的 steps（避免重复记录已完成的 steps）
    # 但 history 会从 checkpoint 加载之前的记录
    future_save_steps = [s for s in all_save_steps if start_step <= s < max_steps]
    
    history_logger = TTTTrainingLogger(
        save_steps=future_save_steps,  # 只记录未来的 steps
        output_dir=config.saver.fileroot,
        is_dp_head=is_dp_head,
        filename='training_history.pkl',
        checkpoint_filename='training_history_checkpoint.pkl',
        aggregate_distributed=True,
    )
    
    if is_dp_head:
        existing_snapshots = len(history_logger.history) if recover_info else 0
        logger.info(f"[TTTLogger] Initialized for step {start_step} to {max_steps}")
        logger.info(f"[TTTLogger] Future save_steps: {future_save_steps}")
        logger.info(f"[TTTLogger] Existing snapshots from checkpoint: {existing_snapshots}")
        logger.info(f"[TTTLogger] Output directory: {config.saver.fileroot}")
    
    logger.info(f"Starting training from step {start_step}/{max_steps}")
    for global_step in range(start_step, max_steps):
        step_info = StepInfo(
            global_step=global_step,
            epoch=global_step,
            epoch_step=global_step,
            steps_per_epoch=max_steps,
        )
        
        # Set current version for staleness calculation
        workflow.set_current_version(global_step)
        
        # Reset workflow buffers (clears any stale pending updates)
        workflow.reset()
        
        # Rollout - workflow updates sampler internally!
        with stats_tracker.record_timing("rollout"):
            batch = actor.prepare_batch(
                train_dataloader,
                workflow=workflow,
                group_size=group_size,
                should_accept_fn=lambda sample: True,
            )
        
        # ============================================================
        # Sampler Synchronization (Unified Interface)
        # ============================================================
        # NOTE: This is a collective operation that MUST be called by ALL ranks.
        # It performs: 1) Gather updates -> 2) Rank 0 applies -> 3) Broadcast to all
        with stats_tracker.record_timing("sampler_update"):
            # Get local updates from this rank (children, parents, and failed parents) + metadata
            # Pass current_step for strict PUCT update mode to prevent cross-batch contamination
            puct_update_result = workflow.get_pending_updates(clear=True, current_step=global_step)
            
            # Handle both old (4-tuple) and new (5-tuple) return formats
            if len(puct_update_result) == 5:
                local_children, local_parents, local_failed, rollout_metadata, cross_batch_info = puct_update_result
            else:
                local_children, local_parents, local_failed, rollout_metadata = puct_update_result
                cross_batch_info = {}
            
            # Log cross-batch contamination prevention (strict mode)
            if cross_batch_info.get('mode') == 'strict' and cross_batch_info.get('n_delayed', 0) > 0:
                logger.info(f"[Step {global_step}] STRICT_PUCT: Prevented cross-batch contamination. "
                           f"This batch: {cross_batch_info['n_this_batch']}, "
                           f"Delayed: {cross_batch_info['n_delayed']}")
            
            # Calculate staleness statistics from metadata
            if rollout_metadata:
                staleness_values = [m['staleness'] for m in rollout_metadata]
                exec_times = [m.get('exec_time_ms', 0) for m in rollout_metadata if m.get('exec_time_ms', 0) > 0]
                
                avg_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0
                max_staleness = max(staleness_values) if staleness_values else 0
                min_staleness = min(staleness_values) if staleness_values else 0
                
                if is_dp_head:
                    logger.info(f"[Step {global_step}] STALENESS_BATCH: n={len(staleness_values)}, "
                               f"avg={avg_staleness:.2f}, min={min_staleness}, max={max_staleness}")
                    if exec_times:
                        avg_exec_time = sum(exec_times) / len(exec_times)
                        logger.info(f"[Step {global_step}] EXEC_TIME: n={len(exec_times)}, avg={avg_exec_time:.2f}ms")
            
            # Optional: Log local update counts (high-level)
            logger.info(f"[Step {global_step}][Rank {actor.dp_rank}] Local updates: "
                       f"children={len(local_children)}, parents={len(local_parents)}, failed={len(local_failed)}")
            
            # Unified synchronization through actor.sync_sampler()
            # This replaces the old: gather_states_and_failures_across_ranks + manual update + sync_sampler_state_from_rank0
            actor.sync_sampler(
                local_children=local_children,
                local_parents=local_parents,
                local_failed=local_failed,
                step=global_step
            )
            
            # Record PUCT update events for three-core-metrics analysis
            if hasattr(workflow, 'log_puct_update') and hasattr(sampler, '_m'):
                for parent in local_parents:
                    pid = parent.id
                    if pid in sampler._m:
                        m_value = sampler._m[pid]
                        n_visits = sampler._n.get(pid, 0)
                        
                        # Get children rewards for this parent
                        children_rewards = [
                            c.value for c in local_children
                            if any(p.get('id') == pid for p in (c.parents or []))
                            and c.value is not None
                        ]
                        
                        score = m_value  # Q-value component
                        
                        try:
                            workflow.log_puct_update(
                                parent_id=pid,
                                update_step=global_step,
                                n_visits=n_visits,
                                m_value=m_value,
                                score=score,
                                children_rewards=children_rewards,
                            )
                        except Exception as e:
                            logger.debug(f"[PUCT_TRACK] Failed to log update for {pid[:8]}: {e}")
            
            # Optional: Post-sync verification (lightweight checksum)
            # Note: Detailed logging is now handled inside actor.sync_sampler()
            if dist.is_initialized() and actor.data_parallel_world_size > 1:
                import hashlib
                if hasattr(sampler, '_states'):
                    state_ids = sorted([s.id for s in sampler._states])
                    hash_input = ','.join(state_ids).encode('utf-8')
                    state_checksum = int(hashlib.md5(hash_input).hexdigest(), 16) & 0xFFFFFFFF
                else:
                    state_checksum = 0
                
                all_checksums = [0] * actor.data_parallel_world_size
                dist.all_gather_object(all_checksums, state_checksum, group=actor.data_parallel_group)
                
                if len(set(all_checksums)) != 1:
                    logger.warning(f"[Step {global_step}] Cross-rank sampler state INCONSISTENT!")
        
        # Log rewards and actual rollout count
        local_rollouts = batch["rewards"].shape[0]
        step_rewards = batch["rewards"].cpu().numpy()
        step_max_reward = float(step_rewards.max())
        step_mean_reward = float(step_rewards.mean())
        
        # ============================================================
        # Global Reward Statistics (All-Reduce across all ranks)
        # ============================================================
        if dist.is_initialized():
            # Convert to tensors for all-reduce
            local_max = torch.tensor([step_max_reward], dtype=torch.float32, device=actor.device)
            local_sum = torch.tensor([step_rewards.sum()], dtype=torch.float32, device=actor.device)
            local_count = torch.tensor([len(step_rewards)], dtype=torch.float32, device=actor.device)
            
            # All-reduce: MAX for max reward, SUM for sum and count
            dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)
            
            # Update with global statistics
            step_max_reward = local_max.item()
            step_mean_reward = (local_sum / local_count).item() if local_count.item() > 0 else 0.0
            global_rollouts = int(local_count.item())
        else:
            global_rollouts = local_rollouts
        
        best_reward = max(best_reward, step_max_reward)
        
        # ============================================================
        # Record Training History (for visualization)
        # ============================================================
        local_step_rewards = step_rewards.tolist()
        
        # Get best solution from sampler (rank 0 only)
        current_best_solution = None
        if is_dp_head and hasattr(sampler, 'get_best_solution'):
            try:
                current_best_solution = sampler.get_best_solution()
            except Exception as e:
                logger.warning(f"Failed to get best solution from sampler: {e}")
        
        # Collect detailed metrics for this step
        step_metrics = {
            'batch_parents': batch_size,
            'group_size': group_size,
            'local_rollouts': local_rollouts,
            'global_rollouts': global_rollouts,
            'step_max_reward': step_max_reward,
            'step_mean_reward': step_mean_reward,
        }
        
        # Add staleness statistics if available
        if rollout_metadata:
            staleness_values = [m['staleness'] for m in rollout_metadata]
            exec_times = [m.get('exec_time_ms', 0) for m in rollout_metadata if m.get('exec_time_ms', 0) > 0]
            step_metrics['avg_staleness'] = sum(staleness_values) / len(staleness_values) if staleness_values else 0
            step_metrics['max_staleness'] = max(staleness_values) if staleness_values else 0
            if exec_times:
                step_metrics['avg_exec_time_ms'] = sum(exec_times) / len(exec_times)
        
        # ============================================================
        # Distributed Data Aggregation (Sync version)
        # Synchronize rollout_metadata and puct_analysis across ranks
        # ============================================================
        if dist.is_initialized() and actor.data_parallel_world_size > 1:
            world_size = actor.data_parallel_world_size
            
            # 1. Sync rollout_metadata (all_gather_object)
            if rollout_metadata:
                all_metadata = [[] for _ in range(world_size)]
                dist.all_gather_object(all_metadata, rollout_metadata)
                # Flatten list of lists
                global_rollout_metadata = []
                for rank_metadata in all_metadata:
                    global_rollout_metadata.extend(rank_metadata)
            else:
                global_rollout_metadata = []
            
            # 2. Sync puct_analysis (merge across ranks)
            puct_analysis_data = None
            if hasattr(workflow, 'get_puct_analysis_data'):
                try:
                    local_puct_data = workflow.get_puct_analysis_data(clear=False)
                    if local_puct_data:
                        all_puct_data = [None] * world_size
                        dist.all_gather_object(all_puct_data, local_puct_data)
                        # Merge: combine parent_episodes and puct_updates from all ranks
                        global_puct_data = {
                            'parent_episodes': {},
                            'puct_updates': [],
                        }
                        for rank_data in all_puct_data:
                            if rank_data:
                                global_puct_data['parent_episodes'].update(
                                    rank_data.get('parent_episodes', {})
                                )
                                global_puct_data['puct_updates'].extend(
                                    rank_data.get('puct_updates', [])
                                )
                        puct_analysis_data = global_puct_data
                except Exception as e:
                    logger.debug(f"[PUCT_TRACK] Failed to get analysis data: {e}")
                    puct_analysis_data = None
            
            # 3. Update step_metrics staleness stats based on global data
            if global_rollout_metadata:
                global_staleness = [m['staleness'] for m in global_rollout_metadata]
                global_exec_times = [m.get('child_exec_time_ms', 0) 
                                    for m in global_rollout_metadata 
                                    if m.get('child_exec_time_ms', 0) > 0]
                
                step_metrics['avg_staleness'] = sum(global_staleness) / len(global_staleness) if global_staleness else 0
                step_metrics['max_staleness'] = max(global_staleness) if global_staleness else 0
                if global_exec_times:
                    step_metrics['avg_exec_time_ms'] = sum(global_exec_times) / len(global_exec_times)
        else:
            # Single rank: use local data directly
            global_rollout_metadata = rollout_metadata if rollout_metadata else []
            puct_analysis_data = None
            if hasattr(workflow, 'get_puct_analysis_data'):
                try:
                    puct_analysis_data = workflow.get_puct_analysis_data(clear=False)
                except Exception as e:
                    logger.debug(f"[PUCT_TRACK] Failed to get analysis data: {e}")
        
        # Record to history logger with synchronized global data
        history_logger.record_step(
            step=global_step,
            rewards=local_step_rewards,
            best_solution=current_best_solution,
            additional_metrics=step_metrics,
            rollout_metadata=global_rollout_metadata if global_rollout_metadata else None,
            puct_analysis_data=puct_analysis_data,
        )
        
        logger.info(f"[Rank {actor.dp_rank}][Step {global_step}] Rollouts: {local_rollouts} (global: {global_rollouts}), "
                   f"batch parents: {batch_size}, group_size: {group_size}")
        
        # Collect metrics for stats_logger
        metrics = {
            "rollout/count": global_rollouts,
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
            "train/entropy": stats.get('ppo_actor/update/entropy/avg', 0.0),
            "train/actor_loss": stats.get('ppo_actor/update/actor_loss/avg', 0.0),
            "train/approx_kl": stats.get('ppo_actor/update/approx_kl/avg', 0.0),
            "train/grad_norm": stats.get('ppo_actor/update/grad_norm', 0.0),
            "train/lr": stats.get('ppo_actor/update/lr', 0.0),
            "train/importance_weight": stats.get('ppo_actor/update/importance_weight/avg', 0.0),
            "train/clip_ratio": stats.get('ppo_actor/update/clip_ratio/avg', 0.0),
        })
        
        metrics.update({
            "train/advantages/avg": stats.get('ppo_actor/advantages/avg', 0.0),
            "train/advantages/max": stats.get('ppo_actor/advantages/max', 0.0),
            "train/advantages/min": stats.get('ppo_actor/advantages/min', 0.0),
        })
        
        # Log to stats_logger (wandb/swanlab/tensorboard)
        if is_dp_head:
            stats_logger.commit(
                epoch=step_info.epoch,
                step=step_info.epoch_step,
                global_step=global_step,
                data=metrics,
            )
            logger.info(
                f"[Step {global_step}] Reward: max={step_max_reward:.4f}, mean={step_mean_reward:.4f}, best={best_reward:.4f} | "
                f"Loss: {metrics['train/actor_loss']:.4f}, KL: {metrics['train/approx_kl']:.4f}, "
                f"Entropy: {metrics['train/entropy']:.4f}, GradNorm: {metrics['train/grad_norm']:.4f}, LR: {metrics['train/lr']:.6f}"
            )
        
        # Save Training History Checkpoint
        # Each rank saves its own checkpoint (logger configured with rank-specific filename)
        checkpoint_path = history_logger.save_checkpoint()
        if checkpoint_path:
            logger.debug(f"[TTTLogger] Saved checkpoint to {checkpoint_path}")
                
        rollout.pause()
        
        with stats_tracker.record_timing("update_weights"):
            logger.info(f"[Rank {actor.dp_rank}][Step {global_step}] Updating weights...")
            actor.update_weights(weight_update_meta)
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
        
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
    
    # ============================================================
    # Save Final Training History
    # ============================================================
    if is_dp_head:
        history_path = history_logger.save(also_save_json=True)
        if history_path:
            logger.info(f"[TTTLogger] Final training history saved to {history_path}")
            
            summary = history_logger.get_summary()
            logger.info(
                f"[TTTLogger] Training Summary:\n"
                f"  - Recorded snapshots: {summary['num_snapshots']}\n"
                f"  - Recorded steps: {summary['recorded_steps']}\n"
                f"  - Overall best reward: {summary['overall_best_reward']:.4f} (step {summary['overall_best_step']})\n"
                f"  - Has Best-of-N baseline: {summary['has_best_of_n']}"
            )
            
            logger.info(
                f"\n[TTTLogger] To generate visualization, run:\n"
                f"  python areal/experimental/ttt_discover/generate_plot.py \\\n"
                f"    --history_path {history_path} \\\n"
                f"    --benchmark_value 2.635983"
            )
    
    # Cleanup
    workflow.shutdown()
    stats_logger.close()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
