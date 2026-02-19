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

from areal.experimental.ttt_discover.config import TTTDPPOActorConfig
from areal.experimental.ttt_discover.actor import TTTDActor
from areal.experimental.ttt_discover.dataloader import create_tttd_dataloader
from areal.experimental.ttt_discover.sampler import create_sampler_from_config
from areal.experimental.ttt_discover.workflow_v2 import TTTDiscoverWorkflowV2
from areal.experimental.ttt_discover.ttt_logger import TTTTrainingLogger


def _ensure_lora_initialized(config: TTTDPPOActorConfig, rank: int = 0) -> str:
    """
    Ensure initial LoRA adapter exists for vLLM to load at startup.
    
    This function should be called BEFORE AReaL framework initialization.
    Only rank 0 performs the actual initialization, other ranks wait via barrier.
    
    Args:
        config: Training configuration
        rank: Current process rank (0 = main rank)
        
    Returns:
        Path to LoRA adapter directory
    """
    import json
    
    # Extract lora_modules path from vllm config
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
    
    # Convert to absolute path
    lora_output_path = os.path.abspath(lora_output_path)
    adapter_config_path = os.path.join(lora_output_path, "adapter_config.json")
    
    # Only rank 0 performs initialization
    if rank == 0:
        if os.path.exists(adapter_config_path):
            logger.info(f"[LoRA Init] Adapter already exists at {lora_output_path}, skipping initialization")
            return lora_output_path
        
        if not config.use_lora:
            logger.info("[LoRA Init] LoRA is not enabled (use_lora=false), skipping initialization")
            return lora_output_path
        
        logger.info("="*80)
        logger.info("[LoRA Init] Creating initial LoRA adapter for vLLM")
        logger.info("="*80)
        logger.info(f"  Base model: {config.path}")
        logger.info(f"  Output path: {lora_output_path}")
        logger.info(f"  LoRA rank: {config.lora_rank}, alpha: {config.lora_alpha}")
        
        try:
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Create parent directory
            parent_dir = os.path.dirname(lora_output_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Load base model
            logger.info("[LoRA Init] Loading base model...")
            model = AutoModelForCausalLM.from_pretrained(
                config.path,
                torch_dtype="auto",
                device_map="cpu",
            )
            
            # Load tokenizer
            tok_path = config.tokenizer_path or config.path
            logger.info(f"[LoRA Init] Loading tokenizer from {tok_path}...")
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
            
            # Configure LoRA
            target_modules = config.target_modules if config.target_modules else ["all-linear"]
            target_mods = "all-linear" if target_modules == ["all-linear"] else target_modules
            
            lora_config = LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                target_modules=target_mods,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # Apply LoRA
            logger.info("[LoRA Init] Applying LoRA configuration...")
            model = get_peft_model(model, lora_config)
            
            # Save
            logger.info(f"[LoRA Init] Saving LoRA adapter to {lora_output_path}...")
            os.makedirs(lora_output_path, exist_ok=True)
            model.save_pretrained(lora_output_path)
            tokenizer.save_pretrained(lora_output_path)
            
            logger.info(f"[LoRA Init] ✓ LoRA adapter created successfully")
            logger.info(f"[LoRA Init] Contents: {os.listdir(lora_output_path)}")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"[LoRA Init] Failed to create LoRA adapter: {e}")
            raise RuntimeError(f"LoRA initialization failed: {e}") from e
    
    return lora_output_path

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


def gather_states_and_failures_across_ranks(actor, local_children, local_parents, local_failed):
    """
    Gather states and failed parents from all data parallel ranks.
    
    IMPORTANT: This is a collective operation that MUST be called by ALL ranks.
    
    Args:
        actor: TTTDActor with data parallel info
        local_children: List of child states from this rank
        local_parents: List of parent states from this rank
        local_failed: List of failed parent states from this rank
        
    Returns:
        (all_children, all_parents, all_failed) on ALL ranks
    """
    if not dist.is_initialized() or actor.data_parallel_world_size <= 1:
        return local_children, local_parents, local_failed
    
    from areal.experimental.ttt_discover.state import state_from_dict
    
    # Serialize states to dicts for communication
    children_dicts = [s.to_dict() for s in local_children]
    parents_dicts = [s.to_dict() for s in local_parents]
    failed_dicts = [s.to_dict() for s in local_failed]
    
    # Gather from all ranks
    all_children_dicts = [None] * actor.data_parallel_world_size
    all_parents_dicts = [None] * actor.data_parallel_world_size
    all_failed_dicts = [None] * actor.data_parallel_world_size
    
    dist.all_gather_object(all_children_dicts, children_dicts, group=actor.data_parallel_group)
    dist.all_gather_object(all_parents_dicts, parents_dicts, group=actor.data_parallel_group)
    dist.all_gather_object(all_failed_dicts, failed_dicts, group=actor.data_parallel_group)
    
    # Deserialize and combine all states on ALL ranks
    all_children = []
    all_parents = []
    all_failed = []
    
    for rank_children, rank_parents, rank_failed in zip(all_children_dicts, all_parents_dicts, all_failed_dicts):
        if rank_children:
            all_children.extend([state_from_dict(d) for d in rank_children])
        if rank_parents:
            all_parents.extend([state_from_dict(d) for d in rank_parents])
        if rank_failed:
            all_failed.extend([state_from_dict(d) for d in rank_failed])
    
    return all_children, all_parents, all_failed


def sync_sampler_state_from_rank0(actor, sampler):
    """
    Synchronize the complete sampler state from rank 0 to all other ranks.
    
    This ensures all ranks have identical sampler state after rank 0 performs
    all the updates (failed rollouts + successful updates).
    
    Args:
        actor: TTTDActor with data parallel info
        sampler: PUCTSampler instance
    """
    if not dist.is_initialized() or actor.data_parallel_world_size <= 1:
        return
    
    from areal.experimental.ttt_discover.state import state_from_dict
    
    # Prepare state data on rank 0
    if actor.dp_rank == 0:
        state_data = {
            'states': [s.to_dict() for s in sampler._states],
            'initial_states': [s.to_dict() for s in sampler._initial_states],
            'T': sampler._T,
            'n': sampler._n,
            'm': sampler._m,
            'current_step': sampler._current_step,
            'last_sampled_states': [s.to_dict() for s in sampler._last_sampled_states],
            'last_sampled_indices': sampler._last_sampled_indices,
            'last_puct_stats': sampler._last_puct_stats,
            'last_scale': sampler._last_scale,
        }
        objects_to_broadcast = [state_data]
    else:
        objects_to_broadcast = [None]
    
    # Broadcast from rank 0 to all ranks
    # Note: broadcast_object_list modifies the list in-place on all ranks
    dist.broadcast_object_list(objects_to_broadcast, src=0, group=actor.data_parallel_group)
    
    # Extract the received data
    state_data = objects_to_broadcast[0]
    
    # Non-rank-0 ranks update their sampler state
    if actor.dp_rank != 0:
        sampler._states = [state_from_dict(d) for d in state_data['states']]
        sampler._initial_states = [state_from_dict(d) for d in state_data['initial_states']]
        sampler._T = state_data['T']
        sampler._n = state_data['n']
        sampler._m = state_data['m']
        sampler._current_step = state_data['current_step']
        sampler._last_sampled_states = [state_from_dict(d) for d in state_data['last_sampled_states']]
        sampler._last_sampled_indices = state_data['last_sampled_indices']
        sampler._last_puct_stats = state_data['last_puct_stats']
        sampler._last_scale = state_data['last_scale']
        
        logger.info(f"[Rank {actor.dp_rank}] Received sampler state from rank 0: "
                    f"T={sampler._T}, n_entries={len(sampler._n)}, m_entries={len(sampler._m)}, "
                    f"states={len(sampler._states)}")


def gather_states_across_ranks(actor, local_children, local_parents):
    """
    Gather states from all data parallel ranks using all_gather_object.
    
    IMPORTANT: This is a collective operation that MUST be called by ALL ranks
    in the data_parallel_group. If any rank skips this call, it will cause deadlock.
    
    Args:
        actor: TTTDActor with data parallel info
        local_children: List of child states from this rank
        local_parents: List of parent states from this rank
        
    Returns:
        (all_children, all_parents) on ALL ranks (not just rank 0)
    """
    if not dist.is_initialized() or actor.data_parallel_world_size <= 1:
        return local_children, local_parents
    
    # Serialize states to dicts for communication
    children_dicts = [s.to_dict() for s in local_children]
    parents_dicts = [s.to_dict() for s in local_parents]
    
    # Gather from all ranks using all_gather_object (handles NCCL automatically)
    all_children_dicts = [None] * actor.data_parallel_world_size
    all_parents_dicts = [None] * actor.data_parallel_world_size
    
    # CRITICAL: all_gather_object is collective - ALL ranks must call this
    dist.all_gather_object(all_children_dicts, children_dicts, group=actor.data_parallel_group)
    dist.all_gather_object(all_parents_dicts, parents_dicts, group=actor.data_parallel_group)
    
    # Deserialize and combine all states on ALL ranks
    # (Previously this only happened on rank 0, causing inconsistency)
    from areal.experimental.ttt_discover.state import state_from_dict
    
    all_children = []
    all_parents = []
    
    for rank_children, rank_parents in zip(all_children_dicts, all_parents_dicts):
        if rank_children:  # Skip empty lists
            all_children.extend([state_from_dict(d) for d in rank_children])
        if rank_parents:
            all_parents.extend([state_from_dict(d) for d in rank_parents])
    
    return all_children, all_parents


def _concat_trajectories(trajectories: list[dict]) -> dict:
    """
    Concatenate a list of trajectory dicts into a single batch dict.
    This is a simpler alternative to redistribute_trajectories that doesn't
    require all_gather from all DP ranks (avoiding the deadlock issue).
    """
    if not trajectories:
        return {}
    
    from areal.utils.data import concat_padded_tensors
    return concat_padded_tensors(trajectories)


def main(args):
    config, _ = load_expr_config(args, TTTDPPOActorConfig)
    config: TTTDPPOActorConfig
    
    # ============================================================
    # Step 0: Initialize LoRA adapter (before AReaL framework)
    # ============================================================
    # Note: We do this early before any distributed initialization
    # Rank 0 creates the adapter, others will wait at the barrier later
    # 
    # IMPORTANT: LoRA is initialized ONLY ONCE per experiment. The adapter is
    # saved to disk and reused across training restarts. To force re-initialization,
    # delete the existing adapter directory (e.g., ./lora_init/) first.
    if not config.skip_lora_init and config.use_lora:
        # At this point, we haven't initialized distributed yet
        # So we use environment variables to determine rank
        init_rank = int(os.environ.get('RANK', 0))
        _ensure_lora_initialized(config, rank=init_rank)
    
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    
    # Create actor first to get correct dp_rank
    actor = TTTDActor(config=config)
    allocation_mode = AllocationMode.from_str(config.allocation_mode)
    parallel_strategy = allocation_mode["actor"].parallel
    assert parallel_strategy is not None
    actor.create_process_group(parallel_strategy=parallel_strategy)
    
    # Use AReaL's dp_rank instead of environment variable RANK
    rank = actor.dp_rank
    world_size = actor.data_parallel_world_size
    
    seeding.set_random_seed(config.seed, key=f"trainer{rank}")
    
    # Create sampler first (before workflow)
    sampler = create_sampler_from_config(
        config=config.sampler,
        log_path=config.saver.fileroot,
        env_type="custom",
    )
    
    batch_size = config.sampler.batch_size  # Total parents per step (e.g., 8)
    group_size = config.gconfig.n_samples   # Rollouts per parent (e.g., 64)
    
    # dp_rank == 0 is the DP head
    is_dp_head = rank == 0
    
    # DEBUG: Log distributed configuration
    if rank == 0:
        logger.info(f"=== Distributed Configuration ===")
        logger.info(f"DP world size: {world_size}")
        logger.info(f"Sampler batch_size: {batch_size}")
        logger.info(f"Group size: {group_size}")
        logger.info(f"Total rollouts: {batch_size * group_size}")
        logger.info(f"Is DP head: {is_dp_head}")
        logger.info(f"================================")
    
    # Distributed Sampler Mode: Each rank samples its own shard
    # States are synchronized across ranks after each step via gather_states_across_ranks
    use_dp_shard = world_size > 1
    if rank == 0:
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
        only_dp_head=False,           # Use distributed sharding (each rank samples its own shard)
    )
    
    ft_spec = FinetuneSpec(
        total_train_epochs=config.total_train_epochs,
        dataset_size=len(train_dataloader) * batch_size,
        train_batch_size=batch_size,
    )
    
    actor.initialize(None, ft_spec)
    
    # ============================================================
    # Synchronize after LoRA initialization
    # ============================================================
    # Ensure all ranks wait for rank 0 to complete LoRA adapter creation
    if dist.is_initialized() and not config.skip_lora_init and config.use_lora:
        logger.info(f"[Rank {rank}] Waiting for LoRA initialization barrier...")
        dist.barrier()
        logger.info(f"[Rank {rank}] LoRA initialization barrier passed")
    
    # ============================================================
    # CONFIGURATION VERIFICATION for Scheme A (DP Head Only)
    # ============================================================
    if rank == 0:
        logger.info("=== Scheme A Configuration Verification ===")
        logger.info(f"Allocation mode: {config.allocation_mode}")
        logger.info(f"Parallel Strategy: {parallel_strategy}")
        logger.info(f"  - DP size: {parallel_strategy.dp_size}")
        logger.info(f"  - TP size: {parallel_strategy.tp_size}")
        logger.info(f"  - PP size: {parallel_strategy.pp_size}")
        logger.info(f"Actor DP rank: {actor.dp_rank}")
        logger.info(f"Actor DP world size: {actor.data_parallel_world_size}")
        logger.info(f"Is DP head: {rank == 0}")
        
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
        if not rank == 0:
            logger.error(
                f"ERROR: Rank {actor.dp_rank} is not DP head but only_dp_head=True. "
                f"This rank will not produce any data!"
            )
        logger.info("==========================================")
    
    # Barrier to ensure all ranks see the verification
    if dist.is_initialized():
        dist.barrier()
    
    # Verify LoRA adapter exists (should have been created by _ensure_lora_initialized)
    # Use absolute path based on current working directory
    import os as _os
    _original_cwd = _os.getcwd()
    
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
            error_msg = (
                f"Initial LoRA adapter not found at {lora_output_path}\n\n"
                f"This should have been created automatically at the start of training.\n"
                f"Possible causes:\n"
                f"  1. LoRA initialization failed on rank 0\n"
                f"  2. The adapter path was changed after initialization\n"
                f"  3. You're resuming from a different working directory\n\n"
                f"Solutions:\n"
                f"  - Check rank 0 logs for initialization errors\n"
                f"  - Run prepare_lora_init.py manually before training\n"
                f"  - Use --skip-lora-init if you have already initialized manually"
            )
            raise RuntimeError(error_msg)
        
        if rank == 0:
            logger.info(f"[LoRA Init] Verified LoRA adapter exists at {lora_output_path}")
    
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
        # Reference model uses DP-only strategy to avoid TP + torch.compile issues
        # This is necessary because TP with torch.compile causes sharding errors
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
    
    # ============================================================
    # Initialize Training History Logger
    # ============================================================
    save_steps = getattr(config, 'save_steps', [i for i in range(0, max_steps)])
    # 过滤只保留在训练范围内的 steps
    save_steps = [s for s in save_steps if start_step <= s < max_steps]
    
    history_logger = TTTTrainingLogger(
        save_steps=save_steps,
        output_dir=config.saver.fileroot,
        is_dp_head=is_dp_head,
        filename='training_history.pkl',
        checkpoint_filename='training_history_checkpoint.pkl',
        aggregate_distributed=True,
    )
    
    if is_dp_head:
        logger.info(f"[TTTLogger] Initialized with save_steps: {save_steps}")
        logger.info(f"[TTTLogger] Output directory: {config.saver.fileroot}")
    
    logger.info(f"Starting training from step {start_step}/{max_steps}")
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
        # NOTE: gather_states_and_failures_across_ranks is a collective operation - ALL ranks must call it
        with stats_tracker.record_timing("sampler_update"):
            # Get local updates from this rank (children, parents, and failed parents)
            local_children, local_parents, local_failed = workflow.get_pending_updates_sync(clear=True)
            
            # === PRE-SYNC LOGGING (All Ranks) ===
            # Log local pending updates before aggregation
            logger.info(f"\n{'='*80}")
            logger.info(f"[Rank {actor.dp_rank}][Step {global_step}] PRE-SYNC: Local Pending Updates")
            logger.info(f"{'='*80}")
            logger.info(f"Local children count: {len(local_children)}")
            logger.info(f"Local parents count: {len(local_parents)}")
            logger.info(f"Local failed count: {len(local_failed)}")
            if local_children:
                logger.info(_format_state_table(local_children, "Local Children States"))
            if local_parents:
                logger.info(_format_state_table(local_parents, "Local Parent States"))
            if local_failed:
                logger.info(_format_state_table(local_failed, "Local Failed Parents"))
            
            # Log current sampler state BEFORE sync
            logger.info(_format_sampler_summary(sampler, f"[Rank {actor.dp_rank}][Step {global_step}] PRE-SYNC Sampler State"))
            
            # Aggregate updates from all ranks using all_gather_object
            # CRITICAL: This must be called by ALL ranks in the DP group
            all_children, all_parents, all_failed = gather_states_and_failures_across_ranks(
                actor, local_children, local_parents, local_failed
            )
            
            # Log aggregated updates from all ranks
            logger.info(f"\n{'='*80}")
            logger.info(f"[Rank {actor.dp_rank}][Step {global_step}] AGGREGATED Updates (All Ranks)")
            logger.info(f"{'='*80}")
            logger.info(f"Total children from all ranks: {len(all_children)}")
            logger.info(f"Total parents from all ranks: {len(all_parents)}")
            logger.info(f"Total failed from all ranks: {len(all_failed)}")
            
            # Group by source rank for clarity
            if all_children:
                parent_counts = {}
                for child, parent in zip(all_children, all_parents):
                    pid = parent.id
                    if pid not in parent_counts:
                        parent_counts[pid] = {"count": 0, "values": [], "timestep": parent.timestep}
                    parent_counts[pid]["count"] += 1
                    parent_counts[pid]["values"].append(child.value if child.value is not None else float('-inf'))
                
                logger.info(f"\nAggregated updates by parent:")
                logger.info(f"{'Parent ID':<36} {'Children':<10} {'Timestep':<10} {'Min Value':<12} {'Max Value':<12}")
                logger.info("-" * 90)
                for pid, info in sorted(parent_counts.items()):
                    min_val = min(info["values"]) if info["values"] else "N/A"
                    max_val = max(info["values"]) if info["values"] else "N/A"
                    min_str = f"{min_val:.4f}" if isinstance(min_val, float) else str(min_val)
                    max_str = f"{max_val:.4f}" if isinstance(max_val, float) else str(max_val)
                    logger.info(f"{pid:<36} {info['count']:<10} {info['timestep']:<10} {min_str:<12} {max_str:<12}")
            
            # === CRITICAL: Only Rank 0 performs all updates ===
            # This ensures _T, _n, _m are computed correctly without duplication
            if rank == 0:
                # Step 1: Record all failed rollouts (updates _T and _n)
                if all_failed:
                    for failed_parent in all_failed:
                        sampler.record_failed_rollout(failed_parent)
                    logger.info(f"[Rank 0][Step {global_step}] Recorded {len(all_failed)} failed rollouts")
                
                # Step 2: Record all successful updates (updates _T, _n, _m, and adds to _states)
                if all_children:
                    sampler.update_states(all_children, all_parents, save=False)
                    logger.info(f"[Rank 0][Step {global_step}] Updated sampler with {len(all_children)} children "
                                f"from {len(set(p.id for p in all_parents))} unique parents")
                
                # Step 3: Flush to disk
                sampler.flush(step=global_step)
            
            # === CRITICAL: Synchronize complete sampler state from rank 0 to all ranks ===
            # This ensures all ranks have identical sampler state for the next sampling
            sync_sampler_state_from_rank0(actor, sampler)
            
            # === POST-SYNC LOGGING (All Ranks) ===
            logger.info(_format_sampler_summary(sampler, f"[Rank {actor.dp_rank}][Step {global_step}] POST-SYNC Sampler State"))
            
            # Cross-rank consistency check
            if dist.is_initialized() and actor.data_parallel_world_size > 1:
                # Get sampler state hash for comparison across ranks
                # Use deterministic hash (hash() is randomized in Python)
                import hashlib
                if hasattr(sampler, '_states'):
                    state_ids = sorted([s.id for s in sampler._states])
                    # Use MD5 for deterministic hash
                    hash_input = ','.join(state_ids).encode('utf-8')
                    state_checksum = int(hashlib.md5(hash_input).hexdigest(), 16) & 0xFFFFFFFF
                else:
                    state_checksum = 0
                
                # Gather checksums from all ranks
                all_checksums = [0] * actor.data_parallel_world_size
                dist.all_gather_object(all_checksums, state_checksum, group=actor.data_parallel_group)
                
                logger.info(f"\n{'='*80}")
                logger.info(f"[Rank {actor.dp_rank}][Step {global_step}] Cross-Rank Consistency Check")
                logger.info(f"{'='*80}")
                logger.info(f"Sampler state checksums across ranks: {all_checksums}")
                if len(set(all_checksums)) == 1:
                    logger.info("✓ All ranks have CONSISTENT sampler state")
                else:
                    logger.warning("✗ Ranks have INCONSISTENT sampler state! Checksums differ!")
                    for r, cs in enumerate(all_checksums):
                        status = "OK" if cs == all_checksums[0] else "DIFF"
                        logger.warning(f"  Rank {r}: checksum={cs} [{status}]")
                logger.info(f"{'='*80}\n")
        
        # Log rewards and actual rollout count
        local_rollouts = batch["rewards"].shape[0]  # Rollouts on this rank
        step_rewards = batch["rewards"].cpu().numpy()
        step_max_reward = float(step_rewards.max())
        step_mean_reward = float(step_rewards.mean())
        best_reward = max(best_reward, step_max_reward)
        
        # ============================================================
        # Record Training History (for visualization)
        # ============================================================
        # 收集所有 rollouts 的 rewards（从 batch 中提取）
        local_step_rewards = step_rewards.tolist()
        
        # 找到当前 step 的最佳解（如果 workflow 支持）
        current_best_solution = None
        if hasattr(workflow, 'get_best_solution'):
            try:
                current_best_solution = workflow.get_best_solution()
            except:
                pass
        
        # 记录当前 step 数据
        history_logger.record_step(
            step=global_step,
            rewards=local_step_rewards,
            best_solution=current_best_solution,
            additional_metrics={
                'batch_parents': batch_size,
                'group_size': group_size,
                'local_rollouts': local_rollouts,
            }
        )
        
        # Log rollout distribution info
        logger.info(f"[Rank {actor.dp_rank}][Step {global_step}] Rank {actor.dp_rank} rollouts: {local_rollouts}, "
                   f"batch parents: {batch_size}, group_size: {group_size}, "
                   f"expected total: {batch_size * group_size}")
        
        # Collect metrics for stats_logger
        metrics = {
            "rollout/count": local_rollouts,
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
        
        # AReaL stats use 'ppo_actor/' prefix with scope paths
        # Correct keys: 'ppo_actor/update/actor_loss/avg', 'ppo_actor/update/entropy/avg', etc.
        metrics.update({
            "train/entropy": stats.get('ppo_actor/update/entropy/avg', 0.0),
            "train/actor_loss": stats.get('ppo_actor/update/actor_loss/avg', 0.0),
            "train/approx_kl": stats.get('ppo_actor/update/approx_kl/avg', 0.0),
            "train/grad_norm": stats.get('ppo_actor/update/grad_norm', 0.0),
            "train/lr": stats.get('ppo_actor/update/lr', 0.0),
            "train/importance_weight": stats.get('ppo_actor/update/importance_weight/avg', 0.0),
            "train/clip_ratio": stats.get('ppo_actor/update/clip_ratio/avg', 0.0),
        })
        
        # Also add advantage stats from ppo_actor scope
        metrics.update({
            "train/advantages/avg": stats.get('ppo_actor/advantages/avg', 0.0),
            "train/advantages/max": stats.get('ppo_actor/advantages/max', 0.0),
            "train/advantages/min": stats.get('ppo_actor/advantages/min', 0.0),
        })
        
        # Log to stats_logger (wandb/swanlab/tensorboard)
        if rank == 0:
            stats_logger.commit(
                epoch=step_info.epoch,
                step=step_info.epoch_step,
                global_step=global_step,
                data=metrics,
            )
            # Print comprehensive training summary
            logger.info(
                f"[Step {global_step}] Reward: max={step_max_reward:.4f}, mean={step_mean_reward:.4f}, best={best_reward:.4f} | "
                f"Loss: {metrics['train/actor_loss']:.4f}, KL: {metrics['train/approx_kl']:.4f}, "
                f"Entropy: {metrics['train/entropy']:.4f}, GradNorm: {metrics['train/grad_norm']:.4f}, LR: {metrics['train/lr']:.6f}"
            )
        
        # ============================================================
        # Save Training History Checkpoint
        # ============================================================
        # 每个 step 后保存 checkpoint，支持中断恢复
        if is_dp_head:
            checkpoint_path = history_logger.save_checkpoint()
            if checkpoint_path:
                logger.debug(f"[TTTLogger] Saved checkpoint to {checkpoint_path}")
                
        rollout.pause()
        
        with stats_tracker.record_timing("update_weights"):
            logger.info(f"[Rank {actor.dp_rank}][Step {global_step}] Updating weights with meta: {weight_update_meta}")
            actor.update_weights(weight_update_meta)
            actor.set_version(global_step + 1)
            rollout.set_version(global_step + 1)
            eval_rollout.set_version(global_step + 1)
        
        logger.info(f"[Step {global_step}] Weights updated, now saving checkpoint")
        with stats_tracker.record_timing("save"):
            saver.save(actor, step_info.epoch, step_info.epoch_step, global_step, tokenizer=tokenizer)
        
        with stats_tracker.record_timing("checkpoint_for_recover"):
            recover_handler.dump(
                actor, step_info, saver, evaluator, stats_logger,
                train_dataloader, tokenizer=tokenizer,
            )
        
        # DEBUG: Log before barrier
        logger.info(f"[Step {global_step}] Rank {actor.dp_rank} (global rank {dist.get_rank() if dist.is_initialized() else 'N/A'}) reaching barrier")
        dist.barrier(group=actor.cpu_group)
        logger.info(f"[Step {global_step}] Rank {actor.dp_rank} passed barrier")
        current_platform.synchronize()
        rollout.resume()
    
    # ============================================================
    # Save Final Training History
    # ============================================================
    if is_dp_head:
        history_path = history_logger.save(also_save_json=True)
        if history_path:
            logger.info(f"[TTTLogger] Final training history saved to {history_path}")
            
            # 打印摘要
            summary = history_logger.get_summary()
            logger.info(
                f"[TTTLogger] Training Summary:\n"
                f"  - Recorded snapshots: {summary['num_snapshots']}\n"
                f"  - Recorded steps: {summary['recorded_steps']}\n"
                f"  - Overall best reward: {summary['overall_best_reward']:.4f} (step {summary['overall_best_step']})\n"
                f"  - Has Best-of-N baseline: {summary['has_best_of_n']}"
            )
            
            # 提示如何生成可视化
            logger.info(
                f"\n[TTTLogger] To generate visualization, run:\n"
                f"  python areal/experimental/ttt_discover/generate_plot.py \\\n"
                f"    --history_path {history_path} \\\n"
                f"    --benchmark_value 2.635983"
            )
    
    # Cleanup
    workflow.shutdown()  # Shutdown code execution thread pool
    stats_logger.close()
    eval_rollout.destroy()
    rollout.destroy()
    if ref is not None:
        ref.destroy()
    actor.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
