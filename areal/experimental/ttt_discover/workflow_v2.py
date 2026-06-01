# areal/experimental/ttt_discover/workflow_v2.py
"""
TTT-Discover Workflow V2 - 直接在内部更新 Sampler，无需外部 metadata 匹配。

这个版本解决了 metadata 长度不匹配的问题：
- Workflow 持有 Sampler 引用
- arun_episode 直接创建 child state 并更新 sampler
- 完全抛弃 metadata 列表和索引匹配

Usage:
    >>> workflow = TTTDiscoverWorkflowV2(
    ...     env=env,
    ...     sampler=sampler,  # 直接注入 sampler
    ...     gconfig=config.gconfig,
    ...     tokenizer=tokenizer,
    ... )
    >>> 
    >>> # 训练脚本中，不需要管理 metadata
    >>> batch = actor.prepare_batch(..., workflow=workflow, group_size=64)
    >>> workflow.flush()  # 确保所有更新完成
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable

import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker

from areal.utils.perf_tracer import atrace_session_phase, trace_session

from .envs.env import BaseEnv, EnvResult
from .reward import tttd_reward_fn

if TYPE_CHECKING:
    from .sampler import StateSampler
    from .state import State

logger = logging.getLogger("TTTDiscoverWorkflowV2")


class TTTDiscoverWorkflowV2(RolloutWorkflow):
    """
    TTT-Discover workflow that updates sampler internally.
    
    Key differences from V1:
    - Holds reference to sampler, updates it directly in arun_episode
    - No metadata list, no indexing issues
    - Thread-safe buffering for batch updates
    """

    def __init__(
        self,
        env: BaseEnv,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        sampler: "StateSampler" | None = None,
        reward_fn: Callable = tttd_reward_fn,
        enable_thinking: bool = False,
        auto_flush: bool = True,
        max_prompt_thinking_tokens: int = 26000,
        max_reward_workers: int | None = None,
        strict_sync_mode: bool = False,
        batch_size: int | None = None,
        group_size: int | None = None,
        puct_update_mode: str = "eager",
        lazy_sampling: bool = False,
        vllm_concurrency: int | None = None,  # Per-rank slot-level. Default: batch_size * group_size
        execution_concurrency: int = 64,  # Per-rank, should match AsyncRewardWrapper max_workers
        dp_rank: int = 0,
        dp_world_size: int = 1,
        hint_fn: Callable | None = None,
    ):
        """
        Initialize TTT-Discover Workflow V2.
        
        Args:
            env: Environment instance (e.g., InequalitiesEnv, CirclePackingEnv)
            gconfig: Generation hyperparameters
            tokenizer: Tokenizer or path to tokenizer
            sampler: PUCTSampler instance (required for lazy_sampling=True)
            reward_fn: Reward function following AReaL convention:
                fn(prompt, completions, prompt_ids, completion_ids, **data) -> float
                Default is tttd_reward_fn which requires _env and _state in data.
            enable_thinking: Whether to enable thinking mode
            auto_flush: Whether to auto-flush sampler updates
            max_prompt_thinking_tokens: Max tokens for prompt + thinking
            max_reward_workers: Max workers for AsyncRewardWrapper. 
                Defaults to TTTD_MAX_CODE_WORKERS env var or 16.
            strict_sync_mode: If True, strictly mimic sync version behavior by only
                processing batch_size * group_size rollouts per step with proper
                parent-child matching. If False (default), process all pending updates.
            batch_size: Number of parent states per step (required if strict_sync_mode=True)
            group_size: Number of rollouts per parent (required if strict_sync_mode=True)
            puct_update_mode: Controls how PUCTSampler is updated to prevent cross-batch contamination:
                - "eager" (default): Use all completed children to update PUCT immediately (async behavior)
                - "strict": Only use children from parents sampled in current step to update PUCT.
                  This prevents cross-batch contamination by delaying PUCT updates for late children.
                  Use this mode to isolate staleness effects on policy training from PUCT update effects.
            lazy_sampling: If True, defer PUCT sampling until VLLM has capacity (minimizes staleness)
            vllm_concurrency: Per-rank max concurrent VLLM generations (for lazy mode). 
                Should match vllm.max_num_seqs per instance (e.g., 128). Not divided by dp_world_size.
            execution_concurrency: Per-rank max concurrent solution executions (for lazy mode). 
                Should match AsyncRewardWrapper max_workers (default 64). Not divided by dp_world_size.
            dp_rank: Data parallel rank for distributed training
            dp_world_size: Total number of data parallel ranks
        """
        self.env = env
        self.hint_fn = hint_fn  # Optional hint function: fn(state) -> str
        self.sampler = sampler  # PUCTSampler reference for lazy sampling
        self.auto_flush = auto_flush
        self.max_prompt_thinking_tokens = max_prompt_thinking_tokens
        
        # Mode configuration
        self.strict_sync_mode = strict_sync_mode
        self.batch_size = batch_size or 8
        self.group_size = group_size or 64
        
        if strict_sync_mode:
            if batch_size is None or group_size is None:
                raise ValueError("batch_size and group_size must be provided when strict_sync_mode=True")
            self.expected_rollouts_per_step = batch_size * group_size
            logger.info(f"TTTDiscoverWorkflowV2: Strict sync mode enabled, expected_rollouts_per_step={self.expected_rollouts_per_step}")
        
        # Lazy PUCT sampling configuration
        self.lazy_sampling = lazy_sampling
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        
        if lazy_sampling:
            if sampler is None:
                raise ValueError("sampler must be provided when lazy_sampling=True")
            # Use per-rank concurrency directly (slot-level: allow pipeline flow)
            # vllm_concurrency default: batch_size * group_size (one full batch per rank)
            if vllm_concurrency is None:
                vllm_concurrency = self.batch_size * self.group_size
                logger.info(f"Auto vllm_concurrency = batch_size({self.batch_size}) * "
                           f"group_size({self.group_size}) = {vllm_concurrency}")
            self._vllm_sem = asyncio.Semaphore(max(1, vllm_concurrency))
            self._exec_sem = asyncio.Semaphore(max(1, execution_concurrency))
            # Batch-level parent caching (version mapping is in sampler)
            self._batch_parents: dict[int, list] = {}
            self._cache_lock = asyncio.Lock()
            logger.info(f"TTTDiscoverWorkflowV2: Lazy sampling enabled, "
                       f"vllm_sem={max(1, vllm_concurrency)} (slot-level per-rank), "
                       f"exec_sem={max(1, execution_concurrency)} (per-rank)")
        
        # PUCT update mode for controlling cross-batch contamination
        self.puct_update_mode = puct_update_mode
        if puct_update_mode not in ["eager", "strict"]:
            raise ValueError(f"puct_update_mode must be 'eager' or 'strict', got {puct_update_mode}")
        
        # Track which step each parent was sampled (needed for cross-batch analysis in all modes)
        self._parent_sample_step: dict[str, int] = {}
        
        if puct_update_mode == "strict":
            logger.info(f"TTTDiscoverWorkflowV2: STRICT PUCT update mode enabled. "
                       f"Cross-batch contamination will be prevented.")
            raise ValueError("Strict sync mode is not fully implemented yet.")
            # Buffer for delayed PUCT updates (children that arrived late)
            self._delayed_puct_children: list[Any] = []
            self._delayed_puct_parents: list[Any] = []
            self._delayed_rollout_metadata: list[dict] = []
        
        # Initialize tokenizer
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer
            self.tokenizer = load_hf_tokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer
            
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        
        # Teacher forcing message for stopping thinking
        self.force_stop_message = "... okay, I am out of thinking tokens. I need to send my final message now"
        
        # Configure parallel code execution with AsyncRewardWrapper
        # Follows AReaL best practice: wrap reward_fn once during initialization
        if max_reward_workers is None:
            max_reward_workers = int(os.environ.get('TTTD_MAX_CODE_WORKERS', '64'))  # Increased from 16
        
        # IMPORTANT: AsyncRewardWrapper timeout includes queue waiting time in ProcessPoolExecutor.
        # To give env.execute() the full eval_timeout budget for actual execution:
        # - wrapper_timeout >> eval_timeout (multiple times to account for queue wait)
        # - actual execution timeout is enforced inside env.execute() by subprocess
        eval_timeout = getattr(env, 'eval_timeout', 60)
        wrapper_timeout = eval_timeout * 10 + 600  # 10x + 10min buffer for queue wait (very generous)
        
        logger.info(f"TTTDiscoverWorkflowV2: max_reward_workers={max_reward_workers}, "
                   f"eval_timeout={eval_timeout}s, wrapper_timeout={wrapper_timeout}s (includes queue wait)")
        
        # Wrap reward function with AsyncRewardWrapper (AReaL standard pattern)
        # This dispatches reward computation to a dedicated process pool
        self.async_reward_fn = AsyncRewardWrapper(
            reward_fn=reward_fn,
            timeout_seconds=wrapper_timeout,  # Very generous buffer for queue + execution
            max_workers=max_reward_workers,
            max_retries=1,
        )
        logger.info(f"AsyncRewardWrapper initialized with {max_reward_workers} workers")
        
        # Async-safe buffer for pending updates (GroupedRolloutWorkflow uses asyncio.gather)
        # Stores (child_state, parent_state) tuples
        self._pending_lock = asyncio.Lock()  # Use asyncio.Lock for coroutine safety
        self._pending_children: list[Any] = []
        self._pending_parents: list[Any] = []
        
        # Track which parents have been processed this batch
        # Key: parent_state.id, Value: list of child rewards
        self._parent_stats: dict[str, list[float]] = {}
        
        # Store detailed rollout metadata for history logger
        # Each entry contains parent-child relationship and execution times for analysis:
        # {
        #   'parent_id': str, 'parent_timestep': int, 'parent_exec_time_ms': float|None,
        #   'child_id': str|None, 'child_timestep': int, 'child_exec_time_ms': float,
        #   'reward': float, 'staleness': int, 'failed': bool (optional)
        # }
        self._rollout_metadata: list[dict] = []
        
        # Cache failed rollouts for delayed update (synced across ranks in distributed training)
        self._failed_parents: list[Any] = []
        
        # Track timing for execute tail latency analysis
        # Records (gpu_done_time, execute_done_time) for each rollout
        self._rollout_timing_pairs: list[tuple[float, float]] = []
        
        # Staleness tracking for research analysis
        # Key: (parent_id, sampled_step), Value: {
        #   'sampled_step': int,            # Step when parent was sampled (sampler counter)
        #   'children_completed': int,
        #   'total_expected': int,
        #   'parent_timestep': int,
        #   'exec_times_ms': list[float],   # Execution times for each child
        # }
        # Staleness = current_step - sampled_step
        # Measures how many training steps delayed from sampling to PUCT update
        # In sync mode: staleness = 0 (immediate update)
        # In async mode: staleness > 0 (delayed update)
        # NOTE: Uses composite key to handle same parent sampled in multiple steps
        self._staleness_tracker: dict[tuple[str, int], dict] = {}
        self._current_version: int = 0  # Current training step, set externally
        self._current_step: int = 0  # For staleness tracking in get_pending_updates

        # Scheme 1 (Sync-like) batch tracking
        # Enables waiting for all parents in a batch to complete before PUCT update
        self._current_batch_step: int = 0
        self._current_batch_parent_ids: set[str] = set()
        self._expected_batch_size: int = 0
        self._expected_n_samples: int = 0
        
        # PUCT behavior tracking for three core metrics:
        # 1. Q-value estimation error
        # 2. Selection switching due to information delay
        # 3. Q-value convergence delay
        self._puct_update_log: list[dict] = []  # Each PUCT update event
        # Key: (parent_id, sampled_step), Value: episode dict
        # Using composite key to handle same parent sampled in multiple steps
        self._parent_episodes: dict[tuple[str, int], dict] = {}

    def get_execute_tail_latency(self, clear: bool = True) -> float:
        """
        Calculate execute tail latency: time from last GPU done to last execute done.
        
        This measures how long it takes to finish all code executions after
        all LLM inferences are complete.
        
        Returns:
            float: Tail latency in seconds, or 0.0 if no data
        """
        if not self._rollout_timing_pairs:
            return 0.0
        
        gpu_times = [t[0] for t in self._rollout_timing_pairs]
        exec_times = [t[1] for t in self._rollout_timing_pairs]
        
        last_gpu_done = max(gpu_times)
        last_exec_done = max(exec_times)
        
        tail_latency = last_exec_done - last_gpu_done
        
        if clear:
            self._rollout_timing_pairs = []
        
        return tail_latency
    
    def get_timing_stats(self, clear: bool = True) -> dict:
        """Get detailed timing statistics for analysis."""
        if not self._rollout_timing_pairs:
            return {}
        
        gpu_times = [t[0] for t in self._rollout_timing_pairs]
        exec_times = [t[1] for t in self._rollout_timing_pairs]
        
        stats = {
            'n_rollouts': len(self._rollout_timing_pairs),
            'first_gpu_done': min(gpu_times),
            'last_gpu_done': max(gpu_times),
            'first_exec_done': min(exec_times),
            'last_exec_done': max(exec_times),
            'gpu_span': max(gpu_times) - min(gpu_times),
            'exec_span': max(exec_times) - min(exec_times),
            'tail_latency': max(exec_times) - max(gpu_times),
        }
        
        if clear:
            self._rollout_timing_pairs = []
        
        return stats

    def _get_prompt(self, state, use_hint: bool = True) -> str:
        """Get prompt for state, optionally appending hint."""
        prompt = self.env.get_prompt(state)
        if use_hint and self.hint_fn is not None and state is not None:
            try:
                hint = self.hint_fn(state)
                if hint:
                    prompt = prompt + hint
            except Exception as e:
                logger.warning(f"[Workflow] hint_fn failed: {e}")
        return prompt

    def _create_trajectory(
        self,
        resp: ModelResponse,
        reward: float,
        state: "State | None" = None,
    ) -> dict[str, torch.Tensor]:
        """Create trajectory tensors from response and reward."""
        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions
        
        return {
            "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor([reward], dtype=torch.float32),
            "_student_prompts": [self._get_prompt(state, use_hint=False) if state is not None else ""],
        }
    
    def _create_failed_trajectory(
        self,
        state: "State | None",
        input_ids: list[int],
        fail_type: str,
        error_msg: str = "",
        resp: "ModelResponse | None" = None,
    ) -> dict[str, torch.Tensor]:
        """
        Create a trajectory for failed rollouts.
        
        If ``resp`` is provided (model already generated a completion but execution
        failed), the full prompt + completion sequence is preserved with a correct
        loss_mask so that prompt_len > 0 and privileged teacher logp alignment works.
        
        Args:
            state: The state object (may be None)
            input_ids: Input token IDs (fallback when resp is None)
            fail_type: Type of failure (timeout, code_extraction_failed, execution_error, missing_state)
            error_msg: Additional error message
            resp: Optional model response containing the generated completion
            
        Returns:
            Dictionary with trajectory tensors
        """
        if resp is not None:
            # Preserve the full generated sequence (prompt + completion) with correct masks
            seq = resp.input_tokens + resp.output_tokens
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            versions = [-1] * resp.input_len + resp.output_versions
            
            # Get reward from env (may vary by fail_type)
            result = self.env.get_failure_result(state, fail_type, error_msg)
            
            trajectory = {
                "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
                "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
                "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
                "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                "rewards": torch.tensor([result.reward], dtype=torch.float32),
            }
        else:
            # Fallback: no response available (e.g. missing state, engine exception)
            trajectory = self.env.create_failed_trajectory(
                state=state,
                input_ids=input_ids,
                tokenizer=self.tokenizer,
                fail_type=fail_type,
                error_msg=error_msg,
            )
        
        trajectory["_student_prompts"] = [self._get_prompt(state, use_hint=False) if state is not None else ""]
        return trajectory

    @trace_session("reward")
    async def _compute_reward(
        self,
        resp: ModelResponse,
        task_data: dict[str, Any],
        gpu_done_time: float | None = None,
    ) -> tuple[float, EnvResult, str, float]:
        """Compute reward by executing code using AsyncRewardWrapper.
        
        This method uses AsyncRewardWrapper with tttd_reward_fn to run env.execute 
        in a ProcessPoolExecutor, enabling true process-level parallelism for code 
        verification. The reward function returns (reward, EnvResult, code, exec_time_ms) tuple
        to avoid re-executing code.
        
        Args:
            resp: Model response from generation
            task_data: Task data including state object
            gpu_done_time: Timestamp when GPU inference completed (for tail latency measurement)
            
        Returns:
            tuple: (reward_value, EnvResult, extracted_code, exec_time_ms)
                - exec_time_ms: Pure execution time in milliseconds (excluding wait time)
        """
        import time
        start_time = time.time()
        
        completion_str = self.tokenizer.decode(resp.output_tokens)
        code = self.env.extract_code(completion_str)
        state = task_data.get("_state_obj")
        
        if code is None:
            logger.warning("Code extraction failed")
            result = self.env.get_failure_result(
                state=state,
                fail_type="code_extraction_failed",
            )
            if gpu_done_time is not None:
                self._rollout_timing_pairs.append((gpu_done_time, time.perf_counter()))
            return result.reward, result, "", 0.0
        
        exec_time_ms = 0.0
        try:
            # Use AsyncRewardWrapper with tttd_reward_fn
            # tttd_reward_fn normally returns (reward, EnvResult, code, exec_time_ms) tuple
            # but AsyncRewardWrapper returns 0 (int) on timeout
            prompt_str = self.tokenizer.decode(resp.input_tokens)
            
            reward_result = await self.async_reward_fn(
                prompt_str,
                completion_str,
                resp.input_tokens,
                resp.output_tokens,
                _env=self.env,
                _state=state,
            )
            
            # Handle both timeout (int) and normal (tuple) return values
            # This handles the AsyncRewardWrapper timeout behavior without modifying AReaL core
            if isinstance(reward_result, int):
                # Timeout case: AsyncRewardWrapper returned 0
                logger.warning(f"Reward computation timeout for state {state.id[:8] if state else 'None'}...")
                result = self.env.get_failure_result(
                    state=state,
                    fail_type="timeout",
                    error_msg="Reward computation timed out",
                )
                reward = float(reward_result)  # Should be 0
                extracted_code = ""
                exec_time_ms = 0.0
            else:
                # Normal case: unpack tuple
                reward, result, extracted_code, exec_time_ms = reward_result
            
            elapsed = time.time() - start_time
            fail_type_info = f", fail_type={result.fail_type}" if result.fail_type else ""
            logger.info(f"reward={result.reward:.4f}, valid={result.is_valid}, exec_time={exec_time_ms:.1f}ms, elapsed={elapsed:.1f}s{fail_type_info}")
            
        except Exception as e:
            logger.warning(f"Execution failed: {e}")
            result = self.env.get_failure_result(
                state=state,
                fail_type="execution_error",
                error_msg=str(e),
            )
            extracted_code = code  # Use originally extracted code on failure
        
        stats_tracker.get("rollout").scalar(
            reward=result.reward,
            is_valid=float(result.is_valid),
            exec_time_ms=exec_time_ms,
        )
        
        if gpu_done_time is not None:
            self._rollout_timing_pairs.append((gpu_done_time, time.perf_counter()))
        
        return result.reward, result, extracted_code, exec_time_ms

    async def _do_lazy_sampling(self, data: dict[str, Any]) -> "State":
        """
        Perform lazy PUCT sampling when VLLM has capacity.
        
        DISTRIBUTED CONSISTENCY:
        - Version mapping stored in sampler._batch_version_mappings (single source of truth)
        - First rollout of a batch assigns the latest available snapshot version
        - After sync_sampler, all ranks see the same mapping for same batch_id
        
        Args:
            data: The placeholder data from dataloader
            
        Returns:
            The sampled State object
        """
        batch_id = data["_batch_id"]
        batch_idx = data["_batch_idx"]
        
        async with self._cache_lock:
            # Get version from sampler (may be assigned by this rank or synced from other rank)
            version = self.sampler.get_batch_version(batch_id)
            
            if version is None:
                # First rollout of this batch: assign latest available version
                available = sorted(self.sampler._version_snapshots.keys())
                version = available[-1] if available else -1
                self.sampler.assign_batch_version(batch_id, version)
                
                logger.info(
                    f"[LAZY_VERSION] batch_id={batch_id} rank={self.dp_rank} "
                    f"assigned_v={version if version != -1 else 'current'} "
                    f"available_snapshots={available}"
                )
            
            # Sample parents (only once per batch per rank)
            if batch_id not in self._batch_parents:
                global_batch = self.batch_size * self.dp_world_size
                all_parents = self.sampler.sample_states_for_version(
                    num_states=global_batch,
                    target_version=version
                )
                
                # Take this rank's slice
                start_idx = self.dp_rank * self.batch_size
                my_parents = all_parents[start_idx:start_idx + self.batch_size]
                self._batch_parents[batch_id] = my_parents
                
                logger.info(
                    f"[LAZY_SAMPLE] batch_id={batch_id} rank={self.dp_rank} "
                    f"version={version} "
                    f"parents={[p.id[:8] for p in my_parents]}"
                )
            
            return self._batch_parents[batch_id][batch_idx]
    
    def get_version_mapping_info(self) -> dict:
        """Get current version mapping state for debugging.
        
        Returns:
            Dict with batch_id -> actual_version mapping and sampler state
        """
        if not self.lazy_sampling:
            return {"enabled": False}
        
        return {
            "enabled": True,
            "batch_mappings": dict(self.sampler._batch_version_mappings) if self.sampler else {},
            "sampler_mappings": dict(self.sampler._version_mapping) if self.sampler else {},
            "available_snapshots": sorted(self.sampler._version_snapshots.keys()) if self.sampler else [],
        }
    
    def validate_version_mapping(self, batch_id: int, expected_target: int) -> bool:
        """Validate that a batch is using the correct version (debug only).
        
        Since version is now sourced directly from sampler, this is mainly
        for detecting logic errors.
        
        Args:
            batch_id: The batch to validate
            expected_target: Expected target version for this batch
            
        Returns:
            True if valid, False otherwise
        """
        if not self.lazy_sampling or not self.sampler:
            return True
        
        actual = self.sampler.get_batch_version(batch_id)
        if actual is None:
            logger.warning(f"[VERSION_VALIDATION] batch_id={batch_id} not found in sampler mappings")
            return False
        
        # Validate: actual should never be > expected (no future versions)
        if actual > expected_target:
            logger.error(
                f"[VERSION_ERROR] batch_id={batch_id}: actual_v={actual} > target_v={expected_target}! "
                f"This should never happen - using future version!"
            )
            return False
        
        return True
    
    def cleanup_old_versions(self, current_step: int, max_history: int = 5):
        """Clean up old batch caches to manage memory.
        
        Args:
            current_step: Current training step
            max_history: Maximum number of batch caches to keep. 
                Should be >= max_head_offpolicyness + 1 to avoid KeyError 
                when old rollouts complete.
        """
        # Skip if lazy sampling is not enabled (attributes not initialized)
        if not self.lazy_sampling:
            return
        
        # Clean up _batch_parents (workflow-local cache)
        sorted_batches = sorted(self._batch_parents.keys())
        if len(sorted_batches) > max_history:
            to_remove = sorted_batches[:-max_history]  # Remove oldest
            
            for bid in to_remove:
                self._batch_parents.pop(bid, None)
            
            logger.info(
                f"[LAZY_CLEANUP] Step {current_step}: Cleaned up {len(to_remove)} old batches, "
                f"remaining: {len(self._batch_parents)}"
            )
    
    def report_version_status(self, current_step: int) -> dict:
        """Report current version mapping status for monitoring.
        
        Args:
            current_step: Current training step
            
        Returns:
            Status dict with version mapping info
        """
        if not self.lazy_sampling:
            return {"enabled": False}
        
        info = self.get_version_mapping_info()
        
        # Calculate statistics
        batch_mappings = info.get("batch_mappings", {})
        if batch_mappings:
            versions_used = set(batch_mappings.values())
            min_batch = min(batch_mappings.keys())
            max_batch = max(batch_mappings.keys())
            
            status = {
                "enabled": True,
                "current_step": current_step,
                "active_batches": len(batch_mappings),
                "versions_in_use": sorted(versions_used),
                "batch_range": (min_batch, max_batch),
                "mappings": batch_mappings,
            }
            
            logger.info(
                f"[VERSION_STATUS] Step {current_step}: "
                f"{status['active_batches']} active batches, "
                f"versions in use: {status['versions_in_use']}, "
                f"batch range: {status['batch_range']}"
            )
        else:
            status = {
                "enabled": True,
                "current_step": current_step,
                "active_batches": 0,
                "versions_in_use": [],
            }
            logger.info(f"[VERSION_STATUS] Step {current_step}: No active batches")
        
        return status

    @trace_session("arun_episode")
    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor] | None:
        """
        Execute a SINGLE rollout episode and update sampler internally.
        
        This method:
        1. Generates response from LLM
        2. Computes reward via env.execute()
        3. Creates child state if valid
        4. Buffers update to sampler (committed on flush)
        
        No external metadata needed - everything is handled internally.
        """
        import time
        
        # === LAZY SAMPLING MODE ===
        if self.lazy_sampling and data.get("_lazy_placeholder"):
            # Wait for VLLM capacity, then sample (ensures fresh PUCT state)
            async with self._vllm_sem:
                # Set sampled_step from batch version for staleness tracking
                # _puct_version is the batch_id which corresponds to the training step
                data['_sampled_step'] = data.get('_puct_version', 0)
                
                # Now VLLM has capacity - sample immediately
                state = await self._do_lazy_sampling(data)
                
                # Execute VLLM generation immediately after sampling
                return await self._execute_rollout(engine, state, data)
        
        # === EAGER MODE (original behavior) ===
        state = data.get("_state_obj")
        if state is None:
            logger.error("Missing '_state_obj' in data")
            return self._create_failed_trajectory(
                state=None,
                input_ids=[0],
                fail_type="missing_state",
            )
        
        return await self._execute_rollout(engine, state, data)
    
    async def _execute_rollout(
        self,
        engine: InferenceEngine,
        state: "State",
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor] | None:
        """Execute the actual rollout (VLLM + Execution)."""
        # Ensure state is available in data for _compute_reward
        data["_state_obj"] = state
        
        # Get the step when this parent was sampled (from dataloader)
        # This is CRITICAL for staleness tracking with composite key
        sampled_step = data.get('_sampled_step', self._current_version if hasattr(self, '_current_version') else 0)
        
        # Track new parent episode for PUCT analysis
        # This is called when parent is sampled and rollout begins
        # NOTE: Use lock to prevent race condition when multiple children
        # of the same parent are processed concurrently
        try:
            expected_children = self.gconfig.n_samples if hasattr(self.gconfig, 'n_samples') else None
            
            async with self._pending_lock:
                # Initialize parent episode tracking (for PUCT analysis)
                # Use composite key (parent_id, sampled_step) to handle same parent in multiple steps
                episode_key = (state.id, sampled_step)
                if episode_key not in self._parent_episodes:
                    # Extract PUCT selection info if available (from dataloader)
                    selection_info = data.get('_puct_selection')
                    
                    self.start_parent_episode(
                        parent_id=state.id,
                        sampled_step=sampled_step,
                        parent_value=state.value if hasattr(state, 'value') else None,
                        expected_children=expected_children,
                        selection_info=selection_info,
                    )
                    logger.debug(f"[PUCT_TRACK] Started new parent episode for {state.id[:8]}... "
                               f"sampled_step={sampled_step}, value={state.value}, expected_children={expected_children}")
                
                # FIX: Initialize staleness tracker at parent sampling time
                # Staleness = current_step - sampled_step
                # Measures how many training steps delayed from sampling to PUCT update
                # NOTE: Use composite key (parent_id, sampled_step) to handle same parent in multiple steps
                staleness_key = (state.id, sampled_step)
                if staleness_key not in self._staleness_tracker:
                    self._staleness_tracker[staleness_key] = {
                        'sampled_step': sampled_step,  # Step when parent was sampled
                        'sample_version': self._current_version,  # For backward compatibility
                        'children_completed': 0,
                        'total_expected': expected_children,
                        'parent_timestep': state.timestep,
                        'exec_times_ms': [],
                    }
                    logger.debug(f"[STALENESS_INIT] parent_id={state.id} sampled_step={sampled_step} "
                               f"current_step={self._current_step} "
                               f"parent_timestep={state.timestep}")
                
                # SCHEME 1: Track parent ID for batch completion waiting
                if hasattr(self, '_current_batch_parent_ids') and sampled_step == self._current_batch_step:
                    self._current_batch_parent_ids.add(state.id)
                    logger.debug(f"[SCHEME1_BATCH][Step {sampled_step}] REGISTER | "
                                f"parent={state.id[:8]}... "
                                f"total={len(self._current_batch_parent_ids)}")
        except Exception as e:
            logger.warning(f"[PUCT_TRACK] Failed to start parent episode: {e}")
        
        try:
            # Generate - Phase 1: Normal thinking
            prompt = self._get_prompt(state)
            messages = [{"role": "user", "content": prompt}]
            input_ids = list(self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            ))
            
            # Paper: limit prompt + thinking tokens to 26000, leave room for final response
            prompt_length = len(input_ids)
            max_context = 32768
            max_thinking_tokens = self.max_prompt_thinking_tokens - prompt_length
            max_new_tokens_phase1 = min(
                self.gconfig.max_new_tokens,
                max_context - prompt_length,
                max(2048, max_thinking_tokens)  # Leave at least 2048 for final response
            )
            
            if prompt_length > self.max_prompt_thinking_tokens:
                logger.warning(f"Prompt length {prompt_length} exceeds limit {self.max_prompt_thinking_tokens}, truncating")
                input_ids = input_ids[:self.max_prompt_thinking_tokens]
            
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1, max_new_tokens=max_new_tokens_phase1),
                tokenizer=self.tokenizer,
            )
            
            async with atrace_session_phase("generate"):
                resp = await engine.agenerate(req)
            
            # Check if generation was truncated or no valid code
            completion_str = self.tokenizer.decode(resp.output_tokens)
            code = self.env.extract_code(completion_str)
            
            # Phase 2: Teacher forcing if no valid code extracted
            if code is None:
                logger.info(f"[Teacher Forcing] No valid code in first generation, forcing final response")
                
                # Append teacher forcing message
                force_message = {"role": "assistant", "content": self.force_stop_message}
                messages_with_force = messages + [force_message]
                
                input_ids_force = list(self.tokenizer.apply_chat_template(
                    messages_with_force,
                    tokenize=True,
                    add_generation_prompt=False,  # Continue from assistant message
                    enable_thinking=self.enable_thinking,
                ))
                
                # Second generation with remaining tokens
                remaining_tokens = max_context - len(input_ids_force)
                
                req_force = ModelRequest(
                    rid=uuid.uuid4().hex + "_force",
                    input_ids=input_ids_force,
                    gconfig=self.gconfig.new(n_samples=1, max_new_tokens=remaining_tokens),
                    tokenizer=self.tokenizer,
                )
                
                async with atrace_session_phase("generate_forced"):
                    resp = await engine.agenerate(req_force)
                
                logger.info(f"[Teacher Forcing] Forced generation completed")
            
            # Record GPU completion time for tail latency measurement
            gpu_done_time = time.perf_counter()
            
            # Compute reward on final response (pass gpu_done_time for timing measurement)
            # Returns: (reward, result, code, exec_time_ms) where exec_time_ms is pure execution time
            # Use execution semaphore for lazy mode (separate from VLLM concurrency)
            if self.lazy_sampling:
                async with self._exec_sem:
                    reward, result, code, exec_time_ms = await self._compute_reward(resp, data, gpu_done_time)
            else:
                reward, result, code, exec_time_ms = await self._compute_reward(resp, data, gpu_done_time)
            
            # Log validation failure (concise)
            if not result.is_valid:
                logger.warning(f"Validation failed: reward={reward:.4f}, fail_type={result.fail_type}")
            
            # Create trajectory - use failed trajectory for invalid results
            if result.is_valid:
                trajectory = self._create_trajectory(resp, reward, state=state)
            else:
                # Use fail_type from result if available, otherwise default to execution_error
                fail_type = result.fail_type or "execution_error"
                trajectory = self._create_failed_trajectory(
                    state=state,
                    input_ids=input_ids,
                    fail_type=fail_type,
                    error_msg=result.observation,
                    resp=resp,
                )
            
            # Create child state and buffer sampler update (thread-safe)
            if result.is_valid:
                # Success: save child state for future sampling
                try:
                    child = self.env.create_state(
                        parent_state=state,
                        code=code,
                        reward=reward,
                        result=result,
                        timestep=state.timestep + 1,
                    )
                    # Store execution time for this child state
                    child.exec_time_ms = exec_time_ms
                    # Store sampled_step for Scheme 1 batch tracking
                    child.sampled_step = sampled_step
                    
                    async with self._pending_lock:
                        self._pending_children.append(child)
                        self._pending_parents.append(state)
                        
                        # Track stats for this parent (atomic under lock)
                        pid = state.id
                        if pid not in self._parent_stats:
                            self._parent_stats[pid] = []
                        self._parent_stats[pid].append(reward)
                        
                        # Staleness tracking: staleness = current_version - sample_version
                        # where sample_version is recorded when parent was SELECTED (in arun_episode)
                        try:
                            current_version = engine.get_version()
                        except Exception:
                            current_version = -1  # Fallback if engine doesn't support versioning
                        
                        # Use composite key (parent_id, sampled_step) to handle same parent in multiple steps
                        staleness_key = (pid, sampled_step)
                        
                        # DEBUG: Log all available keys in staleness_tracker
                        available_keys = list(self._staleness_tracker.keys())[:5]  # Limit to first 5
                        logger.debug(f"[STALENESS_DEBUG] Looking for key=({pid[:8]}..., {sampled_step}), "
                                    f"available={[(p[:8], s) for p, s in available_keys]}, "
                                    f"tracker_size={len(self._staleness_tracker)}")
                        
                        # Parent should already be in tracker (initialized in arun_episode)
                        # But handle fallback case for safety
                        if staleness_key not in self._staleness_tracker:
                            # Fallback: initialize with current version (staleness will be 0)
                            # DEBUG: Check if parent exists with different sampled_step
                            parent_keys = [(p, s) for p, s in self._staleness_tracker.keys() if p == pid]
                            if parent_keys:
                                logger.warning(f"[STALENESS_FALLBACK] parent_id={pid} sampled_step={sampled_step} not initialized, "
                                             f"but found with different steps: {parent_keys}. "
                                             f"This may indicate cross-step contamination.")
                            else:
                                logger.warning(f"[STALENESS_FALLBACK] parent_id={pid} sampled_step={sampled_step} not initialized, "
                                             f"using current_version={current_version} as sample_version")
                            self._staleness_tracker[staleness_key] = {
                                'sample_version': current_version,
                                'children_completed': 0,
                                'total_expected': self.gconfig.n_samples if hasattr(self.gconfig, 'n_samples') else None,
                                'parent_timestep': state.timestep,
                                'exec_times_ms': [],
                            }
                        
                        tracker = self._staleness_tracker[staleness_key]
                        tracker['children_completed'] += 1
                        tracker['exec_times_ms'].append(exec_time_ms)
                        # Staleness = current_step - sampled_step
                        # Measures how many training steps delayed from sampling to PUCT update
                        current_step = self._current_step
                        sampled_step_at_init = tracker['sampled_step']
                        staleness = current_step - sampled_step_at_init
                        
                        logger.info(f"[STALENESS] parent_id={pid} sampled_step={sampled_step_at_init} "
                                   f"child_num={tracker['children_completed']}/{tracker['total_expected']} "
                                   f"current_step={current_step} "
                                   f"staleness={staleness} "
                                   f"parent_timestep={tracker['parent_timestep']} "
                                   f"exec_time_ms={exec_time_ms:.2f}")
                        
                        # Record rollout metadata for history logger
                        # Includes child_id and parent_exec_time_ms for parent-child execution time analysis
                        self._rollout_metadata.append({
                            'parent_id': pid,
                            'parent_timestep': tracker['parent_timestep'],
                            'parent_exec_time_ms': getattr(state, 'exec_time_ms', None),
                            'child_id': child.id,
                            'child_timestep': child.timestep,
                            'child_exec_time_ms': float(exec_time_ms),
                            'reward': float(reward),
                            'staleness': int(staleness),
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to create child state: {e}")
            elif not result.is_valid:
                # Failure: cache failed parent for delayed update (will be synced across ranks)
                try:
                    async with self._pending_lock:
                        self._failed_parents.append(state)
                        
                        # FIX: Increment staleness tracker for failed rollouts too
                        # This ensures wait_for_batch_completion doesn't wait forever
                        staleness_key = (state.id, sampled_step)
                        if staleness_key in self._staleness_tracker:
                            tracker = self._staleness_tracker[staleness_key]
                            tracker['children_completed'] += 1
                            tracker['exec_times_ms'].append(exec_time_ms)
                            logger.debug(f"[STALENESS_FAIL] parent_id={state.id} sampled_step={sampled_step} "
                                        f"completed={tracker['children_completed']}/{tracker['total_expected']}")
                        
                        # Record failed rollout metadata
                        # Staleness = current_step - sampled_step
                        current_step = self._current_step
                        tracker_data = self._staleness_tracker.get(staleness_key, {})
                        sampled_step_at_init = tracker_data.get('sampled_step', current_step)
                        staleness = current_step - sampled_step_at_init
                        self._rollout_metadata.append({
                            'parent_id': state.id,
                            'parent_timestep': state.timestep,
                            'parent_exec_time_ms': getattr(state, 'exec_time_ms', None),
                            'child_id': None,  # Failed rollouts don't create child states
                            'child_timestep': state.timestep + 1,
                            'child_exec_time_ms': float(exec_time_ms),
                            'reward': float(result.reward),
                            'staleness': int(staleness),
                            'failed': True,
                        })
                        logger.debug(f"Cached failed rollout for parent {state.id}")
                except Exception as e:
                    logger.warning(f"Failed to cache failed rollout: {e}")
            
            return trajectory
            
        except Exception as e:
            logger.error(f"arun_episode failed: {e}", exc_info=True)
            # Return failed trajectory - let env decide reward and loss_mask
            if state:
                prompt = self._get_prompt(state)
                messages = [{"role": "user", "content": prompt}]
                try:
                    input_ids = list(self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=self.enable_thinking,
                    ))
                except Exception:
                    input_ids = [self.tokenizer.bos_token_id or 0]
            else:
                input_ids = [0]
            
            return self._create_failed_trajectory(
                state=state,
                input_ids=input_ids,
                fail_type="execution_error",
                error_msg=str(e),
            )

    def reset(self):
        """Clear the internal buffers for pending updates. Should be called at the start of each batch."""
        self._pending_children.clear()
        self._pending_parents.clear()
        self._failed_parents.clear()
        self._parent_stats.clear()
        self._rollout_metadata.clear()

    def set_current_version(self, version: int):
        """Set the current training step for staleness calculation.
        
        Staleness measures how many training steps have passed since a parent
        was sampled until its rollouts are used for PUCT update.
        
        When get_pending_updates() is called, the returned states will be used
        for PUCT update immediately, so staleness = current_version - sampled_step.
        
        In sync mode: staleness = 0 (sample and update in same step)
        In async mode: staleness > 0 (delayed update)
        
        Args:
            version: Current training step (global_step)
        """
        self._current_version = version
        self._current_step = version
        logger.debug(f"[WORKFLOW] Set current step to {version}")
    
    # =========================================================================
    # Scheme 1 (Sync-like) Methods
    # =========================================================================
    
    def set_current_batch_tracking(self, step: int, batch_size: int, n_samples: int):
        """Start tracking a new batch for Scheme 1 (sync-like) mode.
        
        This enables waiting for all parents in a batch to complete before
        updating PUCTSampler, ensuring complete batch updates like sync mode.
        
        Args:
            step: Current training step
            batch_size: Number of parents in this batch
            n_samples: Expected number of children per parent
        """
        # Check if previous batch was cleared properly
        if hasattr(self, '_current_batch_parent_ids') and len(self._current_batch_parent_ids) > 0:
            logger.warning(f"[SCHEME1_BATCH][Step {step}] PREVIOUS_BATCH_NOT_CLEARED | "
                          f"previous_step={self._current_batch_step} "
                          f"remaining_parents={len(self._current_batch_parent_ids)}")
        
        self._current_batch_step = step
        self._current_step = step  # Also set _current_step for staleness tracking
        self._current_batch_parent_ids = set()
        self._expected_batch_size = batch_size
        self._expected_n_samples = n_samples
        logger.info(f"[SCHEME1_BATCH][Step {step}] START | "
                   f"expected_parents={batch_size} "
                   f"expected_children_per_parent={n_samples}")
    
    def get_current_batch_parent_ids(self) -> set[str]:
        """Get the set of parent IDs recorded for current batch."""
        return self._current_batch_parent_ids.copy()
    
    def wait_for_batch_completion(
        self, 
        batch_step: int,
        poll_interval: float = 0.1
    ) -> float:
        """Wait for all parents in the specified batch to complete.
        
        This blocks until all children for all parents in the batch have been
        generated, ensuring complete batch updates for PUCTSampler.
        
        Args:
            batch_step: The step number of the batch to wait for
            poll_interval: Polling interval in seconds
            
        Returns:
            float: Time waited in seconds
        """
        import time
        start_time = time.time()
        
        parent_ids = self._current_batch_parent_ids
        expected_per_parent = self._expected_n_samples
        total_expected_children = len(parent_ids) * expected_per_parent
        
        logger.info(f"[SCHEME1_WAIT][Step {batch_step}] START | "
                   f"parents={len(parent_ids)} "
                   f"expected_children={total_expected_children} "
                   f"mode=BLOCKING")
        
        last_completed = 0
        last_log_time = start_time
        
        while True:
            # Check completion status for all parents in this batch
            all_complete = True
            n_completed = 0
            incomplete_parents = []
            
            for pid in parent_ids:
                key = (pid, batch_step)
                tracker = self._staleness_tracker.get(key)
                
                if tracker:
                    completed = tracker.get('children_completed', 0)
                    expected = tracker.get('total_expected', expected_per_parent)
                    n_completed += completed
                    
                    if completed < expected:
                        all_complete = False
                        incomplete_parents.append((pid, completed, expected))
                else:
                    # Parent not yet tracked (rollout hasn't started)
                    all_complete = False
                    incomplete_parents.append((pid, 0, expected_per_parent))
            
            total_expected = len(parent_ids) * expected_per_parent
            
            # Log progress every 5 seconds
            current_time = time.time()
            if current_time - last_log_time >= 5.0:
                elapsed = current_time - start_time
                # Detailed breakdown of incomplete parents
                not_started = sum(1 for _, c, _ in incomplete_parents if c == 0)
                partial = sum(1 for _, c, e in incomplete_parents if 0 < c < e)
                
                logger.info(f"[SCHEME1_WAIT][Step {batch_step}] PROGRESS | "
                           f"elapsed={elapsed:.1f}s "
                           f"completed={n_completed}/{total_expected} "
                           f"incomplete={len(incomplete_parents)} "
                           f"(not_started={not_started}, partial={partial})")
                
                # Log first few incomplete parents for debugging
                if incomplete_parents:
                    sample = incomplete_parents[:3]
                    logger.debug(f"[SCHEME1_WAIT][Step {batch_step}] INCOMPLETE_SAMPLE | "
                                f"{[(pid[:8], c, e) for pid, c, e in sample]}")
                
                last_log_time = current_time
            
            if all_complete:
                elapsed = time.time() - start_time
                logger.info(f"[SCHEME1_WAIT][Step {batch_step}] SUCCESS | "
                           f"duration={elapsed:.2f}s "
                           f"total_completed={n_completed}")
                return elapsed
            
            time.sleep(poll_interval)
    
    def clear_current_batch_tracking(self):
        """Clear the current batch tracking state."""
        self._current_batch_parent_ids.clear()
        self._expected_batch_size = 0
        self._expected_n_samples = 0
    
    # =========================================================================
    # End Scheme 1 Methods
    # =========================================================================
    
    def get_pending_updates(
        self, 
        clear: bool = True, 
        current_step: int | None = None,
        parent_ids: set[str] | None = None
    ) -> tuple:
        """Return the buffered pending updates (children, parents, failed_parents, rollout_metadata) for this batch.
        
        Args:
            clear: Whether to clear internal buffers after retrieval
            current_step: Current training step. Required for strict PUCT update mode to prevent
                cross-batch contamination. If None, uses self._current_version.
        
        Returns:
            Tuple of (children, parents, failed_parents, rollout_metadata, cross_batch_info)
            where cross_batch_info is a dict with delayed children info (only in strict mode)
        """
        # Determine current step
        if current_step is None:
            current_step = self._current_version
        
        # Get all pending updates
        all_children = self._pending_children.copy()
        all_parents = self._pending_parents.copy()
        all_failed = self._failed_parents.copy()
        all_metadata = self._rollout_metadata.copy()
        
        # CROSS-BATCH TRACKING (for both modes, for analysis)
        cross_batch_info = {
            'mode': self.puct_update_mode,
            'current_step': current_step,
            'n_this_batch': len(all_children),
            'n_delayed': 0,
            'delayed_children_ids': [],
        }
        
        # Determine which children to use for PUCT update
        if self.puct_update_mode == "strict" and current_step is not None:
            # STRICT MODE (方案 1): Only include children from parents sampled in current_step
            filtered_children = []
            filtered_parents = []
            filtered_metadata = []
            
            for child, parent, meta in zip(all_children, all_parents, all_metadata):
                pid = parent.id
                parent_sampled_step = self._parent_sample_step.get(pid)
                
                if parent_sampled_step == current_step:
                    # This parent was sampled in current step
                    filtered_children.append(child)
                    filtered_parents.append(parent)
                    filtered_metadata.append(meta)
                else:
                    # Cross-batch child - track for analysis
                    cross_batch_info['delayed_children_ids'].append(pid)
            
            cross_batch_info['n_this_batch'] = len(filtered_children)
            cross_batch_info['n_delayed'] = len(cross_batch_info['delayed_children_ids'])
            
            # Log cross-batch prevention
            if cross_batch_info['n_delayed'] > 0:
                logger.info(f"[STRICT_PUCT] Step {current_step}: "
                           f"{len(filtered_children)} children this batch, "
                           f"{cross_batch_info['n_delayed']} delayed (cross-batch)")
        else:
            # EAGER MODE (方案 2 - 默认): Use all children for PUCT update
            # This enables Streaming PUCT across batches
            filtered_children = all_children
            filtered_parents = all_parents
            filtered_metadata = all_metadata
            
            # Track cross-batch for analysis (but don't prevent)
            if current_step is not None:
                for child, parent in zip(all_children, all_parents):
                    pid = parent.id
                    parent_sampled_step = self._parent_sample_step.get(pid)
                    if parent_sampled_step != current_step:
                        cross_batch_info['delayed_children_ids'].append(pid)
                cross_batch_info['n_delayed'] = len(cross_batch_info['delayed_children_ids'])
                
                if cross_batch_info['n_delayed'] > 0:
                    logger.debug(f"[STREAMING_PUCT] Step {current_step}: "
                                f"Updating PUCT with {cross_batch_info['n_delayed']} "
                                f"cross-batch children (Streaming mode)")
        
        # SCHEME 1: Filter by parent_ids if specified
        # This is used to get only the updates for a specific batch of parents
        if parent_ids is not None:
            logger.info(f"[SCHEME1_FILTER][Step {current_step}] BEFORE | "
                       f"total_children={len(filtered_children)} "
                       f"total_parents={len(filtered_parents)} "
                       f"total_failed={len(all_failed)} "
                       f"total_metadata={len(filtered_metadata)}")
            
            # Validate parent_ids (only warn if explicitly passed but empty)
            if len(parent_ids) == 0:
                logger.warning("[SCHEME1_FILTER] Empty parent_ids filter! No parents will be returned.")
            
            # Filter children and parents (maintaining 1:1 correspondence)
            scheme1_children = []
            scheme1_parents = []
            
            for child, parent in zip(filtered_children, filtered_parents):
                if parent.id in parent_ids:
                    scheme1_children.append(child)
                    scheme1_parents.append(parent)
            
            # Filter failed_parents (these are parent objects that failed)
            scheme1_failed = [p for p in all_failed if p.id in parent_ids]
            
            # Filter metadata by parent_id field
            scheme1_metadata = [
                m for m in filtered_metadata 
                if m.get('parent_id') in parent_ids
            ]
            
            # Validation: Check counts and consistency
            n_parents_unique = len(set(p.id for p in scheme1_parents))
            n_children = len(scheme1_children)
            n_parents_total = len(scheme1_parents)
            expected_children = len(parent_ids) * self._expected_n_samples if self._expected_n_samples else n_children
            
            # [CHECK 1] Data integrity: children and parents must have 1:1 correspondence
            # Note: This checks internal data consistency, not completion status
            if n_children != n_parents_total:
                logger.error(
                    f"[SCHEME1_CHECK][Step {current_step}] DATA_CORRUPTION | "
                    f"children={n_children} parents={n_parents_total} "
                    f"expected_1-to-1_correspondence - THIS SHOULD NOT HAPPEN!"
                )
            
            # [CHECK 2] All filtered parents should be in the filter set
            orphan_parents = [p for p in scheme1_parents if p.id not in parent_ids]
            if orphan_parents:
                logger.error(
                    f"[SCHEME1_CHECK][Step {current_step}] ORPHAN_PARENTS | "
                    f"count={len(orphan_parents)} "
                    f"ids={[p.id[:8] for p in orphan_parents[:3]]}..."
                )
            
            # [CHECK 3] Batch completion status (including failed trajectories)
            total_completed = n_children + len(scheme1_failed)  # Total finished (success + failed)
            if total_completed < expected_children:
                logger.warning(
                    f"[SCHEME1_CHECK][Step {current_step}] INCOMPLETE_BATCH | "
                    f"success={n_children} failed={len(scheme1_failed)} "
                    f"total_completed={total_completed} expected={expected_children} "
                    f"missing={expected_children - total_completed}"
                )
            elif n_children < expected_children:
                # Some failed, but batch is complete
                logger.info(
                    f"[SCHEME1_CHECK][Step {current_step}] PARTIAL_FAILURE | "
                    f"success={n_children} failed={len(scheme1_failed)} "
                    f"total={total_completed} expected={expected_children} "
                    f"failure_rate={len(scheme1_failed)/expected_children:.1%}"
                )
            elif n_children > expected_children:
                logger.warning(
                    f"[SCHEME1_CHECK][Step {current_step}] EXCESS_CHILDREN | "
                    f"success={n_children} failed={len(scheme1_failed)} "
                    f"expected={expected_children} excess={n_children - expected_children}"
                )
            
            # [CHECK 4] Verify metadata count matches children + failed
            expected_metadata = n_children + len(scheme1_failed)
            actual_metadata = len(scheme1_metadata)
            if actual_metadata != expected_metadata:
                logger.warning(
                    f"[SCHEME1_CHECK][Step {current_step}] METADATA_MISMATCH | "
                    f"metadata={actual_metadata} expected={expected_metadata} "
                    f"(children={n_children} + failed={len(scheme1_failed)})"
                )
            
            # [CHECK 5] Verify staleness_tracker state for each parent in batch
            # This detects parents that were supposed to be sampled but have no tracker entry
            missing_trackers = []
            incomplete_trackers = []
            for pid in parent_ids:
                key = (pid, current_step)
                tracker = self._staleness_tracker.get(key)
                if not tracker:
                    missing_trackers.append(pid)
                else:
                    completed = tracker.get('children_completed', 0)
                    expected = tracker.get('total_expected', self._expected_n_samples or 1)
                    if completed < expected:
                        incomplete_trackers.append((pid, completed, expected))
            
            if missing_trackers:
                logger.error(
                    f"[SCHEME1_CHECK][Step {current_step}] MISSING_TRACKERS | "
                    f"count={len(missing_trackers)} "
                    f"ids={[p[:8] for p in missing_trackers[:3]]}... "
                    f"These parents were in batch but have no staleness record!"
                )
            
            if incomplete_trackers:
                logger.warning(
                    f"[SCHEME1_CHECK][Step {current_step}] INCOMPLETE_TRACKERS | "
                    f"count={len(incomplete_trackers)} "
                    f"details={[(p[:8], c, e) for p, c, e in incomplete_trackers[:3]]}..."
                )
            
            # [CHECK 6] Cross-validate: actual children count vs staleness_tracker
            # Count children per parent from actual returned data
            children_per_parent: dict[str, int] = {}
            for parent in scheme1_parents:
                pid = parent.id
                children_per_parent[pid] = children_per_parent.get(pid, 0) + 1
            
            # Compare with tracker
            tracker_mismatches = []
            for pid in parent_ids:
                key = (pid, current_step)
                tracker = self._staleness_tracker.get(key)
                if tracker:
                    actual_count = children_per_parent.get(pid, 0)
                    tracker_count = tracker.get('children_completed', 0)
                    if actual_count != tracker_count:
                        tracker_mismatches.append((pid, actual_count, tracker_count))
            
            if tracker_mismatches:
                logger.error(
                    f"[SCHEME1_CHECK][Step {current_step}] TRACKER_MISMATCH | "
                    f"count={len(tracker_mismatches)} "
                    f"details={[(p[:8], a, t) for p, a, t in tracker_mismatches[:3]]}... "
                    f"(format: parent_id, actual_children, tracker_children)"
                )
            
            # Replace filtered results with Scheme 1 filtered results
            filtered_children = scheme1_children
            filtered_parents = scheme1_parents
            all_failed = scheme1_failed  # Replace for return
            filtered_metadata = scheme1_metadata
            
            logger.info(f"[SCHEME1_FILTER][Step {current_step}] AFTER | "
                       f"children={len(filtered_children)} "
                       f"parents(unique)={n_parents_unique} "
                       f"parents(total)={n_parents_total} "
                       f"failed={len(all_failed)} "
                       f"metadata={len(filtered_metadata)}")
        
        # Calculate staleness statistics for research analysis
        if self._staleness_tracker and all_children:
            current_step = self._current_step
            staleness_list = []
            incomplete_parents = 0
            
            # Track parents that have all children completed (for cleanup later)
            parents_to_cleanup = set()
            
            for (pid, sampled_step), tracker in self._staleness_tracker.items():
                # Staleness = current_step - sampled_step
                # Measures how many training steps delayed from sampling to PUCT update
                staleness = current_step - tracker['sampled_step']
                staleness_list.append(staleness)
                
                # Check if this parent has all expected children
                total_expected = tracker.get('total_expected') or tracker['children_completed']
                is_complete = tracker['children_completed'] >= total_expected
                
                if not is_complete:
                    incomplete_parents += 1
                    # Log incomplete parent for tracking (regex-friendly)
                    logger.info(f"[STALENESS_INCOMPLETE] parent_id={pid} sampled_step={sampled_step} "
                               f"completed={tracker['children_completed']}/{total_expected} "
                               f"current_step={current_step} "
                               f"staleness={staleness}")
                else:
                    parents_to_cleanup.add((pid, sampled_step))
            
            if staleness_list:
                avg_staleness = sum(staleness_list) / len(staleness_list)
                max_staleness = max(staleness_list)
                # Regex-friendly format for post-processing
                logger.info(f"[STALENESS_BATCH] n_parents={len(staleness_list)} "
                           f"avg_staleness={avg_staleness:.2f} max_staleness={max_staleness} "
                           f"current_step={current_step} "
                           f"incomplete={incomplete_parents} "
                           f"complete={len(parents_to_cleanup)} "
                           f"total_children={len(all_children)}")
        
        if clear:
            self._pending_children.clear()
            self._pending_parents.clear()
            self._failed_parents.clear()
            self._parent_stats.clear()
            self._rollout_metadata.clear()
            # CRITICAL FIX: Only cleanup staleness tracker entries for parents that are COMPLETE
            # Incomplete parents must retain their first_child_time for accurate staleness tracking
            # NOTE: parents_to_cleanup contains tuples (parent_id, step) due to composite key
            if 'parents_to_cleanup' in locals():
                for key in parents_to_cleanup:
                    if key in self._staleness_tracker:
                        pid, step = key
                        logger.debug(f"[STALENESS_CLEANUP] Removing complete parent {pid} step={step} from tracker")
                        del self._staleness_tracker[key]
            
            # CRITICAL: After returning states for PUCT update, increment _current_step
            # This ensures that any states completed after this point will have
            # staleness measured from the NEXT PUCT update cycle
            old_step = self._current_step
            self._current_step += 1
            logger.debug(f"[WORKFLOW] Incremented current step: {old_step} -> {self._current_step} "
                        f"(after get_pending_updates)")
        
        # Record PUCT update timestamp for convergence delay analysis
        # In strict mode, only record for children from current batch
        parents_to_record = filtered_parents if self.puct_update_mode == "strict" else all_parents
        children_to_record = filtered_children if self.puct_update_mode == "strict" else all_children
        
        for parent in parents_to_record:
            pid = parent.id
            episode = self._get_active_episode(pid)
            if episode:
                # Record that update happened at this step
                n_children_in_update = len([c for c in children_to_record 
                                           if any(p.get('id') == pid 
                                                 for p in (c.parents or []))])
                if n_children_in_update > 0:  # Only record if actually updating
                    episode.setdefault('puct_update_steps', []).append({
                        'step': current_step,
                        'timestamp': time.time(),
                        'n_children_in_update': n_children_in_update,
                        'is_strict_mode': self.puct_update_mode == "strict",
                    })
                    
                    # Check if episode is now complete (all children received)
                    total_children = len(episode['children_completed'])
                    expected = episode.get('expected_children')
                    if expected and total_children >= expected:
                        episode['completed'] = True
                        episode['completed_step'] = current_step
                        logger.debug(f"[PUCT_TRACK] Episode {episode['episode_id']} for parent {pid[:8]}... "
                                   f"completed with {total_children} children")
        
        # Return all children for training, but with cross_batch_info for analysis
        # In strict mode, caller should use filtered_* for PUCT update
        if self.puct_update_mode == "strict":
            # Return filtered lists + cross_batch_info
            # Trainer should use filtered_parents for sync_sampler, but all_children for training
            return (
                filtered_children,  # For PUCT update (strict)
                filtered_parents,   # For PUCT update (strict)
                all_failed,         # Failed parents (no change)
                filtered_metadata,  # Metadata for filtered children
                cross_batch_info,   # Info about delayed children
            )
        else:
            # EAGER mode: return all as before
            return all_children, all_parents, all_failed, all_metadata, cross_batch_info

    def get_exec_time_stats(self) -> dict[str, dict]:
        """
        Get execution time statistics for all tracked parents.
        
        Returns a dictionary mapping parent_id to stats including:
        - exec_times_ms: list of execution times for each child
        - avg_exec_time_ms: average execution time
        - total_exec_time_ms: sum of all execution times
        - parent_timestep: timestep of the parent state
        
        This is used for history_logger to analyze relationship between
        parent/child complexity and execution time.
        """
        stats = {}
        for (pid, step), tracker in self._staleness_tracker.items():
            exec_times = tracker.get('exec_times_ms', [])
            if exec_times:
                stats[pid] = {
                    'exec_times_ms': exec_times.copy(),
                    'avg_exec_time_ms': sum(exec_times) / len(exec_times),
                    'total_exec_time_ms': sum(exec_times),
                    'min_exec_time_ms': min(exec_times),
                    'max_exec_time_ms': max(exec_times),
                    'n_children': len(exec_times),
                    'parent_timestep': tracker.get('parent_timestep', -1),
                    'sample_version': tracker.get('sample_version', -1),
                    'sampled_step': step,  # Include the step for reference
                }
        return stats
    
    # ============================================================
    # PUCT Behavior Tracking for Three Core Metrics
    # ============================================================
    
    def start_parent_episode(self, parent_id: str, sampled_step: int,
                             parent_value: float, expected_children: int,
                             selection_info: dict = None):
        """Start tracking a new parent episode for PUCT analysis.
        
        This should be called when a parent is sampled by PUCTSampler.
        Each time a parent is sampled (even if sampled before in a different step), 
        a new episode is created using composite key (parent_id, sampled_step).
        
        Args:
            parent_id: Parent state ID
            sampled_step: Training step when parent was sampled
            parent_value: Parent state's value
            expected_children: Expected number of children for this parent
            selection_info: Optional dict with selection-time info for Q-value analysis:
                - 'q_value': Q-value (m_value) at selection time
                - 'n_visits': Visit count at selection time  
                - 'puct_score': PUCT score at selection time
                - 'candidates': List of candidate parents at selection time
        """
        # Use composite key (parent_id, sampled_step) to handle same parent in multiple steps
        key = (parent_id, sampled_step)
        
        # Count how many times this parent has been sampled before (for episode_id)
        existing_episodes = [ep for ep in self._parent_episodes.values() 
                            if ep['parent_id'] == parent_id]
        episode_id = len(existing_episodes)
        
        new_episode = {
            'episode_id': episode_id,
            'parent_id': parent_id,
            'sampled_step': sampled_step,
            'parent_value': parent_value,
            'expected_children': expected_children,
            'children_completed': [],  # List of completed child info
            'puct_updates': [],        # List of PUCT update events
            'completed': False,        # Whether all children are done
        }
        
        # Add selection-time info if provided (for Q-value estimation error analysis)
        if selection_info is not None:
            new_episode['selection_info'] = selection_info.copy()
        
        self._parent_episodes[key] = new_episode
        
        # Record sample step for strict PUCT update mode
        if self.puct_update_mode == "strict":
            self._parent_sample_step[parent_id] = sampled_step
        
        logger.debug(f"[PUCT_TRACK] Started episode {episode_id} for parent {parent_id[:8]}... "
                   f"step={sampled_step}, value={parent_value}")
    
    def _get_active_episode(self, parent_id: str) -> dict | None:
        """Get the most recent active episode for a parent.
        
        Active = not yet completed (children still being generated).
        Returns None if no active episode found.
        """
        # Search all episodes with matching parent_id
        episodes = [ep for key, ep in self._parent_episodes.items() 
                   if key[0] == parent_id]
        
        if not episodes:
            return None
        
        # Sort by sampled_step to find most recent
        episodes.sort(key=lambda e: e['sampled_step'], reverse=True)
        
        # Find the most recent episode that is not completed
        for episode in episodes:
            if not episode.get('completed', False):
                return episode
        return None
    
    def _get_episode(self, parent_id: str, sampled_step: int) -> dict | None:
        """Get a specific episode by (parent_id, sampled_step).
        
        Returns None if episode not found.
        """
        key = (parent_id, sampled_step)
        return self._parent_episodes.get(key)
    
    def log_child_completion(self, parent_id: str, child_reward: float, 
                             complete_step: int, exec_time_ms: float):
        """Log when a child rollout completes (before PUCT update).
        
        Associates the child with the most recent active episode for this parent.
        """
        episode = self._get_active_episode(parent_id)
        if episode:
            episode['children_completed'].append({
                'reward': child_reward,
                'complete_step': complete_step,
                'exec_time_ms': exec_time_ms,
            })
    
    def log_puct_update(self, parent_id: str, update_step: int,
                        n_visits: int, m_value: float, score: float,
                        children_rewards: list[float],
                        prev_m_value: float = None):
        """Log a PUCT update event for Q-value tracking.
        
        Args:
            parent_id: Parent state ID
            update_step: Step when update occurred
            n_visits: _n[parent] after update
            m_value: _m[parent] after update (this is the Q-value)
            score: PUCT score after update
            children_rewards: Rewards of children included in this update
            prev_m_value: _m[parent] before update (for measuring Q-value change)
        """
        # Record in the most recent active episode
        episode = self._get_active_episode(parent_id)
        if episode:
            update_record = {
                'update_step': update_step,
                'n_visits': n_visits,
                'm_value': m_value,
                'score': score,
                'children_rewards': children_rewards.copy(),
            }
            if prev_m_value is not None:
                update_record['prev_m_value'] = prev_m_value
                update_record['m_value_change'] = m_value - prev_m_value
            episode['puct_updates'].append(update_record)
        
        # Also record in global log for cross-parent analysis
        global_record = {
            'parent_id': parent_id,
            'update_step': update_step,
            'n_visits': n_visits,
            'm_value': m_value,
            'score': score,
            'children_rewards': children_rewards.copy(),
        }
        if prev_m_value is not None:
            global_record['prev_m_value'] = prev_m_value
            global_record['m_value_change'] = m_value - prev_m_value
        self._puct_update_log.append(global_record)
    
    def get_puct_analysis_data(self, clear: bool = True) -> dict:
        """Get data for computing the three core PUCT metrics.
        
        Returns data structure for analyzing:
        1. Q-value estimation error
        2. Selection switching due to information delay  
        3. Q-value convergence delay
        
        Args:
            clear: Whether to clear internal buffers after retrieval
            
        Returns:
            Dictionary with:
            - 'parent_episodes': Detailed episode data for each parent (JSON-safe keys)
            - 'puct_updates': All PUCT update events
        """
        # Convert tuple keys to JSON-safe string keys for serialization
        # Original key format: (parent_id, sampled_step) -> "parent_id:step:sampled_step"
        parent_episodes_json_safe = {}
        for (parent_id, sampled_step), episode in self._parent_episodes.items():
            key = f"{parent_id}:step:{sampled_step}"
            parent_episodes_json_safe[key] = episode
        
        data = {
            'parent_episodes': parent_episodes_json_safe,
            'puct_updates': self._puct_update_log.copy(),
        }
        
        if clear:
            self._puct_update_log.clear()
            # Keep parent_episodes for ongoing tracking
            
        return data
    
    def shutdown(self):
        """Cleanup resources.
        
        AsyncRewardWrapper uses ProcessPoolExecutor with automatic cleanup via weakref.
        No explicit shutdown needed, but we clear the reference.
        """
        if hasattr(self, 'async_reward_fn'):
            logger.info("Cleaning up AsyncRewardWrapper...")
            # AsyncRewardWrapper uses weakref.finalize for automatic cleanup
            # Just clear the reference to allow garbage collection
            self.async_reward_fn = None
    
    def __del__(self):
        """Destructor for compatibility."""
        pass
