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

import asyncio
import os
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
        reward_fn: Callable = tttd_reward_fn,
        enable_thinking: bool = False,
        auto_flush: bool = True,
        max_prompt_thinking_tokens: int = 26000,
        max_reward_workers: int | None = None,
    ):
        """
        Initialize TTT-Discover Workflow V2.
        
        Args:
            env: Environment instance (e.g., InequalitiesEnv, CirclePackingEnv)
            gconfig: Generation hyperparameters
            tokenizer: Tokenizer or path to tokenizer
            reward_fn: Reward function following AReaL convention:
                fn(prompt, completions, prompt_ids, completion_ids, **data) -> float
                Default is tttd_reward_fn which requires _env and _state in data.
            enable_thinking: Whether to enable thinking mode
            auto_flush: Whether to auto-flush sampler updates
            max_prompt_thinking_tokens: Max tokens for prompt + thinking
            max_reward_workers: Max workers for AsyncRewardWrapper. 
                Defaults to TTTD_MAX_CODE_WORKERS env var or 16.
        """
        self.env = env
        self.auto_flush = auto_flush
        self.max_prompt_thinking_tokens = max_prompt_thinking_tokens
        
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
            max_reward_workers = int(os.environ.get('TTTD_MAX_CODE_WORKERS', '16'))
        
        logger.info(f"TTTDiscoverWorkflowV2: max_reward_workers={max_reward_workers}")
        
        # Wrap reward function with AsyncRewardWrapper (AReaL standard pattern)
        # This dispatches reward computation to a dedicated process pool
        self.async_reward_fn = AsyncRewardWrapper(
            reward_fn=reward_fn,
            timeout_seconds=getattr(env, 'eval_timeout', 60) + 10,
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
        
        # Cache failed rollouts for delayed update (synced across ranks in distributed training)
        self._failed_parents: list[Any] = []
        
        # Track timing for execute tail latency analysis
        # Records (gpu_done_time, execute_done_time) for each rollout
        self._rollout_timing_pairs: list[tuple[float, float]] = []

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

    def _create_trajectory(
        self,
        resp: ModelResponse,
        reward: float,
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
        }
    
    def _create_failed_trajectory(
        self,
        state: "State | None",
        input_ids: list[int],
        fail_type: str,
        error_msg: str = "",
    ) -> dict[str, torch.Tensor]:
        """
        Create a trajectory for failed rollouts.
        
        Delegates to the environment to control:
        - Reward value for different failure types
        - Whether the failure contributes to training (loss_mask)
        - Failure-specific observations
        
        Args:
            state: The state object (may be None)
            input_ids: Input token IDs
            fail_type: Type of failure (timeout, code_extraction_failed, execution_error, missing_state)
            error_msg: Additional error message
            
        Returns:
            Dictionary with trajectory tensors
        """
        return self.env.create_failed_trajectory(
            state=state,
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            fail_type=fail_type,
            error_msg=error_msg,
        )

    @trace_session("reward")
    async def _compute_reward(
        self,
        resp: ModelResponse,
        task_data: dict[str, Any],
        gpu_done_time: float | None = None,
    ) -> tuple[float, EnvResult, str]:
        """Compute reward by executing code using AsyncRewardWrapper.
        
        This method uses AsyncRewardWrapper with tttd_reward_fn to run env.execute 
        in a ProcessPoolExecutor, enabling true process-level parallelism for code 
        verification. The reward function returns (reward, EnvResult, code) tuple
        to avoid re-executing code.
        
        Args:
            resp: Model response from generation
            task_data: Task data including state object
            gpu_done_time: Timestamp when GPU inference completed (for tail latency measurement)
            
        Returns:
            tuple: (reward_value, EnvResult, extracted_code)
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
            return result.reward, result, ""
        
        try:
            # Use AsyncRewardWrapper with tttd_reward_fn
            # tttd_reward_fn returns (reward, EnvResult, code) tuple
            prompt_str = self.tokenizer.decode(resp.input_tokens)
            
            reward, result, extracted_code = await self.async_reward_fn(
                prompt_str,
                completion_str,
                resp.input_tokens,
                resp.output_tokens,
                _env=self.env,
                _state=state,
            )
            
            elapsed = time.time() - start_time
            fail_type_info = f", fail_type={result.fail_type}" if result.fail_type else ""
            logger.info(f"reward={result.reward:.4f}, valid={result.is_valid}, elapsed={elapsed:.1f}s{fail_type_info}")
            
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
        )
        
        if gpu_done_time is not None:
            self._rollout_timing_pairs.append((gpu_done_time, time.perf_counter()))
        
        return result.reward, result, extracted_code

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
        
        state = data.get("_state_obj")
        if state is None:
            logger.error("Missing '_state_obj' in data")
            return self._create_failed_trajectory(
                state=None,
                input_ids=[0],
                fail_type="missing_state",
            )
        
        try:
            # Generate - Phase 1: Normal thinking
            prompt = self.env.get_prompt(state)
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
            reward, result, code = await self._compute_reward(resp, data, gpu_done_time)
            
            # Log validation failure (concise)
            if not result.is_valid:
                logger.warning(f"Validation failed: reward={reward:.4f}, fail_type={result.fail_type}")
            
            # Create trajectory - use failed trajectory for invalid results
            if result.is_valid:
                trajectory = self._create_trajectory(resp, reward)
            else:
                # Use fail_type from result if available, otherwise default to execution_error
                fail_type = result.fail_type or "execution_error"
                trajectory = self._create_failed_trajectory(
                    state=state,
                    input_ids=input_ids,
                    fail_type=fail_type,
                    error_msg=result.observation,
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
                    async with self._pending_lock:
                        self._pending_children.append(child)
                        self._pending_parents.append(state)
                        
                        # Track stats for this parent (atomic under lock)
                        pid = state.id
                        if pid not in self._parent_stats:
                            self._parent_stats[pid] = []
                        self._parent_stats[pid].append(reward)
                        
                except Exception as e:
                    logger.warning(f"Failed to create child state: {e}")
            elif not result.is_valid:
                # Failure: cache failed parent for delayed update (will be synced across ranks)
                try:
                    async with self._pending_lock:
                        self._failed_parents.append(state)
                        logger.debug(f"Cached failed rollout for parent {state.id}")
                except Exception as e:
                    logger.warning(f"Failed to cache failed rollout: {e}")
            
            return trajectory
            
        except Exception as e:
            logger.error(f"arun_episode failed: {e}", exc_info=True)
            # Return failed trajectory - let env decide reward and loss_mask
            if state:
                prompt = self.env.get_prompt(state)
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

    def get_pending_updates(self, clear: bool = True) -> tuple[list, list, list]:
        """Return the buffered pending updates (children, parents, failed_parents) for this batch."""
        children = self._pending_children.copy()
        parents = self._pending_parents.copy()
        failed = self._failed_parents.copy()
        
        if clear:
            self._pending_children.clear()
            self._pending_parents.clear()
            self._failed_parents.clear()
            self._parent_stats.clear()
        
        return children, parents, failed

    
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
