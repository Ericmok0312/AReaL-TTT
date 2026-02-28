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
import concurrent.futures
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.perf_tracer import atrace_session_phase, trace_session

from .envs.env import BaseEnv, EnvResult

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
        sampler: "StateSampler | None",  # 直接注入 sampler
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        auto_flush: bool = True,  # 是否自动 flush
        max_code_workers: int = 16,  # Max concurrent code executions
    ):
        self.env = env
        self.auto_flush = auto_flush
        
        # Initialize tokenizer
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer
            self.tokenizer = load_hf_tokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer
            
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        
        # Thread pool for running sync env.execute in background.
        self._code_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_code_workers,
            thread_name_prefix="tttd_code_exec"
        )
        
        # Semaphore limits concurrent code execution to prevent ThreadPool exhaustion.
        # Must be <= max_code_workers to ensure we never overwhelm the pool.
        self._code_semaphore = asyncio.Semaphore(max_code_workers)
        
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
    ) -> tuple[float, EnvResult, str]:
        """Compute reward by executing code in environment.
        
        This method uses run_in_executor to run sync env.execute in a thread pool,
        preventing blocking of the async event loop. This allows multiple rollouts
        to execute code concurrently without stalling AReaL's AsyncTaskRunner.
        """
        completion_str = self.tokenizer.decode(resp.output_tokens)
        
        code = self.env.extract_code(completion_str)
        state = task_data.get("_state_obj")
        
        if code is None:
            logger.warning(f"Code extraction failed")
            # Use env's failure result for consistent handling
            result = self.env.get_failure_result(
                state=state,
                fail_type="code_extraction_failed",
            )
            return result.reward, result, ""
        
        try:
            # Run sync env.execute in dedicated thread pool with semaphore control.
            # The semaphore limits concurrent executions to prevent CPU overload.
            # Add asyncio.timeout to prevent indefinite hanging.
            import time
            start_time = time.time()
            
            async with self._code_semaphore:
                loop = asyncio.get_running_loop()
                # Use asyncio.wait_for to add timeout protection at asyncio level
                # This is in addition to env.execute's internal timeout
                # Get timeout from env, default to 60s, add 5s buffer for asyncio
                env_timeout = getattr(self.env, 'eval_timeout', 60)
                asyncio_timeout = env_timeout + 5.0
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            self._code_executor,
                            self.env.execute,
                            code,
                            state
                        ),
                        timeout=asyncio_timeout  # Dynamic based on env.eval_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Code execution timed out at asyncio level after {asyncio_timeout}s")
                    result = self.env.get_failure_result(
                        state=state,
                        fail_type="async_timeout",
                        error_msg=f"Code execution timed out (asyncio level, timeout={asyncio_timeout}s)",
                    )
            
            elapsed = time.time() - start_time
            fail_type_info = f", fail_type={result.fail_type}" if result.fail_type else ""
            logger.info(f"reward={result.reward:.4f}, valid={result.is_valid}, elapsed={elapsed:.1f}s{fail_type_info}")
        except Exception as e:
            # Execution errors (TimeoutError is typically caught by env and returned as result with fail_type)
            logger.warning(f"Execution failed: {e}")
            # Use env's failure result for consistent handling
            result = self.env.get_failure_result(
                state=state,
                fail_type="execution_error",
                error_msg=str(e),
            )
        
        stats_tracker.get("rollout").scalar(
            reward=result.reward,
            is_valid=float(result.is_valid),
        )
        
        return result.reward, result, code

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
        state = data.get("_state_obj")
        if state is None:
            logger.error("Missing '_state_obj' in data")
            return self._create_failed_trajectory(
                state=None,
                input_ids=[0],
                fail_type="missing_state",
            )
        
        try:
            # Generate
            prompt = self.env.get_prompt(state)
            messages = [{"role": "user", "content": prompt}]
            input_ids = list(self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            ))
            
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            
            async with atrace_session_phase("generate"):
                resp = await engine.agenerate(req)
            
            # Check if generation was truncated
            if resp.stop_reason == "length":
                logger.warning(f"Generation truncated (max_tokens={self.gconfig.max_new_tokens})")
            
            # Compute reward
            reward, result, code = await self._compute_reward(resp, data)
            
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
        """Shutdown the thread pool executor to release resources."""
        if hasattr(self, '_code_executor') and self._code_executor:
            self._code_executor.shutdown(wait=False)
            logger.info("Code execution ThreadPoolExecutor shut down")
    
    def __del__(self):
        """Destructor to ensure executor is cleaned up."""
        try:
            self.shutdown()
        except Exception:
            pass
