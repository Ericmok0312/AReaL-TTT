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
    ):
        self.env = env
        self.sampler = sampler
        self.auto_flush = auto_flush
        
        # Initialize tokenizer
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer
            self.tokenizer = load_hf_tokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer
            
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        
        # Async-safe buffer for pending updates (GroupedRolloutWorkflow uses asyncio.gather)
        # Stores (child_state, parent_state) tuples
        self._pending_lock = asyncio.Lock()  # Use asyncio.Lock for coroutine safety
        self._pending_children: list[Any] = []
        self._pending_parents: list[Any] = []
        
        # Track which parents have been processed this batch
        # Key: parent_state.id, Value: list of child rewards
        self._parent_stats: dict[str, list[float]] = {}

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
    
    def _create_failed_trajectory(self, input_ids: list[int]) -> dict[str, torch.Tensor]:
        """Create a placeholder trajectory for failed rollouts."""
        seq = input_ids + [self.tokenizer.eos_token_id or 0]
        logprobs = [0.0] * len(seq)
        loss_mask = [0] * len(seq)
        versions = [-1] * len(seq)
        
        return {
            "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor([-1.0], dtype=torch.float32),
        }

    @trace_session("reward")
    async def _compute_reward(
        self,
        resp: ModelResponse,
        task_data: dict[str, Any],
    ) -> tuple[float, EnvResult, str]:
        """Compute reward by executing code in environment."""
        completion_str = self.tokenizer.decode(resp.output_tokens)
        code = self.env.extract_code(completion_str)
        if code is None:
            return -1.0, EnvResult(reward=-1.0, observation="Code extraction failed", is_valid=False), ""
        
        state = task_data.get("_state_obj")
        try:
            result = self.env.execute(code, state)
        except Exception as e:
            logger.warning(f"Environment execution failed: {e}")
            result = EnvResult(reward=-1.0, observation=str(e), is_valid=False)
        
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
            return self._create_failed_trajectory([0])
        
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
            
            # Compute reward
            reward, result, code = await self._compute_reward(resp, data)
            
            # Create trajectory
            trajectory = self._create_trajectory(resp, reward)
            
            # Create child state and buffer sampler update (thread-safe)
            if result.is_valid and self.sampler is not None:
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
            
            return trajectory
            
        except Exception as e:
            logger.error(f"arun_episode failed: {e}", exc_info=True)
            # Return failed trajectory
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
            
            return self._create_failed_trajectory(input_ids)

    async def get_pending_updates(self, clear: bool = True) -> tuple[list, list]:
        """
        Get pending sampler updates without applying them.
        
        This is used for distributed training to gather updates from all ranks
        before applying them centrally on rank 0.
        
        Args:
            clear: If True, clear the pending buffers after copying
            
        Returns:
            Tuple of (children_states, parent_states)
        """
        async with self._pending_lock:
            children = self._pending_children.copy()
            parents = self._pending_parents.copy()
            
            if clear:
                self._pending_children.clear()
                self._pending_parents.clear()
                self._parent_stats.clear()
        
        return children, parents

    async def flush(self, save: bool = False, step: int | None = None) -> dict[str, Any]:
        """
        Commit all pending sampler updates.
        
        This should be called after prepare_batch completes.
        
        Args:
            save: Whether to save sampler state to disk
            step: Current training step (for saving)
            
        Returns:
            Stats dict with update counts
        """
        if self.sampler is None:
            return {"updated": 0}
        
        children, parents = await self.get_pending_updates(clear=True)
        
        if not children:
            return {"updated": 0}
        
        # Update sampler (PUCTSampler handles top-k internally)
        self.sampler.update_states(children, parents, save=save, step=step)
        
        # Log stats
        num_parents = len(set(p.id for p in parents))
        logger.info(f"Flushed {len(children)} children from {num_parents} parents to sampler")
        
        return {
            "updated": len(children),
            "num_parents": num_parents,
        }

    async def reset(self):
        """Reset internal buffers (call at start of each step)."""
        async with self._pending_lock:
            self._pending_children.clear()
            self._pending_parents.clear()
            self._parent_stats.clear()

    def reset_sync(self):
        """Synchronous version of reset()."""
        self._pending_children.clear()
        self._pending_parents.clear()
        self._parent_stats.clear()

    def get_pending_updates_sync(self, clear: bool = True) -> tuple[list, list]:
        """Synchronous version of get_pending_updates()."""
        children = self._pending_children.copy()
        parents = self._pending_parents.copy()
        
        if clear:
            self._pending_children.clear()
            self._pending_parents.clear()
            self._parent_stats.clear()
        
        return children, parents

    # Backward compatibility - these are no-ops in V2
    def init_batch_metadata(self):
        """No-op in V2. Use reset_sync() instead."""
        self.reset_sync()
        return []
    
    def reset_batch_metadata(self):
        """No-op in V2. Use reset_sync() instead."""
        pass
    
    @property
    def _batch_metadata(self):
        """Backward compatibility - returns empty list. V2 uses internal buffering."""
        return []
