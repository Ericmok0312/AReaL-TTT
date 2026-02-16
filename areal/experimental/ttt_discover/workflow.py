# areal/experimental/ttt_discover/workflow.py
"""
TTT-Discover Workflow for training with environment-based rewards.

This workflow handles:
1. Converting State to prompt via env.get_prompt()
2. Generating response from LLM
3. Extracting code and computing reward via env.execute()

The workflow is designed to work with AReaL's GroupedRolloutWorkflow for 
group rollout support. Each arun_episode call handles a SINGLE rollout.
Sampler updates are managed externally (e.g., in training loop) to allow
flexible group handling.

Usage with Group Rollout:
    >>> from areal.experimental.ttt_discover.workflow import TTTDiscoverWorkflow
    >>> from areal.experimental.ttt_discover.env import BaseEnv, EnvResult
    >>> from areal.experimental.ttt_discover.state import State
    
    >>> class MyEnv(BaseEnv):
    ...     def get_prompt(self, state):
    ...         return f"Improve this code: {state.code}"
    ...     def execute(self, code, state):
    ...         result = evaluate_code(code, state)
    ...         return EnvResult(reward=result.score, metadata={"construction": result.construction})
    
    >>> # Create simple workflow (single rollout per call)
    >>> workflow = TTTDiscoverWorkflow(
    ...     env=MyEnv(),
    ...     gconfig=config.gconfig,
    ...     tokenizer=tokenizer,
    ... )
    
    >>> # Use with AReaL's group_size for automatic group rollout
    >>> batch = actor.prepare_batch(
    ...     dataloader,
    ...     workflow=workflow,
    ...     group_size=8,  # AReaL handles group rollout
    ... )
    
    >>> # Sampler update is done externally after group completes
    >>> for batch in dataloader:
    ...     results = workflow.rollout_group(engine, batch, group_size=8)
    ...     # Process results and update sampler manually
    ...     best_result = select_best(results)
    ...     sampler.update_states([best_result.state], [batch["_state_obj"]])
"""

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)

from .envs.env import BaseEnv, EnvResult

if TYPE_CHECKING:
    from .state import State

logger = logging.getLogger("TTTDiscoverWorkflow")


@dataclass
class RolloutResult:
    """Result of a single rollout, including all information needed for training and sampler update."""
    # Trajectory tensors for training
    trajectory: dict[str, torch.Tensor]
    
    # Information for sampler update (optional)
    parent_state: "State | None" = None
    child_state: "State | None" = None  # Created by env.create_state()
    reward: float = 0.0
    is_valid: bool = False
    
    # Raw outputs for logging/debugging
    code: str = ""
    observation: str = ""


class TTTDiscoverWorkflow(RolloutWorkflow):
    """
    TTT-Discover workflow for single rollout generation.
    
    This workflow is designed to be simple and composable:
    - Single rollout per arun_episode() call
    - Uses env for prompt generation and reward computation
    - Returns both trajectory (for training) and metadata (for sampler update)
    
    For group rollout, use with AReaL's GroupedRolloutWorkflow or call 
    rollout_group() which handles group logic and returns results for 
    external sampler update.
    
    Attributes:
        env: Environment instance providing get_prompt(), execute(), extract_code()
        tokenizer: Tokenizer for encoding/decoding
        gconfig: Generation hyperparameters
        enable_thinking: Whether to enable thinking tokens
    """

    def __init__(
        self,
        env: BaseEnv,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
    ):
        """
        Initialize TTTDiscoverWorkflow.
        
        Args:
            env: Environment instance with get_prompt(), execute(), extract_code()
            gconfig: Generation hyperparameters
            tokenizer: Tokenizer or path to tokenizer
            enable_thinking: Whether to enable thinking tokens
        """
        # Storage for batch metadata - cleared at start of each batch by training script
        self._batch_metadata: list[dict] = []
        
        self.env = env
        
        # Initialize tokenizer
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer
            self.tokenizer = load_hf_tokenizer(tokenizer)
        else:
            self.tokenizer = tokenizer
            
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking

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
        """Create a placeholder trajectory for failed rollouts.
        
        This ensures alignment between batch and metadata when GroupedRolloutWorkflow
        concatenates results. The trajectory has loss_mask=0 so it doesn't affect training.
        """
        seq = input_ids + [self.tokenizer.eos_token_id or 0]  # Minimal sequence
        logprobs = [0.0] * len(seq)
        loss_mask = [0] * len(seq)  # No training on failed rollouts
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
        """
        Compute reward by executing code in environment.
        
        Returns:
            Tuple of (reward, env_result, extracted_code)
        """
        # Decode completion
        completion_str = self.tokenizer.decode(resp.output_tokens)
        
        # Extract code
        code = self.env.extract_code(completion_str)
        if code is None:
            logger.debug("Failed to extract code from completion")
            return -1.0, EnvResult(reward=-1.0, observation="Code extraction failed", is_valid=False), ""
        
        # Get state from task_data
        state = task_data.get("_state_obj")
        
        # Execute in environment
        try:
            result = self.env.execute(code, state)
        except Exception as e:
            logger.warning(f"Environment execution failed: {e}")
            result = EnvResult(reward=-1.0, observation=str(e), is_valid=False)
        
        # Log metrics
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=result.reward,
            is_valid=float(result.is_valid),
        )
        
        return result.reward, result, code

    @session_context()
    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor] | None:
        """
        Execute a SINGLE rollout episode.
        
        This is the core method called by AReaL's infrastructure.
        For group rollout, AReaL's GroupedRolloutWorkflow will call this 
        multiple times and concatenate results.
        
        Flow:
        1. Extract State from data
        2. Generate prompt via env.get_prompt(state)
        3. Generate response via engine.agenerate()
        4. Extract code and compute reward via env.execute()
        5. Return trajectory tensors
        
        Args:
            engine: Inference engine
            data: Dictionary with '_state_obj' containing the State
            
        Returns:
            Trajectory dict with tensors [batch=1, seq_len], or None on failure
        """
        try:
            # Extract state
            state = data.get("_state_obj")
            if state is None:
                raise ValueError("Missing '_state_obj' in data")
            
            # Generate prompt from state
            prompt = self.env.get_prompt(state)
            
            # Build messages and tokenize
            messages = [{"role": "user", "content": prompt}]
            input_ids = list(self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            ))
            
            # Create request
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
            )
            
            # Generate
            async with atrace_session_phase("generate"):
                resp = await engine.agenerate(req)
            
            # Compute reward
            reward, result, code = await self._compute_reward(resp, data)
            
            # Create trajectory
            trajectory = self._create_trajectory(resp, reward)
            
            # Store metadata in workflow's batch storage
            # This provides a side-channel for non-tensor data
            self._batch_metadata.append({
                "parent_state": state,
                "reward": reward,
                "is_valid": result.is_valid,
                "code": code,
                "observation": result.observation,
                "metadata": result.metadata,
            })
            
            return trajectory
            
        except Exception as e:
            logger.error(f"arun_episode failed: {e}", exc_info=True)
            # Create a placeholder trajectory and metadata for failed rollout
            # This ensures alignment with GroupedRolloutWorkflow (which filters None)
            state = data.get("_state_obj")
            
            # Get input_ids from the original prompt for minimal trajectory
            prompt = self.env.get_prompt(state) if state else ""
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
            
            # Record metadata
            self._batch_metadata.append({
                "parent_state": state,
                "reward": -1.0,
                "is_valid": False,
                "code": "",
                "observation": f"Error: {e}",
                "metadata": {"error": str(e)},
            })
            
            # Return placeholder trajectory (loss_mask=0, won't affect training)
            return self._create_failed_trajectory(input_ids)

    async def rollout_group(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
        group_size: int,
    ) -> list[RolloutResult]:
        """
        Perform group rollout: multiple rollouts for the same state.
        
        This is a convenience method for manual group handling when not using
        AReaL's GroupedRolloutWorkflow.
        
        Args:
            engine: Inference engine
            data: Input data with '_state_obj'
            group_size: Number of rollouts to perform
            
        Returns:
            List of RolloutResult for each rollout in the group
        """
        state = data.get("_state_obj")
        if state is None:
            raise ValueError("Missing '_state_obj' in data")
        
        results = []
        for i in range(group_size):
            logger.debug(f"Group rollout {i+1}/{group_size}")
            
            trajectory = await self.arun_episode(engine, data)
            
            if trajectory is None:
                # Failed rollout
                results.append(RolloutResult(
                    trajectory=self._create_empty_trajectory(-1.0),
                    parent_state=state,
                    reward=-1.0,
                    is_valid=False,
                ))
                continue
            
            # Get metadata from workflow's batch storage
            # Note: In rollout_group, metadata was stored during arun_episode
            metadata = self._batch_metadata[len(results)] if len(results) < len(self._batch_metadata) else {}
            
            # Create child state if valid
            child_state = None
            if metadata.get("is_valid") and metadata.get("code"):
                try:
                    child_state = self.env.create_state(
                        parent_state=state,
                        code=metadata["code"],
                        reward=metadata["reward"],
                        result=EnvResult(
                            reward=metadata["reward"],
                            observation=metadata["observation"],
                            is_valid=metadata["is_valid"],
                            metadata=metadata.get("metadata"),
                        ),
                        timestep=state.timestep + 1,
                    )
                except Exception as e:
                    logger.warning(f"Failed to create child state: {e}")
            
            results.append(RolloutResult(
                trajectory=trajectory,
                parent_state=state,
                child_state=child_state,
                reward=metadata.get("reward", -1.0),
                is_valid=metadata.get("is_valid", False),
                code=metadata.get("code", ""),
                observation=metadata.get("observation", ""),
            ))
        
        return results

    def _create_empty_trajectory(self, reward: float = -1.0) -> dict[str, torch.Tensor]:
        """Create empty trajectory for error cases."""
        return {
            "input_ids": torch.zeros((1, 1), dtype=torch.int32),
            "loss_mask": torch.zeros((1, 1), dtype=torch.int32),
            "logprobs": torch.zeros((1, 1), dtype=torch.float32),
            "versions": torch.zeros((1, 1), dtype=torch.int32),
            "attention_mask": torch.zeros((1, 1), dtype=torch.bool),
            "rewards": torch.tensor([reward], dtype=torch.float32),
        }


