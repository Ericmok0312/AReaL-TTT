# areal/experimental/ttt_discover/workflow.py
"""
TTT-Discover Workflow for training with environment-based rewards.

This module provides TTTDiscoverWorkflow, a custom rollout workflow that:
1. Dynamically generates prompts from State objects using environment-specific logic
2. Executes LLM-generated code in custom environments to compute rewards
3. Supports multiple environments through a registry pattern

Usage:
    1. Define your custom State (inherit from State)
    2. Create a BaseEnv subclass for your problem
    3. Register environments in the workflow

Example:
    >>> from areal.experimental.ttt_discover.workflow import TTTDiscoverWorkflow, BaseEnv
    >>> from areal.experimental.ttt_discover.state import State
    
    >>> class MyState(State):
    ...     def __init__(self, timestep, problem_data, **kwargs):
    ...         super().__init__(timestep, **kwargs)
    ...         self.problem_data = problem_data
    
    >>> class MyEnv(BaseEnv):
    ...     def get_prompt(self, state):
    ...         return f"Solve: {state.problem_data}"
    ...     def execute(self, code, state):
    ...         # Execute code and return reward
    ...         result = run_in_sandbox(code, state.problem_data)
    ...         return EnvResult(reward=result.score)
    
    >>> workflow = TTTDiscoverWorkflow(
    ...     env_registry={"my_problem": MyEnv()},
    ...     gconfig=config.gconfig,
    ...     tokenizer=tokenizer,
    ... )
"""

import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.reward_api import AsyncRewardWrapper
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)
from areal.workflow.rlvr import RLVRWorkflow

from .state import State

logger = logging.getLogger("TTTDiscoverWorkflow")


@dataclass
class EnvResult:
    """
    Result of executing code in an environment.
    
    Attributes:
        reward: The computed reward value (higher is better)
        observation: Optional logs/stdout from execution
        is_valid: Whether the execution was successful and result is valid
        metadata: Additional environment-specific data
    """
    reward: float
    observation: str = ""
    is_valid: bool = True
    metadata: dict[str, Any] | None = None


class BaseEnv(ABC):
    """
    Abstract base class for problem environments.
    
    Users should subclass this to define custom problem domains.
    Each environment handles:
    - Prompt generation for a given State
    - Code execution and reward computation
    - Optional custom code extraction logic
    
    Example:
        >>> class CirclePackingEnv(BaseEnv):
        ...     def get_prompt(self, state):
        ...         return f"Pack circles with radii: {state.radii}"
        ...     
        ...     def execute(self, code, state):
        ...         # Run the generated code in a sandbox
        ...         circles = extract_circles_from_code(code)
        ...         score = compute_packing_density(circles, state.radii)
        ...         return EnvResult(reward=score)
    """

    @abstractmethod
    def get_prompt(self, state: State) -> str:
        """
        Generate a prompt string for the given State.
        
        Args:
            state: The State object containing problem context
            
        Returns:
            A prompt string to be sent to the LLM
        """
        raise NotImplementedError

    @abstractmethod
    def execute(self, code: str, state: State) -> EnvResult:
        """
        Execute the LLM-generated code and compute reward.
        
        This is where the actual environment interaction happens.
        The code should be executed in a sandbox or controlled environment.
        
        Args:
            code: The extracted code from LLM completion
            state: The State object containing problem context (construction, etc.)
            
        Returns:
            EnvResult containing reward and execution metadata
        """
        raise NotImplementedError

    def extract_code(self, completion: str) -> str | None:
        """
        Extract executable code from raw LLM completion.
        
        Override this for custom extraction logic (e.g., different markdown formats).
        
        Args:
            completion: Raw completion string from LLM
            
        Returns:
            Extracted code string, or None if extraction fails
        """
        import re
        
        # Default: extract code from ```python ... ``` blocks
        pattern = r"```python\s+([\s\S]*?)\s*```"
        match = re.search(pattern, completion)
        if match:
            return match.group(1).strip()
        
        # Fallback: try generic code block
        pattern = r"```\s+([\s\S]*?)\s*```"
        match = re.search(pattern, completion)
        if match:
            return match.group(1).strip()
        
        # No code block found, return the whole completion
        return completion.strip()


class TTTDiscoverWorkflow(RLVRWorkflow):
    """
    TTT-Discover workflow with environment-based reward computation.
    
    This workflow extends RLVRWorkflow to support:
    - Dynamic prompt generation from State objects
    - Environment-specific code execution for reward computation
    - Multi-environment support through a registry
    
    Unlike standard RLVRWorkflow which uses static reward functions,
    TTTDiscoverWorkflow executes LLM-generated code in custom environments
    to compute rewards.
    
    Attributes:
        env_registry: Mapping from environment type names to BaseEnv instances
        tokenizer: The tokenizer for encoding/decoding
        gconfig: Generation hyperparameters
        enable_thinking: Whether to enable thinking tokens
    """

    def __init__(
        self,
        env_registry: dict[str, BaseEnv],
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
    ):
        """
        Initialize TTTDiscoverWorkflow.
        
        Args:
            env_registry: Mapping from env type names (e.g., "ac1", "cp") to BaseEnv instances
            gconfig: Generation hyperparameters (temperature, max_tokens, etc.)
            tokenizer: Tokenizer for encoding/decoding, or path to tokenizer
            enable_thinking: Whether to enable thinking tokens in generation
        """
        # Store env registry before calling parent
        self.env_registry = env_registry
        
        # Initialize parent with a placeholder reward_fn
        # We override _compute_rewards so this won't be used
        super().__init__(
            reward_fn=self._placeholder_reward,
            gconfig=gconfig,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
            # We override arun_episode entirely, so these don't matter
            get_input_ids_fn=lambda x, t, e: [],
            data_extract_prompt_fn=lambda x: [],
        )

    def _placeholder_reward(
        self,
        prompt: str,
        completion: str,
        prompt_ids: list[int],
        completion_ids: list[int],
        **data
    ) -> float:
        """Placeholder reward function - should never be called."""
        raise RuntimeError(
            "_placeholder_reward should never be called. "
            "TTTDiscoverWorkflow overrides _compute_rewards."
        )

    def _detect_env_type(self, state: State) -> str:
        """
        Detect environment type from State object.
        
        Maps State class names to environment type names.
        Override this method to add custom mappings.
        
        Args:
            state: The State object
            
        Returns:
            Environment type name (key in env_registry)
        """
        from .state import (
            InequalitiesState,
            CirclePackingState,
            GpuModeState,
            AleBenchState,
            ErdosState,
            DenoisingState,
        )
        
        state_type = type(state).__name__
        
        # Default mappings
        mapping = {
            "InequalitiesState": "ac1",  # Could be ac1 or ac2, need to distinguish
            "CirclePackingState": "cp",
            "GpuModeState": "gpu_mode",
            "AleBenchState": "ale_bench",
            "ErdosState": "erdos",
            "DenoisingState": "denoising",
        }
        
        if state_type in mapping:
            return mapping[state_type]
        
        # If not in mapping, use the state type name directly
        # User should register env with this name
        return state_type

    def _get_env_for_state(self, state: State) -> BaseEnv:
        """
        Get the appropriate environment for a given State.
        
        Args:
            state: The State object
            
        Returns:
            BaseEnv instance from registry
            
        Raises:
            KeyError: If no environment is registered for the state type
        """
        env_type = self._detect_env_type(state)
        if env_type not in self.env_registry:
            raise KeyError(
                f"No environment registered for type '{env_type}'. "
                f"Available: {list(self.env_registry.keys())}. "
                f"State type: {type(state).__name__}"
            )
        return self.env_registry[env_type]

    @trace_session("reward")
    async def _compute_rewards(
        self,
        resp: ModelResponse,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> float:
        """
        Compute reward by executing code in the appropriate environment.
        
        This overrides the parent method to use environment-based reward computation.
        
        Args:
            resp: Model response containing generated tokens
            prompt_str: The original prompt string
            task_data: Additional data including _state_obj
            
        Returns:
            Computed reward value
        """
        # Extract state from task_data
        state = task_data.get("_state_obj")
        if state is None:
            raise ValueError(
                "TTTDiscoverWorkflow requires '_state_obj' in task_data. "
                "Ensure the dataloader yields data with '_state_obj' field."
            )
        
        # Get the appropriate environment
        env = self._get_env_for_state(state)
        
        # Decode the completion
        completion_str = self.tokenizer.decode(resp.output_tokens)
        
        # Extract code from completion
        code = env.extract_code(completion_str)
        if code is None:
            logger.warning(f"Failed to extract code from completion: {completion_str[:200]}...")
            # Return penalty for failed extraction
            return -1.0
        
        # Execute code in environment and get result
        try:
            result = env.execute(code, state)
        except Exception as e:
            logger.warning(f"Environment execution failed: {e}")
            result = EnvResult(reward=-1.0, observation=str(e), is_valid=False)
        
        # Log metrics
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=result.reward,
            is_valid=float(result.is_valid),
        )
        
        return result.reward

    @session_context()
    async def _collect_samples(
        self,
        engine: InferenceEngine,
        req: ModelRequest,
        prompt_str: str,
        task_data: dict[str, Any],
    ) -> tuple[ModelResponse, float]:
        """
        Generate one sample and compute its reward.
        
        Args:
            engine: Inference engine for generation
            req: Model request containing input IDs
            prompt_str: Decoded prompt string for logging
            task_data: Additional data including _state_obj
            
        Returns:
            Tuple of (model response, reward value)
        """
        async with atrace_session_phase("generate"):
            resp = await engine.agenerate(req)
        
        reward = await self._compute_rewards(resp, prompt_str, task_data)
        
        return resp, reward

    async def arun_episode(
        self,
        engine: InferenceEngine,
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """
        Execute one episode: generate response and compute environment-based reward.
        
        This is the main entry point called by the training loop.
        It completely overrides the parent implementation for full control.
        
        Flow:
        1. Extract State from data
        2. Get appropriate environment
        3. Generate prompt from State using env.get_prompt()
        4. Tokenize and create ModelRequest
        5. Generate response via engine.agenerate()
        6. Execute generated code in environment and compute reward
        7. Return tensors for training
        
        Args:
            engine: Inference engine (vLLM, SGLang, etc.)
            data: Dictionary containing at least '_state_obj' key
            
        Returns:
            Dictionary of tensors with shape [batch=1, seq_len]
            
        Raises:
            ValueError: If '_state_obj' is missing from data
            KeyError: If no environment is registered for the state type
        """
        # Extract state from data (required)
        state = data.get("_state_obj")
        if state is None:
            raise ValueError(
                "TTTDiscoverWorkflow requires 'data['_state_obj']' to be a State object. "
                "Ensure your dataloader yields data with '_state_obj' field."
            )
        
        # Get the appropriate environment for this state
        env = self._get_env_for_state(state)
        
        # Generate prompt using the environment
        prompt = env.get_prompt(state)
        
        # Build messages for chat template
        messages = [{"role": "user", "content": prompt}]
        
        # Tokenize using chat template
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        input_ids = list(input_ids)
        
        # Create model request
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
        )
        
        # Decode prompt for logging/reward computation
        prompt_str = self.tokenizer.decode(input_ids)
        
        # Generate and compute reward
        resp, reward = await self._collect_samples(engine, req, prompt_str, data)
        
        # Build result tensors with batch dimension 1
        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions
        
        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(reward, dtype=torch.float32),
        }
        
        # Add batch dimension
        return {k: v.unsqueeze(0) for k, v in res.items()}
