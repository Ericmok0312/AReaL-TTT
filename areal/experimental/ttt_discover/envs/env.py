# SPDX-License-Identifier: Apache-2.0

"""
Environment definitions for TTT-Discover.

This module provides the base environment interface (BaseEnv) and result
data class (EnvResult) for executing LLM-generated code in custom environments.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from ..state import State


@dataclass
class EnvResult:
    """
    Result of executing code in an environment.

    Attributes:
        reward: The computed reward value (higher is better)
        observation: Optional logs/stdout from execution
        is_valid: Whether the execution was successful and result is valid
        metadata: Additional environment-specific data
        fail_type: Type of failure (timeout, code_extraction_failed, execution_error, etc.)
    """

    reward: float
    observation: str = ""
    is_valid: bool = True
    metadata: dict[str, Any] | None = None
    fail_type: str | None = None


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

    @abstractmethod
    def create_state(
        self,
        parent_state: State,
        code: str,
        reward: float,
        result: EnvResult,
        timestep: int,
    ) -> State:
        """
        Create a new child State from the execution result.

        This method creates a new State object that represents the child node
        in the PUCT search tree. The environment determines the specific State
        subclass to instantiate based on its problem domain.

        Args:
            parent_state: The parent State from which this rollout originated
            code: The extracted code from LLM completion
            reward: The computed reward value
            result: The full EnvResult from environment execution
            timestep: The current timestep (parent_state.timestep + 1)

        Returns:
            A new State instance (specific subclass) representing the child node

        Example:
            >>> def create_state(self, parent, code, reward, result, timestep):
            ...     from .state import InequalitiesState
            ...     return InequalitiesState(
            ...         timestep=timestep,
            ...         construction=result.metadata.get("construction"),
            ...         code=code,
            ...         value=reward,
            ...         observation=result.observation,
            ...         parent_values=[parent.value] if parent.value else [],
            ...         parents=[{"id": parent.id, "timestep": parent.timestep}],
            ...     )
        """
        raise NotImplementedError

    def truncate_prompt(
        self, prompt: str, tokenizer: PreTrainedTokenizerFast, max_tokens: int
    ) -> str:
        """
        Truncate a prompt string to fit within ``max_tokens``.

        The default implementation simply truncates token IDs from the end.
        Subclasses can override this to perform domain-aware truncation
        (e.g., drop verbose tool usage sections while preserving the core
        problem statement).

        Args:
            prompt: Prompt string to truncate.
            tokenizer: Tokenizer used to count tokens.
            max_tokens: Maximum number of tokens allowed.

        Returns:
            Truncated prompt string.
        """
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(input_ids) <= max_tokens:
            return prompt
        truncated_ids = input_ids[:max_tokens]
        return tokenizer.decode(truncated_ids)

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

    def get_failure_result(
        self,
        state: State | None,
        fail_type: str,
        error_msg: str = "",
    ) -> EnvResult:
        """
        Create failure result for a rollout.

        Subclasses can override this to customize failure rewards and observations.

        Args:
            state: The state object (may be None if state is missing)
            fail_type: Type of failure (timeout, code_extraction_failed, execution_error, missing_state)
            error_msg: Additional error message

        Returns:
            EnvResult with appropriate reward and observation for the failure type
        """
        if fail_type == "timeout":
            return EnvResult(
                reward=0.0,
                observation=f"Execution timeout: {error_msg}"
                if error_msg
                else "Execution timeout",
                is_valid=False,
                fail_type=fail_type,
            )
        elif fail_type == "code_extraction_failed":
            return EnvResult(
                reward=0.0,
                observation=f"Failed to extract code: {error_msg}"
                if error_msg
                else "Failed to extract code from response",
                is_valid=False,
                fail_type=fail_type,
            )
        elif fail_type == "missing_state":
            return EnvResult(
                reward=-1.0,
                observation=f"Missing state: {error_msg}"
                if error_msg
                else "Missing state object",
                is_valid=False,
                fail_type=fail_type,
            )
        else:  # execution_error or other
            return EnvResult(
                reward=0.0,
                observation=f"Execution error: {error_msg}"
                if error_msg
                else "Execution failed",
                is_valid=False,
                fail_type=fail_type,
            )

    def create_failed_trajectory(
        self,
        state: State | None,
        input_ids: list[int],
        tokenizer: PreTrainedTokenizerFast,
        fail_type: str,
        error_msg: str = "",
    ) -> dict[str, torch.Tensor]:
        """
        Create trajectory for a failed rollout.

        This method allows the environment to control:
        - The reward for different failure types
        - Whether the failure should contribute to training (loss_mask)
        - The observation/error message

        Args:
            state: The state object (may be None)
            input_ids: Input token IDs
            tokenizer: Tokenizer for encoding
            fail_type: Type of failure
            error_msg: Additional error message

        Returns:
            Dictionary with trajectory tensors
        """
        # Get environment-specific failure result
        result = self.get_failure_result(state, fail_type, error_msg)

        seq = input_ids + [tokenizer.eos_token_id or 0]

        # Following ttt-discover logic: all failed rollouts participate in training
        # with reward=0. Only successful rollouts (correctness > 0) create new states.
        # This allows the model to learn from failures (what not to do).

        return {
            "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
            "loss_mask": torch.ones(len(seq), dtype=torch.int32).unsqueeze(
                0
            ),  # Train on all
            "logprobs": torch.zeros(len(seq), dtype=torch.float32).unsqueeze(0),
            "versions": torch.full((len(seq),), -1, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor([result.reward], dtype=torch.float32),
            "raw_scores": torch.tensor([float("nan")], dtype=torch.float32),
        }
