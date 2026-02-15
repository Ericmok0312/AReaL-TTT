"""
Environment definitions for TTT-Discover.

This module provides the base environment interface (BaseEnv) and result
data class (EnvResult) for executing LLM-generated code in custom environments.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

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
