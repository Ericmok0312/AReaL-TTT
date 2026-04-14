from __future__ import annotations
from abc import ABC, abstractmethod
import uuid

import numpy as np


def to_json_serializable(obj):
    """Convert numpy arrays and other non-JSON-serializable types to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    return obj


class State(ABC):
    id: str  # unique identifier for this state
    timestep: int  # the training step this state was first visited at
    value: float  # Expected value of starting from this state (higher = better)
    parent_values: list[float]  # list of ancestor values (most recent first) for terminal value estimation
    parents: list[dict]  # list of parent refs [{"id": ..., "timestep": ...}, ...] (most recent first)
    observation: str  # stdout/logs from the code that created this state
    exec_time_ms: float  # execution time (ms) when this state was created as a child

    def __init__(self, timestep: int, value: float = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None, observation: str = "", exec_time_ms: float = None):
        self.id = id if id is not None else str(uuid.uuid4())
        self.timestep = timestep
        self.value = value
        self.parent_values = parent_values if parent_values is not None else []
        self.parents = parents if parents is not None else []
        self.observation = observation
        self.exec_time_ms = exec_time_ms

    def estimate_value(self, experiences_from_state: list) -> float:
        """Estimate state value from experiences starting from this state."""
        assert len(experiences_from_state) > 0
        rewards = [exp.step_result.reward for exp in experiences_from_state]
        self.value = sum(rewards) / len(rewards)
        return self.value

    def to_prompt(self, target, metric_name: str = "value", maximize: bool = True, language: str = "") -> str:
        """Generate prompt value context from state."""
        value_ctx = f"You are iteratively optimizing {metric_name}."
        improvement_direction = "higher" if maximize else "lower"

        has_code = self.code and self.code.strip()
        if has_code:
            value_ctx += f"\nHere is the last code we ran:\n"
            if language:
                value_ctx += f"```{language}\n{self.code}\n```"
            else:
                value_ctx += f"{self.code}"
        else:
            value_ctx += f"\nNo previous code available."

        # Value context: show before/after if we have parent values
        if self.parent_values and self.value is not None and getattr(self, "construction", None):
            before_value = self.parent_values[0] if maximize else -self.parent_values[0]
            after_value = self.value if maximize else -self.value
            current_gap = target - after_value if maximize else after_value - target
            value_ctx += (
                f"\nHere is the {metric_name} before and after running the code above "
                f"({improvement_direction} is better): {before_value:.6f} -> {after_value:.6f}"
            )
            value_ctx += (
                f"\nTarget: {target}. Current gap: {current_gap:.6f}. "
                f"Further improvements will also be generously rewarded."
            )
        elif self.value is not None:
            after_value = self.value if maximize else -self.value
            current_gap = target - after_value if maximize else after_value - target
            value_ctx += f"\nCurrent {metric_name} ({improvement_direction} is better): {after_value:.6f}"
            value_ctx += (
                f"\nTarget: {target}. Current gap: {current_gap:.6f}. "
                f"Further improvements will also be generously rewarded."
            )
        else:
            value_ctx += f"\nTarget {metric_name}: {target}"

        # Show previous stdout if available
        if self.observation and self.observation.strip():
            stdout = self.observation.strip()
            if len(stdout) > 500:
                stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

        return value_ctx

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize state to dict."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> State:
        """Deserialize state from dict."""
        pass


class InequalitiesState(State):
    construction: list[float]  # the step function construction
    code: str  # the code that generated the construction

    def __init__(self, timestep: int, construction: list[float], code: str, value: float = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None, observation: str = ""):
        super().__init__(timestep, value, parent_values, parents, id, observation)
        self.construction = to_json_serializable(construction)
        self.code = code

    def to_dict(self) -> dict:
        return {
            "type": "InequalitiesState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "exec_time_ms": self.exec_time_ms,
            "construction": to_json_serializable(self.construction),
            "code": self.code,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> InequalitiesState:
        state = cls(
            timestep=d["timestep"],
            construction=d["construction"],
            code=d["code"],
            value=d.get("value"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
        )
        state.exec_time_ms = d.get("exec_time_ms")
        return state


def _to_tuple_of_tuples(obj):
    """Convert nested list [[x,y,r],...] to tuple of tuples for hashability."""
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], (list, tuple)):
        return tuple(tuple(c) for c in obj)
    return obj


class CirclePackingState(State):
    """State for circle packing - holds code and construction (circles)."""
    construction: tuple  # tuple of tuples, each as (x, y, r) - hashable
    code: str  # the code that generated the result

    def __init__(self, timestep: int, construction, code: str, value: float = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None, observation: str = ""):
        super().__init__(timestep, value, parent_values, parents, id, observation)
        self.construction = _to_tuple_of_tuples(construction)
        self.code = code

    def to_dict(self) -> dict:
        return {
            "type": "CirclePackingState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "exec_time_ms": self.exec_time_ms,
            "construction": to_json_serializable(self.construction) if self.construction is not None else None,
            "code": self.code,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> CirclePackingState:
        state = cls(
            timestep=d["timestep"],
            construction=_to_tuple_of_tuples(d.get("construction")),
            code=d["code"],
            value=d.get("value"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
        )
        state.exec_time_ms = d.get("exec_time_ms")
        return state


class GpuModeState(State):
    """State for gpu mode - holds code."""
    code: str  # the code that generated the result

    def __init__(self, timestep: int, code: str, value: float = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None, observation: str = ""):
        super().__init__(timestep, value, parent_values, parents, id, observation)
        self.code = code

    def to_dict(self) -> dict:
        return {
            "type": "GpuModeState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "exec_time_ms": self.exec_time_ms,
            "code": self.code,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> GpuModeState:
        state = cls(
            timestep=d["timestep"],
            code=d["code"],
            value=d.get("value"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
        )
        state.exec_time_ms = d.get("exec_time_ms")
        return state


class AleBenchState(State):
    """State for ALE Bench - holds code."""
    code: str  # the code that generated the result

    def __init__(self, timestep: int, code: str, value: float = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None):
        super().__init__(timestep, value, parent_values, parents, id)
        self.code = code

    def to_dict(self) -> dict:
        return {
            "type": "AleBenchState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "exec_time_ms": self.exec_time_ms,
            "code": self.code,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> AleBenchState:
        state = cls(
            timestep=d["timestep"],
            code=d["code"],
            value=d.get("value"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
        )
        state.exec_time_ms = d.get("exec_time_ms")
        return state


class ErdosState(State):
    """State for Erdos min overlap problem - holds code and construction (h_values)."""
    code: str
    c5_bound: float
    construction: list[float]

    def __init__(self, timestep: int, code: str, value: float = None, c5_bound: float = None, construction: list[float] = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None, observation: str = ""):
        super().__init__(timestep, value, parent_values, parents, id, observation)
        self.code = code
        self.c5_bound = c5_bound
        self.construction = to_json_serializable(construction) if construction is not None else None

    def to_dict(self) -> dict:
        return {
            "type": "ErdosState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "exec_time_ms": self.exec_time_ms,
            "code": self.code,
            "c5_bound": self.c5_bound,
            "construction": to_json_serializable(self.construction) if self.construction is not None else None,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ErdosState":
        state = cls(
            timestep=d["timestep"],
            code=d["code"],
            value=d.get("value"),
            c5_bound=d.get("c5_bound"),
            construction=d.get("construction") or d.get("h_values"),  # backward compat
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
        )
        state.exec_time_ms = d.get("exec_time_ms")
        return state


class DenoisingState(State):
    code: str
    mse: float
    poisson: float

    def __init__(self, timestep: int, code: str, value: float = None, mse: float = None, poisson: float = None, parent_values: list[float] = None, parents: list[dict] = None, id: str = None, observation: str = ""):
        super().__init__(timestep, value, parent_values, parents, id, observation)
        self.code = code
        self.mse = mse
        self.poisson = poisson

    def to_dict(self) -> dict:
        return {
            "type": "DenoisingState",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "exec_time_ms": self.exec_time_ms,
            "code": self.code,
            "mse": self.mse,
            "poisson": self.poisson,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "DenoisingState":
        state = cls(
            timestep=d["timestep"],
            code=d["code"],
            value=d.get("value"),
            mse=d.get("mse"),
            poisson=d.get("poisson"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
        )
        state.exec_time_ms = d.get("exec_time_ms")
        return state


# Registry for state types
STATE_REGISTRY = {
    "InequalitiesState": InequalitiesState,
    "CirclePackingState": CirclePackingState,
    "GpuModeState": GpuModeState,
    "AleBenchState": AleBenchState,
    "ErdosState": ErdosState,
    "DenoisingState": DenoisingState,
}


def state_from_dict(d: dict | None) -> State | None:
    """Deserialize any state type from dict."""
    if d is None:
        return None
    state_type = d.get("type")
    if state_type not in STATE_REGISTRY:
        raise ValueError(f"Unknown state type: {state_type}")
    return STATE_REGISTRY[state_type].from_dict(d)
