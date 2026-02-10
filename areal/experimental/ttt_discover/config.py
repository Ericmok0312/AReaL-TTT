# areal/experimental/ttt_discover/config.py
from dataclasses import dataclass, field
from typing import Optional, Literal
from areal.api.cli_args import PPOActorConfig


@dataclass
class TTTDPPOActorConfig(PPOActorConfig):
    """
    Extended PPO config for TTT-Discover with Entropic Objective support.
    Inherits all standard PPO parameters while adding discovery-specific options.
    """
    
    # Advantage estimator selection
    adv_estimator: Literal["gae", "mean_baseline", "entropic", "entropic_adaptive_beta"] = field(
        default="gae",
        metadata={
            "help": "Advantage estimation method. "
                    "'gae': standard GAE, "
                    "'entropic': TTT-Discover with fixed beta, "
                    "'entropic_adaptive_beta': TTT-Discover with adaptive beta, "
                    "'mean_baseline': simple mean subtraction"
        }
    )
    
    # Entropic parameters
    adv_estimator_beta: float = field(
        default=1.0,
        metadata={"help": "Beta (temperature) for entropic advantage. Higher = more exploration"}
    )
    
    # Adaptive beta parameters
    adv_estimator_target_kl: float = field(
        default=0.693,  # log(2)
        metadata={"help": "Target KL divergence for adaptive beta (default: log(2))"}
    )
    
    adv_estimator_beta_max: float = field(
        default=1e6,
        metadata={"help": "Maximum beta value for adaptive search"}
    )
    
    adv_estimator_beta_iters: int = field(
        default=60,
        metadata={"help": "Binary search iterations for adaptive beta"}
    )
    
    # Grouping strategy
    group_size: Optional[int] = field(
        default=None,
        metadata={"help": "Number of samples per group for advantage calculation. "
                         "If None, infer from batch structure or treat whole batch as one group"}
    )
    
    # Compatibility flag (for type checking)
    is_tttd_config: bool = field(
        default=True,
        repr=False,
        metadata={"help": "Internal flag to identify TTT-D config"}
    )
    
    def __post_init__(self):
        """Validate configuration consistency"""
        super().__post_init__()  # Call parent validation
        
        if self.adv_estimator in ["entropic", "entropic_adaptive_beta"]:
            if self.adv_estimator_beta <= 0 and self.adv_estimator == "entropic":
                raise ValueError(f"adv_estimator_beta must be positive, got {self.adv_estimator_beta}")
            
            # Optional: Warn or auto-adjust if using entropic with default reward scaling
            if not hasattr(self, 'reward_scaling') or self.reward_scaling == 1.0:
                pass