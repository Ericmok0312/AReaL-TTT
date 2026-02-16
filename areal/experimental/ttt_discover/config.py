# areal/experimental/ttt_discover/config.py
from dataclasses import dataclass, field
from typing import Optional, Any
from areal.api.cli_args import (
    PPOActorConfig,
    ClusterSpecConfig,
    StatsLoggerConfig,
    SaverConfig,
    InferenceEngineConfig,
    EvaluatorConfig,
    RecoverConfig,
    GenerationHyperparameters,
    MicroBatchSpec,
    PPOCriticConfig,
)


@dataclass
class SamplerConfig:
    """Configuration for PUCTSampler"""
    type: str = field(
        default="puct",
        metadata={"help": "Sampler type: 'puct' or 'random'"}
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Number of parent states to sample per step"}
    )
    # PUCT parameters
    c_puct: float = field(
        default=1.5,
        metadata={"help": "PUCT exploration constant"}
    )
    gamma: float = field(
        default=0.95,
        metadata={"help": "Discount factor for future rewards"}
    )
    max_children: int = field(
        default=100,
        metadata={"help": "Maximum children per state"}
    )
    # State management
    max_states: int = field(
        default=10000,
        metadata={"help": "Maximum number of states to keep in memory"}
    )
    top_k: int = field(
        default=1000,
        metadata={"help": "Keep top-k states after each iteration"}
    )
    # Exploration
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for state sampling"}
    )
    # Checkpointing
    save_freq: int = field(
        default=100,
        metadata={"help": "Save sampler state every N steps"}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save sampler checkpoints"}
    )
    
    # Initial state
    initial_exp_type: str = field(
        default="best_available",
        metadata={"help": "Initial experience type: 'best_available', 'none', 'random', 'random_no_code'"}
    )
    
    # Environment type for initial state creation
    env_type: str = field(
        default="cp",
        metadata={"help": "Environment type: 'cp', 'ac1', 'ac2', 'mla_decode_nvidia', 'trimul', 'erdos', 'denoising', 'ahc039', 'ahc058'"}
    )
    
    # Environment-specific parameters
    # Circle Packing (cp)
    n_item: int = field(
        default=26,
        metadata={"help": "Number of circles for Circle Packing: 26 or 32"}
    )
    
    # GPU Mode (trimul, mla_decode_nvidia)
    gpu_type: str = field(
        default="H100",
        metadata={"help": "GPU type for Modal execution: H100, H200, etc."}
    )
    eval_timeout: int = field(
        default=60,
        metadata={"help": "Timeout for code execution (seconds)"}
    )
    
    # Erdos (erdos)
    n: int = field(
        default=100,
        metadata={"help": "Size parameter for Erdos construction"}
    )
    
    # Inequalities/AC1 (ac1)
    budget_s: int = field(
        default=1000,
        metadata={"help": "Budget parameter for inequalities"}
    )


@dataclass
class TTTDPPOActorConfig(PPOActorConfig):
    """
    Extended PPO config for TTT-Discover with Entropic Objective support.
    Inherits all standard PPO parameters while adding discovery-specific options.
    """
    
    # Basic configuration fields that YAML expects
    seed: int = field(
        default=1,
        metadata={"help": "Random seed for reproducibility"}
    )
    enable_offload: bool = field(
        default=False,
        metadata={"help": "Enable parameter offloading to CPU"}
    )
    max_steps: int = field(
        default=50,
        metadata={"help": "Maximum training steps for TTT-Discover"}
    )
    total_train_epochs: int = field(
        default=10,
        metadata={"help": "Total training epochs (ignored for TTT-Discover)"}
    )
    tokenizer_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to tokenizer"}
    )
    
    # Nested configuration objects
    cluster: ClusterSpecConfig = field(
        default_factory=ClusterSpecConfig,
        metadata={"help": "Cluster configuration"}
    )
    allocation_mode: str = field(
        default="sglang:d8p1t1+d8p1t1",
        metadata={"help": "GPU allocation mode"}
    )
    scheduler: Optional[dict] = field(
        default=None,
        metadata={"help": "Scheduler configuration"}
    )
    rollout: InferenceEngineConfig = field(
        default_factory=InferenceEngineConfig,
        metadata={"help": "Rollout configuration"}
    )
    gconfig: GenerationHyperparameters = field(
        default_factory=GenerationHyperparameters,
        metadata={"help": "Generation configuration"}
    )
    ref: PPOActorConfig | None = field(
        default=None,
        metadata={"help": "Reference model configuration"}
    )
    critic: PPOCriticConfig | None = field(
        default=None,
        metadata={"help": "Critic model configuration"}
    )
    sglang: dict = field(
        default_factory=dict,
        metadata={"help": "SGLang configuration"}
    )
    vllm: dict = field(
        default_factory=dict,
        metadata={"help": "vLLM configuration"}
    )
    sampler: SamplerConfig = field(
        default_factory=SamplerConfig,
        metadata={"help": "Configuration for PUCTSampler"}
    )
    train_dataset: dict = field(
        default_factory=dict,
        metadata={"help": "Training dataset configuration"}
    )
    valid_dataset: dict = field(
        default_factory=dict,
        metadata={"help": "Validation dataset configuration"}
    )
    saver: SaverConfig = field(
        default_factory=SaverConfig,
        metadata={"help": "Model saver configuration"}
    )
    recover: RecoverConfig = field(
        default_factory=RecoverConfig,
        metadata={"help": "Recovery configuration"}
    )
    evaluator: EvaluatorConfig = field(
        default_factory=EvaluatorConfig,
        metadata={"help": "Evaluator configuration"}
    )
    stats_logger: StatsLoggerConfig = field(
        default_factory=StatsLoggerConfig,
        metadata={"help": "Stats logger configuration"}
    )
    perf_tracer: dict = field(
        default_factory=dict,
        metadata={"help": "Performance tracer configuration"}
    )
    
    # Advantage estimator selection
    adv_estimator: str = field(
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
    
    # Actor field for local launcher compatibility
    # This field exists to satisfy the launcher validation, but the actual
    # actor config is this object itself (flattened structure)
    actor: Any = field(
        default=None,
        metadata={"help": "Actor configuration (for local launcher compatibility)"}
    )
    
    # Enable thinking mode for Qwen3 models
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Enable thinking mode for Qwen3 models (adds enable_thinking=True to chat_template)"}
    )
    
    def __post_init__(self):
        """Validate configuration consistency and convert nested dicts to objects"""
        # Convert mb_spec from dict to MicroBatchSpec if needed
        if isinstance(self.mb_spec, dict):
            self.mb_spec = MicroBatchSpec(**self.mb_spec)
        
        # Call parent validation
        super().__post_init__()
        
        # Validate adv_estimator value
        valid_estimators = ["gae", "mean_baseline", "entropic", "entropic_adaptive_beta"]
        if self.adv_estimator not in valid_estimators:
            raise ValueError(f"adv_estimator must be one of {valid_estimators}, got {self.adv_estimator}")
        
        if self.adv_estimator in ["entropic", "entropic_adaptive_beta"]:
            if self.adv_estimator_beta <= 0 and self.adv_estimator == "entropic":
                raise ValueError(f"adv_estimator_beta must be positive, got {self.adv_estimator_beta}")
            
            # Optional: Warn or auto-adjust if using entropic with default reward scaling
            if not hasattr(self, 'reward_scaling') or self.reward_scaling == 1.0:
                pass