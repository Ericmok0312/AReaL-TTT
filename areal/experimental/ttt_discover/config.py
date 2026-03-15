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
    num_cpus: int = field(
        default=2,
        metadata={"help": "Number of CPUs per task for code execution"}
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
    eval_gconfig: GenerationHyperparameters | None = field(
        default=None,
        metadata={"help": "Generation hyperparameters for evaluation. If None, use gconfig."}
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
    valid_dataset: Optional[dict] = field(
        default=None,
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
    
    # Actor field - contains the training engine configuration
    # This is used by TTTDPPOTrainer to create the actor engine
    actor: PPOActorConfig = field(
        default_factory=PPOActorConfig,
        metadata={"help": "Actor training engine configuration"}
    )
    
    # Enable thinking mode for Qwen3 models
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Enable thinking mode for Qwen3 models (adds enable_thinking=True to chat_template)"}
    )
    
    # Training history recording
    save_steps: list[int] = field(
        default_factory=lambda: [0, 9, 24, 49],
        metadata={"help": "Steps to save training history snapshots for visualization"}
    )
    
    # LoRA check control
    skip_lora_check: bool = field(
        default=False,
        metadata={"help": "Skip LoRA adapter existence check at training start. "
                         "Use only if you are certain the adapter exists at the configured path. "
                         "Note: The LoRA adapter must be created BEFORE training using prepare_lora_init.py "
                         "because vLLM loads it at startup before the training script runs."}
    )
    
    # PUCT stats reset on resume
    reset_puct_stats_on_resume: bool = field(
        default=True,
        metadata={"help": "Reset PUCT statistics (_T, _n, _m) when resuming from checkpoint. "
                         "If True, exploration stats start from 0 (recommended for new experiments). "
                         "If False, stats continue from saved values (for continuing same experiment). "
                         "Default is True to avoid _T inflation across different training runs."}
    )
    
    # Teacher forcing for thinking tokens (paper: limit prompt + thinking to 26000)
    max_prompt_thinking_tokens: int = field(
        default=26000,
        metadata={"help": "Maximum tokens for prompt + thinking phase. Paper: 26000 to leave room for final response. "
                         "If model exceeds this without producing valid code, teacher forcing is applied."}
    )
    
    # Dynamic batch size for async training
    dynamic_bs: bool = field(
        default=False,
        metadata={"help": "Enable dynamic batch sizing for async training (skip slow rollouts)"}
    )
    
    def __post_init__(self):
        """Validate configuration consistency and convert nested dicts to objects"""
        # Convert actor from dict to PPOActorConfig if needed
        if isinstance(self.actor, dict):
            self.actor = PPOActorConfig(**self.actor)
        
        # Convert ref from dict to PPOActorConfig if needed
        if isinstance(self.ref, dict):
            self.ref = PPOActorConfig(**self.ref)
        
        # Convert mb_spec from dict to MicroBatchSpec if needed
        if isinstance(self.mb_spec, dict):
            self.mb_spec = MicroBatchSpec(**self.mb_spec)
        
        # Set default eval_gconfig if not provided
        if self.eval_gconfig is None:
            self.eval_gconfig = self.gconfig.new()
        
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


# Import envs here to avoid circular imports
def create_env_from_config(config):
    """
    Create environment based on config's sampler.env_type.
    
    Supports:
    - 'cp': Circle Packing
    - 'ac1', 'ac2': Inequalities (AC1/AC2)
    
    Args:
        config: Config object with sampler and saver attributes
        
    Returns:
        Environment instance
    """
    # Import here to avoid circular imports
    from .envs import CirclePackingEnv, InequalitiesEnv
    
    env_type = getattr(config.sampler, 'env_type', 'ac1')
    eval_timeout = getattr(config.sampler, 'eval_timeout', 600)
    
    if env_type == 'cp':
        return CirclePackingEnv(
            n_item=getattr(config.sampler, 'n_item', 26),
            eval_timeout=eval_timeout,
            log_dir=config.saver.fileroot,
        )
    elif env_type in ('ac1', 'ac2'):
        return InequalitiesEnv(
            problem_type=env_type,
            budget_s=getattr(config.sampler, 'budget_s', 1000),
            eval_timeout=eval_timeout,
            log_dir=config.saver.fileroot,
            num_cpus=getattr(config.sampler, 'num_cpus', 2),
        )
    else:
        raise ValueError(f"Unknown env_type: {env_type}")