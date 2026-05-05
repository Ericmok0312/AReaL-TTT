# areal/experimental/ttt_discover/config.py
from dataclasses import dataclass, field

from areal.api.cli_args import (
    PPOActorConfig,
    PPOConfig,
)


@dataclass
class SamplerConfig:
    """Configuration for PUCTSampler"""

    type: str = field(
        default="puct",
        metadata={"help": "Sampler type: 'puct' or 'random'"},
    )
    batch_size: int = field(
        default=8,
        metadata={"help": "Number of parent states to sample per step"},
    )
    # PUCT parameters
    c_puct: float = field(
        default=1.5,
        metadata={"help": "PUCT exploration constant"},
    )
    gamma: float = field(
        default=0.95,
        metadata={"help": "Discount factor for future rewards"},
    )
    max_children: int = field(
        default=100,
        metadata={"help": "Maximum children per state"},
    )
    # State management
    max_states: int = field(
        default=10000,
        metadata={"help": "Maximum number of states to keep in memory"},
    )
    top_k: int = field(
        default=1000,
        metadata={"help": "Keep top-k states after each iteration"},
    )
    # Exploration
    temperature: float = field(
        default=1.0,
        metadata={"help": "Temperature for state sampling"},
    )
    # Checkpointing
    save_freq: int = field(
        default=100,
        metadata={"help": "Save sampler state every N steps"},
    )
    checkpoint_dir: str | None = field(
        default=None,
        metadata={"help": "Directory to save sampler checkpoints"},
    )

    # Initial state
    initial_exp_type: str = field(
        default="best_available",
        metadata={"help": "Initial experience type: 'best_available', 'none', 'random', 'random_no_code'"},
    )

    # Environment type for initial state creation
    env_type: str = field(
        default="cp",
        metadata={"help": "Environment type: 'cp', 'ac1', 'ac2', 'mla_decode_nvidia', 'trimul', 'erdos', 'denoising', 'ahc039', 'ahc058'"},
    )

    # Environment-specific parameters
    # Circle Packing (cp)
    n_item: int = field(
        default=26,
        metadata={"help": "Number of circles for Circle Packing: 26 or 32"},
    )

    # GPU Mode (trimul, mla_decode_nvidia)
    gpu_type: str = field(
        default="H100",
        metadata={"help": "GPU type for Modal execution: H100, H200, etc."},
    )
    eval_timeout: int = field(
        default=60,
        metadata={"help": "Timeout for code execution (seconds)"},
    )

    # Erdos (erdos)
    n: int = field(
        default=100,
        metadata={"help": "Size parameter for Erdos construction"},
    )

    # Inequalities/AC1 (ac1)
    budget_s: int = field(
        default=1000,
        metadata={"help": "Budget parameter for inequalities"},
    )
    num_cpus: int = field(
        default=2,
        metadata={"help": "Number of CPUs per task for code execution"},
    )

    # Lazy PUCT Sampling configuration
    lazy_puct_sampling: bool = field(
        default=True,
        metadata={"help": "Enable lazy PUCT sampling: defer sampling until VLLM has capacity"},
    )
    vllm_concurrency: int | None = field(
        default=None,
        metadata={"help": "Per-rank VLLM concurrency limit for lazy sampling. "
                         "If None, auto-computed as batch_size * group_size. "
                         "Each rank (VLLM instance) can have this many concurrent generations. "
                         "Should match or be less than vllm.max_num_seqs."},
    )
    execution_concurrency: int = field(
        default=64,
        metadata={"help": "Per-rank solution execution concurrency limit. "
                         "Each rank can have this many concurrent code executions. "
                         "Should match AsyncRewardWrapper max_workers (default 64)."},
    )
    max_puct_version_history: int = field(
        default=5,
        metadata={"help": "Maximum number of PUCT version snapshots to keep in memory"},
    )


@dataclass
class TTTDPPOActorConfig(PPOActorConfig):
    """
    TTT-Discover-specific extensions to PPOActorConfig for entropic advantage estimation.

    This config only contains actor-level fields. Use TTTDPPOConfig for the top-level
    experiment configuration.
    """

    # Advantage estimator selection
    adv_estimator: str = field(
        default="gae",
        metadata={
            "help": "Advantage estimation method. "
                    "'gae': standard GAE, "
                    "'entropic': TTT-Discover with fixed beta, "
                    "'entropic_adaptive_beta': TTT-Discover with adaptive beta, "
                    "'mean_baseline': simple mean subtraction"
        },
    )

    # Entropic parameters
    adv_estimator_beta: float = field(
        default=1.0,
        metadata={"help": "Beta (temperature) for entropic advantage. Higher = more exploration"},
    )

    # Adaptive beta parameters
    adv_estimator_target_kl: float = field(
        default=0.693,  # log(2)
        metadata={"help": "Target KL divergence for adaptive beta (default: log(2))"},
    )

    adv_estimator_beta_max: float = field(
        default=1e6,
        metadata={"help": "Maximum beta value for adaptive search"},
    )

    adv_estimator_beta_iters: int = field(
        default=60,
        metadata={"help": "Binary search iterations for adaptive beta"},
    )

    # Grouping strategy
    group_size: int | None = field(
        default=None,
        metadata={"help": "Number of samples per group for advantage calculation. "
                         "If None, infer from batch structure or treat whole batch as one group"},
    )

    def __post_init__(self):
        super().__post_init__()
        valid_estimators = ["gae", "mean_baseline", "entropic", "entropic_adaptive_beta"]
        if self.adv_estimator not in valid_estimators:
            raise ValueError(
                f"adv_estimator must be one of {valid_estimators}, got {self.adv_estimator}"
            )
        if self.adv_estimator == "entropic" and self.adv_estimator_beta <= 0:
            raise ValueError(
                f"adv_estimator_beta must be positive for entropic, got {self.adv_estimator_beta}"
            )


@dataclass
class TTTDPPOConfig(PPOConfig):
    """
    TTT-Discover top-level experiment config following AReaL PPOConfig conventions.

    Inherits all standard PPO experiment fields (rollout, ref, critic, gconfig, etc.)
    and adds TTT-Discover-specific experiment-level fields.
    """

    actor: TTTDPPOActorConfig = field(default_factory=TTTDPPOActorConfig)

    # TTT-Discover specific experiment fields
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    max_steps: int = field(
        default=50,
        metadata={"help": "Maximum training steps for TTT-Discover"},
    )
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Enable thinking mode for Qwen3 models (adds enable_thinking=True to chat_template)"},
    )
    max_prompt_thinking_tokens: int = field(
        default=26000,
        metadata={"help": "Maximum tokens for prompt + thinking phase. Paper: 26000 to leave room for final response. "
                         "If model exceeds this without producing valid code, teacher forcing is applied."},
    )
    save_steps: list[int] = field(
        default_factory=lambda: [0, 9, 24, 49],
        metadata={"help": "Steps to save training history snapshots for visualization"},
    )
    skip_lora_check: bool = field(
        default=False,
        metadata={"help": "Skip LoRA adapter existence check at training start. "
                         "Use only if you are certain the adapter exists at the configured path. "
                         "Note: The LoRA adapter must be created BEFORE training using prepare_lora_init.py "
                         "because vLLM loads it at startup before the training script runs."},
    )
    reset_puct_stats_on_resume: bool = field(
        default=True,
        metadata={"help": "Reset PUCT statistics (_T, _n, _m) when resuming from checkpoint. "
                         "If True, exploration stats start from 0 (recommended for new experiments). "
                         "If False, stats continue from saved values (for continuing same experiment). "
                         "Default is True to avoid _T inflation across different training runs."},
    )
    use_scheme_1: bool = field(
        default=False,
        metadata={"help": "Enable Scheme 1 (sync-like) mode: wait for complete batch before PUCT update. "
                         "This ensures complete batch updates like sync mode but allows async rollout. "
                         "Default is False (Scheme 2: streaming/async mode)"},
    )

    # Compatibility flag (for type checking)
    is_tttd_config: bool = field(
        default=True,
        repr=False,
        metadata={"help": "Internal flag to identify TTT-D config"},
    )

    # Override train_dataset type for TTT-Discover compatibility
    train_dataset: dict = field(default_factory=dict)
    valid_dataset: dict | None = field(default=None)

    def __post_init__(self):
        # Convert actor from dict if needed (e.g., manual construction or CLI overrides)
        if isinstance(self.actor, dict):
            self.actor = TTTDPPOActorConfig(**self.actor)
        # Convert ref from dict if needed
        if isinstance(self.ref, dict):
            self.ref = PPOActorConfig(**self.ref)
        # Convert sampler from dict if needed
        if isinstance(self.sampler, dict):
            self.sampler = SamplerConfig(**self.sampler)
        # PPOConfig.__post_init__ handles eval_gconfig and BaseExperimentConfig validation
        super().__post_init__()


@dataclass
class TTTDDistillConfig(TTTDPPOConfig):
    """
    Extended config for TTT-Discover distillation and evaluation.

    Used by both training (train_tttd_distill.py) and evaluation
    (eval_tttd_distill.py) scripts.
    """

    # Teacher model settings
    teacher_path: str = field(
        default="",
        metadata={"help": "Path to teacher model checkpoint (HF format or DCP)"},
    )
    teacher_sampler_checkpoint: str = field(
        default="",
        metadata={"help": "Path to teacher PUCTSampler checkpoint directory"},
    )
    teacher_weight_format: str = field(
        default="hf",
        metadata={"help": "Teacher checkpoint format: 'hf' or 'dcp'", "choices": ["hf", "dcp"]},
    )

    # Distillation settings
    distill_steps: int = field(
        default=3,
        metadata={"help": "Number of distillation steps (no verification)"},
    )
    kl_reward_scale: float = field(
        default=1.0,
        metadata={"help": "Scale factor for KL-based reward (legacy, kept for compatibility)"},
    )
    kl_estimator_type: str = field(
        default="k1",
        metadata={"help": "KL estimator: k1, k2, or k3", "choices": ["k1", "k2", "k3"]},
    )
    total_rollouts_per_step: int = field(
        default=512,
        metadata={"help": "Total number of rollouts per step across all ranks"},
    )
    run_eval_step: bool = field(
        default=True,
        metadata={"help": "Run evaluation with real verification after distillation"},
    )
    teacher_lora_path: str = field(
        default="",
        metadata={"help": "Path to teacher LoRA adapter for eval (defaults to teacher_path if empty)"},
    )
    student_lora_path: str = field(
        default="",
        metadata={"help": "Path to student LoRA adapter for eval"},
    )



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
    from .envs import CirclePackingEnv, DenoisingEnv, ErdosEnv, InequalitiesEnv

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
    elif env_type == 'erdos':
        return ErdosEnv(
            n=getattr(config.sampler, 'n', 200),
            budget_s=getattr(config.sampler, 'budget_s', 1000),
            eval_timeout=eval_timeout,
            log_dir=config.saver.fileroot,
            num_cpus=getattr(config.sampler, 'num_cpus', 2),
        )
    elif env_type == 'denoising':
        return DenoisingEnv(
            eval_timeout=eval_timeout,
            log_dir=config.saver.fileroot,
            num_cpus=getattr(config.sampler, 'num_cpus', 2),
        )
    else:
        raise ValueError(f"Unknown env_type: {env_type}")
