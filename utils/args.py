import tyro
from dataclasses import dataclass


@dataclass
class Args:
    alg: str = "sac"
    """the algorithm to use"""
    env: str = "WindHopper-v5"
    """the environment id of the task. Options:
    originals:  Hopper-v5, HalfCheetah-v5, Walker2d-v5, Ant-v5
    wind:       WindHopper-v5, WindHalfCheetah-v5, WindWalker2d-v5, WindAnt-v5
    """
    wind: bool = True
    """if True, wind will be enabled in the training environments (if using Wind envs). Otherwise wind=0 in training."""
    wind_x_z: tuple = None
    """wind speed intervals in x and z directions, for non-default Wind settings. Example: --wind_x_z -2.5 2.5 -5 5"""
    total_timesteps: int = 3_000_000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = None
    """the replay memory buffer size (by default: total_timesteps * num_envs)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient"""
    batch_size: int = 32
    """the batch size of sample from the replay buffer"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.01
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    eval_freq: int = 10_000
    """how often to evaluate, in terms of environment steps"""
    grad_clip_critic: float = 10.0
    """gradient clipping for the critic networks"""

    # QR-SAC
    num_quantiles: int = 32
    """Number of quantile outputs per critic. (QR‑SAC)"""
    quantile_kappa: float = 1.0
    """Huber threshold for quantile regression loss. (QR-SAC)"""
    top_quantiles_to_drop: int = 0
    """Number of highest quantiles to drop from each critic’s target (0 = keep all). (QR-SAC)"""

    # SPARC
    history_length: int = 50
    """the length of the history to use in SPARC"""
    adapter_lr: float = 3e-4
    """the learning rate of the history adapter optimizer (for SPARC)"""
    sparc_expert_alg: str = "sac"
    """the algorithm to use for the SPARC expert (i.e., sac or qr_sac)"""
    rollout_policy: str = "adapter"
    """the policy to use for the rollout in SPARC (i.e., adapter or expert)"""

    # RMA
    expert_model_path: str = "set/your/path/to/trained_expert_model.pth"
    """the path to the expert model to load for RMA phase 2"""

    # Miscellaneous
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    wandb_mode: str = "online"
    """online, offline, or disabled"""
    wandb_project_name: str = "sparc"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    console_log: str = ""
    """the server's file path to log console outputs"""
    log_freq: int = 1000
    """how often to log the training progress, in terms of environment steps"""


def get_cli_args():
    args = tyro.cli(Args)
    if args.wind_x_z is not None and len(args.wind_x_z) != 4:
        raise ValueError("wind_x_z must be a tuple of 4 values: (x_min, x_max, z_min, z_max)")
    if args.buffer_size is None:
        args.buffer_size = args.total_timesteps * args.num_envs
    return args
