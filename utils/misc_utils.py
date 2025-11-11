import random
import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def set_run_name(args):
    """
    Set the name of the wandb run.
    """
    if 'Wind' in args.env:
        if args.wind_x_z:
            wind = f"-x{args.wind_x_z[0]}-x{args.wind_x_z[1]}-z{args.wind_x_z[2]}-z{args.wind_x_z[3]}"
        else:
            wind = ""
        env = f"{args.env}{wind}"
    else:
        env = args.env
    if args.alg == "sparc":
        expert = f"-{args.sparc_expert_alg}"
    else:
        expert = ""

    return f"{env}_{args.alg}{expert}_{args.seed}"


def copy_modules(src: torch.nn.Module, dst: torch.nn.Module, names: list[str]):
    """
    Copy parameters (in-place) for a list of sub-modules given by attribute names.
    E.g. names = ["fc1", "fc2", "fc3", "fc_mean", "fc_logstd"].
    """
    for n in names:
        getattr(dst, n).load_state_dict(getattr(src, n).state_dict())
