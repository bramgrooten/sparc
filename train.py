import torch
import wandb
from utils import misc_utils
from utils.args import get_cli_args
from algorithms import sparc, sac, qr_sac, oracle, history_input, rma_phase1, rma_phase2


def main():
    args = get_cli_args()
    run_name = misc_utils.set_run_name(args)
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=run_name,
        config=vars(args),
        save_code=True,
    )
    misc_utils.set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    algorithm = args.alg.lower()
    if algorithm == 'sac':
        sac.main(args, run_name, device)
    elif algorithm == 'qr_sac':
        qr_sac.main(args, run_name, device)
    elif algorithm == 'sparc':
        sparc.main(args, run_name, device)
    elif algorithm == 'oracle':
        oracle.main(args, run_name, device)
    elif algorithm == 'history_input':
        history_input.main(args, run_name, device)
    elif algorithm == 'rma_phase1':
        rma_phase1.main(args, run_name, device)
    elif algorithm == 'rma_phase2':
        rma_phase2.main(args, run_name, device)
    else:
        raise NotImplementedError(f"Algorithm {algorithm} is not supported.")


if __name__ == '__main__':
    main()
