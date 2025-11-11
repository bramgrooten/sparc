from algorithms import rma_phase1


def main(args, run_name, device):
    # Run the RMA Phase 1 only. The Expert Policy has access to the ground-truth context, even at eval.
    rma_phase1.main(args, run_name, device)
