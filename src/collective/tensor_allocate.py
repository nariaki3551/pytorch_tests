import os
import argparse

import torch

def main(args):
    print(f"device = {args.device}")

    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_stack=True,
        with_flops=False,
    )

    prof.start()
    torch.zeros(10, device=args.device)
    prof.stop()

    prof.export_chrome_trace(f'trace.json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Allgather example')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run the test on')
    parser.add_argument('--device_id', type=int, default=None, help='Device Id to run the test on')
    args = parser.parse_args()

    if args.device == 'cuda':
        assert args.device_id is not None
        device_id = args.device_id
        args.device = f'cuda:{device_id}'
        torch.cuda.set_device(args.device)

    main(args)
