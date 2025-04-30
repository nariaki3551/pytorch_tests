import os
import argparse

import torch
import torch.distributed as dist

def main(args):
    rank = dist.get_rank()
    size_ = dist.get_world_size()
    count = args.count
    device = args.device

    print(f"Rank {rank}/{size_}: allgather test with count {count} and device {device}")

    source = [rank for _ in range(count)]
    # input_tensor = torch.ones(count, dtype=torch.int32, device=device) * rank
    input_tensor = torch.tensor(source, dtype=torch.int32, device=device)
    output_tensor = torch.zeros(count * size_, dtype=torch.int32, device=device)

    print(f"Rank {rank} allgather input: {input_tensor}")
    dist.all_gather_into_tensor(output_tensor, input_tensor)
    print(f"Rank {rank} allgather received: {output_tensor}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Allgather example')
    parser.add_argument('--backend', type=str, default='mpi', choices=['mpi', 'ucc', 'nccl', 'gloo'], help='Backend to use')
    parser.add_argument('--count', type=int, default=3, help='Number of elements in each tensor')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to run the test on')
    args = parser.parse_args()

    if args.backend == 'mpi':
        dist.init_process_group(backend='mpi')
    elif args.backend == 'ucc':
        dist.init_process_group(backend='ucc', init_method='env://')
    elif args.backend == 'nccl':
        assert args.device == 'cuda', "NCCL backend only supports CUDA device"
        dist.init_process_group(backend='nccl', init_method='env://')
    elif args.backend == 'gloo':
        assert args.device == 'cpu', "Gloo backend only supports CPU device"
        dist.init_process_group(backend='gloo', init_method='env://')

    if args.device == 'cuda':
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        args.device = f'cuda:{local_rank}'
        torch.cuda.set_device(args.device)

    main(args)

    dist.destroy_process_group()
