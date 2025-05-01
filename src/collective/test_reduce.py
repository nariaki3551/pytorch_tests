import os
import argparse

import torch
import torch.distributed as dist


def main(args):
    dtype_ints = [torch.int8, torch.int16, torch.int32, torch.int64]
    dtype_floats = [torch.float32, torch.float64]
    reduce_ops = [dist.ReduceOp.AVG, dist.ReduceOp.SUM, dist.ReduceOp.PRODUCT, dist.ReduceOp.MIN, dist.ReduceOp.MAX]

    for dtype in dtype_ints:
        for reduce_op in reduce_ops:
            if reduce_op == dist.ReduceOp.AVG:
                continue
            if dtype == torch.int16 and args.backend == 'nccl':
                continue # NCCL does not support short (int16)
            if dtype == torch.int16 and args.backend == 'gloo':
                continue # GLOO does not support short (int16)
            test_reduce(args, dtype, reduce_op)

    for dtype in dtype_floats:
        for reduce_op in reduce_ops:
            test_reduce(args, dtype, reduce_op)


def test_reduce(args, dtype, reduce_op):
    rank = dist.get_rank()
    size_ = dist.get_world_size()
    count = args.count
    device = args.device

    print(f"Rank {rank}/{size_}: reduce test with count {count} and device {device}")

    tensor = torch.tensor([rank * count], dtype=dtype, device=device)
    dst = 0
    print(f"Rank {rank} reduce input: {tensor}, reduce op: {reduce_op}")
    dist.reduce(tensor, dst, reduce_op)
    print(f"Rank {rank} reduce received: {tensor}, reduce op: {reduce_op}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Allgather example')
    parser.add_argument('--backend', type=str, default='mpi', choices=['mpi', 'ucc', 'nccl', 'gloo'], help='Backend to use')
    parser.add_argument('--count', type=int, default=3, help='Number of elements in each tensor')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to run the test on')
    args = parser.parse_args()

    if args.backend == 'mpi':
        dist.init_process_group(backend='mpi')
    elif args.backend == 'ucc':
        try:
            dist.init_process_group(backend='ucc', init_method='env://')
        except Exception as e:
            print(f"Error initializing UCC backend: {e}")
            exit(0)
    elif args.backend == 'nccl':
        assert args.device == 'cuda', "NCCL backend only supports CUDA device"
        dist.init_process_group(backend='nccl', init_method='env://')
    elif args.backend == 'gloo':
        dist.init_process_group(backend='gloo', init_method='env://')

    if args.device == 'cuda':
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        args.device = f'cuda:{local_rank}'
        torch.cuda.set_device(args.device)

    main(args)

    dist.destroy_process_group()
