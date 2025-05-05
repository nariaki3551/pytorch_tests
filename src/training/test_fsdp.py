import os
import time
import argparse
import dataclasses
import functools

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
import torch.optim as optim


class SimpleModel(nn.Module):
    def __init__(self, scale: int, rank: int):
        super(SimpleModel, self).__init__()
        self.sequential0 = nn.Linear(512, 1024 * scale)
        self.sequential1 = nn.Linear(1024 * scale, 1024 * scale)
        self.sequential2 = nn.Linear(1024 * scale, 1024 * scale)
        self.sequential3 = nn.Linear(1024 * scale, 512 * scale)
        self.sequential4 = nn.Linear(512 * scale, 10)
      
        def register_hooks(module, module_name):
            def forward_hook(module, input, output):
                print(f"[rank{rank}] Forward pass - {module_name}")
                return output
            def backward_hook(module, grad_input, grad_output):
                print(f"[rank{rank}] Backward pass - {module_name}")
                return grad_input
            module.register_forward_hook(forward_hook)
            module.register_full_backward_hook(backward_hook)
        
        register_hooks(self.sequential0, "Linear0")
        register_hooks(self.sequential1, "Linear1")
        register_hooks(self.sequential2, "Linear2")
        register_hooks(self.sequential3, "Linear3")
        register_hooks(self.sequential4, "Linear4")


    def forward(self, x):
        x = torch.relu(self.sequential0(x))
        x = torch.relu(self.sequential1(x))
        x = torch.relu(self.sequential2(x))
        x = torch.relu(self.sequential3(x))
        return self.sequential4(x)


def get_sharding_strategy(sharding_strategy: str) -> ShardingStrategy:
    if sharding_strategy == 'FULL_SHARD':
        return ShardingStrategy.FULL_SHARD
    elif sharding_strategy == 'HYBRID_SHARD':
        return ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == 'SHARD_GRAD_OP':
        return ShardingStrategy.SHARD_GRAD_OP
    else:
        raise ValueError(f"Invalid sharding strategy: {sharding_strategy}")

@dataclasses.dataclass
class Config:
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    model_scale: int = 1
    num_epochs: int = 5
    device: str = 'cuda'
    profile: bool = False

def fsdp_training(config: Config, local_rank: int):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if config.device == 'cuda':
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    print(f"Rank {rank}/{world_size}: Running FSDP with device {device} and model_scale {config.model_scale} and num_epochs {config.num_epochs}")

    def my_auto_wrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int):
        return isinstance(module, (SimpleModel, nn.Linear))

    model = SimpleModel(config.model_scale, rank).to(device)
    model = FullyShardedDataParallel(
        model,
        device_id=device,
        sharding_strategy=config.sharding_strategy,
        auto_wrap_policy=my_auto_wrap_policy,
        forward_prefetch=True,
        limit_all_gathers=False,
    ).to(device)

    device_name = torch.cuda.get_device_name(local_rank) if config.device == 'cuda' else 'CPU'
    print(f"Rank {rank}/{world_size}: FSDP model created; #params: {sum(p.numel() for p in model.parameters())}; device: {device_name}")

    if rank == 0:
        print(model)

    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    input_data = torch.randn(2, 512).to(device)
    target = torch.randint(0, 10, (2,)).to(device)

    # profiler setting
    if config.profile:
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            with_stack=True
        )
        prof.start()
    else:
        prof = None

    print(f"Rank {rank}/{world_size}: Start training")
    model.train()
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        output = model(input_data)
        loss = nn.CrossEntropyLoss()(output, target)

        start = time.time()
        loss.backward()
        end = time.time()
        optimizer.step()

        if rank == 0:
            print(f"EXP: Rank {rank}/{world_size} Epoch {epoch} backward: {end - start:.3f} seconds")
            print(f"EXP: Rank {rank}/{world_size} Epoch {epoch} loss: {loss.item():.6f}")

    if prof is not None:
        prof.stop()
        if rank == 0:
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSDP Training')
    parser.add_argument(
        '--sharding-strategy', type=str, choices=['FULL_SHARD', 'HYBRID_SHARD', 'SHARD_GRAD_OP'], default='FULL_SHARD',
        help='Sharding strategy to use for training'
    )
    parser.add_argument(
        '--backend', type=str, choices=['mpi', 'nccl', 'ucc', 'gloo'], default='mpi',
        help='Backend to use for training'
    )
    parser.add_argument(
        "--model_scale", type=int, default=1,
        help='Model scale to use for training'
    )
    parser.add_argument(
        "--num_epochs", type=int, default=5,
        help='Number of epochs to use for training'
    )
    parser.add_argument(
        "--device", type=str, default='cuda', choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        "--profile", action='store_true',
        help='Whether to profile the training'
    )
    args = parser.parse_args()

    config = Config(
        sharding_strategy=get_sharding_strategy(args.sharding_strategy),
        model_scale=args.model_scale,
        num_epochs=args.num_epochs,
        device=args.device,
        profile=args.profile,
    )

    local_rank = int(os.getenv("LOCAL_RANK", 0))

    if args.backend == 'mpi':
        dist.init_process_group(backend=args.backend)
    elif args.backend == 'nccl':
        dist.init_process_group(backend="nccl", init_method="env://")
    elif args.backend == 'ucc':
        try:
            dist.init_process_group(backend="ucc", init_method="env://")
        except Exception as e:
            print(f"Error initializing UCC process group: {e}")
            exit(0)
    elif args.backend == 'gloo':
        dist.init_process_group(backend="gloo", init_method="env://")
    print(f"[rank{dist.get_rank()}] backend", args.backend)

    fsdp_training(config, local_rank)
    
    dist.destroy_process_group()

