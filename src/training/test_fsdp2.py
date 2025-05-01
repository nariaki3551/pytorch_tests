import os
import time
import argparse
import dataclasses

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
import torch.optim as optim

class LinearWrapper(nn.Linear):
    ...
class LinearWrapper0(LinearWrapper):
    ...
class LinearWrapper1(LinearWrapper):
    ...
class LinearWrapper2(LinearWrapper):
    ...


class SimpleModel(nn.Module):
    def __init__(self, scale: int, rank: int):
        super(SimpleModel, self).__init__()
        self.sequential0 = LinearWrapper0(512, 1024 * scale)
        self.sequential1 = LinearWrapper1(1024 * scale, 1024 * scale)
        self.sequential2 = LinearWrapper2(1024 * scale, 512 * scale)
        self.last = nn.Linear(512 * scale, 10)

        # Register forward/backward hooks for all sequential layers
        def forward_hook(module, input, output):
            print(f"[rank{rank}] Forward pass - {module.__class__.__name__}")
            return output

        def backward_hook(module, grad_input, grad_output):
            print(f"[rank{rank}] Backward pass - {module.__class__.__name__}")
            return grad_input

        self.sequential0.register_forward_hook(forward_hook)
        self.sequential1.register_forward_hook(forward_hook)
        self.sequential2.register_forward_hook(forward_hook)
        self.last.register_forward_hook(forward_hook)

        self.sequential0.register_full_backward_hook(backward_hook)
        self.sequential1.register_full_backward_hook(backward_hook)
        self.sequential2.register_full_backward_hook(backward_hook)
        self.last.register_full_backward_hook(backward_hook)

    def forward(self, x):
        x = torch.relu(self.sequential0(x))
        x = torch.relu(self.sequential1(x))
        x = torch.relu(self.sequential2(x))
        return self.last(x)


@dataclasses.dataclass
class Config:
    model_scale: int = 1
    num_epochs: int = 5

def fsdp_training(config: Config, local_rank: int):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device('cuda', local_rank)
    print(f"Rank {rank}/{world_size}: Running FSDP with device {device} and model_scale {config.model_scale} and num_epochs {config.num_epochs}")

    model = SimpleModel(config.model_scale, rank).to(device)
    # for module in model.modules():
    #     fully_shard(module)
    fully_shard(
        model,
    )
    model.to_empty(device="cuda")

    gpu_name = torch.cuda.get_device_name(local_rank)
    print(f"Rank {rank}/{world_size}: FSDP model created; #params: {sum(p.numel() for p in model.parameters())}; GPU: {gpu_name}")

    if rank == 0:
        print(model)

    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    input_data = torch.randn(2, 512).to(device)
    target = torch.randint(0, 10, (2,)).to(device)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSDP Training')
    parser.add_argument(
        '--sharding-strategy', type=str, choices=['FULL_SHARD', 'HYBRID_SHARD', 'SHARD_GRAD_OP'], default='FULL_SHARD',
        help='Sharding strategy to use for training'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Debug mode to use for training'
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
    args = parser.parse_args()

    config = Config(
        # sharding_strategy=get_sharding_strategy(args.sharding_strategy),
        model_scale=args.model_scale,
        num_epochs=args.num_epochs,
    )

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if args.backend == 'mpi':
        local_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
    torch.cuda.set_device(local_rank)

    print("local_rank", local_rank, "backend", args.backend)
    if args.backend == 'mpi':
        dist.init_process_group(backend="mpi")
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


    if args.debug:
        import time; time.sleep(20)

    fsdp_training(config, local_rank)
    
    dist.destroy_process_group()

