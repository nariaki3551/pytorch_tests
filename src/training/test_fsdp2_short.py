import os
import time
import argparse
import dataclasses

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
import torch.optim as optim

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.sequential0 = nn.Linear(512, 1024)
        self.sequential1 = nn.Linear(1024, 512)
        self.last = nn.Linear(512, 10)

    def forward(self, x):
        x = torch.relu(self.sequential0(x))
        x = torch.relu(self.sequential1(x))
        return self.last(x)


def fsdp_training(local_rank: int):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device('cuda', local_rank)

    model = SimpleModel().to(device)
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
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(input_data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"Epoch {epoch}/{num_epochs}: Loss {loss.item()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FSDP Training')
    args = parser.parse_args()

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    local_rank = int(os.getenv("OMPI_COMM_WORLD_RANK", 0))
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="mpi")

    fsdp_training(local_rank)
    
    dist.destroy_process_group()

