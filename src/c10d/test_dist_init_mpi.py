import torch
import torch.distributed as dist

def main():

    dist.init_process_group(backend='mpi')
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
