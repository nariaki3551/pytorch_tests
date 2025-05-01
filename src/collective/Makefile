PYTORC_DEBUG ?= 0
ifeq ($(PYTORC_DEBUG), 1)
	DEBUG_FLAGS := TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL
endif

run_all: run_allgather_all run_reduce_all
run_allgather_all: run_allgather_mpi run_allgather_ucc run_allgather_nccl run_allgather_gloo
run_reduce_all: run_reduce_mpi run_reduce_ucc run_reduce_nccl run_reduce_gloo

run_allgather_mpi:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_allgather.py --backend mpi --device cpu
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_allgather.py --backend mpi --device cuda
run_allgather_ucc:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allgather.py --backend ucc --device cpu
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allgather.py --backend ucc --device cuda
run_allgather_nccl:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allgather.py --backend nccl --device cuda
run_allgather_gloo:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allgather.py --backend gloo --device cpu


run_reduce_mpi:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_reduce.py --backend mpi --device cpu
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_reduce.py --backend mpi --device cuda
run_reduce_ucc:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_reduce.py --backend ucc --device cpu
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_reduce.py --backend ucc --device cuda
run_reduce_nccl:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_reduce.py --backend nccl --device cuda
run_reduce_gloo:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_reduce.py --backend gloo --device cpu


