PYTORC_DEBUG ?= 0
ifeq ($(PYTORC_DEBUG), 1)
	DEBUG_FLAGS := TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL
endif

run_all: run_test_allgather_mpi run_test_allgather_ucc run_test_allgather_nccl run_test_allgather_gloo

run_test_allgather_mpi:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_allgather.py --backend mpi --device cpu
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_allgather.py --backend mpi --device cuda

run_test_allgather_ucc:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allgather.py --backend ucc --device cpu
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allgather.py --backend ucc --device cuda

run_test_allgather_nccl:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allgather.py --backend nccl --device cuda

run_test_allgather_gloo:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allgather.py --backend gloo --device cpu


