USE_CUDA ?= 1
PYTORCH_DEBUG ?= 0

ifeq ($(PYTORCH_DEBUG), 1)
	PYTORCH_DEBUG_FLAGS := TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_SHOW_CPP_STACKTRACES=1
endif

run_fsdp_all: run_test_fsdp_mpi run_test_fsdp_nccl run_test_fsdp_ucc run_test_fsdp_gloo
run_fsdp2_all: run_test_fsdp2_mpi run_test_fsdp2_nccl run_test_fsdp2_ucc run_test_fsdp2_gloo

DEVICE = cpu
ifeq ($(USE_CUDA), 1)
	DEVICE = cuda
endif

run_test_fsdp_mpi:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun -n 2 \
		-- python3 test_fsdp.py --model_scale 1 --num_epochs 3 --backend mpi --device $(DEVICE)

run_test_fsdp_nccl:
ifeq ($(USE_CUDA), 1)
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp.py --model_scale 1 --num_epochs 3 --backend nccl --device $(DEVICE)
endif

run_test_fsdp_ucc:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp.py --model_scale 1 --num_epochs 3 --backend ucc --device $(DEVICE)

run_test_fsdp_gloo:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp.py --model_scale 1 --num_epochs 3 --backend gloo --device $(DEVICE)


run_test_fsdp2_mpi:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun -n 2 \
		-- python3 test_fsdp2.py --model_scale 1 --num_epochs 3 --backend mpi --device $(DEVICE)

run_test_fsdp2_nccl:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp2.py --model_scale 1 --num_epochs 3 --backend nccl --device $(DEVICE)

run_test_fsdp2_ucc:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp2.py --model_scale 1 --num_epochs 3 --backend ucc --device $(DEVICE)

run_test_fsdp2_gloo:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp2.py --model_scale 1 --num_epochs 3 --backend gloo --device $(DEVICE)
