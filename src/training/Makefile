PYTORCH_DEBUG ?= 0
ifeq ($(PYTORCH_DEBUG), 1)
	PYTORCH_DEBUG_FLAGS := TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL
endif


run_test_fsdp_mpi:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun -n 2 \
		-- python3 test_fsdp.py --model_scale 1 --num_epochs 2 --backend mpi

run_test_fsdp_gloo:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp.py --model_scale 1 --num_epochs 2 --backend gloo
