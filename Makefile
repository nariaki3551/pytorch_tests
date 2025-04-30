.PHONY: run_test_allgather run_test_fsdp run_cuda_support_check_mpi run_check_ucc_sharp_support

build_mpi:
	$(MAKE) -C ./src/mpi

build_ucc:
	$(MAKE) -C ./src/ucc

run_test_allgather:
	mpirun -n 2 python3 ./src/collective/test_allgather.py

run_test_fsdp_mpi:
	mpirun -n 2 --allow-run-as-root python3 ./src/training/test_fsdp.py --model_scale 1 --num_epochs 2

run_test_fsdp_gloo:
	TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- ./src/training/test_fsdp.py --model_scale 1 --num_epochs 2 --backend gloo

run_test_mpi: build_mpi
	$(MAKE) -C ./src/mpi run_test

run_test_sharp_mpi: build_mpi
	$(MAKE) -C ./src/mpi run_sharp_test

run_test_ucc: build_ucc
	$(MAKE) -C ./src/ucc run_check_sharp_support_ucc
