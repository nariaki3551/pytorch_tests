.PHONY: all run_test_allgather run_test_fsdp run_mpi_cuda_support_check

all: build_check

build_check:
	$(MAKE) -C ./src/check

run_test_allgather: build_check
	mpirun -n 2 --allow-run-as-root python3 ./src/collective/test_allgather.py

run_test_fsdp: build_check
	mpirun -n 2 --allow-run-as-root python3 ./src/training/test_fsdp.py --model_scale 1 --num_epochs 10

run_mpi_cuda_support_check: build_check
	$(MAKE) -C ./src/check run_mpi_cuda_support_check