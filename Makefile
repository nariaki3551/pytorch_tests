include ./make/mpi_config.mk

.PHONY: run_test_allgather run_test_fsdp run_cuda_support_check_mpi run_check_ucc_sharp_support

build_mpi:
	$(MAKE) -C ./src/mpi

build_ucc:
	$(MAKE) -C ./src/ucc

run_test_allgather:
	mpirun -n 2 $(MPI_MCAST_OPTS) $(MCAST_VERBOSE_OPTS) python3 ./src/collective/test_allgather.py

run_test_fsdp:
	mpirun -n 2 $(MPI_MCAST_OPTS) $(MCAST_VERBOSE_OPTS) python3 ./src/training/test_fsdp.py --model_scale 1 --num_epochs 10

run_test_mpi: build_mpi
	$(MAKE) -C ./src/mpi run_test_all

run_test_ucc: build_ucc
	$(MAKE) -C ./src/ucc run_check_sharp_support_ucc
