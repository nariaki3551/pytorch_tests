USE_CUDA ?= 1
USE_SHARP ?= 1
USE_UCC ?= 0
SSH_PORT ?= 12345

.PHONY: run_test_allgather run_test_fsdp run_cuda_support_check_mpi run_check_ucc_sharp_support

all: build_mpi build_ucc build_nccl_tests

build_mpi:
	$(MAKE) -C ./src/mpi USE_CUDA=$(USE_CUDA)

build_ucc:
ifeq ($(USE_UCC), 1)
	$(MAKE) -C ./src/ucc USE_SHARP=$(USE_SHARP) USE_CUDA=$(USE_CUDA)
endif

build_nccl_tests:
ifeq ($(USE_CUDA), 1)
	git submodule update --init --recursive
	$(MAKE) -C ./3rd-party/nccl-tests MPI=1 MPI_HOME=$(MPI_HOME) NCCL_HOME=$(NCCL_HOME) -j
endif

run_test_allgather:
	$(MAKE) -C ./src/collective run_all

run_test_mpi: build_mpi
	$(MAKE) -C ./src/mpi run_test

run_test_sharp_mpi: build_mpi
	$(MAKE) -C ./src/mpi run_sharp_test

run_test_ucc: build_ucc
	$(MAKE) -C ./src/ucc run_check_sharp_support_ucc

run_test_nccl_tests:
	$(MAKE) -f ./Makefile.nccl_tests SSH_PORT=$(SSH_PORT) run_allgather_perf

clean:
	$(MAKE) -C ./src/mpi clean
	$(MAKE) -C ./src/ucc clean
	$(MAKE) -C ./3rd-party/nccl-tests clean

test:
	$(MAKE) -f ./Makefile.nccl_tests test
