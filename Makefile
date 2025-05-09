USE_CUDA ?= 1
USE_SHARP ?= 1
USE_UCC ?= 0

.PHONY: run_test_allgather run_test_fsdp run_cuda_support_check_mpi run_check_ucc_sharp_support

all: build_mpi build_ucc

build_mpi:
	$(MAKE) -C ./src/mpi USE_CUDA=$(USE_CUDA)

build_ucc:
ifeq ($(USE_UCC), 1)
	$(MAKE) -C ./src/ucc USE_SHARP=$(USE_SHARP)
endif

run_test_allgather:
	$(MAKE) -C ./src/collective run_all

run_test_mpi: build_mpi
	$(MAKE) -C ./src/mpi run_test

run_test_sharp_mpi: build_mpi
	$(MAKE) -C ./src/mpi run_sharp_test

run_test_ucc: build_ucc
	$(MAKE) -C ./src/ucc run_check_sharp_support_ucc

clean:
	$(MAKE) -C ./src/mpi clean
	$(MAKE) -C ./src/ucc clean
