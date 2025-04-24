HOSTFILE ?=
SHARP_VERBOSE ?= 0
CXX = mpicxx
INCLUDE_DIRS = -I/usr/local/cuda/include
CXX_FLAGS = -g -O0
LIB_DIRS = -L/usr/local/cuda/lib64
LIBS = -lcudart -lmpi
BIN_DIR = ../../bin

SRCS = check_cuda_support.cpp test_allgather.cpp test_allgather_cuda.cpp test_reducescatter.cpp test_reducescatter_cuda.cpp

BINS = $(SRCS:.cpp=)
TARGETS = $(addprefix $(BIN_DIR)/, $(addsuffix _mpi, $(BINS)))

.PHONY: all clean

all: $(TARGETS)

$(BIN_DIR)/%_mpi: %.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $< $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS)

run_test: run_cuda_support_check_mpi run_test_allgather_mpi run_test_allgather_cuda_mpi run_test_reducescatter_mpi run_test_reducescatter_cuda_mpi
run_sharp_test: run_test_reducescatter_sharp_mpi

ifeq ($(HOSTFILE), )
	HOSTFILE_OPS =
else
	HOSTFILE_OPS = --hostfile $(HOSTFILE)
endif

ifeq ($(SHARP_VERBOSE), 1)
SHARP_VERBOSE_OPTS = --mca coll_ucc_verbose 3 -x UCC_LOG_LEVEL=trace -x UCC_TL_LOG_LEVEL=trace -x UCX_LOG_LEVEL=warning -x SHARP_COLL_LOG_LEVEL=5 -x UCC_TL_SHARP_VERBOSE=3
else
SHARP_VERBOSE_OPTS =
endif

run_cuda_support_check_mpi: $(BIN_DIR)/check_cuda_support_mpi
	mpirun -np 1 $<

run_test_allgather_mpi: $(BIN_DIR)/test_allgather_mpi
	mpirun -n 2 $(HOSTFILE_OPS) $<

run_test_allgather_cuda_mpi: $(BIN_DIR)/test_allgather_cuda_mpi
	mpirun -n 2 $(HOSTFILE_OPS) $<

run_test_reducescatter_mpi: $(BIN_DIR)/test_reducescatter_mpi
	mpirun -n 2 $(HOSTFILE_OPS) $<

run_test_reducescatter_sharp_mpi: $(BIN_DIR)/test_reducescatter_mpi
	@if [ -z "$(HOSTFILE_OPS)" ]; then \
		echo "\n\n\tHOSTFILE is required for sharp tests; \n\tUsage: make (target) HOSTFILE=/path/to/your/hostfile\n\n"; \
		exit 1; \
	fi
	mpirun -n 2 $(HOSTFILE_OPS) \
		--mca coll_ucc_enable 1 --mca coll_ucc_priority 100 -x UCC_MIN_TEAM_SIZE=2 \
		-x UCC_CL_BASIC_TLS=spin,ucp,sharp \
		-x UCC_TL_SHARP_TUNE=reduce_scatter:inf#allreduce:inf \
		-x SHARP_COLL_ENABLE_SAT=1 -x UCC_TL_SHARP_MIN_TEAM_SIZE=2 -x UCC_TL_SHARP_TEAM_MAX_PPN=2 \
		-x CUDA_VISIBLE_DEVICES=1,0 \
		-x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 \
		--mca btl_openib_if_include mlx5_2:1 \
		-x UCC_TL_SHARP_DEVICES=mlx5_2 \
		$(SHARP_VERBOSE_OPTS) \
		$<

run_test_reducescatter_cuda_mpi: $(BIN_DIR)/test_reducescatter_cuda_mpi
	mpirun -n 2 $<

clean:
	rm -f $(TARGETS)
