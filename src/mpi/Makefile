SHARP_VERBOSE ?= 0
MCAST_VERBOSE ?= 0
CXX = mpicxx
INCLUDE_DIRS = -I/usr/local/cuda/include
CXX_FLAGS = -g -O0
LIB_DIRS = -L/usr/local/cuda/lib64
LIBS = -lcudart -lmpi
BIN_DIR = ../../bin
HOSTFILE ?= ../../hostfile

SRCS = check_cuda_support.cpp test_allgather.cpp test_allgather_cuda.cpp test_reducescatter.cpp test_reducescatter_cuda.cpp

BINS = $(SRCS:.cpp=)
TARGETS = $(addprefix $(BIN_DIR)/, $(addsuffix _mpi, $(BINS)))

.PHONY: all clean

all: $(TARGETS)

$(BIN_DIR)/%_mpi: %.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $< $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS)

run_test: run_cuda_support_check_mpi run_test_allgather_mpi run_test_allgather_cuda_mpi run_test_reducescatter_mpi run_test_reducescatter_cuda_mpi
run_test_mcast: run_test_allgather_mcast_mpi
run_sharp_test: run_test_reducescatter_sharp_mpi

ifeq ($(SHARP_VERBOSE), 1)
SHARP_VERBOSE_OPTS = --mca coll_ucc_verbose 3 -x UCC_LOG_LEVEL=trace -x UCC_TL_LOG_LEVEL=trace -x UCX_LOG_LEVEL=warning -x SHARP_COLL_LOG_LEVEL=5 -x UCC_TL_SHARP_VERBOSE=3
else
SHARP_VERBOSE_OPTS =
endif

ifeq ($(MCAST_VERBOSE), 1)
MCAST_VERBOSE_OPTS = -x UCC_LOG_LEVEL=trace -x UCC_TL_LOG_LEVEL=trace -x UCX_LOG_LEVEL=warning
else
MCAST_VERBOSE_OPTS =
endif

run_cuda_support_check_mpi: $(BIN_DIR)/check_cuda_support_mpi
	mpirun -np 1 $<

run_test_allgather_mpi: $(BIN_DIR)/test_allgather_mpi
	mpirun -n 2 $<

run_test_allgather_cuda_mpi: $(BIN_DIR)/test_allgather_cuda_mpi
	mpirun -n 2 $<

un_test_reducescatter_mpi: $(BIN_DIR)/test_reducescatter_mpi
	mpirun -n 2 $<

run_test_allgather_mcast_mpi: $(BIN_DIR)/test_allgather_mpi
	mpirun -n 2 --host snail02:1,snail03:1 \
	    --mca plm_rsh_args "-p 2222" \
		-x CUDA_VISIBLE_DEVICES=1,0 \
		--mca coll_ucc_enable 1 --mca coll_ucc_priority 100 -x UCC_MIN_TEAM_SIZE=2 \
		-x UCC_CL_BASIC_TLS=spin,ucp \
		-x UCC_TL_SPIN_TUNE=allgather:inf \
		-x UCC_TL_SPIN_ALLGATHER_MCAST_ROOTS=1 -x UCC_TL_SPIN_LINK_BW=12.5 -x UCC_TL_SPIN_MAX_RECV_BUF_SIZE=2147483648 \
		-x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 \
		--mca btl_openib_if_include mlx5_2:1 \
		$(MCAST_VERBOSE_OPTS) \
		-- $<

run_test_reducescatter_sharp_mpi: $(BIN_DIR)/test_reducescatter_mpi
	mpirun -n 2 --host snail02:1,snail03:1 \
	    --mca plm_rsh_args "-p 2222" \
		-x CUDA_VISIBLE_DEVICES=1,0 \
		--mca coll_ucc_enable 1 --mca coll_ucc_priority 100 -x UCC_MIN_TEAM_SIZE=2 \
		-x UCC_CL_BASIC_TLS=sharp,ucp \
		-x UCC_TL_SHARP_TUNE=reduce_scatter:inf#allreduce:inf \
		-x SHARP_COLL_ENABLE_SAT=1 -x UCC_TL_SHARP_MIN_TEAM_SIZE=2 -x UCC_TL_SHARP_TEAM_MAX_PPN=2 \
		-x UCC_TL_SHARP_DEVICES=mlx5_2 \
		--mca btl_openib_if_include mlx5_2:1 \
		$(SHARP_VERBOSE_OPTS) \
		-- $<

run_test_reducescatter_cuda_mpi: $(BIN_DIR)/test_reducescatter_cuda_mpi
	mpirun -n 2 $<

clean:
	rm -f $(TARGETS)
