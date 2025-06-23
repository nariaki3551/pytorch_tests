USE_CUDA ?= 1
SSH_PORT ?= 12345
DEBUG ?= 1
SPIN_TIMEOUT_SCALING ?= 5
SPIN_CQ_DEPTH ?= 128
SPIN_SQ_DEPTH ?= 128
SPIN_RQ_DEPTH ?= 128

CXX = mpicxx
INCLUDE_DIRS = -I/usr/local/cuda/include
CXX_FLAGS = -g -O0
LIB_DIRS =
LIBS = -lmpi
SRCS = test_allgather.cpp test_reducescatter.cpp
ifeq ($(USE_CUDA), 1)
	LIB_DIRS = -L/usr/local/cuda/lib64
	LIBS = -lcudart -lmpi
	SRCS += check_cuda_support.cpp test_allgather_cuda.cpp test_reducescatter_cuda.cpp allgather_perf_cuda.cpp reducescatter_perf_cuda.cpp
endif

BIN_DIR = ../../bin
BINS = $(SRCS:.cpp=)
TARGETS = $(addprefix $(BIN_DIR)/, $(addsuffix _mpi, $(BINS)))

.PHONY: all clean

all: $(TARGETS)
clean:
	rm -f $(BIN_DIR)/*_mpi

$(BIN_DIR)/%_mpi: %.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $< $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS)

SHARP_FLAGS = \
	-x UCC_TL_SHARP_TUNE=reduce_scatter:inf\#reduce_scatterv:inf\#allreduce:inf \
	-x SHARP_COLL_ENABLE_SAT=1 \
	-x UCC_TL_SHARP_MIN_TEAM_SIZE=2 \
	-x UCC_TL_SHARP_TEAM_MAX_PPN=2 \
	-x NCCL_COLLNET_ENABLE=1 \
	-x SHARP_COLL_LOCK_ON_COMM_INIT=1 \
	-x SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0
	# -x NCCL_ALGO=CollNet
SPIN_FLAGS = \
	-x UCC_TL_SPIN_TUNE=allgather:inf \
	-x UCC_TL_SPIN_MIN_TEAM_SIZE=2 \
	-x UCC_TL_SPIN_TEAM_MAX_PPN=2 \
	-x UCC_TL_SPIN_MAX_RECV_BUF_SIZE=2147483648 \
	-x UCC_TL_SPIN_TIMEOUT_SCALING=$(SPIN_TIMEOUT_SCALING) \
	-x UCC_TL_SPIN_ALLGATHER_MCAST_ROOTS=1 \
	-x UCC_TL_SPIN_MCAST_CQ_DEPTH=$(SPIN_CQ_DEPTH) \
	-x UCC_TL_SPIN_MCAST_SQ_DEPTH=$(SPIN_SQ_DEPTH) \
	-x UCC_TL_SPIN_MCAST_RQ_DEPTH=$(SPIN_RQ_DEPTH)
UCC_FLAGS = \
	--mca coll_ucc_enable 1 \
	--mca coll_ucc_priority 100 \
	-x UCC_MIN_TEAM_SIZE=2
SNAIL01_V100_2_FLAGS = --host snail01:1 -x CUDA_VISIBLE_DEVICES=2 -x UCC_TL_SHARP_DEVICES=mlx5_0 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_0
SNAIL01_V100_3_FLAGS = --host snail01:1 -x CUDA_VISIBLE_DEVICES=3 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1
SNAIL02_V100_0_FLAGS = --host snail02:1 -x CUDA_VISIBLE_DEVICES=0 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1
SNAIL02_V100_1_FLAGS = --host snail02:1 -x CUDA_VISIBLE_DEVICES=1 -x UCC_TL_SHARP_DEVICES=mlx5_2 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2
SNAIL03_V100_0_FLAGS = --host snail03:1 -x CUDA_VISIBLE_DEVICES=0 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1
SNAIL03_V100_1_FLAGS = --host snail03:1 -x CUDA_VISIBLE_DEVICES=1 -x UCC_TL_SHARP_DEVICES=mlx5_2 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2
TVM01_V100_0_FLAGS   = --host tvm01:1   -x CUDA_VISIBLE_DEVICES=0 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1
TVM01_V100_1_FLAGS   = --host tvm01:1   -x CUDA_VISIBLE_DEVICES=1 -x UCC_TL_SHARP_DEVICES=mlx5_2 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2
TVM02_V100_0_FLAGS   = --host tvm02:1   -x CUDA_VISIBLE_DEVICES=0 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1
TVM02_V100_1_FLAGS   = --host tvm02:1   -x CUDA_VISIBLE_DEVICES=1 -x UCC_TL_SHARP_DEVICES=mlx5_2 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2

FLAGS = --mca plm_rsh_args "-p $(SSH_PORT)"
ifeq ($(DEBUG), 1)
FLAGS += \
	-x UCC_TL_SHARP_VERBOSE=3 \
	-x UCC_TL_SPIN_VERBOSE=3 \
	-x SHARP_COLL_LOG_LEVEL=5 \
	-x UCC_LOG_LEVEL=trace \
	-x UCC_TL_LOG_LEVEL=trace \
	-x NCCL_DEBUG=INFO
endif

run_cuda_support_check: $(BIN_DIR)/check_cuda_support_mpi
	mpirun -np 1 $<

run_multinode_tests: \
	run_test_allgather_multinode_ucp \
	run_test_allgather_multinode_ucp_cuda \
	run_test_allgather_multinode_spin \
	run_test_allgather_multinode_spin_cuda \
	run_test_reducescatter_multinode_ucp \
	run_test_reducescatter_multinode_ucp_cuda \
	run_test_reducescatter_multinode_sharp \
	run_test_reducescatter_multinode_sharp_cuda \

run_test_allgather_multinode_ucp: $(BIN_DIR)/test_allgather_mpi
	@echo "run_test_allgather_multinode_ucp"
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \

run_test_allgather_multinode_ucp_cuda: $(BIN_DIR)/test_allgather_cuda_mpi
	@echo "run_test_allgather_multinode_ucp_cuda"
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \

run_test_allgather_multinode_spin: $(BIN_DIR)/test_allgather_mpi
	@echo "run_test_allgather_multinode_spin"
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \

run_test_allgather_multinode_spin_cuda: $(BIN_DIR)/test_allgather_cuda_mpi
	@echo "run_test_allgather_multinode_spin_cuda"
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< \

run_test_reducescatter_multinode_ucp: $(BIN_DIR)/test_reducescatter_mpi
	@echo "run_test_reducescatter_multinode_ucp"
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \

run_test_reducescatter_multinode_ucp_cuda: $(BIN_DIR)/test_reducescatter_cuda_mpi
	@echo "run_test_reducescatter_multinode_ucp_cuda"
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< \

run_test_reducescatter_multinode_sharp: $(BIN_DIR)/test_reducescatter_mpi
	@echo "run_test_reducescatter_multinode_sharp"
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \

run_test_reducescatter_multinode_sharp_cuda: $(BIN_DIR)/test_reducescatter_cuda_mpi
	@echo "run_test_reducescatter_multinode_sharp_cuda"
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< \


MIN_COUNT ?= 1024
MAX_COUNT ?= 134217728
WORMUP ?= 5
ITERATIONS ?= 20
run_allgather_perf_ucp_cuda: $(BIN_DIR)/allgather_perf_cuda_mpi
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \

run_allgather_perf_spin_cuda: $(BIN_DIR)/allgather_perf_cuda_mpi
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \

run_reducescatter_perf_ucp_cuda: $(BIN_DIR)/reducescatter_perf_cuda_mpi
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \

run_reducescatter_perf_sharp_cuda: $(BIN_DIR)/reducescatter_perf_cuda_mpi
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) \
