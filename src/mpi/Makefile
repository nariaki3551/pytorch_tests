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
SNAIL01_V100_2_FLAGS = --host snail01:1 -x CUDA_VISIBLE_DEVICES=2 -x UCC_TL_SHARP_DEVICES=mlx5_0 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_0 -x UCC_TL_SPIN_START_CORE_ID=0
SNAIL01_V100_3_FLAGS = --host snail01:1 -x CUDA_VISIBLE_DEVICES=3 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1 -x UCC_TL_SPIN_START_CORE_ID=2
SNAIL02_V100_0_FLAGS = --host snail02:1 -x CUDA_VISIBLE_DEVICES=0 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1 -x UCC_TL_SPIN_START_CORE_ID=0
SNAIL02_V100_1_FLAGS = --host snail02:1 -x CUDA_VISIBLE_DEVICES=1 -x UCC_TL_SHARP_DEVICES=mlx5_2 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 -x UCC_TL_SPIN_START_CORE_ID=2
SNAIL03_V100_0_FLAGS = --host snail03:1 -x CUDA_VISIBLE_DEVICES=0 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1 -x UCC_TL_SPIN_START_CORE_ID=0
SNAIL03_V100_1_FLAGS = --host snail03:1 -x CUDA_VISIBLE_DEVICES=1 -x UCC_TL_SHARP_DEVICES=mlx5_2 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 -x UCC_TL_SPIN_START_CORE_ID=2
TVM01_V100_0_FLAGS   = --host tvm01:1   -x CUDA_VISIBLE_DEVICES=0 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1 -x UCC_TL_SPIN_START_CORE_ID=0
TVM01_V100_1_FLAGS   = --host tvm01:1   -x CUDA_VISIBLE_DEVICES=1 -x UCC_TL_SHARP_DEVICES=mlx5_2 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 -x UCC_TL_SPIN_START_CORE_ID=2
TVM02_V100_0_FLAGS   = --host tvm02:1   -x CUDA_VISIBLE_DEVICES=0 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1 -x UCC_TL_SPIN_START_CORE_ID=0
TVM02_V100_1_FLAGS   = --host tvm02:1   -x CUDA_VISIBLE_DEVICES=1 -x UCC_TL_SHARP_DEVICES=mlx5_2 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 -x UCC_TL_SPIN_START_CORE_ID=2


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
STEP ?= 2
run_allgather_perf_ucp_cuda: $(BIN_DIR)/allgather_perf_cuda_mpi
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \

run_allgather_perf_ucp_cuda_perf: $(BIN_DIR)/allgather_perf_cuda_mpi
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- perf record -g fp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- perf record -g fp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- perf record -g fp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- perf record -g fp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- perf record -g fp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- perf record -g fp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- perf record -g fp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- perf record -g fp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- perf record -g fp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- perf record -g fp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \

run_allgather_perf_spin_cuda: $(BIN_DIR)/allgather_perf_cuda_mpi
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \

run_allgather_perf_spin_cuda_data_1: $(BIN_DIR)/allgather_perf_cuda_mpi
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 0 -o /tmp/perf.rank0.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 2 -o /tmp/perf.rank1.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 0 -o /tmp/perf.rank2.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 2 -o /tmp/perf.rank3.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 0 -o /tmp/perf.rank4.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 2 -o /tmp/perf.rank5.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 0 -o /tmp/perf.rank6.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 2 -o /tmp/perf.rank7.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 0 -o /tmp/perf.rank8.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 2 -o /tmp/perf.rank9.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \

run_allgather_perf_spin_cuda_data_2: $(BIN_DIR)/allgather_perf_cuda_mpi
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 1 -o /tmp/perf.rank0.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 3 -o /tmp/perf.rank1.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 1 -o /tmp/perf.rank2.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 3 -o /tmp/perf.rank3.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 1 -o /tmp/perf.rank4.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 3 -o /tmp/perf.rank5.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 1 -o /tmp/perf.rank6.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 3 -o /tmp/perf.rank7.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 1 -o /tmp/perf.rank8.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 3 -o /tmp/perf.rank9.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \

FG=/opt/FlameGraph
run_allgather_perf_spin_cuda_data:
	mkdir -p ./data
	make run_allgather_perf_spin_cuda_data_1 MIN_COUNT=128 MAX_COUNT=128 WORMUP=0 ITERATIONS=10 DEBUG=0 SPIN_CQ_DEPTH=4194303 SPIN_SQ_DEPTH=16384 SPIN_RQ_DEPTH=32768 SPIN_TIMEOUT_SCALING=1000000 DEBUG=0
	make run_allgather_perf_spin_cuda_data_2 MIN_COUNT=128 MAX_COUNT=128 WORMUP=0 ITERATIONS=10 DEBUG=0 SPIN_CQ_DEPTH=4194303 SPIN_SQ_DEPTH=16384 SPIN_RQ_DEPTH=32768 SPIN_TIMEOUT_SCALING=1000000 DEBUG=0 
	ssh -p 12345 snail01 "perf script -i /tmp/perf.rank0.cpu0.data > /tmp/p00.out; $(FG)/stackcollapse-perf.pl /tmp/p00.out > /tmp/p00.folded; $(FG)/flamegraph.pl /tmp/p00.folded > /tmp/perf.rank0.cpu0.svg"
	ssh -p 12345 snail01 "perf script -i /tmp/perf.rank0.cpu1.data > /tmp/p01.out; $(FG)/stackcollapse-perf.pl /tmp/p01.out > /tmp/p01.folded; $(FG)/flamegraph.pl /tmp/p01.folded > /tmp/perf.rank0.cpu1.svg"
	ssh -p 12345 snail01 "perf script -i /tmp/perf.rank1.cpu0.data > /tmp/p10.out; $(FG)/stackcollapse-perf.pl /tmp/p10.out > /tmp/p10.folded; $(FG)/flamegraph.pl /tmp/p10.folded > /tmp/perf.rank1.cpu0.svg"
	ssh -p 12345 snail01 "perf script -i /tmp/perf.rank1.cpu1.data > /tmp/p11.out; $(FG)/stackcollapse-perf.pl /tmp/p11.out > /tmp/p11.folded; $(FG)/flamegraph.pl /tmp/p11.folded > /tmp/perf.rank1.cpu1.svg"
	ssh -p 12345 snail02 "perf script -i /tmp/perf.rank2.cpu0.data > /tmp/p20.out; $(FG)/stackcollapse-perf.pl /tmp/p20.out > /tmp/p20.folded; $(FG)/flamegraph.pl /tmp/p20.folded > /tmp/perf.rank2.cpu0.svg"
	ssh -p 12345 snail02 "perf script -i /tmp/perf.rank2.cpu1.data > /tmp/p21.out; $(FG)/stackcollapse-perf.pl /tmp/p21.out > /tmp/p21.folded; $(FG)/flamegraph.pl /tmp/p21.folded > /tmp/perf.rank2.cpu1.svg"
	ssh -p 12345 snail02 "perf script -i /tmp/perf.rank3.cpu0.data > /tmp/p30.out; $(FG)/stackcollapse-perf.pl /tmp/p30.out > /tmp/p30.folded; $(FG)/flamegraph.pl /tmp/p30.folded > /tmp/perf.rank3.cpu0.svg"
	ssh -p 12345 snail02 "perf script -i /tmp/perf.rank3.cpu1.data > /tmp/p31.out; $(FG)/stackcollapse-perf.pl /tmp/p31.out > /tmp/p31.folded; $(FG)/flamegraph.pl /tmp/p31.folded > /tmp/perf.rank3.cpu1.svg"
	ssh -p 12345 snail03 "perf script -i /tmp/perf.rank4.cpu0.data > /tmp/p40.out; $(FG)/stackcollapse-perf.pl /tmp/p40.out > /tmp/p40.folded; $(FG)/flamegraph.pl /tmp/p40.folded > /tmp/perf.rank4.cpu0.svg"
	ssh -p 12345 snail03 "perf script -i /tmp/perf.rank4.cpu1.data > /tmp/p41.out; $(FG)/stackcollapse-perf.pl /tmp/p41.out > /tmp/p41.folded; $(FG)/flamegraph.pl /tmp/p41.folded > /tmp/perf.rank4.cpu1.svg"
	ssh -p 12345 snail03 "perf script -i /tmp/perf.rank5.cpu0.data > /tmp/p50.out; $(FG)/stackcollapse-perf.pl /tmp/p50.out > /tmp/p50.folded; $(FG)/flamegraph.pl /tmp/p50.folded > /tmp/perf.rank5.cpu0.svg"
	ssh -p 12345 snail03 "perf script -i /tmp/perf.rank5.cpu1.data > /tmp/p51.out; $(FG)/stackcollapse-perf.pl /tmp/p51.out > /tmp/p51.folded; $(FG)/flamegraph.pl /tmp/p51.folded > /tmp/perf.rank5.cpu1.svg"
	ssh -p 12345 tvm01 "perf script -i /tmp/perf.rank6.cpu0.data > /tmp/p60.out; $(FG)/stackcollapse-perf.pl /tmp/p60.out > /tmp/p60.folded; $(FG)/flamegraph.pl /tmp/p60.folded > /tmp/perf.rank6.cpu0.svg"
	ssh -p 12345 tvm01 "perf script -i /tmp/perf.rank6.cpu1.data > /tmp/p61.out; $(FG)/stackcollapse-perf.pl /tmp/p61.out > /tmp/p61.folded; $(FG)/flamegraph.pl /tmp/p61.folded > /tmp/perf.rank6.cpu1.svg"
	ssh -p 12345 tvm01 "perf script -i /tmp/perf.rank7.cpu0.data > /tmp/p70.out; $(FG)/stackcollapse-perf.pl /tmp/p70.out > /tmp/p70.folded; $(FG)/flamegraph.pl /tmp/p70.folded > /tmp/perf.rank7.cpu0.svg"
	ssh -p 12345 tvm01 "perf script -i /tmp/perf.rank7.cpu1.data > /tmp/p71.out; $(FG)/stackcollapse-perf.pl /tmp/p71.out > /tmp/p71.folded; $(FG)/flamegraph.pl /tmp/p71.folded > /tmp/perf.rank7.cpu1.svg"
	ssh -p 12345 tvm02 "perf script -i /tmp/perf.rank8.cpu0.data > /tmp/p80.out; $(FG)/stackcollapse-perf.pl /tmp/p80.out > /tmp/p80.folded; $(FG)/flamegraph.pl /tmp/p80.folded > /tmp/perf.rank8.cpu0.svg"
	ssh -p 12345 tvm02 "perf script -i /tmp/perf.rank8.cpu1.data > /tmp/p81.out; $(FG)/stackcollapse-perf.pl /tmp/p81.out > /tmp/p81.folded; $(FG)/flamegraph.pl /tmp/p81.folded > /tmp/perf.rank8.cpu1.svg"
	ssh -p 12345 tvm02 "perf script -i /tmp/perf.rank9.cpu0.data > /tmp/p90.out; $(FG)/stackcollapse-perf.pl /tmp/p90.out > /tmp/p90.folded; $(FG)/flamegraph.pl /tmp/p90.folded > /tmp/perf.rank9.cpu0.svg"
	ssh -p 12345 tvm02 "perf script -i /tmp/perf.rank9.cpu1.data > /tmp/p91.out; $(FG)/stackcollapse-perf.pl /tmp/p91.out > /tmp/p91.folded; $(FG)/flamegraph.pl /tmp/p91.folded > /tmp/perf.rank9.cpu1.svg"
	scp -P 12345 snail01:/tmp/perf.rank0.cpu0.svg ./data/
	scp -P 12345 snail01:/tmp/perf.rank0.cpu1.svg ./data/
	scp -P 12345 snail01:/tmp/perf.rank1.cpu0.svg ./data/
	scp -P 12345 snail01:/tmp/perf.rank1.cpu1.svg ./data/
	scp -P 12345 snail02:/tmp/perf.rank2.cpu0.svg ./data/
	scp -P 12345 snail02:/tmp/perf.rank2.cpu1.svg ./data/
	scp -P 12345 snail02:/tmp/perf.rank3.cpu0.svg ./data/
	scp -P 12345 snail02:/tmp/perf.rank3.cpu1.svg ./data/
	scp -P 12345 snail03:/tmp/perf.rank4.cpu0.svg ./data/
	scp -P 12345 snail03:/tmp/perf.rank4.cpu1.svg ./data/
	scp -P 12345 snail03:/tmp/perf.rank5.cpu0.svg ./data/
	scp -P 12345 snail03:/tmp/perf.rank5.cpu1.svg ./data/
	scp -P 12345 tvm01:/tmp/perf.rank6.cpu0.svg ./data/
	scp -P 12345 tvm01:/tmp/perf.rank6.cpu1.svg ./data/
	scp -P 12345 tvm01:/tmp/perf.rank7.cpu0.svg ./data/
	scp -P 12345 tvm01:/tmp/perf.rank7.cpu1.svg ./data/
	scp -P 12345 tvm02:/tmp/perf.rank8.cpu0.svg ./data/
	scp -P 12345 tvm02:/tmp/perf.rank8.cpu1.svg ./data/
	scp -P 12345 tvm02:/tmp/perf.rank9.cpu0.svg ./data/
	scp -P 12345 tvm02:/tmp/perf.rank9.cpu1.svg ./data/


run_reducescatter_perf_ucp_cuda: $(BIN_DIR)/reducescatter_perf_cuda_mpi
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \

run_reducescatter_perf_sharp_cuda: $(BIN_DIR)/reducescatter_perf_cuda_mpi
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WORMUP) -n $(ITERATIONS) -s $(STEP) \

