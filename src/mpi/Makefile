USE_CUDA ?= 1

CXX = mpicxx
INCLUDE_DIRS = -I/usr/local/cuda/include
CXX_FLAGS = -g -O0
LIB_DIRS =
LIBS = -lmpi
ifeq ($(USE_CUDA), 1)
	LIB_DIRS = -L/usr/local/cuda/lib64
	LIBS = -lcudart -lmpi
endif
BIN_DIR = ../../bin

SRCS = test_allgather.cpp test_reducescatter.cpp
ifeq ($(USE_CUDA), 1)
	SRCS += check_cuda_support.cpp test_allgather_cuda.cpp test_reducescatter_cuda.cpp test_reducescatter_cuda_unit.cpp
endif

BINS = $(SRCS:.cpp=)
TARGETS = $(addprefix $(BIN_DIR)/, $(addsuffix _mpi, $(BINS)))

.PHONY: all clean

all: $(TARGETS)
clean:
	rm -f $(BIN_DIR)/*_mpi

$(BIN_DIR)/%_mpi: %.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $< $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS)

run_test: run_cuda_support_check_mpi run_test_allgather_mpi run_test_allgather_cuda_mpi run_test_reducescatter_mpi run_test_reducescatter_cuda_mpi
run_sharp_test: run_test_reducescatter_sharp_mpi

SHARP_COMMON_FLAGS = \
	-x UCC_TL_SHARP_TUNE=reduce_scatter:inf\#reduce_scatterv:inf\#allreduce:inf \
	-x SHARP_COLL_ENABLE_SAT=1 \
	-x UCC_TL_SHARP_MIN_TEAM_SIZE=2 \
	-x UCC_TL_SHARP_TEAM_MAX_PPN=2 \
	-x NCCL_COLLNET_ENABLE=1 \
	-x SHARP_COLL_LOCK_ON_COMM_INIT=1 \
	-x SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0
	# -x NCCL_ALGO=CollNet
SPIN_COMMON_FLAGS = \
	-x UCC_TL_SPIN_TUNE=allgather:inf \
	-x UCC_TL_SPIN_MIN_TEAM_SIZE=2 \
	-x UCC_TL_SPIN_TEAM_MAX_PPN=2 \
	-x UCC_TL_SPIN_MAX_RECV_BUF_SIZE=2147483648
UCC_COMMON_FLAGS = \
	--mca coll_ucc_enable 1 \
	--mca coll_ucc_priority 100 \
	-x UCC_MIN_TEAM_SIZE=2
VERBOSE_FLAGS = \
	-x UCC_TL_SHARP_VERBOSE=3 \
	-x UCC_TL_SPIN_VERBOSE=3 \
	-x SHARP_COLL_LOG_LEVEL=5 \
	-x UCC_LOG_LEVEL=trace \
	-x UCC_TL_LOG_LEVEL=trace \
	-x NCCL_DEBUG=INFO
SNAIL01_V100_1_FLAGS = \
	--host snail01:1 \
	--mca plm_rsh_args "-p 2222" \
	--mca btl_openib_if_include mlx5_1:1 \
	--mca btl_tcp_if_exclude lo,docker0,docker_gwbridge \
	-x CUDA_VISIBLE_DEVICES=3 \
	-x UCC_TL_SHARP_DEVICES=mlx5_1 \
	-x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1
SNAIL02_V100_1_FLAGS = \
	--host snail02:1 \
	--mca plm_rsh_args "-p 2222" \
	--mca btl_openib_if_include mlx5_2:1 \
	--mca btl_tcp_if_exclude lo,docker0,docker_gwbridge \
	-x CUDA_VISIBLE_DEVICES=1 \
	-x UCC_TL_SHARP_DEVICES=mlx5_2 \
	-x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2

run_cuda_support_check: $(BIN_DIR)/check_cuda_support_mpi
	mpirun -np 1 $<

run_test_allgather: $(BIN_DIR)/test_allgather_mpi
	mpirun -n 2 $<

run_test_allgather_cuda: $(BIN_DIR)/test_allgather_cuda_mpi
	mpirun -n 2 $<

run_test_allgather_multinode_ucp: $(BIN_DIR)/test_allgather_mpi
	mpirun \
		-n 1 \
		-x UCC_CL_BASIC_TLS=ucp \
		$(SNAIL01_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $< \
		: \
		-n 1 \
		-x UCC_CL_BASIC_TLS=ucp \
		$(SNAIL02_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $<

run_test_allgather_multinode_ucp_cuda: $(BIN_DIR)/test_allgather_cuda_mpi
	mpirun \
		-n 1 \
		-x UCC_CL_BASIC_TLS=ucp \
		$(SNAIL01_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $< \
		: \
		-n 1 \
		-x UCC_CL_BASIC_TLS=ucp \
		$(SNAIL02_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $<

run_test_allgather_multinode_spin: $(BIN_DIR)/test_allgather_mpi
	mpirun \
		-n 1 \
		-x UCC_CL_BASIC_TLS=spin,ucp \
		$(SNAIL01_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(SPIN_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $< --count 64 \
		: \
		-n 1 \
		-x UCC_CL_BASIC_TLS=spin,ucp \
		$(SNAIL02_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(SPIN_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $< --count 64

run_test_allgather_multinode_spin_cuda: $(BIN_DIR)/test_allgather_cuda_mpi
	mpirun \
		-n 1 \
		-x UCC_CL_BASIC_TLS=spin,ucp \
		$(SNAIL01_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(SPIN_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $< --count 64\
		: \
		-n 1 \
		-x UCC_CL_BASIC_TLS=spin,ucp \
		$(SNAIL02_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(SPIN_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $< --count 64

run_test_reducescatter: $(BIN_DIR)/test_reducescatter_mpi
	mpirun -n 2 $<

run_test_reducescatter_cuda: $(BIN_DIR)/test_reducescatter_cuda_mpi
	mpirun -n 2 $<

run_test_reducescatter_multinode_ucp: $(BIN_DIR)/test_reducescatter_mpi
	mpirun \
		-n 1 \
		-x UCC_CL_BASIC_TLS=ucp \
		$(SNAIL01_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $< \
		: \
		-n 1 \
		-x UCC_CL_BASIC_TLS=ucp \
		$(SNAIL02_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $<

run_test_reducescatter_multinode_ucp_cuda: $(BIN_DIR)/test_reducescatter_cuda_mpi
	mpirun \
		-n 1 \
		-x UCC_CL_BASIC_TLS=ucp \
		$(SNAIL01_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $< \
		: \
		-n 1 \
		-x UCC_CL_BASIC_TLS=ucp \
		$(SNAIL02_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $<

run_test_reducescatter_multinode_sharp_cuda: $(BIN_DIR)/test_reducescatter_cuda_mpi
	mpirun \
		-n 1 \
		-x UCC_CL_BASIC_TLS=sharp,ucp \
		$(SNAIL01_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(SHARP_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $< \
		: \
		-n 1 \
		-x UCC_CL_BASIC_TLS=sharp,ucp \
		$(SNAIL02_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(SHARP_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $<

run_test_reducescatter_multinode_sharp_cuda_unit: $(BIN_DIR)/test_reducescatter_cuda_unit_mpi
	mpirun \
		-n 1 \
		-x UCC_CL_BASIC_TLS=sharp,ucp \
		$(SNAIL01_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(SHARP_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $< \
		: \
		-n 1 \
		-x UCC_CL_BASIC_TLS=sharp,ucp \
		$(SNAIL02_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(SHARP_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) -- $<

