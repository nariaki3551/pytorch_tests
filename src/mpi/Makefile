USE_CUDA ?= 1
SSH_PORT ?= 12345
DEBUG ?= 1
SPIN_TIMEOUT_SCALING ?= 1000000
SPIN_CQ_DEPTH ?= 4194303
SPIN_SQ_DEPTH ?= 16384
SPIN_RQ_DEPTH ?= 32768

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
	-x SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0 \
	-x NCCL_SHARP_GROUP_SIZE_THRESH=1 \
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

SNAIL01_V100_2_FLAGS = --host snail01:1 -x CUDA_VISIBLE_DEVICES=2 -x UCC_TL_SHARP_DEVICES=mlx5_0 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_0 -x UCC_TL_SPIN_START_CORE_ID=0 $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(SHARP_FLAGS)
SNAIL01_V100_3_FLAGS = --host snail01:1 -x CUDA_VISIBLE_DEVICES=3 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1 -x UCC_TL_SPIN_START_CORE_ID=2 $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(SHARP_FLAGS)
SNAIL02_V100_0_FLAGS = --host snail02:1 -x CUDA_VISIBLE_DEVICES=0 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1 -x UCC_TL_SPIN_START_CORE_ID=0 $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(SHARP_FLAGS)
SNAIL02_V100_1_FLAGS = --host snail02:1 -x CUDA_VISIBLE_DEVICES=1 -x UCC_TL_SHARP_DEVICES=mlx5_2 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 -x UCC_TL_SPIN_START_CORE_ID=2 $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(SHARP_FLAGS)
SNAIL03_V100_0_FLAGS = --host snail03:1 -x CUDA_VISIBLE_DEVICES=0 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1 -x UCC_TL_SPIN_START_CORE_ID=0 $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(SHARP_FLAGS)
SNAIL03_V100_1_FLAGS = --host snail03:1 -x CUDA_VISIBLE_DEVICES=1 -x UCC_TL_SHARP_DEVICES=mlx5_2 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 -x UCC_TL_SPIN_START_CORE_ID=2 $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(SHARP_FLAGS)
TVM01_V100_0_FLAGS   = --host tvm01:1   -x CUDA_VISIBLE_DEVICES=0 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1 -x UCC_TL_SPIN_START_CORE_ID=0 $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(SHARP_FLAGS)
TVM01_V100_1_FLAGS   = --host tvm01:1   -x CUDA_VISIBLE_DEVICES=1 -x UCC_TL_SHARP_DEVICES=mlx5_2 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 -x UCC_TL_SPIN_START_CORE_ID=2 $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(SHARP_FLAGS)
TVM02_V100_0_FLAGS   = --host tvm02:1   -x CUDA_VISIBLE_DEVICES=0 -x UCC_TL_SHARP_DEVICES=mlx5_1 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1 -x UCC_TL_SPIN_START_CORE_ID=0 $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(SHARP_FLAGS)
TVM02_V100_1_FLAGS   = --host tvm02:1   -x CUDA_VISIBLE_DEVICES=1 -x UCC_TL_SHARP_DEVICES=mlx5_2 -x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 -x UCC_TL_SPIN_START_CORE_ID=2 $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(SHARP_FLAGS)


run_cuda_support_check: $(BIN_DIR)/check_cuda_support_mpi
	mpirun -np 1 $<

run_command:
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) -x UCC_CL_BASIC_TLS=$(UCC_CL_BASIC_TLS) -- $(COMMAND) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) -x UCC_CL_BASIC_TLS=$(UCC_CL_BASIC_TLS) -- $(COMMAND) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) -x UCC_CL_BASIC_TLS=$(UCC_CL_BASIC_TLS) -- $(COMMAND) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) -x UCC_CL_BASIC_TLS=$(UCC_CL_BASIC_TLS) -- $(COMMAND) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) -x UCC_CL_BASIC_TLS=$(UCC_CL_BASIC_TLS) -- $(COMMAND) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) -x UCC_CL_BASIC_TLS=$(UCC_CL_BASIC_TLS) -- $(COMMAND) \
	: -n 1 $(TVM01_V100_0_FLAGS)   -x UCC_CL_BASIC_TLS=$(UCC_CL_BASIC_TLS) -- $(COMMAND) \
	: -n 1 $(TVM01_V100_1_FLAGS)   -x UCC_CL_BASIC_TLS=$(UCC_CL_BASIC_TLS) -- $(COMMAND) \
	: -n 1 $(TVM02_V100_0_FLAGS)   -x UCC_CL_BASIC_TLS=$(UCC_CL_BASIC_TLS) -- $(COMMAND) \
	: -n 1 $(TVM02_V100_1_FLAGS)   -x UCC_CL_BASIC_TLS=$(UCC_CL_BASIC_TLS) -- $(COMMAND) \

run_allgather_ucp: $(BIN_DIR)/test_allgather_mpi
	make run_command UCC_CL_BASIC_TLS=ucp COMMAND=$<

run_allgather_ucp_cuda: $(BIN_DIR)/test_allgather_cuda_mpi
	make run_command UCC_CL_BASIC_TLS=ucp COMMAND=$<

run_allgather_spin: $(BIN_DIR)/test_allgather_mpi
	make run_command UCC_CL_BASIC_TLS=spin,ucp COMMAND=$<

run_allgather_spin_cuda: $(BIN_DIR)/test_allgather_cuda_mpi
	make run_command UCC_CL_BASIC_TLS=spin,ucp COMMAND=$<

run_reducescatter_ucp: $(BIN_DIR)/test_reducescatter_mpi
	make run_command UCC_CL_BASIC_TLS=ucp COMMAND=$<

run_reducescatter_ucp_cuda: $(BIN_DIR)/test_reducescatter_cuda_mpi
	make run_command UCC_CL_BASIC_TLS=ucp COMMAND=$<

run_reducescatter_sharp: $(BIN_DIR)/test_reducescatter_mpi
	make run_command UCC_CL_BASIC_TLS=shapr,ucp COMMAND=$<

run_reducescatter_sharp_cuda: $(BIN_DIR)/test_reducescatter_cuda_mpi
	make run_command UCC_CL_BASIC_TLS=shapr,ucp COMMAND=$<

MIN_COUNT ?= 1024
MAX_COUNT ?= 134217728
WARMUP ?= 5
ITERATIONS ?= 20
STEP ?= 2
perf_allgather_ucp_cuda: $(BIN_DIR)/allgather_perf_cuda_mpi
	make run_command UCC_CL_BASIC_TLS=ucp COMMAND="$< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP)"

perf_allgather_ucp_cuda_perf: $(BIN_DIR)/allgather_perf_cuda_mpi
	make run_command UCC_CL_BASIC_TLS=ucp COMMAND="perf record -g fp -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP)"

perf_allgather_spin_cuda: $(BIN_DIR)/allgather_perf_cuda_mpi
	make run_command UCC_CL_BASIC_TLS=spin,ucp COMMAND="$< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP)"


perf_allgather_spin_cuda_data_1: $(BIN_DIR)/allgather_perf_cuda_mpi
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 0 -o /tmp/perf.rank0.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 2 -o /tmp/perf.rank1.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 0 -o /tmp/perf.rank2.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 2 -o /tmp/perf.rank3.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 0 -o /tmp/perf.rank4.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 2 -o /tmp/perf.rank5.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_0_FLAGS)   -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 0 -o /tmp/perf.rank6.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_1_FLAGS)   -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 2 -o /tmp/perf.rank7.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_0_FLAGS)   -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 0 -o /tmp/perf.rank8.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_1_FLAGS)   -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 2 -o /tmp/perf.rank9.cpu0.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \

perf_allgather_spin_cuda_data_2: $(BIN_DIR)/allgather_perf_cuda_mpi
	mpirun \
	: -n 1 $(SNAIL01_V100_2_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 1 -o /tmp/perf.rank0.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 3 -o /tmp/perf.rank1.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 1 -o /tmp/perf.rank2.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 3 -o /tmp/perf.rank3.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 1 -o /tmp/perf.rank4.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 3 -o /tmp/perf.rank5.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_0_FLAGS)   -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 1 -o /tmp/perf.rank6.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM01_V100_1_FLAGS)   -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 3 -o /tmp/perf.rank7.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_0_FLAGS)   -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 1 -o /tmp/perf.rank8.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \
	: -n 1 $(TVM02_V100_1_FLAGS)   -x UCC_CL_BASIC_TLS=spin,ucp -- perf record -g -a -C 3 -o /tmp/perf.rank9.cpu1.data -- $< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP) \

FG=/opt/FlameGraph
perf_allgather_spin_cuda_data:
	mkdir -p ./data
	make perf_allgather_spin_cuda_data_1 MIN_COUNT=524288 MAX_COUNT=524288 WARMUP=0 ITERATIONS=100 DEBUG=0 SPIN_CQ_DEPTH=4194303 SPIN_SQ_DEPTH=16384 SPIN_RQ_DEPTH=32768 SPIN_TIMEOUT_SCALING=1000000 DEBUG=0
	make perf_allgather_spin_cuda_data_2 MIN_COUNT=524288 MAX_COUNT=524288 WARMUP=0 ITERATIONS=100 DEBUG=0 SPIN_CQ_DEPTH=4194303 SPIN_SQ_DEPTH=16384 SPIN_RQ_DEPTH=32768 SPIN_TIMEOUT_SCALING=1000000 DEBUG=0
	@for rank in 0 1 2 3 4 5 6 7 8 9; do \
		for cpu in 0 1; do \
			case $$rank in \
				0|1) host=snail01 ;; \
				2|3) host=snail02 ;; \
				4|5) host=snail03 ;; \
				6|7) host=tvm01 ;; \
				8|9) host=tvm02 ;; \
			esac; \
			command="ssh -p 12345 $$host \"perf script -i /tmp/perf.rank$$rank.cpu$$cpu.data > /tmp/out; $(FG)/stackcollapse-perf.pl /tmp/out > /tmp/folded; $(FG)/flamegraph.pl /tmp/folded > /tmp/perf.rank$$rank.cpu$$cpu.svg\""; \
			echo $$command; \
			eval $$command; \
		done; \
	done
	@for rank in 0 1 2 3 4 5 6 7 8 9; do \
		for cpu in 0 1; do \
			case $$rank in \
				0|1) host=snail01 ;; \
				2|3) host=snail02 ;; \
				4|5) host=snail03 ;; \
				6|7) host=tvm01 ;; \
				8|9) host=tvm02 ;; \
			esac; \
			command="scp -P 12345 $$host:/tmp/perf.rank$$rank.cpu$$cpu.svg ./data/"; \
			echo $$command; \
			eval $$command; \
		done; \
	done

perf_reducescatter_ucp_cuda: $(BIN_DIR)/reducescatter_perf_cuda_mpi
	make run_command UCC_CL_BASIC_TLS=ucp COMMAND="$< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP)"

perf_reducescatter_sharp_cuda: $(BIN_DIR)/reducescatter_perf_cuda_mpi
	make run_command UCC_CL_BASIC_TLS=sharp,ucp COMMAND="$< -l $(MIN_COUNT) -u $(MAX_COUNT) -w $(WARMUP) -n $(ITERATIONS) -s $(STEP)"

