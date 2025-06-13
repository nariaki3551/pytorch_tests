PROFILE ?= 0
DEBUG ?= 0
SSH_PORT ?= 12345
BACKEND ?= mpi
DEVICE ?= cuda
NUM_EPOCHS := 3
MODEL_SCALE := 1

PROFILE_FLAGS =
ifneq ($(PROFILE), 0)
	PROFILE_FLAGS += --profile $(PROFILE)
endif

TEST_FSDP_ARGS = --model_scale $(MODEL_SCALE) --num_epochs $(NUM_EPOCHS) --backend $(BACKEND) --device $(DEVICE) 
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
	-x UCC_TL_SPIN_MAX_RECV_BUF_SIZE=2147483648
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
	FLAGS+= TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_SHOW_CPP_STACKTRACES=0
endif
FLAGS += \
	-x UCC_TL_SHARP_VERBOSE=3 \
	-x UCC_TL_SPIN_VERBOSE=3 \
	-x SHARP_COLL_LOG_LEVEL=5 \
	-x UCC_LOG_LEVEL=trace \
	-x UCC_TL_LOG_LEVEL=trace \
	-x NCCL_DEBUG=INFO

run_test_fsdp_mpi:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun -n 2 -- python3 test_fsdp.py $(TEST_FSDP_ARGS) $(PROFILE_FLAGS)

run_test_fsdp_mpi_multinode_ucp:
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \

run_test_fsdp_mpi_multinode_sharp:
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \

run_test_fsdp_mpi_multinode_spin:
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \

run_test_fsdp_mpi_multinode_sharp_spin:
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=sharp,spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=sharp,spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=sharp,spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=sharp,spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=sharp,spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=sharp,spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=sharp,spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=sharp,spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=sharp,spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(SPIN_FLAGS) -x UCC_CL_BASIC_TLS=sharp,spin,ucp -- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \

