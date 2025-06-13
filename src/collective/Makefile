DEBUG ?= 0
SSH_PORT ?= 12345
BACKEND ?= mpi
DEVICE ?= cuda

# run_all: run_allgather_all run_reduce_all run_reduce_scatter_all
# run_allgather_all: run_allgather_mpi run_allgather_ucc run_allgather_nccl run_allgather_gloo
# run_reduce_all: run_reduce_mpi run_reduce_ucc run_reduce_nccl run_reduce_gloo
# run_reduce_scatter_all: run_reduce_scatter_mpi run_reduce_scatter_ucc run_reduce_scatter_nccl run_reduce_scatter_gloo
# run_allreduce_all: run_allreduce_mpi run_allreduce_ucc run_allreduce_nccl run_allreduce_gloo

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
	FLAGS += TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL
endif
FLAGS += \
	-x UCC_TL_SHARP_VERBOSE=3 \
	-x UCC_TL_SPIN_VERBOSE=3 \
	-x SHARP_COLL_LOG_LEVEL=5 \
	-x UCC_LOG_LEVEL=trace \
	-x UCC_TL_LOG_LEVEL=trace \
	-x NCCL_DEBUG=INFO


run_allgather_multinode_ucp:
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE)

run_allgather_multinode_spin:
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(UCC_FLAGS) $(SPIN_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(UCC_FLAGS) $(SPIN_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(UCC_FLAGS) $(SPIN_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(UCC_FLAGS) $(SPIN_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(UCC_FLAGS) $(SPIN_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=spin,ucp -- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE)


run_reduce_scatter_multinode_ucp:
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(UCC_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE)

run_reduce_scatter_multinode_sharp:
	mpirun \
	  -n 1 $(SNAIL01_V100_2_FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) --sharp \
	: -n 1 $(SNAIL01_V100_3_FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) --sharp \
	: -n 1 $(SNAIL02_V100_0_FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) --sharp \
	: -n 1 $(SNAIL02_V100_1_FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) --sharp \
	: -n 1 $(SNAIL03_V100_0_FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) --sharp \
	: -n 1 $(SNAIL03_V100_1_FLAGS) $(UCC_FLAGS) $(SHARP_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) --sharp \
	: -n 1 $(TVM01_V100_0_FLAGS)   $(UCC_FLAGS) $(SHARP_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) --sharp \
	: -n 1 $(TVM01_V100_1_FLAGS)   $(UCC_FLAGS) $(SHARP_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) --sharp \
	: -n 1 $(TVM02_V100_0_FLAGS)   $(UCC_FLAGS) $(SHARP_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) --sharp \
	: -n 1 $(TVM02_V100_1_FLAGS)   $(UCC_FLAGS) $(SHARP_FLAGS) $(FLAGS) -x UCC_CL_BASIC_TLS=sharp,ucp -- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) --sharp
