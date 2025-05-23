DEBUG ?= 0
BACKEND ?= mpi
DEVICE ?= cuda

TORCH_DEBUG_FLAGS =
ifeq ($(DEBUG), 1)
	TORCH_DEBUG_FLAGS += TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL
endif

# run_all: run_allgather_all run_reduce_all run_reduce_scatter_all
# run_allgather_all: run_allgather_mpi run_allgather_ucc run_allgather_nccl run_allgather_gloo
# run_reduce_all: run_reduce_mpi run_reduce_ucc run_reduce_nccl run_reduce_gloo
# run_reduce_scatter_all: run_reduce_scatter_mpi run_reduce_scatter_ucc run_reduce_scatter_nccl run_reduce_scatter_gloo
# run_allreduce_all: run_allreduce_mpi run_allreduce_ucc run_allreduce_nccl run_allreduce_gloo

SHARP_COMMON_FLAGS = \
	-x UCC_TL_SHARP_TUNE=reduce_scatter:inf\#reduce_scatterv:inf\#allreduce:inf \
	-x SHARP_COLL_ENABLE_SAT=1 \
	-x UCC_TL_SHARP_MIN_TEAM_SIZE=2 \
	-x UCC_TL_SHARP_TEAM_MAX_PPN=2 \
	-x NCCL_COLLNET_ENABLE=1 \
	-x SHARP_COLL_LOCK_ON_COMM_INIT=1 \
	-x SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0 \
	-x NCCL_SHARP_GROUP_SIZE_THRESH=1 \
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

# run_allgather_mpi:
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_allgather.py --backend mpi --device cpu
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_allgather.py --backend mpi --device cuda
# run_allgather_ucc:
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allgather.py --backend ucc --device cpu
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allgather.py --backend ucc --device cuda
# run_allgather_nccl:
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allgather.py --backend nccl --device cuda
# run_allgather_gloo:
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allgather.py --backend gloo --device cpu


# run_reduce_mpi:
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_reduce.py --backend mpi --device cpu
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_reduce.py --backend mpi --device cuda
# run_reduce_ucc:
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_reduce.py --backend ucc --device cpu
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_reduce.py --backend ucc --device cuda
# run_reduce_nccl:
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_reduce.py --backend nccl --device cuda
# run_reduce_gloo:
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_reduce.py --backend gloo --device cpu


# run_reduce_scatter_mpi:
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_reduce_scatter.py --backend mpi --device cpu
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_reduce_scatter.py --backend mpi --device cuda

run_allgather_multinode_ucp:
	mpirun \
		-n 1 \
		-x UCC_CL_BASIC_TLS=ucp \
		-x RANK=0 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
		$(SNAIL01_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) \
		-- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
		: \
		-n 1 \
		-x UCC_CL_BASIC_TLS=ucp \
		-x RANK=1 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
		$(SNAIL02_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) \
		-- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE)

run_allgather_multinode_spin:
	mpirun \
		-n 1 \
		-x UCC_CL_BASIC_TLS=spin,ucp \
		-x RANK=0 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
		$(SNAIL01_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(SPIN_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) \
		-- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE) \
		: \
		-n 1 \
		-x UCC_CL_BASIC_TLS=spin,ucp \
		-x RANK=1 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
		$(SNAIL02_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(SPIN_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) \
		-- python3 test_allgather.py --backend $(BACKEND) --device $(DEVICE)


run_reduce_scatter_multinode_ucp:
	mpirun \
		-n 1 \
		-x UCC_CL_BASIC_TLS=ucp \
		-x RANK=0 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
		$(SNAIL01_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) \
		-- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) \
		: \
		-n 1 \
		-x UCC_CL_BASIC_TLS=ucp \
		-x RANK=1 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
		$(SNAIL02_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) \
		-- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE)

run_reduce_scatter_multinode_sharp:
	mpirun \
		-n 1 \
		-x UCC_CL_BASIC_TLS=sharp,ucp \
		-x RANK=0 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
		$(SNAIL01_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(SHARP_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) \
		-- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE) \
		: \
		-n 1 \
		-x UCC_CL_BASIC_TLS=sharp,ucp \
		-x RANK=1 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
		$(SNAIL02_V100_1_FLAGS) $(UCC_COMMON_FLAGS) $(SHARP_COMMON_FLAGS) $(VERBOSE_FLAGS) $(TORCH_DEBUG_FLAGS) \
		-- python3 test_reduce_scatter.py --backend $(BACKEND) --device $(DEVICE)


# run_allreduce_mpi:
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_allreduce.py --backend mpi --device cpu
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 mpirun -n 2 -- python3 test_allreduce.py --backend mpi --device cuda
# run_allreduce_ucc:
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allreduce.py --backend ucc --device cpu
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allreduce.py --backend ucc --device cuda
# run_allreduce_nccl:
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allreduce.py --backend nccl --device cuda
# run_allreduce_gloo:
# 	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 -- test_allreduce.py --backend gloo --device cpu
