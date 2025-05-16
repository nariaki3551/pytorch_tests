PROFILE ?= 0
DEBUG ?= 0
BACKEND ?= mpi
DEVICE ?= cuda
NUM_EPOCHS := 3
MODEL_SCALE := 1

TORCH_DEBUG_FLAGS =
ifeq ($(DEBUG), 1)
	TORCH_DEBUG_FLAGS += TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_SHOW_CPP_STACKTRACES=0
endif
PROFILE_FLAGS =
ifneq ($(PROFILE), 0)
	PROFILE_FLAGS += --profile $(PROFILE)
endif

run_fsdp_all: run_test_fsdp_mpi run_test_fsdp_nccl run_test_fsdp_ucc run_test_fsdp_gloo
run_fsdp2_all: run_test_fsdp2_mpi run_test_fsdp2_nccl run_test_fsdp2_ucc run_test_fsdp2_gloo

TEST_FSDP_ARGS = --model_scale $(MODEL_SCALE) --num_epochs $(NUM_EPOCHS) --backend $(BACKEND) --device $(DEVICE) 
SHARP_COMMON_FLAGS = \
	-x UCC_TL_SHARP_TUNE=reduce_scatter:inf\#reduce_scatterv:inf\#allreduce:inf \
	-x SHARP_COLL_ENABLE_SAT=1 \
	-x UCC_TL_SHARP_MIN_TEAM_SIZE=2 \
	-x UCC_TL_SHARP_TEAM_MAX_PPN=2
ifeq ($(BACKEND), nccl)
	SHARP_COMMON_FLAGS += \
		-x NCCL_COLLNET_ENABLE=1 \
		-x SHARP_COLL_LOCK_ON_COMM_INIT=1 \
		-x SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0
		# -x NCCL_ALGO=CollNet
endif
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

run_test_fsdp_mpi:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun -n 2 -- python3 test_fsdp.py $(TEST_FSDP_ARGS) $(PROFILE_FLAGS)

run_test_fsdp_mpi_multinode:
	mpirun \
			-n 1 --host snail01:1 \
			--mca plm_rsh_args "-p 2222" \
			-x UCC_CL_BASIC_TLS=ucp \
			-x CUDA_VISIBLE_DEVICES=3 \
			--mca btl_openib_if_include mlx5_1:1 \
			--mca btl_tcp_if_exclude lo,docker0,docker_gwbridge \
			-x RANK=0 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
			$(TORCH_DEBUG_FLAGS) \
			$(UCC_COMMON_FLAGS) \
			$(VERBOSE_FLAGS) \
			-- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
		: -n 1 --host snail02:1 \
			--mca plm_rsh_args "-p 2222" \
			-x UCC_CL_BASIC_TLS=ucp \
			-x CUDA_VISIBLE_DEVICES=1 \
			--mca btl_openib_if_include mlx5_2:1 \
			--mca btl_tcp_if_exclude lo,docker0,docker_gwbridge \
			-x RANK=1 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
			$(TORCH_DEBUG_FLAGS) \
			$(UCC_COMMON_FLAGS) \
			$(VERBOSE_FLAGS) \
			-- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS)
			# -- gdb -ex "run" --args python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \

run_test_fsdp_mpi_multinode_sharp:
	mpirun \
			-n 1 --host snail01:1 \
			--mca plm_rsh_args "-p 2222" \
			-x UCC_CL_BASIC_TLS=sharp,ucp \
			-x CUDA_VISIBLE_DEVICES=3 \
			--mca btl_openib_if_include mlx5_1:1 \
			-x UCC_TL_SHARP_DEVICES=mlx5_1 \
			-x RANK=0 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
			$(UCC_COMMON_FLAGS) \
			$(SHARP_COMMON_FLAGS) \
			$(VERBOSE_FLAGS) \
			-- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
		: -n 1 --host snail02:1 \
			--mca plm_rsh_args "-p 2222" \
			-x UCC_CL_BASIC_TLS=sharp,ucp \
			-x CUDA_VISIBLE_DEVICES=1 \
			-x UCC_TL_SHARP_DEVICES=mlx5_2 \
			-x RANK=1 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
			$(UCC_COMMON_FLAGS) \
			$(SHARP_COMMON_FLAGS) \
			$(VERBOSE_FLAGS) \
			-- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS)

run_test_fsdp_mpi_multinode_spin:
	mpirun \
			-n 1 --host snail01:1 \
			--mca plm_rsh_args "-p 2222" \
			-x UCC_CL_BASIC_TLS=spin,ucp \
			-x CUDA_VISIBLE_DEVICES=3 \
			-x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1 \
			--mca btl_openib_if_include mlx5_1:1 \
			--mca btl_tcp_if_exclude lo,docker0,docker_gwbridge \
			$(TORCH_DEBUG_FLAGS) \
			$(UCC_COMMON_FLAGS) \
			$(SPIN_COMMON_FLAGS) \
			$(VERBOSE_FLAGS) \
			-x RANK=0 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
			-- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
		: -n 1 --host snail02:1 \
			--mca plm_rsh_args "-p 2222" \
			-x UCC_CL_BASIC_TLS=spin,ucp \
			-x CUDA_VISIBLE_DEVICES=1 \
			-x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 \
			--mca btl_openib_if_include mlx5_2:1 \
			--mca btl_tcp_if_exclude lo,docker0,docker_gwbridge \
			-x RANK=1 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
			$(TORCH_DEBUG_FLAGS) \
			$(UCC_COMMON_FLAGS) \
			$(SPIN_COMMON_FLAGS) \
			$(VERBOSE_FLAGS) \
			-- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS)

run_test_fsdp_mpi_multinode_sharp_spin:
	mpirun \
			-n 1 --host snail01:1 \
			--mca plm_rsh_args "-p 2222" \
			-x UCC_CL_BASIC_TLS=sharp,spin,ucp \
			-x CUDA_VISIBLE_DEVICES=3 \
			-x UCC_TL_SPIN_IB_DEV_NAME=mlx5_1 \
			--mca btl_openib_if_include mlx5_1:1 \
			--mca btl_tcp_if_exclude lo,docker0,docker_gwbridge \
			-x RANK=0 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
			$(UCC_COMMON_FLAGS) \
			$(SHARP_COMMON_FLAGS) \
			$(SPIN_COMMON_FLAGS) \
			$(VERBOSE_FLAGS) \
			-- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
		: -n 1 --host snail02:1 \
			--mca plm_rsh_args "-p 2222" \
			-x CUDA_VISIBLE_DEVICES=1 \
			-x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 \
			--mca btl_openib_if_include mlx5_2:1 \
			--mca btl_tcp_if_exclude lo,docker0,docker_gwbridge \
			-x RANK=1 -x WORLD_SIZE=2 -x MASTER_ADDR=snail01 -x MASTER_PORT=12345 \
			$(UCC_COMMON_FLAGS) \
			$(SHARP_COMMON_FLAGS) \
			$(SPIN_COMMON_FLAGS) \
			$(VERBOSE_FLAGS) \
			-- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS)

run_test_fsdp_mpi_multinode_hello:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun \
		  -n 6 --host snail21:2,snail24:4 --mca plm_rsh_args "-p 2222" \
		  --mca btl ^openib \
		  --mca btl_tcp_if_include eno1 \
		  ./hello

run_test_fsdp_ucc:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp.py --model_scale 1 --num_epochs 3 --backend ucc --device cuda

run_test_fsdp_gloo:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp.py --model_scale 1 --num_epochs 3 --backend gloo --device cuda


run_test_fsdp2_mpi:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun -n 2 \
		-- python3 test_fsdp2.py --model_scale 1 --num_epochs 3 --backend mpi --device cuda

run_test_fsdp2_nccl:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp2.py --model_scale 1 --num_epochs 3 --backend nccl --device cuda

run_test_fsdp2_ucc:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp2.py --model_scale 1 --num_epochs 3 --backend ucc --device cuda

run_test_fsdp2_gloo:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp2.py --model_scale 1 --num_epochs 3 --backend gloo --device cuda
