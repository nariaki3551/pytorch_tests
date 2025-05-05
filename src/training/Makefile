PROFILE ?= 0
DEBUG ?= 0

ifeq ($(DEBUG), 1)
	DEBUG_FLAGS := TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_SHOW_CPP_STACKTRACES=0
endif
PROFILE_FLAGS =
ifeq ($(PROFILE), 1)
	PROFILE_FLAGS += --profile
endif

run_fsdp_all: run_test_fsdp_mpi run_test_fsdp_nccl run_test_fsdp_ucc run_test_fsdp_gloo
run_fsdp2_all: run_test_fsdp2_mpi run_test_fsdp2_nccl run_test_fsdp2_ucc run_test_fsdp2_gloo

run_test_fsdp_mpi:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun -n 2 \
		-- python3 test_fsdp.py --model_scale 1 --num_epochs 100 --backend mpi --device cuda

NUM_EPOCHS := 3
MODEL_SCALE := 1
TEST_FSDP_ARGS = --model_scale $(MODEL_SCALE) --num_epochs $(NUM_EPOCHS) --backend mpi --device cuda
run_test_fsdp_mpi_multinode:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun \
		  -n 1 --host snail21:1 --mca plm_rsh_args "-p 2222" \
			--mca orte_keep_fqdn_hostnames true \
			--mca btl_tcp_if_include eno1 \
			-- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
		: -n 1 --host snail21:1 --mca plm_rsh_args "-p 2222" \
			--mca orte_keep_fqdn_hostnames true \
			--mca btl_tcp_if_include eno1 \
			-- python3 test_fsdp.py --device_id 1 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
		: -n 1 --host snail24:1 --mca plm_rsh_args "-p 2222" \
			--mca orte_keep_fqdn_hostnames true \
			--mca btl_tcp_if_include eno1 \
			-- python3 test_fsdp.py --device_id 0 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
		: -n 1 --host snail24:1 --mca plm_rsh_args "-p 2222" \
			--mca orte_keep_fqdn_hostnames true \
			--mca btl_tcp_if_include eno1 \
			-- python3 test_fsdp.py --device_id 1 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
		: -n 1 --host snail24:1 --mca plm_rsh_args "-p 2222" \
			--mca orte_keep_fqdn_hostnames true \
			--mca btl_tcp_if_include eno1 \
			-- python3 test_fsdp.py --device_id 2 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS) \
		: -n 1 --host snail24:1 --mca plm_rsh_args "-p 2222" \
			--mca orte_keep_fqdn_hostnames true \
			--mca btl_tcp_if_include eno1 \
			-- python3 test_fsdp.py --device_id 3 $(TEST_FSDP_ARGS) $(PROFILE_FLAGS)

run_test_fsdp_mpi_multinode_hello:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun \
		  -n 6 --host snail21:2,snail24:4 --mca plm_rsh_args "-p 2222" \
		  --mca btl ^openib \
		  --mca btl_tcp_if_include eno1 \
		  ./hello


run_test_fsdp_nccl:
	$(DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp.py --model_scale 1 --num_epochs 3 --backend nccl --device cuda

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
