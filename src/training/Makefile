PYTORCH_DEBUG ?= 0

ifeq ($(PYTORCH_DEBUG), 1)
	PYTORCH_DEBUG_FLAGS := TORCH_CPP_LOG_LEVEL=INFO TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_SHOW_CPP_STACKTRACES=0
endif

run_fsdp_all: run_test_fsdp_mpi run_test_fsdp_nccl run_test_fsdp_ucc run_test_fsdp_gloo
run_fsdp2_all: run_test_fsdp2_mpi run_test_fsdp2_nccl run_test_fsdp2_ucc run_test_fsdp2_gloo

run_test_fsdp_mpi:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun -n 2 \
		-- python3 test_fsdp.py --model_scale 1 --num_epochs 100 --backend mpi --device cuda

NUM_EPOCHS := 3
MODEL_SCALE := 1
TEST_FSDP_ARGS = --model_scale $(MODEL_SCALE) --num_epochs $(NUM_EPOCHS) --backend mpi --device cuda
run_test_fsdp_mpi_multinode:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun \
		  -n 1 --host snail21:1 --mca plm_rsh_args "-p 2222" \
			--mca orte_keep_fqdn_hostnames true \
			--mca btl_tcp_if_include eno1 \
			-- python3 test_fsdp.py $(TEST_FSDP_ARGS) --device_id 0 \
		: -n 1 --host snail21:1 --mca plm_rsh_args "-p 2222" \
			--mca orte_keep_fqdn_hostnames true \
			--mca btl_tcp_if_include eno1 \
			-- python3 test_fsdp.py $(TEST_FSDP_ARGS) --device_id 1 \
		: -n 1 --host snail24:1 --mca plm_rsh_args "-p 2222" \
			--mca orte_keep_fqdn_hostnames true \
			--mca btl_tcp_if_include eno1 \
			-- python3 test_fsdp.py $(TEST_FSDP_ARGS) --device_id 0 \
		: -n 1 --host snail24:1 --mca plm_rsh_args "-p 2222" \
			--mca orte_keep_fqdn_hostnames true \
			--mca btl_tcp_if_include eno1 \
			-- python3 test_fsdp.py $(TEST_FSDP_ARGS) --device_id 1 \
		: -n 1 --host snail24:1 --mca plm_rsh_args "-p 2222" \
			--mca orte_keep_fqdn_hostnames true \
			--mca btl_tcp_if_include eno1 \
			-- python3 test_fsdp.py $(TEST_FSDP_ARGS) --device_id 2 \
		: -n 1 --host snail24:1 --mca plm_rsh_args "-p 2222" \
			--mca orte_keep_fqdn_hostnames true \
			--mca btl_tcp_if_include eno1 \
			-- python3 test_fsdp.py $(TEST_FSDP_ARGS) --device_id 3

run_test_fsdp_mpi_multinode_hello:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun \
		  -n 6 --host snail21:2,snail24:4 --mca plm_rsh_args "-p 2222" \
		  --mca btl ^openib \
		  --mca btl_tcp_if_include eno1 \
		  ./hello


run_test_fsdp_nccl:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp.py --model_scale 1 --num_epochs 3 --backend nccl --device cuda

run_test_fsdp_ucc:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp.py --model_scale 1 --num_epochs 3 --backend ucc --device cuda

run_test_fsdp_gloo:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp.py --model_scale 1 --num_epochs 3 --backend gloo --device cuda


run_test_fsdp2_mpi:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		mpirun -n 2 \
		-- python3 test_fsdp2.py --model_scale 1 --num_epochs 3 --backend mpi --device cuda

run_test_fsdp2_nccl:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp2.py --model_scale 1 --num_epochs 3 --backend nccl --device cuda

run_test_fsdp2_ucc:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp2.py --model_scale 1 --num_epochs 3 --backend ucc --device cuda

run_test_fsdp2_gloo:
	$(PYTORCH_DEBUG_FLAGS) OMP_NUM_THREADS=1 \
		torchrun --nproc-per-node 2 --master_addr="0.0.0.0" --master_port=23456 \
		-- test_fsdp2.py --model_scale 1 --num_epochs 3 --backend gloo --device cuda
