SHARP_VERBOSE ?= 0
MCAST_VERBOSE ?= 0

.PHONY: run_test_allgather run_test_fsdp run_cuda_support_check_mpi run_check_ucc_sharp_support


ifeq ($(SHARP_VERBOSE), 1)
SHARP_VERBOSE_OPTS = --mca coll_ucc_verbose 3 -x UCC_LOG_LEVEL=trace -x UCC_TL_LOG_LEVEL=trace -x UCX_LOG_LEVEL=warning -x SHARP_COLL_LOG_LEVEL=5 -x UCC_TL_SHARP_VERBOSE=3
else
SHARP_VERBOSE_OPTS =
endif

ifeq ($(MCAST_VERBOSE), 1)
MCAST_VERBOSE_OPTS = -x UCC_LOG_LEVEL=trace -x UCC_TL_LOG_LEVEL=trace -x UCX_LOG_LEVEL=warning
else
MCAST_VERBOSE_OPTS =
endif

SHARP_OPS = --host snail02:1,snail03:1 \
	    --mca plm_rsh_args \"-p 2222\" \
		-x CUDA_VISIBLE_DEVICES=1,0 \
		--mca coll_ucc_enable 1 --mca coll_ucc_priority 100 -x UCC_MIN_TEAM_SIZE=2 \
		-x UCC_CL_BASIC_TLS=sharp,ucp \
		-x UCC_TL_SHARP_TUNE=reduce_scatter:inf\#allreduce:inf \
		-x SHARP_COLL_ENABLE_SAT=1 -x UCC_TL_SHARP_MIN_TEAM_SIZE=2 -x UCC_TL_SHARP_TEAM_MAX_PPN=2 \
		-x UCC_TL_SHARP_DEVICES=mlx5_2 \
		--mca btl_openib_if_include mlx5_2:1 \
		$(SHARP_VERBOSE_OPTS)

MCAST_OPS = --host snail02:1,snail03:1 \
	    --mca plm_rsh_args \"-p 2222\" \
		-x CUDA_VISIBLE_DEVICES=1,0 \
		--mca coll_ucc_enable 1 --mca coll_ucc_priority 100 -x UCC_MIN_TEAM_SIZE=2 \
		-x UCC_CL_BASIC_TLS=spin,ucp \
		-x UCC_TL_SPIN_TUNE=allgather:inf \
		-x UCC_TL_SPIN_ALLGATHER_MCAST_ROOTS=1 -x UCC_TL_SPIN_LINK_BW=12.5 -x UCC_TL_SPIN_MAX_RECV_BUF_SIZE=2147483648 \
		-x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 \
		--mca btl_openib_if_include mlx5_2:1 \
		$(MCAST_VERBOSE_OPTS)

build_mpi:
	$(MAKE) -C ./src/mpi

build_ucc:
	$(MAKE) -C ./src/ucc

run_test_torch:
	$(MAKE) -C ./src/collective run_test_allgather
	$(MAKE) -C ./src/collective run_test_reducescatter

run_test_mcast_torch:
	$(MAKE) -C ./src/collective run_test_mcast_allgather MCAST_OPS="$(MCAST_OPS)"

run_test_sharp_torch:
	$(MAKE) -C ./src/collective run_test_sharp_reducescatter SHARP_OPS="$(SHARP_OPS)"

run_test_fsdp:
	mpirun -n 2 python3 ./src/training/test_fsdp.py --model_scale 1 --num_epochs 10

run_test_mpi: build_mpi
	$(MAKE) -C ./src/mpi run_test

run_test_sharp_mpi: build_mpi
	$(MAKE) -C ./src/mpi run_sharp_test SHARP_OPS="$(SHARP_OPS)"

run_test_mcast_mpi: build_mpi
	$(MAKE) -C ./src/mpi run_test_mcast_mpi MCAST_OPS="$(MCAST_OPS)"

run_test_ucc: build_ucc
	$(MAKE) -C ./src/ucc run_check_sharp_support_ucc
