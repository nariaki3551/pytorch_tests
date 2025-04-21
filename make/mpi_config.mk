# Multicast options

HOSTFILE ?= ./hostfile
MCAST ?= 0
MCAST_VERBOSE ?= 0

ifeq ($(MCAST), 1)
MPI_MCAST_OPTS = \
	--hostfile $(HOSTFILE) \
	--mca coll_ucc_enable 1 --mca coll_ucc_priority 100 -x UCC_MIN_TEAM_SIZE=2 \
	-x UCC_CL_BASIC_TLS=spin,ucp \
	-x UCC_TL_SPIN_ALLGATHER_MCAST_ROOTS=1 -x UCC_TL_SPIN_LINK_BW=12.5 -x UCC_TL_SPIN_MAX_RECV_BUF_SIZE=2147483648 \
	-x CUDA_VISIBLE_DEVICES=1,0 \
	-x UCC_TL_SPIN_IB_DEV_NAME=mlx5_2 \
	--mca btl_openib_if_include mlx5_2:1 \
	-x UCC_TL_SHARP_DEVICES=mlx5_2
else
MPI_MCAST_OPTS =
endif

ifeq ($(MCAST_VERBOSE), 1)
MCAST_VERBOSE_OPTS = --mca coll_ucc_verbose 3 -x UCC_LOG_LEVEL=trace -x UCC_TL_LOG_LEVEL=trace -x UCX_LOG_LEVEL=warning
else
MCAST_VERBOSE_OPTS =
endif