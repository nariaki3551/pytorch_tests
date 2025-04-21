include ../../make/mpi_config.mk

CXX = mpicxx
INCLUDE_DIRS = -I/usr/local/cuda/include
CXX_FLAGS = -g -O0
LIB_DIRS = -L/usr/local/cuda/lib64
LIBS = -lcudart -lmpi
BIN_DIR = ../../bin
HOSTFILE ?= ../../hostfile

SRCS = check_cuda_support.cpp test_allgather.cpp test_allgather_cuda.cpp

BINS = $(SRCS:.cpp=)
TARGETS = $(addprefix $(BIN_DIR)/, $(addsuffix _mpi, $(BINS)))

.PHONY: all clean

all: $(TARGETS)

$(BIN_DIR)/%_mpi: %.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $< $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS)

run_test_all: run_cuda_support_check_mpi run_test_allgather_mpi run_test_allgather_cuda_mpi

run_cuda_support_check_mpi: $(BIN_DIR)/check_cuda_support_mpi
	mpirun -np 1 $(MPI_MCAST_OPTS) $(MCAST_VERBOSE_OPTS) $<

run_test_allgather_mpi: $(BIN_DIR)/test_allgather_mpi
	mpirun -n 2 $(MPI_MCAST_OPTS) $(MCAST_VERBOSE_OPTS) $<

run_test_allgather_mpi_verbose: $(BIN_DIR)/test_allgather_mpi
	mpirun -n 2 --allow-run-as-root \
		--mca coll_ucc_enable 1 --mca coll_ucc_priority 100 -x UCC_MIN_TEAM_SIZE=2 \
		-x UCC_CL_BASIC_TLS=spin,ucp \
		-x UCC_TL_SPIN_TUNE=allgather:inf \
		--mca coll_ucc_verbose 3 -x UCC_LOG_LEVEL=trace -x UCC_TL_LOG_LEVEL=trace \
		-- $<

run_test_allgather_cuda_mpi: $(BIN_DIR)/test_allgather_cuda_mpi
	mpirun -n 2 $(MPI_MCAST_OPTS) $(MCAST_VERBOSE_OPTS) $<

clean:
	rm -f $(TARGETS)
