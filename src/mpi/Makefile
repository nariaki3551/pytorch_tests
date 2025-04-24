
SHARP_OPS ?= 
MCAST_OPS ?= 

CXX = mpicxx
INCLUDE_DIRS = -I/usr/local/cuda/include
CXX_FLAGS = -g -O0
LIB_DIRS = -L/usr/local/cuda/lib64
LIBS = -lcudart -lmpi
BIN_DIR = ../../bin
HOSTFILE ?= ../../hostfile
MPIRUN := $(shell which mpirun)

SRCS = check_cuda_support.cpp test_allgather.cpp test_allgather_cuda.cpp test_reducescatter.cpp test_reducescatter_cuda.cpp

BINS = $(SRCS:.cpp=)
TARGETS = $(addprefix $(BIN_DIR)/, $(addsuffix _mpi, $(BINS)))

.PHONY: all clean

all: $(TARGETS)

$(BIN_DIR)/%_mpi: %.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $< $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS)

run_test: run_cuda_support_check_mpi run_test_allgather_mpi run_test_allgather_cuda_mpi run_test_reducescatter_mpi run_test_reducescatter_cuda_mpi
run_test_mcast: run_test_allgather_mcast_mpi
run_sharp_test: run_test_reducescatter_sharp_mpi run_test_reducescatter_sharp_cuda_mpi

run_cuda_support_check_mpi: $(BIN_DIR)/check_cuda_support_mpi
	$(MPIRUN) -np 1 $<

run_test_allgather_mpi: $(BIN_DIR)/test_allgather_mpi
	$(MPIRUN) -n 2 $<

run_test_allgather_cuda_mpi: $(BIN_DIR)/test_allgather_cuda_mpi
	$(MPIRUN) -n 2 $<

un_test_reducescatter_mpi: $(BIN_DIR)/test_reducescatter_mpi
	$(MPIRUN) -n 2 $<

run_test_allgather_mcast_mpi: $(BIN_DIR)/test_allgather_mpi
	$(MPIRUN) -n 2 $(MCAST_OPS) -- $<

run_test_reducescatter_sharp_mpi: $(BIN_DIR)/test_reducescatter_mpi
	$(MPIRUN) -n 2 $(SHARP_OPS) -- $<

run_test_reducescatter_cuda_mpi: $(BIN_DIR)/test_reducescatter_cuda_mpi
	$(MPIRUN) -n 2 $<

run_test_reducescatter_sharp_cuda_mpi: $(BIN_DIR)/test_reducescatter_cuda_mpi
	$(MPIRUN) -n 2 $(SHARP_OPS) $<

clean:
	rm -f $(TARGETS)
