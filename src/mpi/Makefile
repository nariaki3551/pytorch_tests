CXX = mpicxx
INCLUDE_DIRS = -I/usr/local/cuda/include
CXX_FLAGS = -g -O0
LIB_DIRS = -L/usr/local/cuda/lib64
LIBS = -lcudart -lmpi
BIN_DIR = ../../bin

SRCS = check_cuda_support.cpp test_allgather.cpp

BINS = $(SRCS:.cpp=)
TARGETS = $(addprefix $(BIN_DIR)/, $(addsuffix _mpi, $(BINS)))

.PHONY: all clean

all: $(TARGETS)

$(BIN_DIR)/%_mpi: %.cpp
	$(CXX) $(CXX_FLAGS) -o $@ $< $(INCLUDE_DIRS) $(LIB_DIRS) $(LIBS)

run_cuda_support_check_mpi: $(BIN_DIR)/check_cuda_support_mpi
	mpirun -np 1 --allow-run-as-root $<

run_test_allgather_mpi: $(BIN_DIR)/test_allgather_mpi
	mpirun -n 2 --allow-run-as-root $<

clean:
	rm -f $(TARGETS)