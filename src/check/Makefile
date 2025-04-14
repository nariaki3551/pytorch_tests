CC = mpicc
CXX = mpicxx
CFLAGS = -I/usr/local/cuda/include -g -O0
CXXFLAGS = -I/usr/local/cuda/include -g -O0
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lmpi
BIN_DIR = ../../bin

SRCS = mpi_cuda_support_check.cpp ucc_sharp_support_check.c
BINS = $(SRCS:.cpp=) $(SRCS:.c=)

.PHONY: all clean

all: build_mpi_cuda_support_check build_ucc_sharp_support_check

build_mpi_cuda_support_check:
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/mpi_cuda_support_check mpi_cuda_support_check.cpp $(LDFLAGS)

build_ucc_sharp_support_check:
	$(CC) $(CFLAGS) -o $(BIN_DIR)/ucc_sharp_support_check ucc_sharp_support_check.c $(LDFLAGS) -lucc

run_mpi_cuda_support_check:
	mpirun -np 1 --allow-run-as-root $(BIN_DIR)/mpi_cuda_support_check

run_ucc_sharp_support_check:
	mpirun -np 1 --allow-run-as-root $(BIN_DIR)/ucc_sharp_support_check

clean:
	rm -f $(BIN_DIR)/mpi_cuda_support_check $(BIN_DIR)/ucc_sharp_support_check
