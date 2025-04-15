CC = mpicc
CXX = mpicxx
CFLAGS = -I/usr/local/cuda/include -g -O0
CXXFLAGS = -I/usr/local/cuda/include -g -O0
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lmpi
BIN_DIR = ../../bin

SRCS_CPP = mpi_cuda_support_check.cpp
SRCS_C   = ucc_sharp_support_check.c
OBJS_CPP = $(SRCS_CPP:.cpp=.o)
OBJS_C   = $(SRCS_C:.c=.o)
TARGETS = $(BIN_DIR)/mpi_cuda_support_check $(BIN_DIR)/ucc_sharp_support_check

.PHONY: all clean run_mpi_cuda_support_check run_ucc_sharp_support_check

all: $(BIN_DIR)/mpi_cuda_support_check
# all: $(TARGETS)  ← どちらでもOK

$(BIN_DIR)/mpi_cuda_support_check: mpi_cuda_support_check.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

$(BIN_DIR)/ucc_sharp_support_check: ucc_sharp_support_check.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -lucc

run_mpi_cuda_support_check: $(BIN_DIR)/mpi_cuda_support_check
	mpirun -np 1 --allow-run-as-root $<

run_ucc_sharp_support_check: $(BIN_DIR)/ucc_sharp_support_check
	mpirun -np 1 --allow-run-as-root $<

clean:
	rm -f $(TARGETS)