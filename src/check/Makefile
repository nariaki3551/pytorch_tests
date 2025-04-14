CXX = mpicxx
CXXFLAGS = -I/usr/local/cuda/include -g -O0
LDFLAGS = -L/usr/local/cuda/lib64 -lcudart -lmpi -lopen-pal -lopen-rte -lcuda
BIN_DIR = ../../bin

SRCS = cuda_support_check.cpp
BINS = $(SRCS:.cpp=)

all: $(BINS)

%: %.cpp
	$(CXX) $(CXXFLAGS) -o $(BIN_DIR)/$@ $< $(LDFLAGS)

run_cuda_support_check:
	mpirun -np 1 --allow-run-as-root $(BIN_DIR)/cuda_support_check

clean:
	rm -f $(addprefix $(BIN_DIR)/, $(BINS))
