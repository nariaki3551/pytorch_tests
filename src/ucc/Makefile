USE_CUDA ?= 1
USE_SHARP ?= 1

CC = mpicc
CFLAGS = -I/usr/local/cuda/include -g -O0
LIB_DIRS = -L/usr/local/cuda/lib64
LIBS = -lcudart -lmpi -lucc
BIN_DIR = ../../bin

SRCS_C =
ifeq ($(USE_SHARP), 1)
ifeq ($(USE_CUDA), 1)
	SRCS_C += check_sharp_support.c
endif
endif

BINS = $(SRCS_C:.c=)
TARGETS = $(addprefix $(BIN_DIR)/, $(addsuffix _ucc, $(BINS)))

.PHONY: all clean run_check_sharp_support_ucc

all: $(TARGETS)

$(BIN_DIR)/%_ucc: %.c
	$(CC) $(CFLAGS) -o $@ $< $(LIB_DIRS) $(LIBS)

run_check_sharp_support_ucc: $(BIN_DIR)/check_sharp_support_ucc
	mpirun -np 1 --allow-run-as-root $<

clean:
	rm -f $(TARGETS)
