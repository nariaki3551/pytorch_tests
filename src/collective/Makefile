SHARP_OPS ?= 
MCAST_OPS ?= 
MPIRUN := $(shell which mpirun)
PYTHON := $(shell which python3)

run_test_mcast: run_test_mcast_allgather run_test_mcast_allgather_cuda
run_test_sharp: run_test_sharp_reducescatter run_test_sharp_reducescatter_cuda

run_test_allgather:
	$(MPIRUN) -n 2 -- $(PYTHON) ./test_allgather.py

run_test_mcast_allgather:
	$(MPIRUN) -n 2 $(MCAST_OPS) -- $(PYTHON) ./test_allgather.py --device cpu

run_test_mcast_allgather_cuda:
	$(MPIRUN) -n 2 $(MCAST_OPS) -- $(PYTHON) ./test_allgather.py --device cuda

run_test_reducescatter:
	$(MPIRUN) -n 2 -- $(PYTHON) ./test_reducescatter.py

run_test_sharp_reducescatter:
	$(MPIRUN) -n 2 $(SHARP_OPS) -- $(PYTHON) ./test_reducescatter.py --device cpu --count 128

run_test_sharp_reducescatter_cuda:
	$(MPIRUN) -n 2 $(SHARP_OPS) -- $(PYTHON) ./test_reducescatter.py --device cuda --count 128
