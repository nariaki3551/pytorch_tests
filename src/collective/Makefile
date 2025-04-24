SHARP_OPS ?= 
MCAST_OPS ?= 

run_test_allgather:
	mpirun -n 2 python3 ./test_allgather.py

run_test_mcast_allgather:
	mpirun -n 2 $(MCAST_OPS) python3 ./test_allgather.py

run_test_reducescatter:
	mpirun -n 2 python3 ./test_reducescatter.py

run_test_sharp_reducescatter:
	mpirun -n 2 $(SHARP_OPS) python3 ./test_reducescatter.py
