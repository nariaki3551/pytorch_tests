run_test_allgather:
	mpirun -n 2 --allow-run-as-root python3 ./src/collective/test_allgather.py

run_test_fsdp:
	mpirun -n 2 --allow-run-as-root python3 ./src/training/test_fsdp.py --model_scale 1 --num_epochs 10
