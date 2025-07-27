
for host in snail02 snail03 tvm01 tvm02
do
	COMMAND="scp -P 12345 /app/pytorch_tests/bin/allgather_perf_cuda_mpi ${host}:/app/pytorch_tests/bin/"
	echo $COMMAND
	$COMMAND
	COMMAND="scp -P 12345 /app/pytorch_tests/bin/reducescatter_perf_cuda_mpi ${host}:/app/pytorch_tests/bin/"
	echo $COMMAND
	$COMMAND
done
