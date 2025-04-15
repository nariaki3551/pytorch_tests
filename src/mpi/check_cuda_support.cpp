#include <mpi.h>
#include <stdio.h>
#include <mpi-ext.h> 
#include <cuda_runtime.h>

int main(int argc, char **argv) {
    int provided, cuda_supported;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int n;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("cudaGetDeviceCount: %d\n", n);

#if defined(MPIX_CUDA_AWARE_SUPPORT)
    printf("MPIX_CUDA_AWARE_SUPPORT is defined.\n");
    cuda_supported = MPIX_Query_cuda_support();
#else
    printf("MPIX_CUDA_AWARE_SUPPORT is not defined.\n");
    cuda_supported = 0;
#endif

    if (cuda_supported) {
        printf("MPI CUDA support is ENABLED.\n");
    } else {
        printf("MPI CUDA support is DISABLED.\n");
    }

    MPI_Finalize();
    return 0;
}
