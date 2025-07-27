#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <string>

struct Context {
    int rank;
    int size;
    int count;
    int* buff;
    int root;
};

double do_bcast(Context *ctx) {
    double elapsed_time = - MPI_Wtime();
    MPI_Bcast(
        ctx->buff, ctx->count, MPI_INT,
        ctx->root, MPI_COMM_WORLD
    );
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

double do_ibcast(Context *ctx) {
    MPI_Request request;
    double elapsed_time = - MPI_Wtime();
    MPI_Ibcast(
        ctx->buff, ctx->count, MPI_INT,
        ctx->root, MPI_COMM_WORLD, &request
    );
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

void check_buff(Context *ctx, int* host_buff) {
    // assert that the buff contains the expected values
    cudaMemcpy(host_buff, ctx->buff, ctx->count * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < ctx->count; ++i) {
        if (host_buff[i] != ctx->root) {
            fprintf(stderr, "rank: %d, buff[%d] is %d, expected %d\n", ctx->rank, i, host_buff[i], ctx->root);
            fflush(stderr);
            sleep(10);
            abort();
        }
    }
}

int main(int argc, char** argv) {
    Context ctx;
    ctx.count = 1024 * 1024 * 128;
    ctx.root = 0;

    bool debug = false;
    double elapsed_time;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--count" && i + 1 < argc) {
            ctx.count = std::stoi(argv[++i]);
        } else if (arg == "--root" && i + 1 < argc) {
            ctx.root = std::stoi(argv[++i]);
        } else if (arg == "--debug") {
            debug = true;
        }
    }

    if (debug) {
        sleep(20);
    }

    // initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);

    cudaMalloc((void**)&ctx.buff, ctx.count * sizeof(int));

    int* host_buff = (int*)malloc(ctx.count * sizeof(int));
    
    // Initialize buffer with rank value
    for (int i = 0; i < ctx.count; ++i) {
        host_buff[i] = ctx.rank;
    }
    cudaMemcpy(ctx.buff, host_buff, ctx.count * sizeof(int), cudaMemcpyHostToDevice);

    printf("rank %d: start ibcast with count %d, root: %d, buff: %p\n", ctx.rank, ctx.count, ctx.root, ctx.buff);
    elapsed_time = do_ibcast(&ctx);
    printf("rank %d: end ibcast, time: %f\n", ctx.rank, elapsed_time);
    check_buff(&ctx, host_buff);

    // Reset buffer for next test
    for (int i = 0; i < ctx.count; ++i) {
        host_buff[i] = ctx.rank;
    }
    cudaMemcpy(ctx.buff, host_buff, ctx.count * sizeof(int), cudaMemcpyHostToDevice);

    printf("rank %d: start bcast with count %d, root: %d, buff: %p\n", ctx.rank, ctx.count, ctx.root, ctx.buff);
    elapsed_time = do_bcast(&ctx);
    printf("rank %d: end bcast, time: %f\n", ctx.rank, elapsed_time);
    check_buff(&ctx, host_buff);

    printf("rank %d: success\n", ctx.rank);

    // sleep(30);
    fflush(stdout);

    // cleanup
    cudaFree(ctx.buff);
    free(host_buff);

    MPI_Finalize();

    return 0;
}
