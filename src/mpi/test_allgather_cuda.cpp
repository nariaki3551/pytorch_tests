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
    int* send_buff;
    int* recv_buff;
};

double do_allgather(Context *ctx) {
    double elapsed_time = - MPI_Wtime();
    MPI_Allgather(
        ctx->send_buff, ctx->count, MPI_INT,
        ctx->recv_buff, ctx->count, MPI_INT,
        MPI_COMM_WORLD
    );
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

double do_iallgather(Context *ctx) {
    MPI_Request request;
    double elapsed_time = - MPI_Wtime();
    MPI_Iallgather(
        ctx->send_buff, ctx->count, MPI_INT,
        ctx->recv_buff, ctx->count, MPI_INT,
        MPI_COMM_WORLD, &request
    );
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

void check_recv_buff(Context *ctx, int* host_buff) {
    // assert that the recv_buff is the same as the send_buff
    cudaMemcpy(host_buff, ctx->recv_buff, ctx->count * ctx->size * sizeof(int), cudaMemcpyDeviceToHost);

    int i = 0;
    for (int rank = 0; rank < ctx->size; ++rank) {
        for (int k = 0; k < ctx->count; ++k) {
            if (host_buff[i] != rank) {
                fprintf(stderr, "rank: %d, recv_buff[%d] is %d, expected %d\n", ctx->rank, i, ctx->recv_buff[i], rank);
                fflush(stderr);
                abort();
            }
            i++;
        }
    }
}

int main(int argc, char** argv) {
    Context ctx;
    ctx.count = 1024 * 1024 * 128;

    bool debug = false;
    double elapsed_time;

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--count" && i + 1 < argc) {
            ctx.count = std::stoi(argv[++i]);
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

    cudaMalloc((void**)&ctx.send_buff, ctx.count * sizeof(int));
    cudaMalloc((void**)&ctx.recv_buff, ctx.count * ctx.size * sizeof(int));

    int* host_buff = (int*)malloc(ctx.count * ctx.size * sizeof(int));
    for (int i = 0; i < ctx.count; ++i) {
        host_buff[i] = ctx.rank;
    }
    cudaMemcpy(ctx.send_buff, host_buff, ctx.count * sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 0; i < ctx.count * ctx.size; ++i) {
        host_buff[i] = -1;
    }
    cudaMemcpy(ctx.recv_buff, host_buff, ctx.count * ctx.size * sizeof(int), cudaMemcpyHostToDevice);


    printf("rank %d: start iallgather with count %d\n", ctx.rank, ctx.count);
    elapsed_time = do_iallgather(&ctx);
    printf("rank %d: end iallgather, time: %f\n", ctx.rank, elapsed_time);
    check_recv_buff(&ctx, host_buff);

    printf("rank %d: start allgather with count %d\n", ctx.rank, ctx.count);
    elapsed_time = do_allgather(&ctx);
    printf("rank %d: end allgather, time: %f\n", ctx.rank, elapsed_time);
    check_recv_buff(&ctx, host_buff);

    printf("rank %d: success\n", ctx.rank);

    // sleep(30);
    fflush(stdout);

    // cleanup
    cudaFreeHost(ctx.send_buff);
    cudaFreeHost(ctx.recv_buff);
    free(host_buff);

    MPI_Finalize();

    return 0;
}
