#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>

struct Context {
    int rank;
    int size;
    int count;
    int* send_buff;
    int* recv_buff;
};

double do_reduce_scatter(Context *ctx) {
    double elapsed_time = - MPI_Wtime();
    MPI_Reduce_scatter_block(
        ctx->send_buff, ctx->recv_buff, ctx->count,
        MPI_INT, MPI_SUM,
	MPI_COMM_WORLD
    );
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

double do_ireduce_scatter(Context *ctx) {
    MPI_Request request;
    double elapsed_time = - MPI_Wtime();
    MPI_Ireduce_scatter_block(
        ctx->send_buff, ctx->recv_buff, ctx->count,
        MPI_INT, MPI_SUM,
	MPI_COMM_WORLD, &request
    );
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

void check_recv_buff(Context *ctx) {
    int sum = 0;
    for (int i = 0; i < ctx->size; ++i) {
        sum += i;
    }
    for (int i = 0; i < ctx->count; ++i) {
        if (ctx->recv_buff[i] != sum) {
            printf("rank: %d, recv_buff[%d] is %d, expected %d\n", ctx->rank, i, ctx->recv_buff[i], sum);
            abort();
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

    posix_memalign((void**)&ctx.send_buff, 4096, ctx.count * ctx.size * sizeof(int));
    posix_memalign((void**)&ctx.recv_buff, 4096, ctx.count * sizeof(int));
    for (int i = 0; i < ctx.count * ctx.size; ++i) {
        ctx.send_buff[i] = ctx.rank;
    }

    printf("rank %d: start reducescatter with count %d\n", ctx.rank, ctx.count);
    elapsed_time = do_reduce_scatter(&ctx);
    printf("rank %d: end reducescatter, time: %f\n", ctx.rank, elapsed_time);
    check_recv_buff(&ctx);

    printf("rank %d: start ireducescatter with count %d\n", ctx.rank, ctx.count);
    elapsed_time = do_ireduce_scatter(&ctx);
    printf("rank %d: end ireducescatter, time: %f\n", ctx.rank, elapsed_time);
    check_recv_buff(&ctx);

    printf("rank %d: finish all\n", ctx.rank);

    // sleep(30);
    fflush(stdout);

    // cleanup
    free(ctx.send_buff);
    free(ctx.recv_buff);

    MPI_Finalize();

    return 0;
}
