#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>
#include <getopt.h>
#include <vector>

struct Context {
    int rank;
    int size;
    int count;
    int* send_buff;
    int* recv_buff;
};

double do_allgather(Context *ctx) {
    MPI_Request request;
    double elapsed_time = - MPI_Wtime();
    MPI_Allgather(
        ctx->send_buff, ctx->count, MPI_INT,
        ctx->recv_buff, ctx->count, MPI_INT,
        MPI_COMM_WORLD
    );
    elapsed_time += MPI_Wtime();
    return elapsed_time;
}

void check_recv_buff(Context *ctx) {
    // assert that the recv_buff is the same as the send_buff
    int i = 0;
    for (int rank = 0; rank < ctx->size; ++rank) {
        for (int k = 0; k < ctx->count; ++k) {
            if (ctx->recv_buff[i] != rank) {
                fprintf(stderr, "rank %d: recv_buff[%d] is %d, expected %d\n", ctx->rank, i, ctx->recv_buff[i], rank);
                fflush(stderr);
                abort();
            }
            i++;
        }
    }
}

int main(int argc, char** argv) {
    Context ctx;
    size_t min_count = 1024;                  // default: 1K ints
    size_t max_count = 1024 * 1024 * 128;     // default: 128M ints
    int iterations = 20;
    int warmup = 5;
    double step_factor = 2.0;

    int opt;
    while ((opt = getopt(argc, argv, "l:u:w:n:s:")) != -1) {
        switch (opt) {
            case 'l': min_count = std::stoul(optarg); break;
            case 'u': max_count = std::stoul(optarg); break;
            case 'w': warmup = std::stoi(optarg); break;
            case 'n': iterations = std::stoi(optarg); break;
            case 's': step_factor = std::stod(optarg); break;
            default:
                fprintf(stderr, "Usage: %s [-l min_count] [-u max_count] [-w warmup] [-n iterations] [-s stepfactor]\n", argv[0]);
                exit(EXIT_FAILURE);
        }
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.size);

    if (ctx.rank == 0) {
        printf("# allgather benchmark\n");
        printf("# min: %lu, max: %lu, step: %.1f, warmup: %d, iters: %d\n", min_count, max_count, step_factor, warmup, iterations);
        printf("# size is the data size which each rank sends\n");
        printf("# dtype\tcount\t\tsize(Kbytes)\tavg_time(s)\n");
    }
    fflush(stdout);

    for (size_t count = min_count; count <= max_count; count = static_cast<size_t>(count * step_factor)) {
        ctx.count = static_cast<int>(count);
        posix_memalign((void**)&ctx.send_buff, 4096, ctx.count * sizeof(int));
        posix_memalign((void**)&ctx.recv_buff, 4096, ctx.count * ctx.size * sizeof(int));
        for (int i = 0; i < ctx.count; ++i) {
            ctx.send_buff[i] = ctx.rank;
        }

        for (int i = 0; i < warmup; ++i) {
            do_allgather(&ctx);
        }

        double total_time = 0.0;
        for (int i = 0; i < iterations; ++i) {
            total_time += do_allgather(&ctx);
        }

        check_recv_buff(&ctx);
        double avg_time = total_time / iterations;
        if (ctx.rank == 0) {
            printf("int\t%ld\t\t%.6f\t%.6f\n", count, count * sizeof(int) / 1024.0, avg_time);
        }
        fflush(stdout);

        free(ctx.send_buff);
        free(ctx.recv_buff);
    }

    double total_sent_words = 0;
    double total_received_words = 0;
    for (size_t count = min_count; count <= max_count; count = static_cast<size_t>(count * step_factor)) {
        int rounds = warmup + iterations;
        total_sent_words += static_cast<double>(count * sizeof(int) * rounds);
        total_received_words += static_cast<double>(count * sizeof(int) * ctx.size * rounds);
    }

    if (ctx.rank == 0) {
        double n = static_cast<double>(ctx.size);
        printf("# Total data sent per rank in multicast: %.1f words\n", total_sent_words / 4.0);
        printf("# Total data received per rank in multicast: %.1f words\n", total_received_words / 4.0);
        printf("# Total data sent per rank in ring algo: %.1f words\n", (n-1)/ n * total_received_words / 4.0);
        printf("# Total data received per rank in ring algo: %.1f words\n", (n-1)/ n * total_received_words / 4.0);
    }

    MPI_Finalize();
    return 0;
}
