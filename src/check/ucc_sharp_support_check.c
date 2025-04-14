#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucc/api/ucc.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t e = cmd; \
        if (e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void init_cuda_data(float *d_buf, size_t count, float value) {
    float *h_buf = (float *)malloc(count * sizeof(float));
    for (size_t i = 0; i < count; ++i) h_buf[i] = value;
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf, count * sizeof(float), cudaMemcpyHostToDevice));
    free(h_buf);
}

int test_allgather(ucc_team_h team, size_t count) {
    size_t team_size = 1; // single rank test
    size_t total_count = count * team_size;

    void *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, total_count * sizeof(float)));

    init_cuda_data((float *)d_src, count, 1.0f);

    ucc_coll_args_t coll;
    memset(&coll, 0, sizeof(coll));
    coll.coll_type = UCC_COLL_TYPE_ALLGATHER;

    coll.src.info.buffer    = d_src;
    coll.src.info.count     = count;
    coll.src.info.datatype  = UCC_DT_FLOAT32;
    coll.src.info.mem_type  = UCC_MEMORY_TYPE_CUDA;

    coll.dst.info.buffer    = d_dst;
    coll.dst.info.count     = total_count;
    coll.dst.info.datatype  = UCC_DT_FLOAT32;
    coll.dst.info.mem_type  = UCC_MEMORY_TYPE_CUDA;

    ucc_coll_req_h req;
    ucc_status_t st = ucc_collective_init(&coll, &req, team);
    if (st == UCC_OK) {
        printf("✅ UCC ALLGATHER initialized with CUDA memory\n");
        ucc_collective_finalize(req);
    } else {
        fprintf(stderr, "❌ UCC ALLGATHER failed: %s\n", ucc_status_string(st));
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    return st == UCC_OK ? 0 : 1;
}

int test_reducescatter(ucc_team_h team, size_t count) {
    size_t team_size = 1;
    size_t total_count = count * team_size;

    void *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, total_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, count * sizeof(float)));

    init_cuda_data((float *)d_src, total_count, 1.0f);

    ucc_coll_args_t coll;
    memset(&coll, 0, sizeof(coll));
    coll.coll_type = UCC_COLL_TYPE_REDUCE_SCATTER;

    coll.src.info.buffer    = d_src;
    coll.src.info.count     = total_count;
    coll.src.info.datatype  = UCC_DT_FLOAT32;
    coll.src.info.mem_type  = UCC_MEMORY_TYPE_CUDA;

    coll.dst.info.buffer    = d_dst;
    coll.dst.info.count     = count;
    coll.dst.info.datatype  = UCC_DT_FLOAT32;
    coll.dst.info.mem_type  = UCC_MEMORY_TYPE_CUDA;

    coll.op = UCC_OP_SUM;

    ucc_coll_req_h req;
    ucc_status_t st = ucc_collective_init(&coll, &req, team);
    if (st == UCC_OK) {
        printf("✅ UCC REDUCESCATTER initialized with CUDA memory\n");
        ucc_collective_finalize(req);
    } else {
        fprintf(stderr, "❌ UCC REDUCESCATTER failed: %s\n", ucc_status_string(st));
    }

    cudaFree(d_src);
    cudaFree(d_dst);
    return st == UCC_OK ? 0 : 1;
}

int main() {
    const size_t count = 16;
    const int team_size = 1;

    // --- UCC Setup ---
    ucc_lib_h lib;
    ucc_context_h context;
    ucc_team_h team;

    ucc_lib_config_h lib_config;
    ucc_context_config_h ctx_config;

    ucc_lib_params_t lib_params = {
        .mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCC_THREAD_SINGLE
    };

    if (ucc_lib_config_read(NULL, NULL, &lib_config) != UCC_OK ||
        ucc_init(&lib_params, lib_config, &lib) != UCC_OK) {
        fprintf(stderr, "UCC init failed\n");
        return EXIT_FAILURE;
    }
    ucc_lib_config_release(lib_config);

    ucc_context_params_t ctx_params = {
        .mask = UCC_CONTEXT_PARAM_FIELD_TYPE,
    };

    if (ucc_context_config_read(lib, NULL, &ctx_config) != UCC_OK ||
        ucc_context_create(lib, &ctx_params, ctx_config, &context) != UCC_OK) {
        fprintf(stderr, "UCC context create failed\n");
        return EXIT_FAILURE;
    }
    ucc_context_config_release(ctx_config);

    ucc_team_params_t team_params = {
        .mask = UCC_TEAM_PARAM_FIELD_EP |
                UCC_TEAM_PARAM_FIELD_EP_RANGE |
                UCC_TEAM_PARAM_FIELD_TEAM_SIZE,
        .ep = 0,
        .ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG,
        .team_size = team_size
    };

    if (ucc_team_create_post(&context, 1, &team_params, &team) != UCC_OK) {
        fprintf(stderr, "UCC team create failed\n");
        return EXIT_FAILURE;
    }

    while (ucc_team_create_test(team) == UCC_INPROGRESS) {}

    if (ucc_team_create_test(team) != UCC_OK) {
        fprintf(stderr, "UCC team test failed\n");
        return EXIT_FAILURE;
    }

    // --- Run tests ---
    test_allgather(team, count);
    test_reducescatter(team, count);

    // --- Cleanup ---
    ucc_team_destroy(team);
    ucc_context_destroy(context);
    ucc_finalize(lib);

    return 0;
}
