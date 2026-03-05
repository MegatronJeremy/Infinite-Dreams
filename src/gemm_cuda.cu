#include "gemm_kernels.cuh"

#include "generated/gemm_dispatch_decl_gen.inc"

void gemm_forward_cuda_launcher(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    int tile_m, int tile_n, int tile_k,
    int tm, int tn,
    cudaStream_t stream
) {
    #include "generated/gemm_dispatch_calls_gen.inc"
}