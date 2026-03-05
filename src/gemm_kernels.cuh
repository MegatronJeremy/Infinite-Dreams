#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<int TILE_M, int TILE_N, int TILE_K, int TM, int TN>
__global__ void gemm_tiled_reg(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ half As[TILE_M][TILE_K];
    __shared__ half Bs[TILE_K][TILE_N];

    float acc[TM][TN] = {0};

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        int numThreads = blockDim.x * blockDim.y;

        for (int idx = tid; idx < TILE_M * TILE_K; idx += numThreads) {
            int r = idx / TILE_K;
            int c = idx % TILE_K;
            int gr = blockIdx.y * TILE_M + r;
            int gc = k0 + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }

        for (int idx = tid; idx < TILE_K * TILE_N; idx += numThreads) {
            int r = idx / TILE_N;
            int c = idx % TILE_N;
            int gr = k0 + r;
            int gc = blockIdx.x * TILE_N + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            float aReg[TM];
            float bReg[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int r = threadIdx.y * TM + i;
                aReg[i] = __half2float(As[r][kk]);
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = threadIdx.x * TN + j;
                bReg[j] = __half2float(Bs[kk][c]);
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += aReg[i] * bReg[j];
                }
            }
        }

        __syncthreads();
    }

    const int row0 = blockIdx.y * TILE_M + threadIdx.y * TM;
    const int col0 = blockIdx.x * TILE_N + threadIdx.x * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int r = row0 + i;
        if (r < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int c = col0 + j;
                if (c < N) C[r * N + c] = __float2half(acc[i][j]);
            }
        }
    }
}

static inline int div_up(int a, int b) { return (a + b - 1) / b; }

#define TRY_CFG(TILE_M, TILE_N, TILE_K, TM, TN)                                        \
    if (tile_m == (TILE_M) && tile_n == (TILE_N) && tile_k == (TILE_K) &&              \
        tm == (TM) && tn == (TN))                                                      \
    {                                                                                  \
        dim3 block((TILE_N) / (TN), (TILE_M) / (TM), 1);                               \
        dim3 grid(div_up(N, (TILE_N)), div_up(M, (TILE_M)), 1);                        \
        gemm_tiled_reg<(TILE_M), (TILE_N), (TILE_K), (TM), (TN)>                       \
            <<<grid, block, 0, stream>>>(A, B, C, M, N, K);                            \
        return true;                                                                   \
    }