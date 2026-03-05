#include "generated/gemm_cfg_menu.h"

#include <torch/extension.h>
#include <vector>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>

void gemm_forward_cuda_launcher(
    const half* A,
    const half* B,
    half* C,
    int M, int N, int K,
    int tile_m, int tile_n, int tile_k,
    int tm, int tn,
    cudaStream_t stream
);

// ------------------------------------------------------------

static torch::Tensor gemm_forward_cuda_impl(
    torch::Tensor A,
    torch::Tensor B,
    int tile_m,
    int tile_n,
    int tile_k,
    int tm,
    int tn
) {
    const int M = (int)A.size(0);
    const int K = (int)A.size(1);
    const int N = (int)B.size(1);

    auto C = torch::empty({M, N}, A.options());

    const int device_index = A.get_device();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_index).stream();

    gemm_forward_cuda_launcher(
        (const half*)A.data_ptr<at::Half>(),
        (const half*)B.data_ptr<at::Half>(),
        (half*)C.data_ptr<at::Half>(),
        M, N, K,
        tile_m, tile_n, tile_k,
        tm, tn,
        stream
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

// ------------------------------------------------------------
// C++ interface
// ------------------------------------------------------------
torch::Tensor gemm_forward(
    torch::Tensor A,
    torch::Tensor B,
    int tile_m,
    int tile_n,
    int tile_k,
    int tm,
    int tn
) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");

    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");

    TORCH_CHECK(A.dim() == 2, "A must be 2D (M x K)");
    TORCH_CHECK(B.dim() == 2, "B must be 2D (K x N)");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimensions must match: A is (M x K), B is (K x N)");

    TORCH_CHECK(tile_m > 0 && tile_n > 0 && tile_k > 0, "tile_m/tile_n/tile_k must be > 0");
    TORCH_CHECK(tm > 0 && tn > 0, "tm/tn must be > 0");

    TORCH_CHECK((tile_m % tm) == 0, "tile_m must be divisible by tm");
    TORCH_CHECK((tile_n % tn) == 0, "tile_n must be divisible by tn");

    const int threads = (tile_m / tm) * (tile_n / tn);
    TORCH_CHECK(threads > 0 && threads <= 1024, "Invalid threads per block for this config");

    c10::cuda::CUDAGuard device_guard(A.device());
    TORCH_CHECK(A.get_device() == B.get_device(), "A and B must be on the same CUDA device");

    // Ensure contiguous memory for simple pointer arithmetic in the CUDA kernel
    A = A.contiguous();
    B = B.contiguous();

    return gemm_forward_cuda_impl(A, B, tile_m, tile_n, tile_k, tm, tn);
}

torch::Tensor gemm_forward_cfg(
    torch::Tensor A,
    torch::Tensor B,
    int cfg_id
) {
    TORCH_CHECK(cfg_id >= 0, "cfg_id must be >= 0");
    const auto& menu = gemm_cfg_menu();
    TORCH_CHECK(cfg_id < (int)menu.size(), "cfg_id out of range");

    const GemmCfg cfg = menu[(size_t)cfg_id];
    return gemm_forward(A, B, cfg.tile_m, cfg.tile_n, cfg.tile_k, cfg.tm, cfg.tn);
}

torch::Tensor gemm_cfg_table_cpu() {
    const auto& menu = gemm_cfg_menu();
    auto out = torch::empty({(long)menu.size(), 5}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto acc = out.accessor<int, 2>();
    for (size_t i = 0; i < menu.size(); ++i) {
        acc[(int)i][0] = menu[i].tile_m;
        acc[(int)i][1] = menu[i].tile_n;
        acc[(int)i][2] = menu[i].tile_k;
        acc[(int)i][3] = menu[i].tm;
        acc[(int)i][4] = menu[i].tn;
    }
    return out;
}

// ------------------------------------------------------------
PYBIND11_MODULE(infinite_dreams_ext, m) {
    m.def("gemm_forward", &gemm_forward,
          "FP16 GEMM forward (CUDA) with (tile_m,tile_n,tile_k,tm,tn)");

    m.def("gemm_forward_cfg", &gemm_forward_cfg,
          "FP16 GEMM forward (CUDA) using cfg_id from static menu");

    m.def("gemm_cfg_table_cpu", &gemm_cfg_table_cpu,
          "Returns CPU int32 tensor [num_cfg,5] with columns: tile_m,tile_n,tile_k,tm,tn");
}