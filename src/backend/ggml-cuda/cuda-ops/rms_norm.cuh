#include "ggml.h"
#include "norm.cuh"
#include <cuda_runtime.h>

template<int block_size>
static __global__ void rms_norm_f32_with_weight(
    const float *src, const float *weight, float *dst, const int ncols, const float eps
) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y; // row
    const int tid = threadIdx.x; // 0 - 1023

    float tmp{0.0f};

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = src[row * ncols + col];
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[row * ncols + col] = scale * weight[row * ncols + col] * src[row * ncols + col];
    }
}

static void rms_norm_f32_with_weight_cuda(
    const float *src, const float *weight, float *dst, 
    const int ncols, const int nrows, const float eps,
    cudaStream_t stream
) {
    GGML_ASSERT(ncols % WARP_SIZE == 0);
    if (ncols < 1024) {
        const dim3 block_dim{WARP_SIZE, 1, 1};
        rms_norm_f32_with_weight<WARP_SIZE><<<nrows, block_dim, 0, stream>>>(src, weight, dst, ncols, eps);
    } else {
        const dim3 block_dim{1024, 1, 1};
        rms_norm_f32_with_weight<1024><<<nrows, block_dim, 0, stream>>>(src, weight, dst, ncols, eps);
    }
}

void rms_norm_with_weight(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
    const auto src0{dst->src[0]};
    const auto src1{dst->src[1]};

    auto src0_data{reinterpret_cast<const float *>(src0->data)}, src1_data{reinterpret_cast<const float *>(src1->data)};
    auto dst_data{reinterpret_cast<float *>(dst->data)};

    auto stream{ctx.stream()};

    const auto n_cols{dst->ne[0]};
    const auto n_rows{ggml_nrows(dst)};

    float eps{0.f};
    memcpy(&eps, dst->op_params, sizeof(float));

    rms_norm_f32_with_weight_cuda(src0_data, src1_data, dst_data, n_cols, n_rows, eps, stream);
}