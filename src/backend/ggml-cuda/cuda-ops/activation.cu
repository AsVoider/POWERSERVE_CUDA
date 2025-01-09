#include "activation.cuh"

static __global__ void silu_with_mul(const float *gate, const float *up, float *dst, int k) {
    const int i{blockDim.x * blockIdx.x + threadIdx.x};
    if (i >= k) {
        return;
    }

    dst[i] = up[i] * (gate[i] / (1.0f + expf(-gate[i])));
}

static __global__ void relu_with_mul(const float *gate, const float *up, float *dst, int k) {
    const int i{blockDim.x * blockIdx.x + threadIdx.x};
    if (i >= k) {
        return;
    } 

    dst[i] = gate[i] <= 0 ? 0.f : gate[i] * up[i];
}

void ps_silu_with_mul(const float *gate, const float *up, float *dst, int k, cudaStream_t stream) {
    const int num_blocks{(k + CUDA_SILU_BLOCK_SIZE - 1) / CUDA_SILU_BLOCK_SIZE};
    silu_with_mul<<<num_blocks, CUDA_SILU_BLOCK_SIZE, 0, stream>>>(gate, up, dst, k);
}

void ps_relu_with_mul(const float *gate, const float *up, float *dst, int k, cudaStream_t stream) {
    const int num_blocks{(k + CUDA_RELU_BLOCK_SIZE - 1) / CUDA_RELU_BLOCK_SIZE};
    relu_with_mul<<<num_blocks, CUDA_RELU_BLOCK_SIZE, 0, stream>>>(gate, up, dst, k);
}