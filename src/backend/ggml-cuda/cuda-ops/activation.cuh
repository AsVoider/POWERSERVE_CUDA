#include "common.cuh"
#include "unary.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

void ps_silu_with_mul(const float *gate, const float *up, float *dst, int k, cudaStream_t stream);
void ps_relu_with_mul(const float *gate, const float *up, float *dst, int k, cudaStream_t stream);

void activate_without_mul(ggml_backend_cuda_context &ctx, ggml_tensor *dst);

void activate_with_mul(ggml_backend_cuda_context &ctx, ggml_tensor *dst);
