#include "activation.cuh"

static __global__ void silu_with_mul(const float *gate, const float *up, float *dst, int k) {
    const int i{static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x)};
    if (i >= k) {
        return;
    }

    dst[i] = up[i] * (gate[i] / (1.0f + expf(-gate[i])));
}

static __global__ void relu_with_mul(const float *gate, const float *up, float *dst, int k) {
    const int i{static_cast<int>(blockDim.x * blockIdx.x + threadIdx.x)};
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

void activate_without_mul(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
    auto op{ggml_get_unary_op(dst)};
    switch (op) {
    case GGML_UNARY_OP_NEG:
        ggml_cuda_op_neg(ctx, dst);
        break;
    case GGML_UNARY_OP_STEP:
        ggml_cuda_op_step(ctx, dst);
        break;
    case GGML_UNARY_OP_GELU:
        ggml_cuda_op_gelu(ctx, dst);
        break;
    case GGML_UNARY_OP_SILU:
        ggml_cuda_op_silu(ctx, dst);
        break;
    case GGML_UNARY_OP_GELU_QUICK:
        ggml_cuda_op_gelu_quick(ctx, dst);
        break;
    case GGML_UNARY_OP_TANH:
        ggml_cuda_op_tanh(ctx, dst);
        break;
    case GGML_UNARY_OP_RELU:
        ggml_cuda_op_relu(ctx, dst);
        break;
    case GGML_UNARY_OP_SIGMOID:
        ggml_cuda_op_sigmoid(ctx, dst);
        break;
    case GGML_UNARY_OP_HARDSIGMOID:
        ggml_cuda_op_hardsigmoid(ctx, dst);
        break;
    case GGML_UNARY_OP_HARDSWISH:
        ggml_cuda_op_hardswish(ctx, dst);
        break;
    case GGML_UNARY_OP_EXP:
        ggml_cuda_op_exp(ctx, dst);
        break;
    default:
        break;
    }
}

void activate_with_mul(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
    auto op{ggml_get_unary_op(dst)};
    auto gate{dst->src[0]}, up{dst->src[1]};
    auto gate_data{static_cast<float *>(gate->data)}, up_data{static_cast<float *>(up->data)}, dst_data{static_cast<float *>(dst->data)};
    switch (op) {
    case GGML_UNARY_OP_SILU:
        ps_silu_with_mul(gate_data, up_data, dst_data, ggml_nelements(gate), ctx.stream());
        break;
    case GGML_UNARY_OP_RELU:
        ps_relu_with_mul(gate_data, up_data, dst_data, ggml_nelements(gate), ctx.stream());
        break;
    default:
        GGML_ASSERT(false and "unsupported unary type\n");
    }
}