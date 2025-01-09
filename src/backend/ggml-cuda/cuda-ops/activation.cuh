#include "common.cuh"
#include "unary.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

void ps_silu_with_mul(const float *gate, const float *up, float *dst, int k, cudaStream_t stream);
void ps_relu_with_mul(const float *gate, const float *up, float *dst, int k, cudaStream_t stream);

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
