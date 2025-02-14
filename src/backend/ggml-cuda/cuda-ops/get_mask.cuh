#include "common.cuh"

constexpr int CUDA_DIAG_MASK_INF_BLOCK_SIZE = 32;

static __global__ void mask_inf_f32(float * data, const int number_cols, const int number_rows, const int first_pos) {
    const int col = blockDim.y * blockIdx.y + threadIdx.y; // 32 * col_block_id + thread_id
    const int row = blockDim.x * blockIdx.x + threadIdx.x; // 1 * row_id + 0

    if (col >= number_cols) {
        return;
    }

    const int i = row * number_cols + col;
    data[i] = col > (first_pos + row % number_rows) ? -INFINITY : 0.0f;
}

static void mask_inf_f32_cuda(float * data, const int kv_size, const int batch_size, const int first_pos, cudaStream_t stream) {
    const dim3 block_dims{1, CUDA_DIAG_MASK_INF_BLOCK_SIZE, 1};
    const int block_num_x = (kv_size + CUDA_DIAG_MASK_INF_BLOCK_SIZE - 1) / CUDA_DIAG_MASK_INF_BLOCK_SIZE;
    const dim3 block_nums{static_cast<uint32_t>(batch_size), static_cast<uint32_t>(block_num_x), 1};
    mask_inf_f32<<<block_nums, block_dims, 0, stream>>>(data, kv_size, batch_size, first_pos);
}

void ggml_get_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    cudaStream_t stream = ctx.stream();

    const auto first_pos = dst->op_params[0];
    const auto pos_size = dst->op_params[1];
    const auto kv_number = dst->op_params[2];
    const auto batch_size = dst->op_params[3];

    GGML_ASSERT(pos_size == batch_size);

    mask_inf_f32_cuda((float *)dst->data, kv_number, batch_size, first_pos, stream);
}