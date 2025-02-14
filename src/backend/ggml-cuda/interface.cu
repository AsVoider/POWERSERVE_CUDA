#include "backend/ggml-cuda/interface.cuh"
#include "backend/ggml-cuda/cuda-ops/rms_norm.cuh"
#include "backend/ggml-cuda/cuda-ops/mat_mul.cuh"
#include "backend/ggml-cuda/cuda-ops/activation.cuh"
#include "backend/ggml-cuda/cuda-ops/copy_ext.cuh"
#include "backend/ggml-cuda/cuda-ops/get_mask.cuh"

#include "common.cuh"
#include "cpy.cuh"
#include "softmax.cuh"
#include "rope.cuh"
#include "binbcast.cuh"
#include "ggml-quants.h"

#include <exception>
#include <functional>
#include <string>

namespace powerserve::ggml_cuda
{

cuda_context_warp::cuda_context_warp() {
    auto err{cudaGetDeviceCount(&device_count)};
    if (err not_eq cudaSuccess) {
        auto err_str{std::string{"Error occurs when getting cuda device number: "} + std::to_string(static_cast<int>(err))};
        throw std::runtime_error(err_str);
    }

    ctx = std::allocator<ggml_backend_cuda_context>().allocate(device_count);
    auto ctx_ptr{static_cast<ggml_backend_cuda_context *>(ctx)};
    for (int i{0}; i < device_count; ++i) {
        std::construct_at(&ctx_ptr[i], i);
    }
}

auto cuda_context_warp::malloc_cuda_buffer(void **ptr, size_t size) -> int {
    return static_cast<int>(cudaMalloc(ptr, size));
}

auto cuda_context_warp::free_cuda_buffer(void *ptr) -> int {
    return static_cast<int>(cudaFree(ptr));
}

auto cuda_context_warp::device_sync() -> int {
    return static_cast<int>(cudaDeviceSynchronize());
}

auto cuda_context_warp::copy_memory_host_to_host(void *dst, void *src, size_t size) -> int{
    return static_cast<int>(cudaMemcpy(dst, src, size, cudaMemcpyHostToHost));
}

auto cuda_context_warp::copy_memory_host_to_device(void *dst, void *src, size_t size) -> int{
    return static_cast<int>(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
}

auto cuda_context_warp::copy_memory_device_to_host(void *dst, void *src, size_t size) -> int{
    return static_cast<int>(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
}

auto cuda_context_warp::copy_memory_device_to_device(void *dst, void *src, size_t size) -> int{
    return static_cast<int>(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

auto cuda_context_warp::copy_memory_host_to_host_async(void *dst, void *src, size_t size, void *context) -> int {
    if (context == nullptr) {
        return static_cast<int>(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToHost));
    } else {
        auto stream{static_cast<ggml_backend_cuda_context *>(context)->stream()};
        return static_cast<int>(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToHost, stream));
    } 
}

auto cuda_context_warp::copy_memory_host_to_device_async(void *dst, void *src, size_t size, void *context) -> int {
    if (context == nullptr) {
        return static_cast<int>(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice));
    } else {
        auto stream{static_cast<ggml_backend_cuda_context *>(context)->stream()};
        return static_cast<int>(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream));
    } 
}

auto cuda_context_warp::copy_memory_device_to_host_async(void *dst, void *src, size_t size, void *context) -> int {
    if (context == nullptr) {
        return static_cast<int>(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost));
    } else {
        auto stream{static_cast<ggml_backend_cuda_context *>(context)->stream()};
        return static_cast<int>(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream));
    } 
}

auto cuda_context_warp::copy_memory_device_to_device_async(void *dst, void *src, size_t size, void *context) -> int {
    if (context == nullptr) {
        return static_cast<int>(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice));
    } else {
        auto stream{static_cast<ggml_backend_cuda_context *>(context)->stream()};
        return static_cast<int>(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
    } 
}

auto cuda_context_warp::device_memset(void *dst, int value, size_t size) -> int {
    return static_cast<int>(cudaMemset(dst, value, size));
}

auto cuda_context_warp::device_memset_async(void *dst, int value, size_t size, void *stream_ptr) -> int {
    return static_cast<int>(cudaMemsetAsync(dst, value, size, static_cast<cudaStream_t>(stream_ptr)));
}

op_interface op_interfaces::op_get_embedding = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    // auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};
    // TODO:
    // return 0;
    GGML_UNUSED(dst);
};

op_interface op_interfaces::op_mat_mul = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};

    auto src0{dst->src[0]}, src1{dst->src[1]};

    bool use_mul_mat_vec   = (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src0->ne[0] % 2 == 0 && src1->ne[1] == 1;
    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
    bool use_mul_mat_q     = ggml_is_quantized(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    bool any_gpus_with_slow_fp16{false};
    bool any_gpus_without_fp16_mma{false};
    bool split{false};

    if (split) {
        
    } else {
        const int cc              = ggml_cuda_info().devices[cuda_context_ptr[0].device].cc;
        use_mul_mat_q             = use_mul_mat_q             && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]);
        any_gpus_with_slow_fp16   = any_gpus_with_slow_fp16   || !fast_fp16_available(cc);
        any_gpus_without_fp16_mma = any_gpus_without_fp16_mma || !fp16_mma_available(cc);
    }

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    if (!split && use_mul_mat_vec && dst->ne[3] == 1 && (src0->ne[1] < MMV_MAX_ROWS || any_gpus_without_fp16_mma)) {
        // the custom F16 vector kernel can be used over batched cuBLAS GEMM
        // but this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec(cuda_context_ptr[0], src0, src1, dst);
    } else if (!split && src0->type == GGML_TYPE_F16 && (src1->type == GGML_TYPE_F16 || !any_gpus_with_slow_fp16)
               && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        // general KQ + KQV multi-batch without FlashAttention
        ggml_cuda_mul_mat_batched_cublas(cuda_context_ptr[0], src0, src1, dst);
    } else if (use_mul_mat_vec) {
        ggml_cuda_op_mul_mat(cuda_context_ptr[0], src0, src1, dst, ggml_cuda_op_mul_mat_vec, nullptr);
    } else if (use_mul_mat_vec_q) {
        ggml_cuda_op_mul_mat(cuda_context_ptr[0], src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    } else if (use_mul_mat_q) {
        ggml_cuda_op_mul_mat(cuda_context_ptr[0], src0, src1, dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    } else {
        ggml_cuda_op_mul_mat(cuda_context_ptr[0], src0, src1, dst, ggml_cuda_op_mul_mat_cublas, nullptr);
    }
};

op_interface op_interfaces::op_rms_norm = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};

    if (dst->src[1] == nullptr) {
        ggml_cuda_op_rms_norm(cuda_context_ptr[0], dst);
    } else {
        // TODO:
        rms_norm_with_weight(cuda_context_ptr[0], dst);
    }
};

op_interface op_interfaces::op_softmax = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    // with mask or without mask
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};

    ggml_cuda_op_soft_max(cuda_context_ptr[0], dst);
};

op_interface op_interfaces::op_rope = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};

    ggml_cuda_op_rope(cuda_context_ptr[0], dst);
};

op_interface op_interfaces::op_add = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};

    ggml_cuda_op_add(cuda_context_ptr[0], dst);
};

op_interface op_interfaces::op_cont = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};

    ggml_cuda_dup(cuda_context_ptr[0], dst);
};

op_interface op_interfaces::op_copy = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};

    ggml_cuda_dup(cuda_context_ptr[0], dst);
};

op_interface op_interfaces::op_print = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    // auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};
    // GGML_UNUSED(cuda_context_ptr);
    // auto file_name{std::string{dst->name}}; TODO:
    auto file_name{std::string{"output.txt"}};
    auto file{fopen(file_name.c_str(), "a+")};

    int64_t rows_to_print{static_cast<int64_t>(dst->op_params[0])};
    if (rows_to_print == 0L) {
        rows_to_print = dst->ne[1] * dst->ne[2] * dst->ne[3];
    }

    switch (dst->type) {
    case GGML_TYPE_F32: {
        float *buffer_to_print{new float[rows_to_print * dst->ne[0]]};
        cudaMemcpy(buffer_to_print, dst->data, rows_to_print * dst->ne[0] * sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (int64_t row{0}; row < rows_to_print; ++row) {
            for (int64_t col{0}; col < dst->ne[0]; ++col) {
                fprintf(file, "%f ", buffer_to_print[row * dst->ne[0] + col]);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n\n");
    } break;
    
    case GGML_TYPE_F16: {
        half *buffer_to_print{new half[rows_to_print * dst->ne[0]]};
        cudaMemcpy(buffer_to_print, dst->data, rows_to_print * dst->ne[0] * sizeof(half), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (int64_t row{0}; row < rows_to_print; ++row) {
            for (int64_t col{0}; col < dst->ne[0]; ++col) {
                float data_to_print{static_cast<float>(buffer_to_print[row * dst->ne[0] + col])};
                fprintf(file, "%f ", data_to_print);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n\n");
        delete[] buffer_to_print;
    } break;
    
    case GGML_TYPE_Q4_0: {
        char *buffer_to_convert{new char[ggml_nbytes(dst)]};
        float *buffer_to_print{new float[rows_to_print * dst->ne[0]]};
        cudaMemcpy(buffer_to_convert, dst->data, ggml_nbytes(dst), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        dequantize_row_q4_0((block_q4_0 *)buffer_to_convert, buffer_to_print, rows_to_print * dst->ne[0]);

        for (int64_t row{0}; row < rows_to_print; ++row) {
            for (int64_t col{0}; col < dst->ne[0]; ++col) {
                fprintf(file, "%f ", buffer_to_print[row * dst->ne[0] + col]);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n\n");

        delete[] buffer_to_print;
        delete[] buffer_to_convert;
    } break;

    case GGML_TYPE_Q8_0: {
        char *buffer_to_convert{new char[ggml_nbytes(dst)]};
        float *buffer_to_print{new float[rows_to_print * dst->ne[0]]};
        cudaMemcpy(buffer_to_convert, dst->data, ggml_nbytes(dst), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        dequantize_row_q8_0((block_q8_0 *)buffer_to_convert, buffer_to_print, rows_to_print * dst->ne[0]);

        for (int64_t row{0}; row < rows_to_print; ++row) {
            for (int64_t col{0}; col < dst->ne[0]; ++col) {
                fprintf(file, "%f ", buffer_to_print[row * dst->ne[0] + col]);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n\n");

        delete[] buffer_to_print;
        delete[] buffer_to_convert;
    } break;

    default: {
        fclose(file);
        exit(0);
    }
    }

    fclose(file);
    return;
};

op_interface op_interfaces::op_silu_and_mul = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};

    GGML_ASSERT(dst->op_params[0] == GGML_UNARY_OP_SILU);

    activate_with_mul(cuda_context_ptr[0], dst);
};

op_interface op_interfaces::op_append_v_cache = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};

    copy_permuted_v_cache(cuda_context_ptr[0], dst);
};

op_interface op_interfaces::op_get_mask = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};

    ggml_get_mask(cuda_context_ptr[0], dst);
};

} // namespace powerserve::ggml_cuda
