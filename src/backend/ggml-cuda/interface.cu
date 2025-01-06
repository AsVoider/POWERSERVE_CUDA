#include "backend/ggml-cuda/interface.cuh"
#include "backend/ggml-cuda/cuda-ops/rms_norm.cuh"
#include "backend/ggml-cuda/cuda-ops/mat_mul.cuh"

#include "common.cuh"
#include "cpy.cuh"
#include "softmax.cuh"
#include "rope.cuh"
#include "binbcast.cuh"

#include <exception>
#include <functional>
#include <string>

namespace smart::ggml_cuda
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

op_interface op_interfaces::op_get_embedding = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};
    // TODO:
    // return 0;
};

op_interface op_interfaces::op_mat_mul = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    auto src0{dst->src[0]}, src1{dst->src[1]};
    auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};
    auto split{dst->op_params[15] == 1 ? true : false};
    auto deq_mat_mul_vec{ggml_cuda_dmmv_type_supported(src0->type) and src1->type == GGML_TYPE_F32 and dst->type == GGML_TYPE_F32
        and src0->ne[0] % (GGML_CUDA_DMMV_X * 2) == 0 and src1->ne[1] == 1};
    auto mat_mul_vec{ggml_is_quantized(src0->type) and src1->type == GGML_TYPE_F32 and dst->type == GGML_TYPE_F32
        and src1->ne[1] <= MMVQ_MAX_BATCH_SIZE};
    auto mat_mul_q{ggml_is_quantized(src0->type) and src1->type == GGML_TYPE_F32 and dst->type == GGML_TYPE_F32};

    auto slow_fp16{false};

    if (split) {
        // TODO:
    } else {
        const auto cc{ggml_cuda_info().devices[cuda_context_ptr->device].cc};
        mat_mul_q = mat_mul_q && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]);
        slow_fp16 = slow_fp16 or !fast_fp16_available(cc);
    }

    if (not split and slow_fp16 and src0->type == GGML_TYPE_F16 and ggml_is_permuted(src0) and ggml_is_permuted(src1) and src1->ne[1] == 1) {
        ggml_cuda_mul_mat_vec_p021(*cuda_context_ptr, src0, src1, dst);
    } else if (not split and slow_fp16 and src0->type == GGML_TYPE_F16 and not ggml_is_contiguous(src0) and not ggml_is_transposed(src1) and src1->ne[1] == 1) {
        ggml_cuda_mul_mat_vec_nc(*cuda_context_ptr, src0, src1, dst);
    } else if (not split and src0->type == GGML_TYPE_F16 and (src1->type == GGML_TYPE_F16 or not slow_fp16) and not ggml_is_transposed(src0)
                and not ggml_is_transposed(src1) and src1->ne[2] * src1->ne[3] > 1) {
        ggml_cuda_mul_mat_batched_cublas(*cuda_context_ptr, src0, src1, dst);
    } else if (deq_mat_mul_vec) {
        ggml_cuda_op_mul_mat(*cuda_context_ptr, src0, src1, dst, ggml_cuda_op_dequantize_mul_mat_vec, nullptr);
    } else if (mat_mul_vec) {
        ggml_cuda_op_mul_mat(*cuda_context_ptr, src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    } else if (mat_mul_q) {
        ggml_cuda_op_mul_mat(*cuda_context_ptr, src0, src1, dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    } else {
        ggml_cuda_op_mul_mat(*cuda_context_ptr, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, nullptr);
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

    ggml_cuda_dup(*cuda_context_ptr, dst);
};

op_interface op_interfaces::op_copy = [] (cuda_context_warp &ctx, ggml_tensor *dst) -> void {
    if (ctx.ctx == nullptr) [[unlikely]] {
        exit(1);
    }

    auto cuda_context_ptr{static_cast<ggml_backend_cuda_context *>(ctx.ctx)};

    ggml_cuda_dup(*cuda_context_ptr, dst);
};

} // namespace smart::ggml_cuda
