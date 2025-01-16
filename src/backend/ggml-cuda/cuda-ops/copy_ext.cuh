#include "common.cuh"

using copy_func = void(*)(const char *, char *, const int, const int);
constexpr int copy_block_size = 64;

static __device__ void copy_func_f32_to_f32(const char *v, char *v_cache, const int token_num, const int src_stride) {
#pragma unroll    
    for (int i{0}; i < token_num; ++i) {
        const float *v_ptr = reinterpret_cast<const float *>(v + i * src_stride);
        float *dst_ptr = reinterpret_cast<float *>(v_cache + i * sizeof(float));
        *dst_ptr = *v_ptr;
    }
}

static __device__ void copy_func_f32_to_f16(const char *v, char *v_cache, const int token_num, const int src_stride) {
#pragma unroll
    for (int i{0}; i < token_num; ++i) {
        const float *v_ptr = reinterpret_cast<const float *>(v + i * src_stride);
        half *dst_ptr = reinterpret_cast<half *>(v_cache + i * sizeof(half));
        *dst_ptr = __float2half(*v_ptr);
    }
}

static __device__ void copy_func_f16_to_f32(const char *v, char *v_cache, const int token_num, const int src_stride) {
#pragma unroll
    for (int i{0}; i < token_num; ++i) {
        const half *v_ptr = reinterpret_cast<const half *>(v + i * src_stride);
        float *dst_ptr = reinterpret_cast<float *>(v_cache + i * sizeof(float));
        *dst_ptr = *v_ptr;
    }
}

static __device__ void copy_func_f16_to_f16(const char *v, char *v_cache, const int token_num, const int src_stride) {
#pragma unroll
    for (int i{0}; i < token_num; ++i) {
        const half *v_ptr = reinterpret_cast<const half *>(v + i * src_stride);
        half *dst_ptr = reinterpret_cast<half *>(v_cache + i * sizeof(half));
        *dst_ptr = *v_ptr;
    }
}

template<copy_func Func>
static __global__ void copy_v_cache_kernel(const char *v, char *v_cache, 
    const int dst_offset, const int dst_stride, const int src_nb, const int src_stride, const int kv_dim, const int token_number) {
    
    const int64_t i = blockDim.x * blockIdx.x + threadIdx.x; // token idx
    if (i >= kv_dim) {
        return;
    }

    auto v_cache_ptr{v_cache + dst_offset + i * dst_stride};
    auto v_ptr{v + i * src_nb};

    Func(v_ptr, v_cache_ptr, token_number, src_stride);
}

template<ggml_type TypeDst, ggml_type TypeSrc>
static void copy_v_cache(const char *v, char *v_cache, 
    const int n_ctx, const int next_token, const int kv_dim, const int token_number,
    cudaStream_t stream) {

    const int num_blocks{(kv_dim + copy_block_size - 1) / copy_block_size};

    if constexpr (TypeDst == GGML_TYPE_F32) {
        const int dst_offset{next_token * static_cast<int>(sizeof(float))};
        const int dst_stride{n_ctx * static_cast<int>(sizeof(float))};
        if constexpr (TypeSrc == GGML_TYPE_F32) {
            const int src_stride{kv_dim * static_cast<int>(sizeof(float))};
            copy_v_cache_kernel<copy_func_f32_to_f32><<<num_blocks, copy_block_size, 0, stream>>>(
                v, v_cache, dst_offset, dst_stride, sizeof(float), src_stride, kv_dim, token_number
            );
            return;
        } else {
            const int src_stride{kv_dim * static_cast<int>(sizeof(half))};
            copy_v_cache_kernel<copy_func_f16_to_f32><<<num_blocks, copy_block_size, 0, stream>>>(
                v, v_cache, dst_offset, dst_stride, sizeof(half), src_stride, kv_dim, token_number
            );
            return;
        }
    } 

    if constexpr (TypeDst == GGML_TYPE_F16) {
        const int dst_offset{next_token * static_cast<int>(sizeof(half))};
        const int dst_stride{n_ctx * static_cast<int>(sizeof(half))};
        if constexpr (TypeSrc == GGML_TYPE_F32) {
            const int src_stride{kv_dim * static_cast<int>(sizeof(float))};
            copy_v_cache_kernel<copy_func_f32_to_f16><<<num_blocks, copy_block_size, 0, stream>>>(
                v, v_cache, dst_offset, dst_stride, sizeof(float), src_stride, kv_dim, token_number
            );
            return;
        } else {
            const int src_stride{kv_dim * static_cast<int>(sizeof(half))};
            copy_v_cache_kernel<copy_func_f16_to_f16><<<num_blocks, copy_block_size, 0, stream>>>(
                v, v_cache, dst_offset, dst_stride, sizeof(half), src_stride, kv_dim, token_number
            );
            return;
        }
    }
}

void copy_permuted_v_cache(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {

    auto src0{dst->src[0]};
    auto dst_type{dst->type}, src_type{src0->type};
    auto dst_data{static_cast<char *>(dst->data)}, src_data{static_cast<char *>(src0->data)};
    auto stream{ctx.stream()};

    int n_ctx{dst->op_params[0]};
    int next_token{dst->op_params[1]};
    int kv_dim{dst->op_params[2]};
    int num_token{dst->op_params[3]};

    if (dst_type == GGML_TYPE_F32 and src_type == GGML_TYPE_F32) {
        copy_v_cache<GGML_TYPE_F32, GGML_TYPE_F32>(src_data, dst_data, n_ctx, next_token, kv_dim, num_token, stream);
    } else if (dst_type == GGML_TYPE_F16 and src_type == GGML_TYPE_F32) {
        copy_v_cache<GGML_TYPE_F16, GGML_TYPE_F32>(src_data, dst_data, n_ctx, next_token, kv_dim, num_token, stream);
    } else if (dst_type == GGML_TYPE_F32 and src_type == GGML_TYPE_F16) {
        copy_v_cache<GGML_TYPE_F32, GGML_TYPE_F16>(src_data, dst_data, n_ctx, next_token, kv_dim, num_token, stream);
    } else if (dst_type == GGML_TYPE_F16 and src_type == GGML_TYPE_F16) {
        copy_v_cache<GGML_TYPE_F16, GGML_TYPE_F16>(src_data, dst_data, n_ctx, next_token, kv_dim, num_token, stream);
    } else {
        exit(1);
    }
}