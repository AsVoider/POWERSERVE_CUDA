#include "common.cuh"

using copy_func = void (*) (const char *, char *);

static __device__ void copy_func_f32_to_f32(const char *v, char *v_cache) {

}

static __device__ void copy_func_f32_to_f16(const char *v, char *v_cache) {

}

static __device__ void copy_func_f16_to_f32(const char *v, char *v_cache) {

}

static __device__ void copy_func_f16_to_f16(const char *v, char *v_cache) {

}

template<copy_func Func>
static __global__ void copy_v_cache(const char *v, char *v_cache, 
    const int n_ctx, const int next_token, const int kv_dim) {

}

void copy_permuted_v_cache(ggml_backend_cuda_context &ctx, ggml_tensor *dst) {
    
}