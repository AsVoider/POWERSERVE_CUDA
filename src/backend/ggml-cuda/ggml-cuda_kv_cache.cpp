#include "backend/ggml-cuda/ggml-cuda_kv_cache.hpp"
#include "backend/ggml-cuda/interface.cuh"

namespace powerserve::ggml_cuda{
    
GGML_CUDAKV::GGML_CUDAKV(const ModelConfig::LLMConfig &config) : config{config} {
    kv_shape.kv_dim = config.kv_dim;
    kv_shape.kv_heads = config.n_kv_heads;
    kv_shape.n_ctx = config.seq_len;
    kv_shape.n_layers = config.n_layers;
    kv_shape.head_size = config.head_size;
    kv_shape.batch_size = 1UL;
    kv_shape.type = DataType::FP32;

    init_cache();
}    

auto GGML_CUDAKV::init_cache() -> void {
    k_cache.resize(kv_shape.n_layers);
    v_cache.resize(kv_shape.n_layers);

    auto k_size{kv_shape.get_k_size(kv_shape.n_ctx)};
    auto v_size{kv_shape.get_v_size(kv_shape.n_ctx)};

    for (size_t i{0}; i < kv_shape.n_layers; ++i) {
        cuda_context_warp::malloc_cuda_buffer(reinterpret_cast<void **>(&k_cache[i].cache_data_ptr), k_size);
        cuda_context_warp::malloc_cuda_buffer(reinterpret_cast<void **>(&v_cache[i].cache_data_ptr), v_size);
    }
}

auto GGML_CUDAKV::get_k_cache_tensor(size_t layer_id) -> Tensor * {
    auto ggml_tp{convert_datatype_to_ggml(kv_shape.type)};
    Stride t_stride{
        get_type_size(kv_shape.type), 
        ggml_row_size(ggml_tp, kv_shape.kv_dim),
        ggml_row_size(ggml_tp, kv_shape.head_size),
        ggml_row_size(ggml_tp, kv_shape.kv_dim * k_cache[layer_id].next_position / kv_shape.get_k_size(1)),
    }; 
    Shape t_shape{

    };
    // auto ret{new Tensor{kv_shape.type, }}
}

auto GGML_CUDAKV::get_v_cache_tensor(size_t layer_id) -> Tensor * {

}

auto GGML_CUDAKV::clear_cache(size_t trunc_idx) -> void {
    for (int i{0}; i < kv_shape.n_layers; ++i) {
        const auto k_cur_size{k_cache[i].next_position};
        const auto v_cur_size{v_cache[i].next_position};

        const auto k_aft_size{kv_shape.get_k_size(trunc_idx)};
        const auto v_aft_size{kv_shape.get_v_size(trunc_idx)};

        if (const size_t clear_k_size{k_cur_size - k_aft_size}; clear_k_size > 0) {
            cuda_context_warp::device_memset(k_cache[i].cache_data_ptr + k_aft_size, 0, clear_k_size);
        }

        if (const size_t clear_v_size{v_cur_size - v_aft_size}; clear_v_size > 0) {
            cuda_context_warp::device_memset(k_cache[i].cache_data_ptr + v_aft_size, 0, clear_v_size);
        }

        k_cache[i].next_position = k_aft_size;
        v_cache[i].next_position = v_aft_size;
    }
}

auto GGML_CUDAKV::get_k_cache(size_t layer_id) -> uint8_t * {
    return k_cache[layer_id].cache_data_ptr;
}

auto GGML_CUDAKV::get_v_cache(size_t layer_id) -> uint8_t * {
    return v_cache[layer_id].cache_data_ptr;
}

auto GGML_CUDAKV::append_k_cache(const void *k_data, size_t layer_id, size_t token_nums) -> void {
    const size_t target_size{kv_shape.get_k_size(token_nums)};
    auto dst_ptr{reinterpret_cast<void *>(k_cache[layer_id].cache_data_ptr + k_cache[layer_id].next_position)};
    auto src_ptr{const_cast<void *>(k_data)};
    cuda_context_warp::copy_memory<3>(dst_ptr, src_ptr, target_size);
    k_cache[layer_id].next_position += target_size;
}

auto GGML_CUDAKV::append_v_cache(const void *v_data, size_t layer_id, size_t token_nums) -> void {
    const size_t target_size{kv_shape.get_v_size(token_nums)};
    auto dst_ptr{reinterpret_cast<void *>(v_cache[layer_id].cache_data_ptr + v_cache[layer_id].next_position)};
    auto src_ptr{const_cast<void *>(v_data)};
    cuda_context_warp::copy_memory<3>(dst_ptr, src_ptr, target_size);
    v_cache[layer_id].next_position += target_size;
}

auto GGML_CUDAKV::KVCacheShape::get_k_size(size_t token_nums) -> size_t {
    auto ggml_tp{convert_datatype_to_ggml(type)};
    return ggml_row_size(ggml_tp, head_size) * kv_heads * n_ctx;
}

auto GGML_CUDAKV::KVCacheShape::get_v_size(size_t token_nums) -> size_t {
    auto ggml_tp{convert_datatype_to_ggml(type)};
    return flash_attn ? ggml_row_size(ggml_tp, head_size) * kv_heads * n_ctx :
        ggml_row_size(ggml_tp, n_ctx) * head_size * kv_heads;
}

} // namespace powerserve::ggml_cuda
