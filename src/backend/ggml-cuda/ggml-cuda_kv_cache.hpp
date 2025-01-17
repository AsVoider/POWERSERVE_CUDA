#include "backend/ggml-cuda/common.hpp"

namespace powerserve::ggml_cuda { 

class GGML_CUDAKV {
public:
    static constexpr int KVPaddingSize{256};
    static constexpr int KVBlockSize{64};

    struct GGML_CUDACHUNK {
        uint8_t *cache_data_ptr{nullptr};
        size_t next_position{0};
        size_t valid_idx{0};
    };

    struct KVCacheShape {
        size_t kv_dim{0}; // 1024
        size_t kv_heads{0}; // 8
        size_t n_ctx{0}; // n
        size_t n_layers{0}; // 32
        size_t head_size{0}; // ? 128 ?
        size_t batch_size{0}; // always 1
        size_t kv_size{0}; //
        DataType type{DataType::FP32};
        bool flash_attn{false};

        auto get_k_size(size_t token_nums) -> size_t;
        auto get_v_size(size_t token_nums) -> size_t;
    };

public:
    const ModelConfig::LLMConfig &config;
    KVCacheShape kv_shape;

    std::vector<GGML_CUDACHUNK> k_cache;
    std::vector<GGML_CUDACHUNK> v_cache;

    GGML_CUDAKV(const ModelConfig::LLMConfig &config);
    ~GGML_CUDAKV() = default;

public:
    auto get_cache_position() -> size_t;
    auto get_cache(size_t layer_id) -> std::pair<Tensor *, Tensor *>;
    auto clear_cache(size_t trunc_idx) -> void;
    auto append_k_cache(const Tensor *k_tensor, size_t layer_id, size_t token_nums) -> void;
    auto append_v_cache(const Tensor *v_tensor, size_t layer_id, size_t token_nums) -> void;
    auto get_k_cache_tensor(size_t layer_id) -> Tensor *;
    auto get_v_cache_tensor(size_t layer_id) -> Tensor *;

private:
    auto init_cache() -> void;
    auto get_k_cache(size_t layer_id) -> uint8_t *;
    auto get_v_cache(size_t layer_id) -> uint8_t *;
};

} // namespace powerinfer::ggml_cuda
