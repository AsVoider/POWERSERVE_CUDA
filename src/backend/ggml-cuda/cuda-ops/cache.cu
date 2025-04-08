#include "cache.cuh"

namespace powerserve::ggml_cuda {

template <KVSparsityType tp, size_t head_size, typename T = half>
__global__ static void calculate() {
}

template<KVSparsityType tp, size_t head_size>
void init_meta_data_inner(const uint8_t *prefill_cache, uint8_t *meta_data, const size_t length, const size_t start, cudaStream_t stream) {
    constexpr size_t stride{tp == KVSparsityType::QUEST ? 2UL : 1UL};
    const auto cache_start_ptr{prefill_cache + start * head_size};
    auto meta_start_ptr{meta_data + start / KVBlockSize * head_size * stride};

    const dim3 block_num{(length + KVBlockSize - 1) / KVBlockSize * KVBlockSize, 1, 1};
    const dim3 block_dim{32U, 1, 1};
    calculate<tp, head_size><<<block_num, block_dim, 0, stream>>>();
}

void init_meta_data(const uint8_t *prefill_cache, uint8_t *meta_data, const size_t length, const size_t start, const size_t head_size, KVSparsityType type, cudaStream_t stream) {
    // TODO: Other types
    auto stream_cuda{static_cast<cudaStream_t>(stream)};
    if (type == KVSparsityType::QUEST) {
        if (head_size == 4096UL) {
            constexpr size_t HEAD_SIZE{4096UL};
            init_meta_data_inner<KVSparsityType::QUEST, HEAD_SIZE>(prefill_cache, meta_data, length, start, stream_cuda);
        }
    
        if (head_size == 3072UL) {
            constexpr size_t HEAD_SIZE{4096UL};
            init_meta_data_inner<KVSparsityType::QUEST, HEAD_SIZE>(prefill_cache, meta_data, length, start, stream_cuda);
        }
    }
}

} // namespace powerserve::ggml_cuda