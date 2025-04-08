#pragma once

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace powerserve::ggml_cuda {

static constexpr int KVPaddingSize{256};
static constexpr int KVBlockSize{64};
static constexpr int InitBlockNum{2};
static constexpr int LocalBlockNum{2};

enum class KVSparsityType : int32_t {
    QUEST = 0,
    MEAN = 1,
    MAX = 2,
    NONE = 3,
};

void init_meta_data(const uint8_t *prefill_cache, uint8_t *meta_data, const size_t length, const size_t head_dim, const size_t head_num, KVSparsityType type);

} // namespace powerserve::ggml_cuda


