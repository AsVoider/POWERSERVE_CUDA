#include "kv_cache.hpp"

#include "graph/node.hpp"

namespace smart {

void KVCache::add_key_cache(Graph &g, TensorNode *tensor, size_t L, size_t pos) {
    auto c      = g.add_tensor(key_cache);
    auto offset = L * seq_len * kv_dim + pos * kv_dim;
    g.copy(c, tensor, offset);
}

void KVCache::add_value_cache(Graph &g, TensorNode *tensor, size_t L, size_t pos) {
    auto c      = g.add_tensor(value_cache);
    auto offset = L * seq_len * kv_dim + pos * kv_dim;
    g.copy(c, tensor, offset);
}

TensorNode *KVCache::add_key_cache_node(Graph &g) {
    return g.add_tensor(key_cache);
}

TensorNode *KVCache::add_value_cache_node(Graph &g) {
    return g.add_tensor(value_cache);
}

} // namespace smart