#include "kv_cache.hpp"
#include "graph/node.hpp"

namespace smart {

void KVCache::add_cache(Graph &g, TensorNode *tensor, size_t offset) {
	auto c = g.add_tensor(cache);
	g.copy(c, tensor, offset);
}

TensorNode *KVCache::add_cache_node(Graph &g) {
	return g.add_tensor(cache);
}

} // namespace smart