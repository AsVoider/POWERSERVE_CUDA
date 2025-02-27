#include "graph_split.hpp"

namespace powerserve {

GraphSplit::GraphSplit (std::vector<std::shared_ptr<OpNode>> &ops, size_t &first_or_end) {
    POWERSERVE_ASSERT(not ops.empty());
    graph_backend = ops[first_or_end]->compute_backend;

    for (auto i{first_or_end}; ;++i) {
        if (ops[i]->compute_backend not_eq graph_backend) {
            first_or_end = i;
            break;
        }
        split_ops.emplace_back(ops[i]);

        if (i == ops.size() - 1) {
            first_or_end = i + 1;
            break;
        }
    }

    POWERSERVE_ASSERT(not split_ops.empty());
}

GraphSplit::GraphSplit(GraphSplit &&right) : 
    graph_backend{right.graph_backend}, 
    split_ops{std::move(right.split_ops)}, 
    backend{right.backend} {} 


auto GraphSplit::run_graph_compute() -> void {
    POWERSERVE_ASSERT(backend != nullptr);
    backend->graph_compute(split_ops);
}

} // namespace powerserve