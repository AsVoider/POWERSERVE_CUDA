#include "graph.hpp"
#include "backend/backend.hpp"

namespace powerserve {

class GraphSplit {
public:
    TensorBackend graph_backend{TensorBackend::UNKNOWN};
    std::vector<std::shared_ptr<OpNode>> split_ops{};
    Backend *backend{nullptr};

    GraphSplit() = default;

    GraphSplit(const GraphSplit &) = delete;

    GraphSplit(GraphSplit &&);

    GraphSplit (std::vector<std::shared_ptr<OpNode>> &ops, size_t &first_or_end);

    ~GraphSplit() = default;

    auto set_backend(Backend *backend) -> void {
        this->backend = backend;
    }

    auto run_graph_compute() -> void;
};

} // namespace powerserve