#include "executor/executor.hpp"

namespace smart {

void Executor::run(const Platform &platform, const Graph &graph) {
    for (auto op : graph.ops) {
        switch (op->op) {
            case OpType::MAT_MUL: {
                auto src = op->prev[0]->tensor();
                auto weight = op->prev[1]->tensor();
                auto dst = op->output();

                platform.ggml_backend.matmul(dst, weight, src);
            } break;

            default: SMART_ASSERT(false);
        }
    }
}

}
