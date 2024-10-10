#include "graph/node.hpp"

namespace smart {

auto Node::tensor() -> TensorNode * {
    return static_cast<TensorNode *>(this);
}

auto Node::op() -> OpNode * {
    return static_cast<OpNode *>(this);
}

}
