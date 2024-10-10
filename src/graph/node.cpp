#include "graph/node.hpp"

namespace smart {

auto Node::tensor() -> TensorNode * {
    return dynamic_cast<TensorNode *>(this);
}

auto Node::op() -> OpNode * {
    return dynamic_cast<OpNode *>(this);
}

}
