#include "graph/node.hpp"

namespace smart {

auto Node::tensor() -> Tensor * {
	return dynamic_cast<Tensor *>(this);
}

auto Node::op() -> OpNode * {
	return dynamic_cast<OpNode *>(this);
}

} // namespace smart
