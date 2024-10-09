#pragma once

#include "graph/node.hpp"
#include <map>
#include <memory>
#include <vector>
namespace smart {

// DAG
class Graph {
public:
	std::vector<std::shared_ptr<Node>> nodes; // manage all locals nodes lifecycle
	// x, +wei, op1, xo
	std::map<int, std::vector<std::shared_ptr<Operator>>> DAG;
	std::shared_ptr<Node> root;

	Graph() : root(std::make_shared<Operator>(OpType::OP_NONE)) { nodes.push_back(root); };
	~Graph() = default;
};

} // namespace smart