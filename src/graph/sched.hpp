#pragma once

#include "backend/platform.hpp"
#include "graph/graph.hpp"
#include "graph/node.hpp"
#include <memory>
namespace smart {

class Sched {
private:
	void execute_op(std::shared_ptr<Operator> op, Platform &platform);

public:
	void run(Graph &graph, Platform &platform); // DAG -> platform.ggml_tensor.xxx

	Sched()	 = default;
	~Sched() = default;
};

} // namespace smart