#pragma once

#include "graph/graph.hpp"
#include <string>
namespace smart {

class Model {
public:
	std::string filename;

	virtual Graph *prefill() = 0;
	virtual Graph *decode()	 = 0;

	Model() = default;
	~Model() = default;
};

} // namespace smart