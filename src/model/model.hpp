#pragma once

#include "graph/graph.hpp"

#include <string>

namespace smart {

struct Model {
	std::string filename;

	virtual Graph *prefill() = 0;
	virtual Graph *decode()	 = 0;

	Model(const std::string &filename) : filename(filename) {}

	~Model() = default;
};

} // namespace smart
