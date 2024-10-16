#pragma once

#include "graph/graph.hpp"

#include <string>

namespace smart {

struct Model {

public:
    std::string m_filename;

public:
    Model(const std::string &filename) : m_filename(filename) {}

    virtual ~Model() = default;

public:
    virtual Graph *prefill() = 0;
    virtual Graph *decode()  = 0;
};

} // namespace smart
