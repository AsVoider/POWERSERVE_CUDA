#pragma once

#include <vector>
#include <string>

#include "core/tensor.hpp"

namespace smart {

enum class NodeType {
    TENSOR,
    OPERATOR,
};

struct TensorNode;
struct OpNode;

struct Node {
    NodeType type;
    std::string name = "";
    std::vector<Node *> prev;
    std::vector<Node *> next;

    Node(NodeType type_) : type(type_) {}
    virtual ~Node() = default;

    void connect(Node &other) {
        connect(&other);
    }

    void connect(Node *other) {
        next.push_back(other);
        other->prev.push_back(this);
    }

    auto set_name(const std::string &name) {
        this->name = name;
        return this;
    }

    auto tensor() -> TensorNode *;
    auto op() -> OpNode *;
};

struct TensorNode : Node, Tensor {
    TensorNode(const Tensor &tensor) :
        Node(NodeType::TENSOR),
        Tensor(tensor)
    {}

    auto prev_op() const -> OpNode * {
        SMART_ASSERT(prev.size() == 1);
        return prev[0]->op();
    }
};

enum class OpType {
	NONE = 0,

	ADD,
	MAT_MUL,
	SIN,
	COS,
	SUM,
	RMS_NORM,
	SILU_HADAMARD,
	ROPE,
	SOFT_MAX,

	MHA,
};

struct OpNode : Node {
    OpType op;

    OpNode(OpType op_) :
        Node(NodeType::OPERATOR),
        op(op_)
    {}

    auto set_inputs(const std::vector<TensorNode *> &tensors) {
        for (auto tensor : tensors) {
            tensor->connect(this);
        }
        return this;
    }

    auto set_outputs(const std::vector<TensorNode *> &tensors) {
        for (auto tensor : tensors) {
            connect(tensor);
        }
        return this;
    }

    size_t n_outputs() const {
        return next.size();
    }

    auto output() const -> TensorNode * {
        SMART_ASSERT(n_outputs() == 1);
        return next[0]->tensor();
    }
};

}
