#pragma once

#include <vector>
#include <string>

#include "core/tensor.hpp"
#include "graph/op_type.hpp"
#include "graph/op_params.hpp"

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

    auto tensor() -> Tensor *;
    auto op() -> OpNode *;
};

struct TensorNode : Tensor, Node {
    TensorNode(const Tensor &tensor) :
        Node(NodeType::TENSOR),
        Tensor(tensor)
    {}

    TensorNode(DataType dtype, Tensor::Shape shape) :
        Node(NodeType::TENSOR),
        Tensor(dtype, shape)
    {}

    auto prev_op() const -> OpNode * {
        SMART_ASSERT(prev.size() == 1);
        return prev[0]->op();
    }
};

struct OpNode : Node {
    OpType op;
    std::unique_ptr<OpParams> params;

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

    template <typename Params>
    auto set_params(const Params &params) {
        this->params.reset(new Params(params));
        return this;
    }

    template <typename Params>
    auto get_params() const {
        return *static_cast<Params *>(params.get());
    }

    size_t n_outputs() const {
        return next.size();
    }

    auto output() const -> Tensor * {
        SMART_ASSERT(n_outputs() == 1);
        return next[0]->tensor();
    }
};

}
