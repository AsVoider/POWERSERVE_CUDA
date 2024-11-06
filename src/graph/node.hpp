#pragma once

#include "core/tensor.hpp"
#include "graph/op_params.hpp"
#include "graph/op_type.hpp"

#include <string>
#include <vector>

namespace smart {

enum class NodeType {
    TENSOR,
    OPERATOR,
    TENSOR_VIEW,
};

struct Graph;
struct TensorNode;
struct TensorViewNode;
struct OpNode;

struct Node {

public:
    NodeType type;
    std::string name = "";
    std::vector<Node *> prev;
    std::vector<Node *> next;

public:
    virtual ~Node() = default;

public:
    void connect(Node &other) {
        connect(&other);
    }

    void connect(Node *other) {
        next.push_back(other);
        other->prev.push_back(this);
    }

    void set_name(const std::string &name) {
        this->name = name;
    }

    auto tensor() -> Tensor *;
    auto op() -> OpNode *;
    auto tensor_view() -> TensorViewNode *;

protected:
    Node(NodeType type) : type(type) {}
};

struct TensorNode : Tensor, Node {
private:
    friend struct Graph;

protected:
    TensorNode(const Tensor &tensor) : Tensor(tensor), Node(NodeType::TENSOR) {}

    TensorNode(DataType dtype, const Tensor::Shape &shape) : Tensor(dtype, shape), Node(NodeType::TENSOR) {}

public:
    ~TensorNode() override = default;

public:
    auto prev_op() const -> OpNode * {
        SMART_ASSERT(prev.size() == 1);
        return prev[0]->op();
    }
};

struct TensorViewNode : TensorNode {
private:
    friend struct Graph;
public:
    Tensor *parent;

private:
    TensorViewNode(const Tensor &tensor, Tensor::Shape shape) : TensorNode(tensor) {
        type = NodeType::TENSOR_VIEW;
        parent = const_cast<Tensor *>(&tensor);
        SMART_ASSERT(parent->n_elements() == n_elements());
        m_shape = shape;
    }
public:
    ~TensorViewNode() override = default;
};

struct OpNode : Node {
public:
    OpType op;
    std::unique_ptr<OpParams> params;

private:
    OpNode(OpType op) : Node(NodeType::OPERATOR), op(op) {}

public:
    ~OpNode() override = default;

private:
    friend struct Graph;

public:
    void set_inputs(const std::vector<TensorNode *> &tensors) {
        for (auto tensor : tensors) {
            tensor->connect(this);
        }
    }

    void set_outputs(const std::vector<TensorNode *> &tensors) {
        for (auto tensor : tensors) {
            connect(tensor);
        }
    }

    template <typename T>
    void set_params(const T &params) {
        this->params.reset(new OpParamWrapper<T>(params));
    }

    template <typename T>
    const auto &get_params() const {
        return dynamic_cast<OpParamWrapper<T> *>(params.get())->value;
    }

    size_t n_outputs() const {
        return next.size();
    }

    auto output() const -> Tensor * {
        SMART_ASSERT(n_outputs() == 1);
        return next[0]->tensor();
    }
};

} // namespace smart
