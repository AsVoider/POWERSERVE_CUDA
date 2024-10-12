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
};

struct Graph;
struct TensorNode;
struct OpNode;

struct Node {
	NodeType type;
	std::string name = "";
	std::vector<Node *> prev;
	std::vector<Node *> next;

	virtual ~Node() = default;

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

protected:
	Node(NodeType type) : type(type) {}
};

struct TensorNode : Tensor, Node {
	auto prev_op() const -> OpNode * {
		SMART_ASSERT(prev.size() == 1);
		return prev[0]->op();
	}

private:
	TensorNode(const Tensor &tensor) : Tensor(tensor), Node(NodeType::TENSOR) {}

	TensorNode(DataType dtype, const Tensor::Shape &shape) : Tensor(dtype, shape), Node(NodeType::TENSOR) {}

	friend struct Graph;
};

struct OpNode : Node {
	OpType op;
	std::unique_ptr<OpParams> params;

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

private:
	OpNode(OpType op) : Node(NodeType::OPERATOR), op(op) {}

	friend struct Graph;
};

} // namespace smart
