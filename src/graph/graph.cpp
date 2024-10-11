#include "graph/graph.hpp"
#include "fmt/base.h"

namespace smart {
// Add a tensorNode from existing tensor to graph and this tensor have allocated memory
auto Graph::add_tensor(const Tensor &tensor) -> TensorNode * {
	return tensors.emplace_back(new TensorNode(tensor)).get();
}
// Create a new tensorNode and allocate memory when call Executor::allocate_buffers
auto Graph::new_tensor(DataType dtype, Tensor::Shape shape) -> TensorNode * {
	return tensors.emplace_back(new TensorNode(dtype, shape)).get();
}

auto Graph::new_op(OpType type) -> OpNode * {
	return ops.emplace_back(new OpNode(type)).get();
}
// Duplicate a tensorNode(datatype + shape) and **Note**: but not share the same memory
auto Graph::dup_tensor(TensorNode *tensor) -> TensorNode * {
	return new_tensor(tensor->dtype_, tensor->shape_);
}

auto Graph::add(TensorNode *a, TensorNode *b) -> TensorNode * {
	SMART_ASSERT(a->dtype_ == b->dtype_);
	SMART_ASSERT(a->shape_ == b->shape_);

	auto out = dup_tensor(a);
	new_op(OpType::ADD)
		->set_inputs({a, b})
		->set_outputs({out});

	return out;
}

auto Graph::mat_mul(TensorNode *x, TensorNode *weight) -> TensorNode * {
	// TODO: Add checks
	SMART_ASSERT(x->shape_[0] == weight->shape_[0]);

	auto shape = x->shape_;
	shape[0]   = weight->n_elements() / weight->shape_[0];
	auto out   = new_tensor(x->dtype_, shape);
	new_op(OpType::MAT_MUL)
		->set_inputs({x, weight})
		->set_outputs({out});

	return out;
}

auto Graph::rms_norm(TensorNode *x, TensorNode *weight) -> TensorNode * {
	SMART_ASSERT(weight->n_dims() == 1);
	SMART_ASSERT(x->dtype_ == weight->dtype_);
	SMART_ASSERT(x->shape_[0] == weight->shape_[0]);

	auto out = dup_tensor(x);
	new_op(OpType::RMS_NORM)
		->set_inputs({x, weight})
		->set_outputs({out});

	return out;
}

auto Graph::silu_hadamard(TensorNode *gate, TensorNode *up) -> TensorNode * {
	SMART_ASSERT(gate->dtype_ == up->dtype_);
	SMART_ASSERT(gate->shape_ == up->shape_);

	auto out = dup_tensor(gate);
	new_op(OpType::SILU_HADAMARD)
		->set_inputs({gate, up})
		->set_outputs({out});

	return out;
}

auto Graph::rope(TensorNode *q, TensorNode *k, TensorNode *pos) -> RopeResult {
	// TODO: Add checks

	auto q_out = dup_tensor(q);
	auto k_out = dup_tensor(k);
	new_op(OpType::ROPE)
		->set_inputs({q, k, pos})
		->set_outputs({q_out, k_out});

	return {.q_out = q_out, .k_out = k_out};
}

auto Graph::softmax(TensorNode *x) -> TensorNode * {
	auto out = dup_tensor(x);
	new_op(OpType::SOFTMAX)
		->set_inputs({x})
		->set_outputs({out});

	return out;
}

auto Graph::mha(TensorNode *q, TensorNode *key_cache, TensorNode *val_cache, TensorNode *pos, size_t layer_id) -> TensorNode * {
	// TODO: Add checks

	auto out = dup_tensor(q);
	new_op(OpType::MHA)
		->set_inputs({q, key_cache, val_cache, pos})
		->set_outputs({out})
		->set_params(MHAParams(layer_id));

	return out;
}

auto Graph::copy(TensorNode *dst, TensorNode *src, size_t off) -> void {
	new_op(OpType::COPY)
		->set_inputs({dst, src})
		->set_params(CopyParams(off));
}

} // namespace smart
