#include "graph/graph.hpp"

namespace smart {

// Add a tensorNode from existing tensor to graph and this tensor have allocated memory
auto Graph::add_tensor(const Tensor &tensor) -> TensorNode * {
	return tensors.emplace_back(new TensorNode(tensor)).get();
}

// Create a new tensorNode and allocate memory when call Executor::allocate_buffers
auto Graph::new_tensor(DataType dtype, const Tensor::Shape &shape) -> TensorNode * {
	return tensors.emplace_back(new TensorNode(dtype, shape)).get();
}

auto Graph::new_op(OpType type) -> OpNode * {
	return ops.emplace_back(new OpNode(type)).get();
}

// Duplicate a tensorNode(datatype + shape) and **Note**: but not share the same memory
auto Graph::dup_tensor(TensorNode *tensor) -> TensorNode * {
	return new_tensor(tensor->dtype, tensor->shape);
}

auto Graph::add(TensorNode *a, TensorNode *b) -> TensorNode * {
	SMART_ASSERT(a->dtype == b->dtype);
	SMART_ASSERT(a->shape == b->shape);

	auto out = dup_tensor(a);
	auto op	 = new_op(OpType::ADD);
	op->set_inputs({a, b});
	op->set_outputs({out});

	return out;
}

auto Graph::mat_mul(TensorNode *x, TensorNode *weight) -> TensorNode * {
	// TODO: Add checks
	SMART_ASSERT(x->shape[0] == weight->shape[0]);

	auto shape = x->shape;
	shape[0]   = weight->n_elements() / weight->shape[0];
	auto out   = new_tensor(x->dtype, shape);
	auto op	   = new_op(OpType::MAT_MUL);
	op->set_inputs({x, weight});
	op->set_outputs({out});

	return out;
}

auto Graph::rms_norm(TensorNode *x, TensorNode *weight) -> TensorNode * {
	SMART_ASSERT(weight->n_dims() == 1);
	SMART_ASSERT(x->dtype == weight->dtype);
	SMART_ASSERT(x->shape[0] == weight->shape[0]);

	auto out = dup_tensor(x);
	auto op	 = new_op(OpType::RMS_NORM);
	op->set_inputs({x, weight});
	op->set_outputs({out});

	return out;
}

auto Graph::silu_hadamard(TensorNode *gate, TensorNode *up) -> TensorNode * {
	SMART_ASSERT(gate->dtype == up->dtype);
	SMART_ASSERT(gate->shape == up->shape);

	auto out = dup_tensor(gate);
	auto op	 = new_op(OpType::SILU_HADAMARD);
	op->set_inputs({gate, up});
	op->set_outputs({out});

	return out;
}

auto Graph::rope(TensorNode *q, TensorNode *k, TensorNode *pos) -> RopeResult {
	// TODO: Add checks

	auto q_out = dup_tensor(q);
	auto k_out = dup_tensor(k);
	auto op	   = new_op(OpType::ROPE);
	op->set_inputs({q, k, pos});
	op->set_outputs({q_out, k_out});

	return {.q_out = q_out, .k_out = k_out};
}

auto Graph::softmax(TensorNode *x) -> TensorNode * {
	auto out = dup_tensor(x);
	auto op	 = new_op(OpType::SOFTMAX);
	op->set_inputs({x});
	op->set_outputs({out});

	return out;
}

auto Graph::mha(TensorNode *q, TensorNode *key_cache, TensorNode *val_cache, TensorNode *pos, size_t layer_id)
	-> TensorNode * {
	// TODO: Add checks

	auto out = dup_tensor(q);
	auto op	 = new_op(OpType::MHA);
	op->set_inputs({q, key_cache, val_cache, pos});
	op->set_outputs({out});
	op->set_params(MHAParams{.layer_id = layer_id});

	return out;
}

auto Graph::copy(TensorNode *dst, TensorNode *src, size_t off) -> void {
	auto op = new_op(OpType::COPY);
	op->set_inputs({dst, src});
	op->set_params(CopyParams{.off = off});
}

} // namespace smart
