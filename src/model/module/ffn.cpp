#include "model/module/ffn.hpp"
#include "graph/graph.hpp"
namespace smart {

void FFN::build_graph(Graph &g, std::shared_ptr<LlamaConfig> config, std::shared_ptr<LlamaWeight> weights, std::shared_ptr<LlamaBuffer> buffer, int64_t L, int64_t pos) {
	auto ffn_rms_norm = std::make_shared<Operator>(OpType::OP_RMS_NORM);
	auto ffn_norm_xb  = buffer->get_new_node_dim("xb");
	auto x			  = buffer->get_new_node_dim("x");
	g.nodes.push_back(ffn_rms_norm);
	g.nodes.push_back(ffn_norm_xb);
	g.nodes.push_back(x);
	add_input(ffn_rms_norm, x);
	add_input(ffn_rms_norm, weights->lw[L].ffn_norm);
	add_output(ffn_rms_norm, ffn_norm_xb);

	auto mulmat_gate = std::make_shared<Operator>(OpType::OP_MUL_MAT);
	auto hb			 = buffer->get_new_node_hidden_dim("hb");
	g.nodes.push_back(mulmat_gate);
	g.nodes.push_back(hb);
	add_input(mulmat_gate, ffn_norm_xb);
	add_input(mulmat_gate, weights->lw[L].ffn_gate);
	add_output(mulmat_gate, hb);

	auto mulmat_up = std::make_shared<Operator>(OpType::OP_MUL_MAT);
	auto hb2	   = buffer->get_new_node_hidden_dim("hb2");
	g.nodes.push_back(mulmat_up);
	g.nodes.push_back(hb2);
	add_input(mulmat_up, ffn_norm_xb);
	add_input(mulmat_up, weights->lw[L].ffn_up);
	add_output(mulmat_up, hb2);

	auto silu = std::make_shared<Operator>(OpType::OP_SILU_HADAMARD);
	g.nodes.push_back(silu);
	add_input(silu, hb);
	add_input(silu, hb2);

	auto mulmat_down = std::make_shared<Operator>(OpType::OP_MUL_MAT);
	g.nodes.push_back(mulmat_down);
	add_input(mulmat_down, hb);
	add_input(mulmat_down, weights->lw[L].ffn_down);
	add_output(mulmat_down, ffn_norm_xb);

	auto ffn_res_conn = std::make_shared<Operator>(OpType::OP_RES_CONN);
	auto x2			  = buffer->get_new_node_dim("x");
	g.nodes.push_back(ffn_res_conn);
	g.nodes.push_back(x2);
	add_input(ffn_res_conn, x2);
	add_input(ffn_res_conn, ffn_norm_xb);
}

} // namespace smart