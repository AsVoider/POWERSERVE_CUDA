#include "model/module/ffn.hpp"

#include "graph/graph.hpp"
#include "graph/node.hpp"

namespace smart {

TensorNode *FFN::build(Graph &g, TensorNode *attn_o, int64_t L) {
    auto ffn_norm_w = g.add_tensor(m_weights->lw[L].ffn_norm);
    auto ffn_norm_o = g.rms_norm(attn_o, ffn_norm_w, m_config.norm_eps);

    auto gate_w = g.add_tensor(m_weights->lw[L].ffn_gate);
    auto gate_o = g.mat_mul(gate_w, ffn_norm_o);

    auto up_w = g.add_tensor(m_weights->lw[L].ffn_up);
    auto up_o = g.mat_mul(up_w, ffn_norm_o);

    // {hidden_dim, bs, 1, 1}
    auto silu = g.silu_hadamard(gate_o, up_o);

    auto down_w = g.add_tensor(m_weights->lw[L].ffn_down);
    auto down_o = g.mat_mul(down_w, silu);

    // {embed_dim, bs, 1, 1}
    auto res_conn = g.add(attn_o, down_o);

    return res_conn;
}

} // namespace smart
