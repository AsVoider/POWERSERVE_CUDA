// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "model/module/ffn.hpp"

#include "graph/graph.hpp"
#include "graph/node.hpp"

namespace powerserve {

TensorNode *FFN::build(Graph &g, TensorNode *attn_o, int64_t L) {
    auto ffn_norm_w = g.add_tensor(m_weights->lw[L].ffn_norm);
    auto ffn_norm_o = g.rms_norm(attn_o, ffn_norm_w, m_config.norm_eps);
    ffn_norm_o->m_name = fmt::format("ffn_norm_o_{}", L);

    auto gate_w = g.add_tensor(m_weights->lw[L].ffn_gate);
    auto gate_o = g.mat_mul(gate_w, ffn_norm_o);
    gate_o->m_name = fmt::format("gate_o_{}", L);

    auto up_w = g.add_tensor(m_weights->lw[L].ffn_up);
    auto up_o = g.mat_mul(up_w, ffn_norm_o);
    up_o->m_name = fmt::format("up_o_{}", L);

    // {hidden_dim, bs, 1, 1}
    auto silu = g.silu_hadamard(gate_o, up_o);
    silu->m_name = fmt::format("silu_{}", L);

    auto down_w = g.add_tensor(m_weights->lw[L].ffn_down);
    auto down_o = g.mat_mul(down_w, silu);
    down_o->m_name = fmt::format("down_o_{}", L);

    // {embed_dim, bs, 1, 1}
    auto res_conn = g.add(attn_o, down_o);

    return res_conn;
}

} // namespace powerserve
