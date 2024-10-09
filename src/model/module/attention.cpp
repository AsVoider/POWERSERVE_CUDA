#include "attention.hpp"
#include "graph/graph.hpp"
namespace smart {

void Attention::build_graph(Graph &g) {
    return;
    // Operator *rmsnorm_attn = new Operator(GGML_OP_RMS_NORM);
    // Operator *mulmat_q = new Operator(GGML_OP_MUL_MAT);
    // Operator *mulmat_k = new Operator(GGML_OP_MUL_MAT);
    // Operator *mulmat_v = new Operator(GGML_OP_MUL_MAT);
    // Operator *rope = new Operator(GGML_OP_ROPE);
    // // TODO: MHA + residual connection
    // Operator *mha = new Operator(GGML_OP_MAP_CUSTOM1);
    // Operator *attn_output = new Operator(GGML_OP_MUL_MAT);
    // Operator *residual_conn = new Operator(GGML_OP_MAP_CUSTOM2);

    // Tensor *attn_input = new Tensor(); // shared ptr

}

} // namespace smart