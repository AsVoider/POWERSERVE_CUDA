#include <ggml.h>

int main() {

    struct ggml_init_params params = {
        .mem_size   = 16 * 1024 * 1024,
        .mem_buffer = NULL,
    };

    // memory allocation happens here
    struct ggml_context *ctx = ggml_init(params);
    struct ggml_tensor *a    = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 64, 2);
    // ggml_set_param(ctx, a); // a is an input variable
    struct ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 64, 128);

    struct ggml_tensor *a2 = ggml_mul_mat(ctx, w, a);

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, a2);
    // ggml_set_2d();
    ggml_graph_compute_with_ctx(ctx, gf, 1);
    return 0;

    // ggml_quantize_chunk(
    // );
}
