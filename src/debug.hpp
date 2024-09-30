#pragma once

#include "debug.hpp"
#include "ggml.h"
#include "model.hpp"

namespace smart {
void debug_meta_info(gguf_context *gguf_ctx, ggml_context *ggml_ctx);
void debug_tensors_info(gguf_context *gguf_ctx, ggml_context * ggml_ctx);
void debug_config_info(Transformer::Config *c);
void debug_weight_info(std::string name, OpTensor *opt);
void debug_weights_info(Transformer::TransformerWeights *w);
}