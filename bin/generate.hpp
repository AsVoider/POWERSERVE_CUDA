#pragma once

#include "model.hpp"
#include "sampler.hpp"
#include "llama_tokenizer.hpp"

namespace smart {

// common funcs
long time_in_ms();
void rmsnorm(OpTensor *o, OpTensor *x, OpTensor *weight);
void softmax(OpTensor *x, int64_t size);
void matmul(float* xout, float* x, float* w, int n, int d);
void safe_printf(std::string piece);
float* forward(Transformer* tf, int token, int pos);

void generate(Transformer *tf, smart::LlamaTokenizer *tk, Sampler *sampler, std::string prompt, int steps);



} // namespace smart