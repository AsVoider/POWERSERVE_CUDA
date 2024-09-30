#pragma once

#include "ggml.h"
#include <cassert>
#include <cstdint>
namespace smart {

#define QK8_0 32

struct block_q8_0{
    uint16_t d;        // delta
    int8_t   qs[QK8_0]; // quants
};

#define QK4_0 32

struct block_q4_0{
    uint16_t d;           // delta
    uint8_t  qs[QK4_0 / 2]; // nibbles / quants
};

OpTensor *get_optensor_from_ggml(ggml_tensor *t);
void free_optensor_deep(OpTensor *opt);
void free_optensor(OpTensor *opt);

void dequantize_row_q8_0(const block_q8_0 *x, float * y, int64_t k);
void dequantize_row_q4_0(const block_q4_0 * x, float * y, int64_t k);

void rmsnorm_internal(float* o, float* x, float* weight, int64_t size);
void rmsnorm(OpTensor *o, OpTensor *x, OpTensor *weight);
void softmax_internal(float* x, int64_t size);
void softmax(OpTensor *x, int64_t size);
void matmul(float* xout, float* x, float* w, int n, int d);

} // namespace smart
