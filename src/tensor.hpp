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


inline void dequantize_row_q8_0(const block_q8_0 *x, float * y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

inline void dequantize_row_q4_0(const block_q4_0 * x, float * y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = ggml_fp16_to_fp32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}



} // namespace smart
