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

#include "ggml-quants.h"
#include "ggml.hpp"

#include <iostream>

namespace powerserve::ggml {

using std::atomic_int;

void GGMLBackend::matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    auto dst_tensor  = convert_to_ggml(dst);
    auto src0_tensor = convert_to_ggml(src0);
    auto src1_tensor = convert_to_ggml(src1);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        params.thread_pool = (void *)m_thread_pool.get();
        params.barrier_fn  = [](void *opaque) {
            auto thread_pool = (ThreadPool *)opaque;
            thread_pool->barrier();
        };
        params.current_chunk = (atomic_int *)&m_current_chunk;

        powerserve_compute_forward_mul_mat(&params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get());
    });

    // if (dst->m_name == "attn_o_0_5") {
    //     auto file{fopen("matmul_attn_o.txt_ref", "w")};
    //     float *dst_buffer{new float[dst->m_shape[0] * dst->m_shape[1] * dst->m_shape[2] * dst->m_shape[3]]};
    //     memcpy(dst_buffer, dst->get<CPUBuffer>().m_data, dst->m_shape[0] * dst->m_shape[1] * dst->m_shape[2] *
    //     dst->m_shape[3] * sizeof(float));

    //     for (size_t i{0}; i < dst->m_shape[3]; ++i) {
    //         for (size_t j{0}; j < dst->m_shape[2]; ++j) {
    //             for (size_t k{0}; k < dst->m_shape[1]; ++k) {
    //                 for (size_t l{0}; l < dst->m_shape[0]; ++l) {
    //                     fprintf(file, "%f ", dst_buffer[i * dst->m_shape[2] * dst->m_shape[1] * dst->m_shape[0] + j *
    //                     dst->m_shape[1] * dst->m_shape[0] + k * dst->m_shape[0] + l]);
    //                 }
    //                 fprintf(file, "\n\n");
    //             }
    //             fprintf(file, "\n\n");
    //         }
    //         fprintf(file, "\n\n");
    //     }
    //     exit(0);
    // }

    // if (dst->m_name == "logits_7") {

    //     printf("src0 shape %ld %ld %ld %ld, stride %ld %ld %ld %ld\n", src0->m_shape[0], src0->m_shape[1],
    //     src0->m_shape[2], src0->m_shape[3], src0->get<CPUBuffer>().m_stride[0],
    //     src0->get<CPUBuffer>().m_stride[1], src0->get<CPUBuffer>().m_stride[2],
    //     src0->get<CPUBuffer>().m_stride[3]); printf("src1 shape %ld %ld %ld %ld, stride %ld %ld %ld %ld\n",
    //     src1->m_shape[0], src1->m_shape[1], src1->m_shape[2], src1->m_shape[3], src1->get<CPUBuffer>().m_stride[0],
    //     src1->get<CPUBuffer>().m_stride[1], src1->get<CPUBuffer>().m_stride[2],
    //     src1->get<CPUBuffer>().m_stride[3]); printf("dst shape %ld %ld %ld %ld, stride %ld %ld %ld %ld\n",
    //     dst->m_shape[0], dst->m_shape[1], dst->m_shape[2], dst->m_shape[3], dst->get<CPUBuffer>().m_stride[0],
    //     dst->get<CPUBuffer>().m_stride[1], dst->get<CPUBuffer>().m_stride[2],
    //     dst->get<CPUBuffer>().m_stride[3]); float *src0_buffer{new float[src0->m_shape[0]]};
    //     memcpy(
    //         src0_buffer, src0->get<CPUBuffer>().m_data, src0->m_shape[0] * sizeof(float)
    //     );

    //     float *src1_buffer{new float[src1->m_shape[0] * src1->m_shape[1]]};
    //     memcpy(
    //         src1_buffer, src1->get<CPUBuffer>().m_data, src1->m_shape[0] * src1->m_shape[1] * sizeof(float)
    //     );

    //     float *dst_buffer{new float[dst->m_shape[0]]};
    //     memcpy(
    //         dst_buffer, dst->get<CPUBuffer>().m_data, dst->m_shape[0] * sizeof(float)
    //     );

    //     auto file{fopen("matmul_final_ref.txt", "w")};
    //     fprintf(file, "src0:\n");

    //     for (size_t j{0}; j < src0->m_shape[0]; ++j) {
    //         fprintf(file, "%f ", src0_buffer[j]);
    //     }
    //     fprintf(file, "\n\n");

    //     fprintf(file, "src1:\n");
    //     for (size_t i{0}; i < src1->m_shape[1]; ++i) {
    //         for (size_t j{0}; j < src1->m_shape[0]; ++j) {
    //             fprintf(file, "%f ", src1_buffer[i * src1->m_shape[0] + j]);
    //         }
    //         fprintf(file, "\n\n");
    //     }
    //     fprintf(file, "\n\n");

    //     fprintf(file, "dst:\n");
    //     for (size_t j{0}; j < dst->m_shape[0]; ++j) {
    //         fprintf(file, "%f ", dst_buffer[j]);
    //     }
    //     fprintf(file, "\n\n");

    //     fclose(file);
    //     exit(0);
    // }
}

void GGMLBackend::rmsnorm(const Tensor *out, const Tensor *x, const Tensor *weight, float eps) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);
    auto src1_tensor = convert_to_ggml(weight);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_rms_norm(&params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get(), eps);
    });

    // if (weight->m_name == "output_norm.weight") {
    //     {
    //         std::cout << "eps: " << eps << std::endl;

    //         float *src_buffer{new float[x->m_shape[0] * x->m_shape[1]]};
    //         memcpy(src_buffer, x->get<CPUBuffer>().m_data, x->m_shape[0] *
    //         x->m_shape[1] * sizeof(float));

    //         float *weight_buffer{new float[weight->m_shape[0]]};
    //         memcpy(weight_buffer, weight->get<CPUBuffer>().m_data,
    //         weight->m_shape[0] * sizeof(float));

    //         float *dst_buffer{new float[out->m_shape[0] * out->m_shape[1]]};
    //         memcpy(dst_buffer, out->get<CPUBuffer>().m_data, out->m_shape[0] *
    //         out->m_shape[1] * sizeof(float));

    //         auto file{fopen("rms_norm_final_ref.txt", "w")};
    //         fprintf(file, "src0:\n");
    //         for (size_t i{0}; i < x->m_shape[1]; ++i) {
    //             for (size_t j{0}; j < x->m_shape[0]; ++j) {
    //                 fprintf(file, "%f ", src_buffer[i * x->m_shape[0] + j]);
    //             }
    //             fprintf(file, "\n\n");
    //         }
    //         fprintf(file, "\n\n");

    //         fprintf(file, "weight:\n");
    //         for (size_t i{0}; i < weight->m_shape[0]; ++i) {
    //             fprintf(file, "%f ", weight_buffer[i]);
    //         }
    //         fprintf(file, "\n\n");

    //         fprintf(file, "dst:\n");
    //         for (size_t i{0}; i < out->m_shape[1]; ++i) {
    //             for (size_t j{0}; j < out->m_shape[0]; ++j) {
    //                 fprintf(file, "%f ", dst_buffer[i * out->m_shape[0] + j]);
    //             }
    //             fprintf(file, "\n\n");
    //         }
    //         fprintf(file, "\n\n");

    //         fclose(file);
    //         exit(0);
    //     }
    // }
}

void GGMLBackend::softmax(const Tensor *out, const Tensor *x) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_soft_max(&params, dst_tensor.get(), src0_tensor.get());
    });
}

void GGMLBackend::rope(
    Tensor *out,
    const Tensor *src,
    const Tensor *rope_factors,
    const std::vector<int> &pos,
    const ModelConfig::LLMConfig::RopeConfig &rope_cfg
) const {
    POWERSERVE_UNUSED(rope_factors);
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(src);
    auto src1_tensor = std::make_unique<ggml_tensor>();
    {
        src1_tensor->data  = (void *)pos.data();
        src1_tensor->type  = GGML_TYPE_I32;
        src1_tensor->ne[0] = pos.size();
        src1_tensor->ne[1] = src1_tensor->ne[2] = src1_tensor->ne[3] = 1;
        src1_tensor->nb[0]                                           = sizeof(int32_t);
        src1_tensor->nb[1] = src1_tensor->nb[2] = src1_tensor->nb[3] = pos.size() * sizeof(int32_t);
    }

    rope_compute_params rope_params = {
        .n_dims      = rope_cfg.n_dims,
        .n_ctx_orig  = rope_cfg.n_ctx_orig,
        .freq_base   = rope_cfg.freq_base,
        .freq_scale  = rope_cfg.freq_scale,
        .ext_factor  = rope_cfg.ext_factor,
        .attn_factor = rope_cfg.attn_factor,
        .beta_fast   = rope_cfg.beta_fast,
        .beta_slow   = rope_cfg.beta_slow,
        .mode        = rope_cfg.rope_type,
    };

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_rope(
            &params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get(), nullptr, &rope_params
        );
    });

    // DEBUG
    // {
    //     auto file{fopen("rope_ref.txt", "w")};
    //     float *src0_buffer{new float[src->m_shape[0] * src->m_shape[1] * src->m_shape[2] * src->m_shape[3]]};
    //     memcpy(src0_buffer, src->get<CPUBuffer>().m_data, src->m_shape[0] * src->m_shape[1] * src->m_shape[2] *
    //     src->m_shape[3] * sizeof(float));

    //     float *src2_buffer{new float[rope_factors->m_shape[0] * rope_factors->m_shape[1] * rope_factors->m_shape[2] *
    //     rope_factors->m_shape[3]]}; memcpy(src2_buffer, rope_factors->get<CPUBuffer>().m_data,
    //     rope_factors->m_shape[0] * rope_factors->m_shape[1] * rope_factors->m_shape[2] * rope_factors->m_shape[3] *
    //     sizeof(float));

    //     float *dst_buffer{new float[out->m_shape[0] * out->m_shape[1] * out->m_shape[2] * out->m_shape[3]]};
    //     memcpy(dst_buffer, out->get<CPUBuffer>().m_data, out->m_shape[0] * out->m_shape[1] * out->m_shape[2] *
    //     out->m_shape[3] * sizeof(float));

    //     fprintf(file, "src0:\n");
    //     for (size_t i{0}; i < src->m_shape[3]; ++i) {
    //         for (size_t j{0}; j < src->m_shape[2]; ++j) {
    //             for (size_t k{0}; k < src->m_shape[1]; ++k) {
    //                 for (size_t l{0}; l < src->m_shape[0]; ++l) {
    //                     fprintf(file, "%f ", src0_buffer[i * src->m_shape[2] * src->m_shape[1] * src->m_shape[0] + j
    //                     * src->m_shape[1] * src->m_shape[0] + k * src->m_shape[0] + l]);
    //                 }
    //                 fprintf(file, "\n\n");
    //             }
    //             fprintf(file, "\n\n");
    //         }
    //         fprintf(file, "\n\n");
    //     }
    //     fprintf(file, "\n\n");

    //     fprintf(file, "src2:\n");
    //     for (size_t i{0}; i < rope_factors->m_shape[3]; ++i) {
    //         for (size_t j{0}; j < rope_factors->m_shape[2]; ++j) {
    //             for (size_t k{0}; k < rope_factors->m_shape[1]; ++k) {
    //                 for (size_t l{0}; l < rope_factors->m_shape[0]; ++l) {
    //                     fprintf(file, "%f ", src2_buffer[i * rope_factors->m_shape[2] * rope_factors->m_shape[1] *
    //                     rope_factors->m_shape[0] + j * rope_factors->m_shape[1] * rope_factors->m_shape[0] + k *
    //                     rope_factors->m_shape[0] + l]);
    //                 }
    //                 fprintf(file, "\n\n");
    //             }
    //             fprintf(file, "\n\n");
    //         }
    //         fprintf(file, "\n\n");
    //     }
    //     fprintf(file, "\n\n");

    //     fprintf(file, "dst:\n");
    //     for (size_t i{0}; i < out->m_shape[3]; ++i) {
    //         for (size_t j{0}; j < out->m_shape[2]; ++j) {
    //             for (size_t k{0}; k < out->m_shape[1]; ++k) {
    //                 for (size_t l{0}; l < out->m_shape[0]; ++l) {
    //                     fprintf(file, "%f ", dst_buffer[i * out->m_shape[2] * out->m_shape[1] * out->m_shape[0] + j *
    //                     out->m_shape[1] * out->m_shape[0] + k * out->m_shape[0] + l]);
    //                 }
    //                 fprintf(file, "\n\n");
    //             }
    //             fprintf(file, "\n\n");
    //         }
    //         fprintf(file, "\n\n");
    //     }
    //     fprintf(file, "\n\n");

    //     fclose(file);
    //     exit(0);
    // }
}

void GGMLBackend::add(const Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    auto dst_tensor  = convert_to_ggml(dst);
    auto src0_tensor = convert_to_ggml(src0);
    auto src1_tensor = convert_to_ggml(src1);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_add(&params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get());
    });

    // if (dst->m_name == "ffn_o_31_6") {
    //     printf("src0 shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", src0->m_shape[0], src0->m_shape[1],
    //     src0->m_shape[2], src0->m_shape[3],
    //         src0->get<CPUBuffer>().m_stride[0], src0->get<CPUBuffer>().m_stride[1],
    //         src0->get<CPUBuffer>().m_stride[2], src0->get<CPUBuffer>().m_stride[3]);
    //     printf("src1 shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", src1->m_shape[0], src1->m_shape[1],
    //     src1->m_shape[2], src1->m_shape[3],
    //         src1->get<CPUBuffer>().m_stride[0], src1->get<CPUBuffer>().m_stride[1],
    //         src1->get<CPUBuffer>().m_stride[2], src1->get<CPUBuffer>().m_stride[3]);
    //     printf("dst shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", dst->m_shape[0], dst->m_shape[1],
    //     dst->m_shape[2], dst->m_shape[3],
    //         dst->get<CPUBuffer>().m_stride[0], dst->get<CPUBuffer>().m_stride[1],
    //         dst->get<CPUBuffer>().m_stride[2], dst->get<CPUBuffer>().m_stride[3]);

    //     float *dst_buffer{new float[dst->m_shape[0] * dst->m_shape[1] * dst->m_shape[2] * dst->m_shape[3]]};
    //     memcpy(dst_buffer, dst->get<CPUBuffer>().m_data, dst->m_shape[0] *
    //     dst->m_shape[1] * dst->m_shape[2] * dst->m_shape[3] * sizeof(float));

    //     auto file{fopen("add_ffn_o_ref.txt", "w")};
    //     for (size_t i = 0; i < dst->m_shape[3]; i++) {
    //         for (size_t j = 0; j < dst->m_shape[2]; j++) {
    //             for (size_t k = 0; k < dst->m_shape[1]; k++) {
    //                 for (size_t l = 0; l < dst->m_shape[0]; l++) {
    //                     fprintf(file, "%f ", dst_buffer[i * dst->m_shape[2] * dst->m_shape[1] * dst->m_shape[0] + j *
    //                     dst->m_shape[1] * dst->m_shape[0] + k * dst->m_shape[0] + l]);
    //                 }
    //                 fprintf(file, "\n\n");
    //             }
    //             fprintf(file, "\n\n\n");
    //         }
    //         fprintf(file, "\n\n");
    //     }
    //     fclose(file);
    //     delete[] dst_buffer;
    //     exit(0);
    // }
}

void GGMLBackend::permute(const Tensor *out, const Tensor *x, Shape axes) const {
    Stride stride{};
    stride[axes[0]] = x->get<CPUBuffer>().m_stride[0];
    stride[axes[1]] = x->get<CPUBuffer>().m_stride[1];
    stride[axes[2]] = x->get<CPUBuffer>().m_stride[2];
    stride[axes[3]] = x->get<CPUBuffer>().m_stride[3];

    out->get<CPUBuffer>().m_stride = stride;
}

void GGMLBackend::cont(const Tensor *out, const Tensor *x) const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_dup(&params, dst_tensor.get(), src0_tensor.get());
    });
}

void GGMLBackend::copy(const Tensor *dst, const Tensor *src) const {
    auto dst_tensor  = convert_to_ggml(dst);
    auto src0_tensor = convert_to_ggml(src);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_dup(&params, dst_tensor.get(), src0_tensor.get());
    });
}

void GGMLBackend::softmax_ext(const Tensor *out, const Tensor *x, const Tensor *mask, float scale, float max_bias)
    const {
    auto dst_tensor  = convert_to_ggml(out);
    auto src0_tensor = convert_to_ggml(x);
    auto src1_tensor = convert_to_ggml(mask);

    m_thread_pool->run([&](size_t thread_id) {
        op_compute_params params = m_params;

        params.ith = thread_id;
        params.nth = m_thread_pool->size();

        powerserve_compute_forward_softmax_ext(
            &params, dst_tensor.get(), src0_tensor.get(), src1_tensor.get(), scale, max_bias
        );
    });

    // if (out->m_name == "kq_0_5") {
    //     auto file{fopen("softmax_ref.txt", "w")};
    //     float *dst_buffer{new float[out->m_shape[0] * out->m_shape[1] * out->m_shape[2] * out->m_shape[3]]};
    //     memcpy(dst_buffer, out->get<CPUBuffer>().m_data, out->m_shape[0] * out->m_shape[1] * out->m_shape[2] *
    //     out->m_shape[3] * sizeof(float)); printf("dst shape is %ld %ld %ld %ld, nb is %ld %ld %ld %ld\n",
    //         out->m_shape[0], out->m_shape[1], out->m_shape[2], out->m_shape[3],
    //             out->get<CPUBuffer>().m_stride[0], out->get<CPUBuffer>().m_stride[1],
    //             out->get<CPUBuffer>().m_stride[2], out->get<CPUBuffer>().m_stride[3]);
    //     fprintf(file, "dst:\n");
    //     for (size_t i{0}; i < out->m_shape[3]; ++i) {
    //         for (size_t j{0}; j < out->m_shape[2]; ++j) {
    //             for (size_t k{0}; k < out->m_shape[1]; ++k) {
    //                 for (size_t l{0}; l < out->m_shape[0]; ++l) {
    //                     fprintf(file, "%f ", dst_buffer[i * out->m_shape[2] * out->m_shape[1] * out->m_shape[0] + j *
    //                     out->m_shape[1] * out->m_shape[0] + k * out->m_shape[0] + l]);
    //                 }
    //                 fprintf(file, "\n\n");
    //             }
    //             fprintf(file, "\n\n");
    //         }
    //         fprintf(file, "\n\n");
    //     }
    //     exit(0);
    // }
}

void GGMLBackend::get_embedding(const Tensor *dst, const Tensor *weight, const std::vector<int> &tokens) const {
    auto embd_tb = static_cast<char *>(weight->get<CPUBuffer>().m_data);
    auto dst_tb  = static_cast<float *>(dst->get<CPUBuffer>().m_data);

    auto dim        = dst->m_shape[0];
    auto batch_size = tokens.size();
    POWERSERVE_ASSERT(batch_size == dst->m_shape[1]);
    auto weight_strip = weight->get<CPUBuffer>().m_stride;

    for (size_t i = 0; i < batch_size; i++) {
        auto token = tokens[i];
        // printf("weightstrip is %ld, token is %d\n", weight_strip[1], token);
        auto src = embd_tb + weight_strip[1] * token;
        POWERSERVE_ASSERT(src < embd_tb + weight_strip[2]);
        switch (weight->m_dtype) {
        case DataType::FP32: {
            memcpy(dst_tb + i * dim, src, dim * sizeof(float));
        } break;

        case DataType::FP16: {
            auto ptr = dst_tb + i * dim;
            for (size_t j = 0; j < dim; ++j) {
                auto src_ptr = (ggml_fp16_t *)src;
                ptr[j]       = ggml_fp16_to_fp32(src_ptr[j]);
            }
        } break;

        case DataType::GGML_Q4_0: {
            dequantize_row_q4_0((block_q4_0 *)src, dst_tb + i * dim, dim);
        } break;

        case DataType::GGML_Q8_0: {
            dequantize_row_q8_0((block_q8_0 *)src, dst_tb + i * dim, dim);
        } break;

        default:
            POWERSERVE_ASSERT(false);
        }
    }

    // {
    //     if (tokens.size() == 1) {
    //         auto file{fopen("get_embedding_ref.txt", "w")};
    //         for (size_t i = 0; i < dst->m_shape[0]; i++) {
    //             fprintf(file, "%.6f ", dst_tb[i]);
    //         }
    //         fprintf(file, "\n");

    //         fclose(file);
    //         exit(0);
    //     }
    // }
}

bool GGMLBackend::is_contiguous(const Tensor *tensor, int n) const {
    POWERSERVE_ASSERT(n >= 0 && n <= 2);
    if (n == 0) {
        return ggml_is_contiguous_0(convert_to_ggml(tensor).get());
    } else if (n == 1) {
        return ggml_is_contiguous_1(convert_to_ggml(tensor).get());
    } else if (n == 2) {
        return ggml_is_contiguous_2(convert_to_ggml(tensor).get());
    }
    return false;
}

int GGMLBackend::get_n_tasks(std::shared_ptr<OpNode> op) {
    int n_tasks = 1;

    switch (op->op) {
    // custom ops
    case OpType::SILU_HADAMARD:
    case OpType::ADD_CACHE:
    case OpType::PRINT:
    case OpType::VIEW:
    case OpType::TRANSPOSE:
    case OpType::COPY: {
        n_tasks = 1;
    } break;

    // ggml wrapper ops
    case OpType::PERMUTE:
    case OpType::GET_MASK:
    case OpType::GET_EMBEDDING: {
        n_tasks = 1;
    } break;

    case OpType::ROPE:
    case OpType::RMS_NORM:
    case OpType::CONT:
    case OpType::MAT_MUL:
    case OpType::ADD: {
        n_tasks = num_threads;
    } break;

    case OpType::SOFTMAX_EXT:
    case OpType::SOFTMAX: {
        n_tasks = std::min((int64_t)num_threads, op->prev[0]->tensor()->nrows());
    } break;

#if defined(POWERSERVE_WITH_QNN)
    case OpType::QNN_FORWARD: {
        n_tasks = 1;
    } break;
    case OpType::QNN_FORWARD_VL: {
        n_tasks = 1;
    } break;
#endif

    default: {
        fmt::println("op not implemented: {}", int(op->op));
        POWERSERVE_ASSERT(false);
    }
    }

    return n_tasks;
}

ggml_type GGMLBackend::get_vec_dot_type(const Tensor *tensor) {
    auto t = convert_to_ggml(tensor);
    return powerserve_get_vec_dot_type(t.get());
}

} // namespace powerserve::ggml
