#include "ggml-cuda.hpp"

#include "backend/ggml-cuda/buffer.hpp"
#include "ggml-quants.h"
#include "ggml.h"

#include <algorithm>

namespace powerserve::ggml_cuda {

void GGML_CUDABackend::get_embedding(Tensor *dst, const Tensor *weight, const std::vector<int> &tokens)
    const { // finish
    auto ggml_tensor_dst{convert_to_ggml_tensor(dst)};
    auto ggml_tensor_weight{convert_to_ggml_tensor(weight)};
    auto ggml_tensor_tokens{new ggml_tensor{}};
    auto buffer_stride{
        Stride{sizeof(int), sizeof(int) * tokens.size(), sizeof(int) * tokens.size(), sizeof(int) * tokens.size()}
    };
    auto tensor_shape{Shape{tokens.size(), 1, 1, 1}};
    void *cuda_ptr{nullptr};
    cuda_context_warp::malloc_cuda_buffer(&cuda_ptr, sizeof(int) * tokens.size());
    cuda_context_warp::copy_memory<1>(
        cuda_ptr, reinterpret_cast<void *>(const_cast<int *>(tokens.data())), sizeof(int) * tokens.size()
    );
    ggml_tensor_tokens->data = cuda_ptr;
    ggml_tensor_tokens->type = GGML_TYPE_I32;
    memcpy(ggml_tensor_tokens->ne, tensor_shape.data(), tensor_shape.size() * sizeof(Shape::size_type));
    memcpy(ggml_tensor_tokens->nb, buffer_stride.data(), buffer_stride.size() * sizeof(Stride::size_type));

    ggml_tensor_dst->src[0] = ggml_tensor_weight.get();
    ggml_tensor_dst->src[1] = ggml_tensor_tokens;
    op_interfaces::op_get_embedding(*warp, ggml_tensor_dst.get());

    // DEBUG
    // if (tokens.size() == 1) {
    //     cuda_context_warp::device_sync();
    //     auto file{fopen("get_embedding.txt", "a+")};
    //     float *dst_buffer{new float[dst->m_shape[0] * dst->m_shape[1]]};
    //     cuda_context_warp::copy_memory<2>(dst_buffer, dst->m_data->m_data_device, dst->m_shape[0] *
    //     dst->m_shape[1] * sizeof(float)); cuda_context_warp::device_sync();

    //     for (size_t i{0}; i < dst->m_shape[1]; ++i) {
    //         for (size_t j{0}; j < dst->m_shape[0]; ++j) {
    //             fprintf(file, "%f ", dst_buffer[i * dst->m_shape[0] + j]);
    //         }
    //         fprintf(file, "\n\n");
    //     }

    //     fclose(file);
    //     exit(0);
    // }
}

void GGML_CUDABackend::matmul(Tensor *dst, const Tensor *src0, const Tensor *src1) const { // finish
    auto split{src0->m_backend == TensorBackend::GGML_GPU_SPLIT};
    auto ggml_tensor_dst{convert_to_ggml_tensor(dst)};
    auto ggml_tensor_src0{convert_to_ggml_tensor(src0)};
    auto ggml_tensor_src1{convert_to_ggml_tensor(src1)};

    ggml_tensor_dst->src[0] = ggml_tensor_src0.get();
    ggml_tensor_dst->src[1] = ggml_tensor_src1.get();

    ggml_tensor_dst->op_params[15] = split ? 1 : 0; // 16th param: split or not?

    op_interfaces::op_mat_mul(*warp, ggml_tensor_dst.get());

    // printf("dst name is %s\n", dst->m_name.c_str());
    // DEBUG
    // if (dst->m_name == "attn_o_0_0") {
    //     printf("src1 shape %ld %ld %ld %ld, stride %ld %ld %ld %ld\n", src1->m_shape[0], src1->m_shape[1],
    //            src1->m_shape[2], src1->m_shape[3], src1->m_data->m_stride[0],
    //            src1->m_data->m_stride[1], src1->m_data->m_stride[2],
    //            src1->m_data->m_stride[3]);
    //     cuda_context_warp::device_sync();
    //     auto file{fopen("matmul_attn_o.txt", "w")};
    //     float *src_buffer{new float[src1->m_shape[0] * src1->m_shape[1] * src1->m_shape[2] * src1->m_shape[3]]};
    //     float *dst_buffer{new float[dst->m_shape[0] * dst->m_shape[1] * dst->m_shape[2] * dst->m_shape[3]]};
    //     cuda_context_warp::copy_memory<2>(
    //         src_buffer, src1->m_data->m_data_device, src1->m_shape[0] * src1->m_shape[1] * src1->m_shape[2] *
    //         src1->m_shape[3] * sizeof(float)
    //     );
    //     cuda_context_warp::copy_memory<2>(
    //         dst_buffer, dst->m_data->m_data_device, dst->m_shape[0] * dst->m_shape[1] * dst->m_shape[2] *
    //         dst->m_shape[3] * sizeof(float)
    //     );
    //     cuda_context_warp::device_sync();

    //     for (size_t i{0}; i < src1->m_shape[3]; ++i) {
    //         for (size_t j{0}; j < src1->m_shape[2]; ++j) {
    //             for (size_t k{0}; k < src1->m_shape[1]; ++k) {
    //                 for (size_t l{0}; l < src1->m_shape[0]; ++l) {
    //                     fprintf(file, "%f ", src_buffer[i * src1->m_shape[2] * src1->m_shape[1] * src1->m_shape[0] + j *
    //                     src1->m_shape[1] * src1->m_shape[0] + k * src1->m_shape[0] + l]);
    //                 }
    //                 fprintf(file, "\n\n");
    //             }
    //             fprintf(file, "\n\n");
    //         }
    //         fprintf(file, "\n\n");
    //     }

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
    //     fclose(file);
    //     exit(0);
    // }
    // DEBUG
    // if (dst->m_name.substr(0, 7) == "logits_") {
    //     cuda_context_warp::device_sync();
    //     dst->m_data->m_data_host = new float[dst->m_shape[0]];
    //     cuda_context_warp::copy_memory<2>(
    //         dst->m_data->m_data_host, dst->m_data->m_data_device, dst->m_shape[0] * sizeof(float)
    //     );
    //     printf("dst cuda ptr is %p\n", dst->m_data->m_data_device);
    //     cuda_context_warp::device_sync();

    //     if (dst->m_name == "logits_7") {
    //         cuda_context_warp::device_sync();
    //         printf("src0 shape %ld %ld %ld %ld, stride %ld %ld %ld %ld\n", src0->m_shape[0], src0->m_shape[1],
    //         src0->m_shape[2], src0->m_shape[3], src0->m_data->m_stride[0],
    //         src0->m_data->m_stride[1], src0->m_data->m_stride[2],
    //         src0->m_data->m_stride[3]); printf("src1 shape %ld %ld %ld %ld, stride %ld %ld %ld %ld\n",
    //         src1->m_shape[0], src1->m_shape[1], src1->m_shape[2], src1->m_shape[3],
    //         src1->m_data->m_stride[0], src1->m_data->m_stride[1],
    //         src1->m_data->m_stride[2], src1->m_data->m_stride[3]); printf("dst shape %ld %ld
    //         %ld %ld, stride %ld %ld %ld %ld\n", dst->m_shape[0], dst->m_shape[1], dst->m_shape[2], dst->m_shape[3],
    //         dst->m_data->m_stride[0], dst->m_data->m_stride[1],
    //         dst->m_data->m_stride[2], dst->m_data->m_stride[3]); float *src0_buffer{new
    //         float[src0->m_shape[0]]}; cuda_context_warp::copy_memory<2>(
    //             src0_buffer, src0->m_data->m_data_device, src0->m_shape[0] * sizeof(float)
    //         );

    //         float *src1_buffer{new float[src1->m_shape[0] * src1->m_shape[1]]};
    //         cuda_context_warp::copy_memory<2>(
    //             src1_buffer, src1->m_data->m_data_device, src1->m_shape[0] * src1->m_shape[1] *
    //             sizeof(float)
    //         );

    //         cuda_context_warp::device_sync();

    //         auto file{fopen("matmul_final.txt", "w")};
    //         fprintf(file, "src0:\n");

    //         for (size_t j{0}; j < src0->m_shape[0]; ++j) {
    //             fprintf(file, "%f ", src0_buffer[j]);
    //         }
    //         fprintf(file, "\n\n");

    //         fprintf(file, "src1:\n");
    //         for (size_t i{0}; i < src1->m_shape[1]; ++i) {
    //             for (size_t j{0}; j < src1->m_shape[0]; ++j) {
    //                 fprintf(file, "%f ", src1_buffer[i * src1->m_shape[0] + j]);
    //             }
    //             fprintf(file, "\n\n");
    //         }
    //         fprintf(file, "\n\n");

    //         fprintf(file, "dst:\n");
    //         float *host_ptr = static_cast<float *>(dst->m_data->m_data_host);
    //         for (size_t j{0}; j < dst->m_shape[0]; ++j) {
    //             fprintf(file, "%f ", host_ptr[j]);
    //         }
    //         fprintf(file, "\n\n");

    //         fclose(file);
    //     }
    //     exit(0);
    // }
}

void GGML_CUDABackend::rmsnorm(Tensor *o, const Tensor *x, const Tensor *weight, float eps) const { // finish
    auto ggml_tensor_o{convert_to_ggml_tensor(o)};
    POWERSERVE_ASSERT(ggml_tensor_o->data not_eq nullptr);
    memcpy(&ggml_tensor_o->op_params[0], &eps, sizeof(float));

    auto ggml_tensor_x{convert_to_ggml_tensor(x)};
    auto ggml_tensor_weight{convert_to_ggml_tensor(weight)};
    ggml_tensor_o->src[0] = ggml_tensor_x.get();
    ggml_tensor_o->src[1] = ggml_tensor_weight.get();

    op_interfaces::op_rms_norm(*warp, ggml_tensor_o.get());

    // DEBUG
    // if (weight->m_name == "output_norm.weight") {
    //     {
    //         cuda_context_warp::device_sync();
    //         std::cout << "eps: " << eps << std::endl;

    //         float *src_buffer{new float[x->m_shape[0] * x->m_shape[1]]};
    //         cuda_context_warp::copy_memory<2>(src_buffer, x->m_data->m_data_device, x->m_shape[0] *
    //         x->m_shape[1] * sizeof(float));

    //         float *weight_buffer{new float[weight->m_shape[0]]};
    //         cuda_context_warp::copy_memory<2>(weight_buffer, weight->m_data->m_data_device,
    //         weight->m_shape[0] * sizeof(float));

    //         float *dst_buffer{new float[o->m_shape[0] * o->m_shape[1]]};
    //         cuda_context_warp::copy_memory<2>(dst_buffer, o->m_data->m_data_device, o->m_shape[0] *
    //         o->m_shape[1] * sizeof(float));

    //         cuda_context_warp::device_sync();

    //         auto file{fopen("rms_norm_final.txt", "w")};
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
    //         for (size_t i{0}; i < o->m_shape[1]; ++i) {
    //             for (size_t j{0}; j < o->m_shape[0]; ++j) {
    //                 fprintf(file, "%f ", dst_buffer[i * o->m_shape[0] + j]);
    //             }
    //             fprintf(file, "\n\n");
    //         }
    //         fprintf(file, "\n\n");

    //         fclose(file);
    //         exit(0);
    //     }
    // }
}

void GGML_CUDABackend::softmax(Tensor *out, const Tensor *x, const Tensor *mask, float scale, float bias) const {
    auto ggml_tensor_out{convert_to_ggml_tensor(out)};
    auto ggml_tensor_x{convert_to_ggml_tensor(x)};
    auto ggml_tensor_mask{convert_to_ggml_tensor(mask)};

    memcpy(&ggml_tensor_out->op_params[0], &scale, sizeof(float));
    memcpy(&ggml_tensor_out->op_params[1], &bias, sizeof(float));

    ggml_tensor_out->src[0] = ggml_tensor_x.get();
    ggml_tensor_out->src[1] = ggml_tensor_mask.get();

    op_interfaces::op_softmax(*warp, ggml_tensor_out.get());

    // DEBUG
    // if (out->m_name == "kq_0_5") {
    //     cuda_context_warp::device_sync();
    //     auto file{fopen("softmax.txt", "w")};
    //     float *dst_buffer{new float[out->m_shape[0] * out->m_shape[1] * out->m_shape[2] * out->m_shape[3]]};
    //     cuda_context_warp::copy_memory<2>(dst_buffer, out->m_data->m_data_device, out->m_shape[0] *
    //     out->m_shape[1] * out->m_shape[2] * out->m_shape[3] * sizeof(float)); cuda_context_warp::device_sync();
    //     printf("dst shape is %ld %ld %ld %ld, nb is %ld %ld %ld %ld\n",
    //         out->m_shape[0], out->m_shape[1], out->m_shape[2], out->m_shape[3],
    //             out->m_data->m_stride[0], out->m_data->m_stride[1],
    //             out->m_data->m_stride[2], out->m_data->m_stride[3]);
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

void GGML_CUDABackend::rope(
    Tensor *out,
    const Tensor *src,
    const Tensor *rope_frators,
    const std::vector<int> &pos,
    const ModelConfig::LLMConfig::RopeConfig &rope_cfg
) const {
    auto ggml_tensor_out{convert_to_ggml_tensor(out)};
    auto ggml_tensor_src{convert_to_ggml_tensor(src)};
    auto ggml_tensor_rope_factors{convert_to_ggml_tensor(rope_frators)};

    auto ggml_tensor_pos{std::make_unique<ggml_tensor>()};
    {
        void *pos_data_ptr{nullptr};
        void *cpu_data_ptr = static_cast<void *>(const_cast<int *>(pos.data()));
        cuda_context_warp::malloc_cuda_buffer(&pos_data_ptr, pos.size() * sizeof(int));
        cuda_context_warp::copy_memory_async<1>(pos_data_ptr, cpu_data_ptr, pos.size() * sizeof(int), nullptr);
        ggml_tensor_pos->data  = pos_data_ptr;
        ggml_tensor_pos->type  = GGML_TYPE_I32;
        ggml_tensor_pos->ne[0] = pos.size();
        ggml_tensor_pos->ne[1] = ggml_tensor_pos->ne[2] = ggml_tensor_pos->ne[3] = 1;
        ggml_tensor_pos->nb[0]                                                   = sizeof(int32_t);
        ggml_tensor_pos->nb[1] = ggml_tensor_pos->nb[2] = ggml_tensor_pos->nb[3] = pos.size() * sizeof(int32_t);
    }

    const auto &out_buffer{*out->m_data};
    const auto &src_buffer{*src->m_data};
    POWERSERVE_ASSERT(out_buffer.m_data_device and src_buffer.m_data_device);

    ggml_tensor_out->src[0] = ggml_tensor_src.get();
    ggml_tensor_out->src[1] = ggml_tensor_pos.get();
    ggml_tensor_out->src[2] = ggml_tensor_rope_factors.get();

    int arr_i[5]{0, rope_cfg.n_dims, rope_cfg.rope_type, 0, rope_cfg.n_ctx_orig};
    float arr_f[6]{
        rope_cfg.freq_base,
        rope_cfg.freq_scale,
        rope_cfg.ext_factor,
        rope_cfg.attn_factor,
        rope_cfg.beta_fast,
        rope_cfg.beta_slow
    };

    // DEBUG
    // {
    //     printf("ndims is %d, rope type is %d, n_ctx_orig is %d, freq_base is %f, freq_scale is %f, ext_f is %f,
    //     attn_factor is %f, beta_fast is %f, beta_slow is %f\n", rope_cfg.n_dims, rope_cfg.rope_type,
    //     rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale, rope_cfg.ext_factor, rope_cfg.attn_factor,
    //     rope_cfg.beta_fast, rope_cfg.beta_slow); exit(0);
    // }

    memcpy(&ggml_tensor_out->op_params[0], arr_i, sizeof(arr_i));
    memcpy(&ggml_tensor_out->op_params[5], arr_f, sizeof(arr_f));

    op_interfaces::op_rope(*warp, ggml_tensor_out.get());

    // DEBUG
    // {
    //     cuda_context_warp::device_sync();
    //     auto file{fopen("rope.txt", "w")};
    //     float *src0_buffer{new float[src->m_shape[0] * src->m_shape[1] * src->m_shape[2] * src->m_shape[3]]};
    //     cuda_context_warp::copy_memory<2>(src0_buffer, src->m_data->m_data_device, src->m_shape[0] *
    //     src->m_shape[1] * src->m_shape[2] * src->m_shape[3] * sizeof(float));

    //     float *src2_buffer{new float[rope_frators->m_shape[0] * rope_frators->m_shape[1] * rope_frators->m_shape[2] *
    //     rope_frators->m_shape[3]]}; cuda_context_warp::copy_memory<2>(src2_buffer,
    //     rope_frators->m_data->m_data_device, rope_frators->m_shape[0] * rope_frators->m_shape[1] *
    //     rope_frators->m_shape[2] * rope_frators->m_shape[3] * sizeof(float));

    //     float *dst_buffer{new float[out->m_shape[0] * out->m_shape[1] * out->m_shape[2] * out->m_shape[3]]};
    //     cuda_context_warp::copy_memory<2>(dst_buffer, out->m_data->m_data_device, out->m_shape[0] *
    //     out->m_shape[1] * out->m_shape[2] * out->m_shape[3] * sizeof(float)); cuda_context_warp::device_sync();

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
    //     for (size_t i{0}; i < rope_frators->m_shape[3]; ++i) {
    //         for (size_t j{0}; j < rope_frators->m_shape[2]; ++j) {
    //             for (size_t k{0}; k < rope_frators->m_shape[1]; ++k) {
    //                 for (size_t l{0}; l < rope_frators->m_shape[0]; ++l) {
    //                     fprintf(file, "%f ", src2_buffer[i * rope_frators->m_shape[2] * rope_frators->m_shape[1] *
    //                     rope_frators->m_shape[0] + j * rope_frators->m_shape[1] * rope_frators->m_shape[0] + k *
    //                     rope_frators->m_shape[0] + l]);
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

void GGML_CUDABackend::permute(Tensor *out, const Tensor *x, Shape axes) const {
    Stride stride{};
    const auto &buffer_x{*x->m_data};
    stride[axes[0]] = buffer_x.m_stride[0];
    stride[axes[1]] = buffer_x.m_stride[1];
    stride[axes[2]] = buffer_x.m_stride[2];
    stride[axes[3]] = buffer_x.m_stride[3];

    out->m_data->m_stride = stride;
    // DEBUG
    // {
    //     std::cout << "stride is ";
    //     for (auto &&s : stride) {
    //         std::cout << s << " ";
    //     }
    //     std::cout << std::endl;
    //     exit(0);
    // }
}

void GGML_CUDABackend::add(Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    auto ggml_tensor_dst{convert_to_ggml_tensor(dst)};
    auto ggml_tensor_src0{convert_to_ggml_tensor(src0)};
    auto ggml_tensor_src1{convert_to_ggml_tensor(src1)};

    // POWERSERVE_ASSERT();

    ggml_tensor_dst->src[0] = ggml_tensor_src0.get();
    ggml_tensor_dst->src[1] = ggml_tensor_src1.get();

    op_interfaces::op_add(*warp, ggml_tensor_dst.get());

    // DEBUG
    // if (dst->m_name == "ffn_o_31_6") {
    //     cuda_context_warp::device_sync();
    //     printf("src0 shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", src0->m_shape[0], src0->m_shape[1],
    //     src0->m_shape[2], src0->m_shape[3],
    //         src0->m_data->m_stride[0], src0->m_data->m_stride[1],
    //         src0->m_data->m_stride[2], src0->m_data->m_stride[3]);
    //     printf("src1 shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", src1->m_shape[0], src1->m_shape[1],
    //     src1->m_shape[2], src1->m_shape[3],
    //         src1->m_data->m_stride[0], src1->m_data->m_stride[1],
    //         src1->m_data->m_stride[2], src1->m_data->m_stride[3]);
    //     printf("dst shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", dst->m_shape[0], dst->m_shape[1],
    //     dst->m_shape[2], dst->m_shape[3],
    //         dst->m_data->m_stride[0], dst->m_data->m_stride[1],
    //         dst->m_data->m_stride[2], dst->m_data->m_stride[3]);

    //     float *dst_buffer{new float[dst->m_shape[0] * dst->m_shape[1] * dst->m_shape[2] * dst->m_shape[3]]};
    //     cuda_context_warp::copy_memory<2>(dst_buffer, dst->m_data->m_data_device, dst->m_shape[0] *
    //     dst->m_shape[1] * dst->m_shape[2] * dst->m_shape[3] * sizeof(float)); cuda_context_warp::device_sync();

    //     auto file{fopen("add_ffn_o.txt", "w")};
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

bool GGML_CUDABackend::is_contiguous(const Tensor *tensor, int n) const {
    POWERSERVE_ASSERT(n >= 0 and n <= 2);
    if (n == 0) {
        return ggml_is_contiguous_0(convert_to_ggml_tensor(tensor).get());
    } else if (n == 1) {
        return ggml_is_contiguous_1(convert_to_ggml_tensor(tensor).get());
    } else if (n == 2) {
        return ggml_is_contiguous_2(convert_to_ggml_tensor(tensor).get());
    }

    return false;
}

void GGML_CUDABackend::cont(Tensor *out, const Tensor *x) const {
    auto ggml_tensor_out{convert_to_ggml_tensor(out)};
    auto ggml_tensor_x{convert_to_ggml_tensor(x)};
    ggml_tensor_out->src[0] = ggml_tensor_x.get();

    op_interfaces::op_cont(*warp, ggml_tensor_out.get());

    // DEBUG
    // {
    //     cuda_context_warp::device_sync();
    //     printf("src shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", x->m_shape[0], x->m_shape[1],
    //     x->m_shape[2], x->m_shape[3],
    //         x->m_data->m_stride[0], x->m_data->m_stride[1], x->m_data->m_stride[2],
    //         x->m_data->m_stride[3]);
    //     printf("dst shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", out->m_shape[0], out->m_shape[1],
    //     out->m_shape[2], out->m_shape[3],
    //         out->m_data->m_stride[0], out->m_data->m_stride[1],
    //         out->m_data->m_stride[2], out->m_data->m_stride[3]);

    //     float *dst_buffer{new float[out->m_shape[0] * out->m_shape[1] * out->m_shape[2] * out->m_shape[3]]};
    //     cuda_context_warp::copy_memory<2>(dst_buffer, out->m_data->m_data_device, out->m_shape[0] *
    //     out->m_shape[1] * out->m_shape[2] * out->m_shape[3] * sizeof(float)); cuda_context_warp::device_sync();

    //     auto file{fopen("cont.txt", "w")};
    //     for (size_t i = 0; i < out->m_shape[3]; i++) {
    //         for (size_t j = 0; j < out->m_shape[2]; j++) {
    //             for (size_t k = 0; k < out->m_shape[1]; k++) {
    //                 for (size_t l = 0; l < out->m_shape[0]; l++) {
    //                     fprintf(file, "%f ", dst_buffer[i * out->m_shape[2] * out->m_shape[1] * out->m_shape[0] + j *
    //                     out->m_shape[1] * out->m_shape[0] + k * out->m_shape[0] + l]);
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

void GGML_CUDABackend::silu_and_mul(Tensor *out, const Tensor *gate, const Tensor *up) const {
    auto ggml_tensor_out{convert_to_ggml_tensor(out)};
    auto ggml_tensor_gate{convert_to_ggml_tensor(gate)};
    auto ggml_tensor_up{convert_to_ggml_tensor(up)};

    ggml_tensor_out->src[0]       = ggml_tensor_gate.get();
    ggml_tensor_out->src[1]       = ggml_tensor_up.get();
    ggml_tensor_out->op           = GGML_OP_UNARY;
    ggml_tensor_out->op_params[0] = GGML_UNARY_OP_SILU; // first param: act method

    op_interfaces::op_silu_and_mul(*warp, ggml_tensor_out.get());
}

void GGML_CUDABackend::copy(Tensor *out, const Tensor *src) const {
    auto ggml_tensor_out{convert_to_ggml_tensor(out)};
    auto ggml_tensor_src{convert_to_ggml_tensor(src)};

    ggml_tensor_out->src[0] = ggml_tensor_src.get();
    op_interfaces::op_copy(*warp, ggml_tensor_out.get());

    // DEBUG K
    // if (out->m_name == "k_cache_view_0_5") {
    //     printf("src shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", src->m_shape[0], src->m_shape[1],
    //     src->m_shape[2], src->m_shape[3],
    //         src->m_data->m_stride[0], src->m_data->m_stride[1],
    //         src->m_data->m_stride[2], src->m_data->m_stride[3]);
    //     printf("dst shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", out->m_shape[0], out->m_shape[1],
    //     out->m_shape[2], out->m_shape[3],
    //         out->m_data->m_stride[0], out->m_data->m_stride[1],
    //         out->m_data->m_stride[2], out->m_data->m_stride[3]);
    //     cuda_context_warp::device_sync();
    //     auto file{fopen("copy_k.txt", "w")};
    //     float *src_buffer{new float[src->m_shape[0] * src->m_shape[1] * src->m_shape[2] * src->m_shape[3]]};
    //     cuda_context_warp::copy_memory<2>(src_buffer, src->m_data->m_data_device, src->m_shape[0] *
    //     src->m_shape[1] * src->m_shape[2] * src->m_shape[3] * sizeof(float));

    //     float *dst_buffer{new float[out->m_shape[0] * out->m_shape[1] * out->m_shape[2] * out->m_shape[3]]};
    //     cuda_context_warp::copy_memory<2>(dst_buffer, out->m_data->m_data_device, out->m_shape[0] *
    //     out->m_shape[1] * out->m_shape[2] * out->m_shape[3] * sizeof(float)); cuda_context_warp::device_sync();

    //     fprintf(file, "src:\n");
    //     for (size_t i{0}; i < src->m_shape[3]; ++i) {
    //         for (size_t j{0}; j < src->m_shape[2]; ++j) {
    //             for (size_t k{0}; k < src->m_shape[1]; ++k) {
    //                 for (size_t l{0}; l < src->m_shape[0]; ++l) {
    //                     fprintf(file, "%f ", src_buffer[i * src->m_shape[2] * src->m_shape[1] * src->m_shape[0] + j *
    //                     src->m_shape[1] * src->m_shape[0] + k * src->m_shape[0] + l]);
    //                 }
    //                 fprintf(file, "\n\n");
    //             }
    //             fprintf(file, "\n\n");
    //         }
    //         fprintf(file, "\n\n");
    //     }
    //     fprintf(file, "\n\n");

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
    //     delete[] src_buffer;
    //     delete[] dst_buffer;
    //     exit(0);
    // }
}

void GGML_CUDABackend::print(const Tensor *x, size_t rows) const {
    // SMART_UNUSED(size);
    POWERSERVE_ASSERT((rows & 0xFFFF'8000) == 0); // for safe convert
    auto ggml_tensor_to_print{convert_to_ggml_tensor(x)};
    ggml_tensor_to_print->op_params[0] = static_cast<int>(rows);
    op_interfaces::op_print(*warp, ggml_tensor_to_print.get());
}

void GGML_CUDABackend::get_mask(Tensor *out, const std::vector<int> &pos, size_t kv_number, size_t batch_size) {
    auto ggml_tensor_out{convert_to_ggml_tensor(out)};
    POWERSERVE_ASSERT(pos.size() > 0);
    const auto first_pos{pos[0]};
    const auto pos_size{pos.size()};

    ggml_tensor_out->op_params[0] = static_cast<int>(first_pos);
    ggml_tensor_out->op_params[1] = static_cast<int>(pos_size);
    ggml_tensor_out->op_params[2] = static_cast<int>(kv_number);
    ggml_tensor_out->op_params[3] = static_cast<int>(batch_size);

    op_interfaces::op_get_mask(*warp, ggml_tensor_out.get());
}

void GGML_CUDABackend::append_kv_cache(
    const Tensor *src, const size_t layer_id, const size_t token_num, bool is_k_cache
) {
    POWERSERVE_ASSERT(src->m_dtype == DataType::FP32);

    if (is_k_cache) {
        m_kv->append_k_cache(src, layer_id, token_num);
    } else {
        if (m_kv->kv_shape.flash_attn) {
            m_kv->append_v_cache(src, layer_id, token_num);
        } else {
            auto ggml_tensor_dst{new ggml_tensor()}, ggml_tensor_src{new ggml_tensor()};
            ggml_tensor_dst->data = m_kv->v_cache[layer_id].cache_data_ptr;
            ggml_tensor_src->data = src->m_data->m_data_device;

            ggml_tensor_dst->op_params[0] = static_cast<int32_t>(m_kv->kv_shape.n_ctx);
            ggml_tensor_dst->op_params[1] = static_cast<int32_t>(m_kv->v_cache[layer_id].valid_idx);
            ggml_tensor_dst->op_params[2] = static_cast<int32_t>(m_kv->kv_shape.kv_dim);
            ggml_tensor_dst->op_params[3] = static_cast<int32_t>(token_num);
            ggml_tensor_dst->src[0]       = ggml_tensor_src;
            op_interfaces::op_append_v_cache(*warp, ggml_tensor_dst);
        }
    }
}

void GGML_CUDABackend::transpose(Tensor *out, const Tensor *x) const {
    const auto &buffer_x{*x->m_data};
    auto &buffer_out{*out->m_data};
    auto stride{buffer_x.m_stride};
    stride[0] = buffer_x.m_stride[1];
    stride[1] = buffer_x.m_stride[0];

    buffer_out.m_stride = stride;
}

void GGML_CUDABackend::advance(const size_t &size) {
    m_kv->advanced_kv_cache_size(size);
}

void GGML_CUDABackend::reset_kv_batch_size(const size_t &size) {
    m_kv->reset_kv_batch_size(size);
}

std::pair<Tensor *, Tensor *> GGML_CUDABackend::get_kv_cache(size_t layer_id) {
    return m_kv->get_cache(layer_id);
}

void GGML_CUDABackend::graph_compute(std::vector<std::shared_ptr<OpNode>> &ops) {
    for (auto &op : ops) {
        switch (op->op) {
        case OpType::ADD: {
            auto src0 = op->prev[0]->tensor();
            auto src1 = op->prev[1]->tensor();
            auto out  = op->output();
            POWERSERVE_ASSERT(
                src0->m_backend == src1->m_backend and src0->m_backend == TensorBackend::GGML_GPU and
                out->m_backend == TensorBackend::GGML_GPU
            );
            add(out, src0, src1);
        } break;

        case OpType::MAT_MUL: {
            auto src0 = op->prev[0]->tensor();
            auto src1 = op->prev[1]->tensor();
            auto out  = op->output();
            POWERSERVE_ASSERT(
                src0->m_backend == src1->m_backend and src0->m_backend == TensorBackend::GGML_GPU and
                out->m_backend == TensorBackend::GGML_GPU
            );
            matmul(out, src0, src1);
        } break;

        case OpType::RMS_NORM: {
            auto x      = op->prev[0]->tensor();
            auto weight = op->prev[1]->tensor();
            auto out    = op->output();
            auto [eps]  = op->get_params<RMSNormParams>();
            POWERSERVE_ASSERT(
                x->m_backend == TensorBackend::GGML_GPU and x->m_backend == TensorBackend::GGML_GPU and
                out->m_backend == TensorBackend::GGML_GPU
            );
            rmsnorm(out, x, weight, eps);
        } break;

        case OpType::SILU_HADAMARD: {
            auto gate = op->prev[0]->tensor();
            auto up   = op->prev[1]->tensor();
            auto out  = op->output();
            POWERSERVE_ASSERT(
                gate->m_backend == TensorBackend::GGML_GPU and up->m_backend == TensorBackend::GGML_GPU and
                out->m_backend == TensorBackend::GGML_GPU
            );
            silu_and_mul(out, gate, up);
        } break;

        case OpType::ROPE: {
            // printf("ROPE\n");
            auto src             = op->prev[0]->tensor();
            auto rope_factors    = op->prev[1]->tensor();
            auto out             = op->output();
            auto [pos, rope_cfg] = op->get_params<RopeParams>();
            POWERSERVE_ASSERT(src->m_backend == TensorBackend::GGML_GPU and out->m_backend == TensorBackend::GGML_GPU);
            rope(out, src, rope_factors, pos, rope_cfg);
        } break;

        case OpType::SOFTMAX: {
            // printf("SOFTMAX\n");
            auto src = op->prev[0]->tensor();
            auto out = op->output();
            POWERSERVE_ASSERT(src->m_backend == TensorBackend::GGML_GPU and out->m_backend == TensorBackend::GGML_GPU);
            softmax(out, src, nullptr, 1.0, 0.0);
        } break;

        case OpType::COPY: {
            // printf("COPY\n");
            // get input tensor and output tensor, check backend, then call copy on GPU backend
            auto dst = op->prev[0]->tensor();
            auto src = op->prev[1]->tensor();
            POWERSERVE_ASSERT(src->m_backend == TensorBackend::GGML_GPU && dst->m_backend == TensorBackend::GGML_GPU);
            copy(dst, src);
        } break;

        case OpType::PRINT: {
            // printf("PRINT\n");
            // get input tensor and size, check backend, then call print on GPU backend
            auto x    = op->prev[0]->tensor();
            auto size = op->get_params<PrintParams>().size;
            POWERSERVE_ASSERT(x->m_backend == TensorBackend::GGML_GPU);
            print(x, size);
        } break;

        case OpType::GET_EMBEDDING: {
            // printf("GET_EMBEDDING\n");
            // get weight tensor, output tensor and tokens, check backend, then call get_embedding on GPU backend
            auto weight   = op->prev[0]->tensor();
            auto out      = op->output();
            auto [tokens] = op->get_params<GetEmbeddingParams>();
            POWERSERVE_ASSERT(out->m_backend == TensorBackend::GGML_GPU);
            get_embedding(out, weight, tokens);
        } break;

        case OpType::ADD_CACHE: {
            // printf("ADD_CACHE\n");
            // get k tensor, v tensor, L, pos and head_id, check backend, then call add_cache on GPU backend
            auto k                 = op->prev[0]->tensor();
            auto v                 = op->prev[1]->tensor();
            auto [L, pos, head_id] = op->get_params<AddCacheParams>();
            POWERSERVE_ASSERT(k->m_backend == v->m_backend and k->m_backend == TensorBackend::GGML_GPU);
            append_kv_cache(k, L, pos.size(), true);
            append_kv_cache(v, L, pos.size(), false);
        } break;

        case OpType::PERMUTE: {
        } break;

        case OpType::CONT: {
            // printf("CONT\n");
            // get input tensor and output tensor, check backend, then call cont on GPU backend
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            POWERSERVE_ASSERT(x->m_backend == TensorBackend::GGML_GPU and out->m_backend == TensorBackend::GGML_GPU);
            cont(out, x);
        } break;

        case OpType::VIEW: {
        } break;

        case OpType::SOFTMAX_EXT: {
            // printf("SOFTMAX_EXT\n");
            // get output tensor, input tensor, mask tensor, scale and max_bias, check backend, then call softmax_ext on
            // GPU backend
            auto out               = op->output();
            auto x                 = op->prev[0]->tensor();
            auto mask              = op->prev[1]->tensor();
            auto [scale, max_bias] = op->get_params<SoftmaxExtParams>();
            POWERSERVE_ASSERT(
                x->m_backend == TensorBackend::GGML_GPU and mask->m_backend == TensorBackend::GGML_GPU and
                out->m_backend == TensorBackend::GGML_GPU
            );
            softmax(out, x, mask, scale, max_bias);
        } break;

        case OpType::GET_MASK: {
            // printf("GET_MASK\n");
            // get output tensor, mask tensor and pos, check backend, then set mask on GPU backend
            auto out         = op->output();
            auto [mask, pos] = op->get_params<GetMaskParams>();
            POWERSERVE_ASSERT(out->m_dtype == DataType::FP32 and out->m_backend == TensorBackend::GGML_GPU);
            auto n_kv       = out->m_shape[0];
            auto batch_size = out->m_shape[1];
            get_mask(out, pos, n_kv, batch_size);
        } break;

        case OpType::TRANSPOSE: {
        } break;

        default:
            POWERSERVE_ABORT("Unknown OpType: {}", static_cast<int>(op->op));
        }
    }

    auto &last_op{ops.back()};
    auto last_out{last_op->output()};
    // if (last_out->m_name.substr(0, 7) == "logits_") {
    cuda_context_warp::stream_sync(warp->get_stream());
    auto num_element{std::reduce(last_out->m_shape.begin(), last_out->m_shape.end(), 1, std::multiplies<size_t>())};
    last_out->m_data->m_data_host      = malloc(num_element * sizeof(float));
    last_out->m_data->m_is_host_malloc = true;
    cuda_context_warp::copy_memory<2>(
        last_out->m_data->m_data_host, last_out->m_data->m_data_device, num_element * sizeof(float)
    );
    // }
}

} // namespace powerserve::ggml_cuda
