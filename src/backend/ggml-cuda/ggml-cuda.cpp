#include "ggml-cuda.hpp"

#include "backend/ggml-cuda/buffer.hpp"

#include "ggml.h"
#include "ggml-quants.h"

namespace powerserve::ggml_cuda {

void GGML_CUDABackend::get_embedding(Tensor *dst, const Tensor *weight, const std::vector<int> &tokens) const { // finish
    auto ggml_tensor_dst{convert_to_ggml_tensor(dst)};
    auto ggml_tensor_weight{convert_to_ggml_tensor(weight)};
    auto ggml_tensor_tokens{new ggml_tensor{}};
    auto buffer_stride{Stride{sizeof(int), sizeof(int) * tokens.size(), sizeof(int) * tokens.size(), sizeof(int) * tokens.size()}};
    auto tensor_shape{Shape{tokens.size(), 1, 1, 1}};
    void *cuda_ptr{nullptr};
    cuda_context_warp::malloc_cuda_buffer(&cuda_ptr, sizeof(int) * tokens.size());
    cuda_context_warp::copy_memory<1>(cuda_ptr, reinterpret_cast<void *>(const_cast<int*>(tokens.data())), sizeof(int) * tokens.size());
    ggml_tensor_tokens->data = cuda_ptr;
    ggml_tensor_tokens->type = GGML_TYPE_I32;
    memcpy(ggml_tensor_tokens->ne, tensor_shape.data(), tensor_shape.size() * sizeof(Shape::size_type));
    memcpy(ggml_tensor_tokens->nb, buffer_stride.data(), buffer_stride.size() * sizeof(Stride::size_type));

    ggml_tensor_dst->src[0] = ggml_tensor_weight.get();
    ggml_tensor_dst->src[1] = ggml_tensor_tokens;
    op_interfaces::op_get_embedding(*warp, ggml_tensor_dst.get());

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
}

void GGML_CUDABackend::rope(Tensor *out, const Tensor *src, const Tensor *rope_frators, const std::vector<int> &pos, const ModelConfig::LLMConfig::RopeConfig &rope_cfg) const {
    auto ggml_tensor_out{convert_to_ggml_tensor(out)};
    auto ggml_tensor_src{convert_to_ggml_tensor(src)};
    auto ggml_tensor_rope_factors{convert_to_ggml_tensor(rope_frators)};

    auto ggml_tensor_pos{std::make_unique<ggml_tensor>()};
    {
        void *pos_data_ptr{nullptr};
        void *cpu_data_ptr = static_cast<void *>(const_cast<int *>(pos.data()));
        cuda_context_warp::malloc_cuda_buffer(&pos_data_ptr, pos.size() * sizeof(int));
        cuda_context_warp::copy_memory_async<1>(pos_data_ptr, cpu_data_ptr, pos.size() * sizeof(int), nullptr);
        ggml_tensor_pos->data = pos_data_ptr;
        ggml_tensor_pos->type = GGML_TYPE_I32;
        ggml_tensor_pos->ne[0] = pos.size();
        ggml_tensor_pos->ne[1] = ggml_tensor_pos->ne[2] = ggml_tensor_pos->ne[3] = 1;
        ggml_tensor_pos->nb[0] = sizeof(int32_t);
        ggml_tensor_pos->nb[1] = ggml_tensor_pos->nb[2] = ggml_tensor_pos->nb[3] = pos.size() * sizeof(int32_t);
    }

    const auto &out_buffer{out->get<Buffer_CUDA>()};
    const auto &src_buffer{src->get<Buffer_CUDA>()};
    POWERSERVE_ASSERT(out_buffer.m_data_cuda and src_buffer.m_data_cuda);

    ggml_tensor_out->src[0] = ggml_tensor_src.get();
    ggml_tensor_out->src[1] = ggml_tensor_pos.get();
    ggml_tensor_out->src[2] = ggml_tensor_rope_factors.get();

    int arr_i[5]{0, rope_cfg.n_dims, rope_cfg.rope_type, 0, rope_cfg.n_ctx_orig};
    float arr_f[6]{rope_cfg.freq_base, rope_cfg.freq_scale, rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow};

    // DEBUG
    // {
    //     printf("ndims is %d, rope type is %d, n_ctx_orig is %d, freq_base is %f, freq_scale is %f, ext_f is %f, attn_factor is %f, beta_fast is %f, beta_slow is %f\n", rope_cfg.n_dims, rope_cfg.rope_type, rope_cfg.n_ctx_orig, rope_cfg.freq_base, rope_cfg.freq_scale, rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow);
    //     exit(0);
    // }
    
    memcpy(&ggml_tensor_out->op_params[0], arr_i, sizeof(arr_i));
    memcpy(&ggml_tensor_out->op_params[5], arr_f, sizeof(arr_f));

    op_interfaces::op_rope(*warp, ggml_tensor_out.get());
}

void GGML_CUDABackend::permute(Tensor *out, const Tensor *x, Shape axes) const {
    Stride stride{};
    const auto &buffer_x{x->get<Buffer_CUDA>()};
    stride[axes[0]] = buffer_x.m_stride[0];
    stride[axes[1]] = buffer_x.m_stride[1];
    stride[axes[2]] = buffer_x.m_stride[2];
    stride[axes[3]] = buffer_x.m_stride[3];

    out->get<Buffer_CUDA>().m_stride = stride;
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
    // if (dst->m_name == "ffn_o_31") {
    //     cuda_context_warp::device_sync();
    //     printf("src0 shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", src0->m_shape[0], src0->m_shape[1], src0->m_shape[2], src0->m_shape[3], 
    //         src0->get<Buffer_CUDA>().m_stride[0], src0->get<Buffer_CUDA>().m_stride[1], src0->get<Buffer_CUDA>().m_stride[2], src0->get<Buffer_CUDA>().m_stride[3]);
    //     printf("src1 shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", src1->m_shape[0], src1->m_shape[1], src1->m_shape[2], src1->m_shape[3], 
    //         src1->get<Buffer_CUDA>().m_stride[0], src1->get<Buffer_CUDA>().m_stride[1], src1->get<Buffer_CUDA>().m_stride[2], src1->get<Buffer_CUDA>().m_stride[3]);
    //     printf("dst shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", dst->m_shape[0], dst->m_shape[1], dst->m_shape[2], dst->m_shape[3], 
    //         dst->get<Buffer_CUDA>().m_stride[0], dst->get<Buffer_CUDA>().m_stride[1], dst->get<Buffer_CUDA>().m_stride[2], dst->get<Buffer_CUDA>().m_stride[3]);

    //     float *dst_buffer{new float[dst->m_shape[0] * dst->m_shape[1] * dst->m_shape[2] * dst->m_shape[3]]};
    //     cuda_context_warp::copy_memory<2>(dst_buffer, dst->get<Buffer_CUDA>().m_data_cuda, dst->m_shape[0] * dst->m_shape[1] * dst->m_shape[2] * dst->m_shape[3] * sizeof(float));
    //     cuda_context_warp::device_sync();

    //     auto file{fopen("add_ffn_o.txt", "w")};
    //     for (size_t i = 0; i < dst->m_shape[3]; i++) {
    //         for (size_t j = 0; j < dst->m_shape[2]; j++) {
    //             for (size_t k = 0; k < dst->m_shape[1]; k++) {
    //                 for (size_t l = 0; l < dst->m_shape[0]; l++) {
    //                     fprintf(file, "%f ", dst_buffer[i * dst->m_shape[2] * dst->m_shape[1] * dst->m_shape[0] + j * dst->m_shape[1] * dst->m_shape[0] + k * dst->m_shape[0] + l]);
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
    //     printf("src shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", x->m_shape[0], x->m_shape[1], x->m_shape[2], x->m_shape[3], 
    //         x->get<Buffer_CUDA>().m_stride[0], x->get<Buffer_CUDA>().m_stride[1], x->get<Buffer_CUDA>().m_stride[2], x->get<Buffer_CUDA>().m_stride[3]);
    //     printf("dst shape is %ld %ld %ld %ld, stride is %ld %ld %ld %ld\n", out->m_shape[0], out->m_shape[1], out->m_shape[2], out->m_shape[3], 
    //         out->get<Buffer_CUDA>().m_stride[0], out->get<Buffer_CUDA>().m_stride[1], out->get<Buffer_CUDA>().m_stride[2], out->get<Buffer_CUDA>().m_stride[3]);
        
    //     float *dst_buffer{new float[out->m_shape[0] * out->m_shape[1] * out->m_shape[2] * out->m_shape[3]]};
    //     cuda_context_warp::copy_memory<2>(dst_buffer, out->get<Buffer_CUDA>().m_data_cuda, out->m_shape[0] * out->m_shape[1] * out->m_shape[2] * out->m_shape[3] * sizeof(float));
    //     cuda_context_warp::device_sync();

    //     auto file{fopen("cont.txt", "w")};
    //     for (size_t i = 0; i < out->m_shape[3]; i++) {
    //         for (size_t j = 0; j < out->m_shape[2]; j++) {
    //             for (size_t k = 0; k < out->m_shape[1]; k++) {
    //                 for (size_t l = 0; l < out->m_shape[0]; l++) {
    //                     fprintf(file, "%f ", dst_buffer[i * out->m_shape[2] * out->m_shape[1] * out->m_shape[0] + j * out->m_shape[1] * out->m_shape[0] + k * out->m_shape[0] + l]);
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

    ggml_tensor_out->src[0] = ggml_tensor_gate.get();
    ggml_tensor_out->src[1] = ggml_tensor_up.get();
    ggml_tensor_out->op = GGML_OP_UNARY;
    ggml_tensor_out->op_params[0] = GGML_UNARY_OP_SILU; // first param: act method

    op_interfaces::op_silu_and_mul(*warp, ggml_tensor_out.get());
}
void GGML_CUDABackend::copy(Tensor *out, const Tensor *src) const {
    auto ggml_tensor_out{convert_to_ggml_tensor(out)};
    auto ggml_tensor_src{convert_to_ggml_tensor(src)};

    ggml_tensor_out->src[0] = ggml_tensor_src.get();
    op_interfaces::op_copy(*warp, ggml_tensor_out.get());
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

void GGML_CUDABackend::append_kv_cache(const Tensor *src, const size_t layer_id, const size_t token_num, bool is_k_cache) {
    POWERSERVE_ASSERT(src->m_dtype == DataType::FP32);

    if (is_k_cache) {
        m_kv->append_k_cache(src, layer_id, token_num);
    } else {
        if (m_kv->kv_shape.flash_attn) {
            m_kv->append_v_cache(src, layer_id, token_num);
        } else {
            auto ggml_tensor_dst{new ggml_tensor()}, ggml_tensor_src{new ggml_tensor()};
            ggml_tensor_dst->data = m_kv->v_cache[layer_id].cache_data_ptr;
            ggml_tensor_src->data = src->get<Buffer_CUDA>().m_data_cuda;

            ggml_tensor_dst->op_params[0] = static_cast<int32_t>(m_kv->kv_shape.n_ctx);
            ggml_tensor_dst->op_params[1] = static_cast<int32_t>(m_kv->v_cache[layer_id].valid_idx);
            ggml_tensor_dst->op_params[2] = static_cast<int32_t>(m_kv->kv_shape.kv_dim);
            ggml_tensor_dst->op_params[3] = static_cast<int32_t>(token_num);
            ggml_tensor_dst->src[0] = ggml_tensor_src;
            op_interfaces::op_append_v_cache(*warp, ggml_tensor_dst);
        }
    }
}

void GGML_CUDABackend::transpose(Tensor *out, const Tensor *x) const {
    const auto &buffer_x{x->get<Buffer_CUDA>()};
    auto &buffer_out{out->get<Buffer_CUDA>()};
    auto stride{buffer_x.m_stride};
    stride[0] = buffer_x.m_stride[1];
    stride[1] = buffer_x.m_stride[0];

    buffer_out.m_data_cuda = buffer_x.m_data_cuda;
    buffer_out.m_data_host = buffer_x.m_data_host;
    buffer_out.m_stride = stride;
    buffer_out.m_size = buffer_x.m_size;
}

} // namespace powerserve::ggml_cuda
