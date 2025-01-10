#include "ggml-cuda.hpp"

#include "backend/ggml-cuda/buffer.hpp"

#include "ggml.h"
#include "ggml-quants.h"

namespace powerserve::ggml_cuda {

void GGML_CUDABackend::get_embedding(Tensor *dst, const Tensor *weight, const std::vector<int> &tokens) const {
    auto weight_buffer{weight->get<Buffer_CUDA>()};
    auto dst_buffer{dst->get<Buffer_CUDA>()};
    
    auto embedding_table{static_cast<char *>(weight_buffer.m_data_host)};
    auto dst_table{static_cast<float *>(dst_buffer.m_data_host)};

    POWERSERVE_ASSERT(embedding_table not_eq nullptr and dst_table not_eq nullptr);

    auto dim{dst->m_shape[0]};
    auto batch_size{tokens.size()};

    POWERSERVE_ASSERT(batch_size == dst->m_shape[1]);
    auto weight_strip{weight_buffer.m_stride};

    for (decltype(batch_size) i{0}; i < batch_size; ++i) {
        auto token{tokens[i]};
        auto src{embedding_table + weight_strip[1] * token};

        POWERSERVE_ASSERT(src < embedding_table + weight_strip[2]);

        switch (weight->m_dtype) {
        case DataType::FP32: {
            memcpy(dst_table + i * dim, src, dim * sizeof(float));
        } break;
        
        case DataType::GGML_Q4_0: {
            dequantize_row_q4_0((block_q4_0 *)src, dst_table + i * dim, dim);
        } break;

        case DataType::GGML_Q8_0: {
            dequantize_row_q8_0((block_q8_0 *)src, dst_table + i * dim, dim);
        } break;

        default: {
            POWERSERVE_ASSERT(false);
        }
        }
    }

    POWERSERVE_ASSERT(dst_buffer.m_data_cuda not_eq nullptr and dst_buffer.m_size >= batch_size * dim * sizeof(float));
    cuda_context_warp::copy_memory_async<1>(dst_buffer.m_data_cuda, dst_buffer.m_data_host, 
        batch_size * dim * sizeof(float), warp->ctx);
    
}

void GGML_CUDABackend::matmul(Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    auto split{src0->get<Buffer_CUDA>().m_buffer_type == buffer_type::GGML_GPU_SPLIT};
    auto ggml_tensor_dst{convert_to_ggml_tensor(dst)};
    auto ggml_tensor_src0{convert_to_ggml_tensor(src0)};
    auto ggml_tensor_src1{convert_to_ggml_tensor(src1)};

    ggml_tensor_dst->src[0] = ggml_tensor_src0.get();
    ggml_tensor_dst->src[1] = ggml_tensor_src1.get();

    ggml_tensor_dst->op_params[15] = split ? 1 : 0; // 16th param: split or not?

    op_interfaces::op_mat_mul(*warp, ggml_tensor_dst.get());
}

void GGML_CUDABackend::rmsnorm(Tensor *o, const Tensor *x, const Tensor *weight, float eps) const {
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

void GGML_CUDABackend::rope(Tensor *out, const Tensor *src, const Tensor *pos, const Tensor *freq_factors, const ModelConfig::LLMConfig::RopeConfig &rope_cfg) const {
    auto ggml_tensor_out{convert_to_ggml_tensor(out)};
    auto ggml_tensor_src{convert_to_ggml_tensor(src)};
    auto ggml_tensor_pos{convert_to_ggml_tensor(pos)};
    auto ggml_tensor_freq_factors{freq_factors ? convert_to_ggml_tensor(freq_factors) : nullptr};

    auto out_buffer{out->get<Buffer_CUDA>()};
    auto src_buffer{src->get<Buffer_CUDA>()};
    auto pos_buffer{pos->get<Buffer_CUDA>()};
    POWERSERVE_ASSERT(out_buffer.m_data_cuda and src_buffer.m_data_cuda and pos_buffer.m_data_cuda);

    ggml_tensor_out->src[0] = ggml_tensor_src.get();
    ggml_tensor_out->src[1] = ggml_tensor_pos.get();
    ggml_tensor_out->src[2] = ggml_tensor_freq_factors.get();

    int arr_i[5]{0, rope_cfg.n_dims, rope_cfg.rope_type, 0, rope_cfg.n_ctx_orig};
    float arr_f[6]{rope_cfg.freq_base, rope_cfg.freq_scale, rope_cfg.ext_factor, rope_cfg.attn_factor, rope_cfg.beta_fast, rope_cfg.beta_slow};
    
    memcpy(&ggml_tensor_out->op_params[0], arr_i, sizeof(arr_i));
    memcpy(&ggml_tensor_out->op_params[5], arr_f, sizeof(arr_f));

    op_interfaces::op_rope(*warp, ggml_tensor_out.get());
}

void GGML_CUDABackend::permute(Tensor *out, const Tensor *x, Shape axes) const {
    Stride stride{};
    auto buffer_x{x->get<Buffer_CUDA>()};
    stride[axes[0]] = buffer_x.m_stride[0];
    stride[axes[1]] = buffer_x.m_stride[1];
    stride[axes[2]] = buffer_x.m_stride[2];
    stride[axes[3]] = buffer_x.m_stride[3];

    out->get<Buffer_CUDA>().m_stride = stride;
    // TODO: where is data?
}

void GGML_CUDABackend::add(Tensor *dst, const Tensor *src0, const Tensor *src1) const {
    auto ggml_tensor_dst{convert_to_ggml_tensor(dst)};
    auto ggml_tensor_src0{convert_to_ggml_tensor(src0)};
    auto ggml_tensor_src1{convert_to_ggml_tensor(src1)};

    // POWERSERVE_ASSERT();

    ggml_tensor_dst->src[0] = ggml_tensor_src0.get();
    ggml_tensor_dst->src[1] = ggml_tensor_src1.get();

    op_interfaces::op_add(*warp, ggml_tensor_dst.get());
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
}

void GGML_CUDABackend::silu_and_mul(Tensor *out, const Tensor *gate, const Tensor *up) const {
    auto ggml_tensor_out{convert_to_ggml_tensor(out)};
    auto ggml_tensor_gate{convert_to_ggml_tensor(gate)};
    auto ggml_tensor_up{convert_to_ggml_tensor(up)};

    ggml_tensor_out->src[0] = ggml_tensor_gate.get();
    ggml_tensor_out->src[1] = ggml_tensor_up.get();
    ggml_tensor_out->op_params[0] = 10; // first param: act method

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
    POWERSERVE_ASSERT((rows and 0xFFFF'8000) == 0); // for safe convert
    auto ggml_tensor_to_print{convert_to_ggml_tensor(x)};
    ggml_tensor_to_print->op_params[0] = static_cast<int>(rows);
    op_interfaces::op_print(*warp, ggml_tensor_to_print.get());
}

void GGML_CUDABackend::transpose(Tensor *out, const Tensor *x) const {
    auto buffer_x{x->get<Buffer_CUDA>()};
    auto buffer_out{out->get<Buffer_CUDA>()};
    auto stride{buffer_x.m_stride};
    stride[0] = buffer_x.m_stride[1];
    stride[1] = buffer_x.m_stride[0];

    buffer_out.m_data_cuda = buffer_x.m_data_cuda;
    buffer_out.m_data_host = buffer_x.m_data_host;
    buffer_out.m_stride = stride;
    buffer_out.m_size = buffer_x.m_size;
}

} // namespace powerserve::ggml_cuda
