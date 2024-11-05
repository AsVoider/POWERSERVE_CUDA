#pragma once

#include "backend/backend.hpp"
#include "backend/ggml/buffer.hpp"
#include "common.hpp"
#include "core/data_type.hpp"
#include "core/tensor.hpp"
#include "core/thread_pool.hpp"
#include "ggml.h"
#include "model/common/config.hpp"
#include "model/module/region.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace smart::ggml {

static ggml_type convert_datatype_to_ggml(DataType dtp) {
    switch (dtp) {
    case DataType::FP32:
        return GGML_TYPE_F32;
    case DataType::FP16:
        return GGML_TYPE_F16;
    case DataType::GGML_Q4_0:
        return GGML_TYPE_Q4_0;
    case DataType::GGML_Q8_0:
        return GGML_TYPE_Q8_0;
    case DataType::INT32:
        return GGML_TYPE_I32;
    default:
        SMART_ASSERT(false);
    }
}

static DataType convert_datatype_from_ggml(ggml_type tp) {
    switch (tp) {
    case GGML_TYPE_F32:
        return DataType::FP32;
    case GGML_TYPE_F16:
        return DataType::FP16;
    case GGML_TYPE_Q4_0:
        return DataType::GGML_Q4_0;
    case GGML_TYPE_Q8_0:
        return DataType::GGML_Q8_0;
    default:
        SMART_ASSERT(false);
    }
}

static Tensor convert_from_ggml(ggml_tensor *t) {
    SMART_ASSERT(t != nullptr);
    Tensor::Shape shape;
    Buffer::Stride stride;
    for (size_t i = 0; i < Tensor::max_n_dims; i++) {
        shape[i]  = t->ne[i];
        stride[i] = t->nb[i];
    }
    Tensor tensor(convert_datatype_from_ggml(t->type), shape);
    tensor.m_data = std::make_shared<Buffer>(stride, t->data);
    return tensor;
}

static std::unique_ptr<ggml_tensor> convert_to_ggml(const Tensor *tensor) {
    auto gt  = std::make_unique<ggml_tensor>();
    gt->data = tensor->get<ggml::Buffer>().m_data;
    gt->type = convert_datatype_to_ggml(tensor->m_dtype);
    for (size_t i = 0; i < Tensor::max_n_dims; i++) {
        gt->ne[i] = tensor->m_shape[i];
        gt->nb[i] = tensor->get<ggml::Buffer>().m_stride[i];
    }
    return gt;
}

// debug functions
static void debug_meta_info(gguf_context *gguf_ctx, ggml_context *ggml_ctx) {
    {
        fmt::println("version     : {:10}", gguf_get_version(gguf_ctx));
        fmt::println("n_kv        : {:10}", gguf_get_n_kv(gguf_ctx));
        fmt::println("n_tensors   : {:10}", gguf_get_n_tensors(gguf_ctx));
        fmt::println("alignment   : {:10}", gguf_get_alignment(gguf_ctx));
        fmt::println("meta size   : {:10}", gguf_get_meta_size(gguf_ctx));
        fmt::println("data offset : {:10}", gguf_get_data_offset(gguf_ctx));
    }

    {
        for (auto i = 0; i < gguf_get_n_kv(gguf_ctx); i++) {
            auto key      = gguf_get_key(gguf_ctx, i);
            auto v_type   = gguf_get_kv_type(gguf_ctx, i);
            auto type_str = gguf_type_name(v_type);
            fmt::println("{:40}: {:4}", key, type_str);
        }
    }

    {
        for (auto i = 0; i < gguf_get_n_tensors(gguf_ctx); i++) {
            auto name   = gguf_get_tensor_name(gguf_ctx, i);
            auto t_type = gguf_get_tensor_type(gguf_ctx, i);
            fmt::println("{:40}: {:6}: {:10}", name, ggml_type_name(t_type), gguf_get_tensor_offset(gguf_ctx, i));
        }
    }

    {
        fmt::println("GGML used mem        : {:10}", ggml_used_mem(ggml_ctx));
        fmt::println("GGML no alloc        : {:10}", ggml_get_no_alloc(ggml_ctx));
        fmt::println("GGML mem buffer      : {:10}", ggml_get_mem_buffer(ggml_ctx));
        fmt::println("GGML mem size        : {:10}", ggml_get_mem_size(ggml_ctx));
        fmt::println("GGML max tensor size : {:10}", ggml_get_max_tensor_size(ggml_ctx));
    }
}

static void debug_tensors_info(gguf_context *gguf_ctx, ggml_context *ggml_ctx) {
    for (auto i = 0; i < gguf_get_n_tensors(gguf_ctx); i++) {
        auto t = ggml_get_tensor(ggml_ctx, gguf_get_tensor_name(gguf_ctx, i));

        fmt::println(
            "{:40}|{:>5}|({:6},{:6},{:1},{:1})|{:10}|{:4}|{:4}|{:10}",
            ggml_get_name(t),
            ggml_type_name(t->type),
            t->ne[0],
            t->ne[1],
            t->ne[2],
            t->ne[3],
            ggml_get_data(t),
            ggml_type_size(t->type),
            ggml_blck_size(t->type),
            ggml_row_size(t->type, ggml_nelements(t)) // ne * ggml_type_size / ggml_blk_size (bytes)
        );
    }
}

static void debug_system_info(void) {
    std::string s{};

    s += "AVX = " + std::to_string(ggml_cpu_has_avx()) + " | ";
    s += "AVX_VNNI = " + std::to_string(ggml_cpu_has_avx_vnni()) + " | ";
    s += "AVX2 = " + std::to_string(ggml_cpu_has_avx2()) + " | ";
    s += "AVX512 = " + std::to_string(ggml_cpu_has_avx512()) + " | ";
    s += "AVX512_VBMI = " + std::to_string(ggml_cpu_has_avx512_vbmi()) + " | ";
    s += "AVX512_VNNI = " + std::to_string(ggml_cpu_has_avx512_vnni()) + " | ";
    s += "AVX512_BF16 = " + std::to_string(ggml_cpu_has_avx512_bf16()) + " | ";
    s += "FMA = " + std::to_string(ggml_cpu_has_fma()) + " | ";
    s += "NEON = " + std::to_string(ggml_cpu_has_neon()) + " | ";
    s += "SVE = " + std::to_string(ggml_cpu_has_sve()) + " | ";
    s += "ARM_FMA = " + std::to_string(ggml_cpu_has_arm_fma()) + " | ";
    s += "F16C = " + std::to_string(ggml_cpu_has_f16c()) + " | ";
    s += "FP16_VA = " + std::to_string(ggml_cpu_has_fp16_va()) + " | ";
    s += "RISCV_VECT = " + std::to_string(ggml_cpu_has_riscv_v()) + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = " + std::to_string(ggml_cpu_has_blas()) + " | ";
    s += "SSE3 = " + std::to_string(ggml_cpu_has_sse3()) + " | ";
    s += "SSSE3 = " + std::to_string(ggml_cpu_has_ssse3()) + " | ";
    s += "VSX = " + std::to_string(ggml_cpu_has_vsx()) + " | ";
    s += "MATMUL_INT8 = " + std::to_string(ggml_cpu_has_matmul_int8()) + " | ";
    s += "LLAMAFILE = " + std::to_string(ggml_cpu_has_llamafile()) + " | ";

    fmt::println("system info: {}", s);
}

// **Note**: Backend receives Tensor not TensorNode
struct GGMLBackend : Backend {
public:
    op_compute_params m_params;
    std::vector<char> m_wdata;
    std::shared_ptr<Config> m_config;

public:
    explicit GGMLBackend(std::shared_ptr<Config> config, int n_threads = 1) :
        m_wdata(config->tf_cfg.dim * 32),
        m_config(config) {
        m_params = {
            .ith           = 0,
            .nth           = 1,
            .wsize         = (size_t)config->tf_cfg.dim * 32,
            .wdata         = m_wdata.data(),
            .thread_pool   = nullptr,
            .barrier_fn    = nullptr,
            .current_chunk = nullptr,
        };

        std::vector<ThreadConfig> configs;
        for (int i = 0; i < n_threads; i++) {
            configs.emplace_back(ThreadConfig{.cpu_ids = {(size_t)i}});
        }
        // create thread num has limit
        m_thread_pool = std::make_unique<ThreadPool>(configs);
        // fmt::println("\nCreated thread pool with {} threads", configs.size());
    }

    ~GGMLBackend() override = default;

public:
    // void compute_forward(thread_compute_params *params, const OpNode *op) const override;

public:
    void matmul(const Tensor *dst, const Tensor *src0, const Tensor *src1) const;
    void rmsnorm(const Tensor *o, const Tensor *x, const Tensor *weight) const;
    void softmax(const Tensor *out, const Tensor *x) const;
    // void rope(Tensor *q_out, Tensor *k_out, const Tensor *q, const Tensor *k, const Tensor *pos) const;
    void rope(
        Tensor *out,
        const Tensor *src,
        const Tensor *pos,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
    ) const;
    void multihead_attention(
        const Tensor *out,
        const Tensor *q,
        const Tensor *key_cache,
        const Tensor *val_cache,
        const Tensor *pos,
        const int64_t L
    ) const;
    void silu_hadamard(const Tensor *out, const Tensor *hb, const Tensor *hb2) const;
    void add(const Tensor *dst, const Tensor *src0, const Tensor *src1) const;
    void copy(const Tensor *dst, const Tensor *src, const int64_t off) const;
    void quest_attention(
        const Tensor *out,
        const Tensor *q,
        const Tensor *key_cache,
        const Tensor *val_cache,
        const Tensor *pos,
        const int64_t L,
        std::vector<Region> &regions
    ) const;
    void cos_sim(const Tensor *src0, const Tensor *src1) const;
    void print(const Tensor *x, size_t size) const;

public:
    template <typename T>
    auto create_buffer(Tensor::Shape shape) -> BufferPtr {
        Buffer::Stride stride;
        stride[0] = sizeof(T);
        for (size_t i = 1; i < shape.size(); i++) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        size_t size = stride.back() * shape.back();

        return std::make_shared<Buffer>(stride, malloc(size), true);
    }

private:
    std::unique_ptr<ThreadPool> m_thread_pool;
    std::atomic<int> m_current_chunk = 0;

private:
    void rmsnorm_internal(float *o, float *x, float *weight, int64_t size) const;
    void softmax_internal(float *out, float *x, size_t size) const;
    void cos_sim_internal(float *out_, float *x_, size_t size) const;
};

} // namespace smart::ggml
