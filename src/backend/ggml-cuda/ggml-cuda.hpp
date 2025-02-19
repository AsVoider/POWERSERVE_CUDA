#include "backend/backend.hpp"
#include "backend/ggml-cuda/buffer.hpp"
#include "core/config.hpp"
#include "core/tensor.hpp"
#include "ggml.h"

#include <memory>
#include <numeric>
#include <vector>


namespace smart::ggml_cuda {

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

static Tensor convert_from_ggml_with_data_copied(ggml_tensor *t) {
    SMART_ASSERT(t != nullptr);
    Shape tensor_shape{static_cast<size_t>(t->ne[0]), static_cast<size_t>(t->ne[1]), static_cast<size_t>(t->ne[2]), static_cast<size_t>(t->ne[3])};
    Stride tensor_stride{t->nb[0], t->nb[1], t->nb[2], t->nb[3]};
    Tensor tensor{convert_datatype_from_ggml(t->type), std::move(tensor_shape)};
   
    void *cuda_ptr{nullptr};
    cuda_context_warp::malloc_cuda_buffer(&cuda_ptr, ggml_nbytes(t));
    cuda_context_warp::copy_memory_async<1>(cuda_ptr, t->data, ggml_nbytes(t));

    tensor.m_data = std::make_shared<Buffer_CUDA>(tensor_stride, cuda_ptr, t->data, usage::WEIGHT, ggml_nbytes(t), true, false);
    return tensor;
}

static std::unique_ptr<ggml_tensor> convert_to_ggml_tensor(const Tensor *t) {
    if (t == nullptr) {
        return nullptr;
    }

    auto gt = std::make_unique<ggml_tensor>();
    auto buffer_t = t->get<Buffer_CUDA>();
    gt->type = convert_datatype_to_ggml(t->m_dtype);

    // Copy if need
    if (buffer_t.m_data_cuda == nullptr) {
        SMART_ASSERT(buffer_t.m_data_host not_eq nullptr and "Data host is nullptr\n");
        cuda_context_warp::copy_memory_async<1>(buffer_t.m_data_cuda, buffer_t.m_data_host, buffer_t.m_size);
    }

    gt->data = buffer_t.m_data_cuda;
    memcpy(gt->ne, t->m_shape.data(), t->m_shape.size() * sizeof(Shape::size_type));
    memcpy(gt->nb, buffer_t.m_stride.data(), buffer_t.m_stride.size() * sizeof(Stride::size_type));

    //  FIXME: if no sync?
    cuda_context_warp::device_sync();
    return gt;
}

class GGML_CUDABackend : Backend {
public:
    // op_compute_params m_params;
    cuda_context_warp *warp;

    explicit GGML_CUDABackend(const std::shared_ptr<ModelConfig::LLMConfig> &config, const HyperParams &hparams) : warp{new cuda_context_warp()} { }

    ~GGML_CUDABackend() override = default;

public: // ! Math Ops
    void add(Tensor *dst, const Tensor *src0, const Tensor *src1) const;
    void get_embedding(Tensor *dst, const Tensor *weight, const std::vector<int> &tokens) const;
    void matmul(Tensor *dst, const Tensor *src0, const Tensor *src1) const;
    void rmsnorm(Tensor *o, const Tensor *x, const Tensor *weight, float eps) const;
    void softmax(Tensor *out, const Tensor *x, const Tensor *mask, float scale, float bias) const;
    void rope(Tensor *out, const Tensor *src, const Tensor *pos, const Tensor *freq_factors, const ModelConfig::LLMConfig::RopeConfig &rope_cfg) const;
    void permute(Tensor *out, const Tensor *x, Shape axes) const;
    bool is_contiguous(const Tensor *tensor, int n) const;
    void cont(Tensor *out, const Tensor *x) const;
    void copy(Tensor *out, const Tensor *src) const;
    // void quest_attention(
    //     const Tensor *out,
    //     const Tensor *q,
    //     const std::vector<int> &pos,
    //     const int64_t L,
    //     std::vector<Region> &regions,
    //     const uint32_t n_heads
    // ) const;
    // void cos_sim(const Tensor *src0, const Tensor *src1) const;
    // void print(const Tensor *x, size_t size) const;
    // void reset_kv_batch_size(const size_t batch_size) const;
    // void add_cache(const Tensor *src, size_t L, const std::vector<int> &pos, size_t head_id, bool is_k);
    void transpose(Tensor *out, const Tensor *x) const;

public: // ! Mem Ops
    template <typename T>
    auto create_cuda_buffer(Shape shape, usage use = usage::ANY, buffer_type buf_t = buffer_type::GGML_GPU) -> BufferPtr {
        Stride stride;
        stride[0] = sizeof(T);
        for (size_t i{1}; i < shape.size(); ++i) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        auto size{stride.back() * shape.back()};

        void *cuda_ptr{nullptr};
        cuda_context_warp::malloc_cuda_buffer(&cuda_ptr, size);
        return std::make_shared<Buffer_CUDA>(stride, cuda_ptr, nullptr, use, size, true, false);
    }

    template <typename T>
    auto create_cuda_buffer_view(Buffer_CUDA &parent, Shape shape) {
        Stride stride;
        stride[0] = sizeof(T);
        for (size_t i{1}; i < shape.size(); ++i) {
            stride[i] = stride[i - 1] * shape[i - 1];
        }
        SMART_ASSERT(parent.m_data_cuda != nullptr);


        auto b{std::make_shared<Buffer_CUDA>(stride, nullptr, nullptr, usage::ANY, parent.m_size, false, false)};
        b->m_data_cuda = parent.m_data_cuda;
        b->m_data_host = parent.m_data_host;
        return b;
    }

    template <DataType D_Type>
    static void print_tensor(const Tensor &x, size_t n_rows = 0UL) {
        if (n_rows == 0UL) {
            n_rows = std::accumulate(std::next(x.m_shape.begin()), x.m_shape.end(), 1, std::multiplies<size_t>());
        }

        if constexpr (D_Type == DataType::FP16) {
            // half *mem_buffer = new half[n_rows * x.m_shape[0]];
            // cudaMemcpy(mem_buffer, x.get<Buffer_CUDA>().m_data_cuda, n_rows * x.m_shape[0] * sizeof(half), cudaMemcpyDeviceToHost);
            
            // for (size_t i{0UL}; i < n_rows; ++i) {
            //     for (size_t j{0UL}; j < x.m_shape[0]; ++j) {
            //         auto num_to_print{static_cast<float>(mem_buffer[j + i * x.m_shape[0]])};
            //         printf("%f ", num_to_print);
            //     }
            //     printf("\n");
            // }
            // printf("\n\n");
        } else if constexpr (D_Type == DataType::FP32) {
            float *mem_buffer = new float[n_rows * x.m_shape[0]];
            // cudaMemcpy(mem_buffer, x.get<Buffer_CUDA>().m_data_cuda, n_rows * x.m_shape[0] * sizeof(float), cudaMemcpyDeviceToHost);
            for (size_t i{0UL}; i < n_rows; ++i) {
                for (size_t j{0UL}; j < x.m_shape[0]; ++j) {
                    auto num_to_print{mem_buffer[j + i * x.m_shape[0]]};
                    printf("%f ", num_to_print);
                }
                printf("\n");
            }
            printf("\n\n");
        } else if constexpr (D_Type == DataType::GGML_Q4_0) {

        } else if constexpr (D_Type == DataType::GGML_Q8_0) {

        } else {
            SMART_ASSERT(false and "not supported tensor type");
        }
    }
};

} // namespace smart::ggml-cuda
