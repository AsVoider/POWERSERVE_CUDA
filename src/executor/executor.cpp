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

#include "executor/executor.hpp"

#include "core/logger.hpp"

#include <cstdint>

namespace powerserve {

void Executor::shed_op_to_backend() {
    for (auto &op : m_graph.ops) {
        switch (op->op) {
        case OpType::ADD:
        case OpType::MAT_MUL:
        case OpType::RMS_NORM:
        case OpType::SILU_HADAMARD:
        case OpType::ROPE:
        case OpType::SOFTMAX:
        case OpType::GET_EMBEDDING:
        case OpType::PERMUTE:
        case OpType::CONT:
        case OpType::VIEW:
        case OpType::SOFTMAX_EXT:
        case OpType::GET_MASK:
        case OpType::TRANSPOSE:
        {
            op->compute_backend = op->output()->m_backend;
        } break;

        case OpType::COPY:
        case OpType::PRINT:
        case OpType::ADD_CACHE:
        {
            op->compute_backend = op->prev[0]->tensor()->m_backend;
        } break;

        default:
            POWERSERVE_ASSERT(false and "op not implemented");            
        }
    }
}
    
void Executor::allocate_buffers() {
    for (auto tensor : m_graph.tensors) {
        if (tensor->m_data) {
            continue;
        }

        switch (tensor->m_dtype) {
        case DataType::FP32: {
            create_cpu_buffer<float>(tensor);
        } break;

        case DataType::INT32: {
            create_cpu_buffer<int32_t>(tensor);
        } break;
        case DataType::INT64: {
            create_cpu_buffer<int64_t>(tensor);
        } break;

        default:
            POWERSERVE_ABORT("could not allocate buffer for data type: {}", static_cast<int>(tensor->m_dtype));
        }
    }
}

void Executor::allocate_buffer_with_backend() {
    for (auto tensor : m_graph.tensors) {
        if (tensor->m_data) {
            continue;
        }
        POWERSERVE_ASSERT(tensor->m_backend != TensorBackend::UNKNOWN);

        switch (tensor->m_dtype) {
            case DataType::FP32: {
                create_backend_buffer<float>(tensor);
            } break;

            case DataType::INT32: {
                create_backend_buffer<int32_t>(tensor);
            } break;

            case DataType::INT64: {
                create_backend_buffer<int64_t>(tensor);
            } break;

            default:
                POWERSERVE_ABORT("could not allocate buffer for data type: {}", static_cast<int>(tensor->m_dtype));
        }
    }
}

void Executor::plan() {
    m_platform.ggml_backends[m_graph.m_model_id]->plan(m_graph.ops);
}

void Executor::run() {
    auto &model_id = m_graph.m_model_id;
    plan();

    for (auto op : m_graph.ops) {
        switch (op->op) {
        case OpType::GET_EMBEDDING: {
            auto weight   = op->prev[0]->tensor();
            auto out      = op->output();
            auto [tokens] = op->get_params<GetEmbeddingParams>();
            m_platform.ggml_backends[model_id]->get_embedding(out, weight, tokens);
        } break;

        case OpType::ADD: {
            auto a   = op->prev[0]->tensor();
            auto b   = op->prev[1]->tensor();
            auto out = op->output();
            m_platform.ggml_backends[model_id]->add(out, a, b);
        } break;

        case OpType::MAT_MUL: {
            auto a   = op->prev[0]->tensor();
            auto b   = op->prev[1]->tensor();
            auto out = op->output();
            m_platform.ggml_backends[model_id]->matmul(out, a, b);
        } break;

        case OpType::RMS_NORM: {
            auto x      = op->prev[0]->tensor();
            auto weight = op->prev[1]->tensor();
            auto out    = op->output();
            auto [eps]  = op->get_params<RMSNormParams>();
            m_platform.ggml_backends[model_id]->rmsnorm(out, x, weight, eps);
        } break;

        case OpType::SILU_HADAMARD: {
            auto gate = op->prev[0]->tensor();
            auto up   = op->prev[1]->tensor();
            auto out  = op->output();
            m_platform.ggml_backends[model_id]->silu_hadamard(out, gate, up);
        } break;

        case OpType::ROPE: {
            auto src             = op->prev[0]->tensor();
            auto out             = op->next[0]->tensor();
            auto [pos, rope_cfg] = op->get_params<RopeParams>();
            m_platform.ggml_backends[model_id]->rope(out, src, pos, rope_cfg);
        } break;

        case OpType::SOFTMAX: {
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            m_platform.ggml_backends[model_id]->softmax(out, x);
        } break;

        case OpType::COPY: {
            auto dst = op->prev[0]->tensor();
            auto src = op->prev[1]->tensor();
            m_platform.ggml_backends[model_id]->copy(dst, src);
        } break;

#if defined(POWERSERVE_WITH_QNN)
        case OpType::QNN_FORWARD: {
            auto x     = op->prev[0]->tensor();
            auto out   = op->output();
            auto pos   = op->get_params<QNNForwardParams>().pos;
            auto &mask = op->get_params<QNNForwardParams>().mask;
            m_platform.qnn_backend->forward(m_graph.m_model_id, out, x, pos, mask);
        } break;
        case OpType::QNN_FORWARD_VL: {
            auto x                  = op->prev[0]->tensor();
            auto out                = op->output();
            auto pos                = op->get_params<QNNForwardVLParams>().pos;
            auto &mask              = op->get_params<QNNForwardVLParams>().mask;
            auto &pixel_values_list = op->get_params<QNNForwardVLParams>().pixel_values_list;
            auto &img_infos         = op->get_params<QNNForwardVLParams>().img_infos;
            m_platform.qnn_backend->forward(m_graph.m_model_id, out, x, pixel_values_list, img_infos, pos, mask);
            pixel_values_list.clear();
            img_infos.clear();
        } break;
#endif

        case OpType::PRINT: {
            auto x    = op->prev[0]->tensor();
            auto size = op->get_params<PrintParams>().size;
            m_platform.ggml_backends[model_id]->print(x, size);

        } break;

        case OpType::ADD_CACHE: {
            auto k                 = op->prev[0]->tensor();
            auto v                 = op->prev[1]->tensor();
            auto [L, pos, head_id] = op->get_params<AddCacheParams>();
            m_platform.ggml_backends[model_id]->add_cache(k, v, L, pos, head_id);
        } break;

        case OpType::PERMUTE: {
            auto x      = op->prev[0]->tensor();
            auto out    = op->output();
            auto [axes] = op->get_params<PermuteParams>();
            m_platform.ggml_backends[model_id]->permute(out, x, axes);
        } break;

        case OpType::CONT: {
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            m_platform.ggml_backends[model_id]->cont(out, x);
        } break;

        case OpType::VIEW: {
            auto out                       = op->output();
            auto [stride, offset]          = op->get_params<ViewParams>();
            out->get<CPUBuffer>().m_stride = stride;
            out->get<CPUBuffer>().m_data   = (char *)out->get<CPUBuffer>().m_data + offset;
        } break;

        case OpType::SOFTMAX_EXT: {
            auto out               = op->output();
            auto x                 = op->prev[0]->tensor();
            auto mask              = op->prev[1]->tensor();
            auto [scale, max_bias] = op->get_params<SoftmaxExtParams>();

            m_platform.ggml_backends[model_id]->softmax_ext(out, x, mask, scale, max_bias);
        } break;

        case OpType::GET_MASK: {
            auto out         = op->output();
            auto [mask, pos] = op->get_params<GetMaskParams>();
            auto n_kv        = out->m_shape[0];
            auto batch_size  = out->m_shape[1];

            POWERSERVE_ASSERT(out->m_dtype == DataType::FP32);
            auto mask_buf = (float *)out->get<CPUBuffer>().m_data;
            for (size_t i = 0; i < batch_size; i++) {
                size_t cur_pos = pos[i];
                for (size_t j = 0; j < n_kv; j++) {
                    mask_buf[j + i * n_kv] = (j <= cur_pos) ? 0.f : -INFINITY;
                }
            }
        } break;

        case OpType::TRANSPOSE: {
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            m_platform.ggml_backends[model_id]->transpose(out, x);
        } break;
        default:
            POWERSERVE_ABORT("Unknown OpType: {}", static_cast<int>(op->op));
        }
    }
}

// fix this function, accept a parameter to print graph to a file
void Executor::print_graph(std::ostream &os) {
    os << "total tensor num is " << m_graph.tensors.size() << std::endl;

    for (auto op : m_graph.ops) {
        switch (op->op) {
        case OpType::ADD: {
            auto a = op->prev[0]->tensor();
            auto b = op->prev[1]->tensor();
            auto c = op->output();
            os << "ADD: src0 " << static_cast<int>(a->m_backend) <<  " shape is ";
            for (auto &&p : a->m_shape) {
                os << p << " ";
            }
            os << "src1 " << static_cast<int>(b->m_backend) << " shape is ";
            for (auto &&p : b->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(c->m_backend) << " shape is ";
            for (auto &&p : c->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::MAT_MUL: {
            auto a = op->prev[0]->tensor();
            auto b = op->prev[1]->tensor();
            auto c = op->output();
            os << "MAT_MUL: src0 " << static_cast<int>(a->m_backend) <<  " shape is ";
            for (auto &&p : a->m_shape) {
                os << p << " ";
            }
            os << "src1 " << static_cast<int>(b->m_backend) << " shape is ";
            for (auto &&p : b->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(c->m_backend) << " shape is ";
            for (auto &&p : c->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::RMS_NORM: {
            auto x      = op->prev[0]->tensor();
            auto weight = op->prev[1]->tensor();
            auto c      = op->output();
            os << "RMS_NORM: src " << static_cast<int>(x->m_backend) <<  " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            if (weight != nullptr) {
                os << "weight " << static_cast<int>(weight->m_backend) << " shape is ";
                for (auto &&p : weight->m_shape) {
                    os << p << " ";
                }
            }
            os << "dst " << static_cast<int>(c->m_backend) << " shape is ";
            for (auto &&p : c->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::SILU_HADAMARD: {
            auto gate = op->prev[0]->tensor();
            auto up   = op->prev[1]->tensor();
            auto c    = op->output();
            os << "SILU_HADAMARD: gate " << static_cast<int>(gate->m_backend) <<  " shape is ";
            for (auto &&p : gate->m_shape) {
                os << p << " ";
            }
            os << "up " << static_cast<int>(up->m_backend) << " shape is ";
            for (auto &&p : up->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(c->m_backend) << " shape is ";
            for (auto &&p : c->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::ROPE: {
            auto src = op->prev[0]->tensor();
            auto c   = op->output();
            auto [pos, rope_cfg] = op->get_params<RopeParams>();
            os << "ROPE: src " << static_cast<int>(src->m_backend) <<  " shape is ";
            for (auto &&p : src->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(c->m_backend) << " shape is ";
            for (auto &&p : c->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::SOFTMAX: {
            auto x = op->prev[0]->tensor();
            auto c = op->output();
            os << "SOFTMAX: src " << static_cast<int>(x->m_backend) <<  " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(c->m_backend) << " shape is ";
            for (auto &&p : c->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::COPY: {
            auto dst = op->prev[0]->tensor();
            auto src = op->prev[1]->tensor();
            os << "COPY: src " << static_cast<int>(src->m_backend) <<  " shape is ";
            for (auto &&p : src->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(dst->m_backend) << " shape is ";
            for (auto &&p : dst->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::PRINT: {
            auto x = op->prev[0]->tensor();
            auto [size] = op->get_params<PrintParams>();
            os << "PRINT: src " << static_cast<int>(x->m_backend) <<  " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            os << "size " << size << std::endl;
        } break;

        case OpType::GET_EMBEDDING: {
            auto weight = op->prev[0]->tensor();
            auto out    = op->output();
            auto [tokens] = op->get_params<GetEmbeddingParams>();
            os << "GET_EMBEDDING: weight " << static_cast<int>(weight->m_backend) <<  " shape is ";
            for (auto &&p : weight->m_shape) {
                os << p << " ";
            }
            os << "out " << static_cast<int>(out->m_backend) << " shape is ";
            for (auto &&p : out->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::ADD_CACHE: {
            auto k = op->prev[0]->tensor();
            auto v = op->prev[1]->tensor();
            auto [L, pos, head_id] = op->get_params<AddCacheParams>();
            os << "ADD_CACHE: k " << static_cast<int>(k->m_backend) <<  " shape is ";
            for (auto &&p : k->m_shape) {
                os << p << " ";
            }
            os << "v " << static_cast<int>(v->m_backend) << " shape is ";
            for (auto &&p : v->m_shape) {
                os << p << " ";
            }
            os << "L " << L << " pos " << pos[0] << " head_id " << head_id << std::endl;
        } break;

        case OpType::PERMUTE: {
            auto x = op->prev[0]->tensor();
            auto out = op->output();
            auto [axes] = op->get_params<PermuteParams>();
            os << "PERMUTE: src " << static_cast<int>(x->m_backend) <<  " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            os << "out " << static_cast<int>(out->m_backend) << " shape is ";
            for (auto &&p : out->m_shape) {
                os << p << " ";
            }
            os << "axes ";
            for (auto &&p : axes) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::CONT: {
            auto x = op->prev[0]->tensor();
            auto out = op->output();
            os << "CONT: src " << static_cast<int>(x->m_backend) <<  " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            os << "out " << static_cast<int>(out->m_backend) << " shape is ";
            for (auto &&p : out->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::VIEW: {
            auto out = op->output();
            auto [stride, offset] = op->get_params<ViewParams>();
            os << "VIEW: dst " << static_cast<int>(out->m_backend);
            os << " stride is ";
            for (auto &&p : stride) {
                os << p << " ";
            }
            os << "offset is " << offset << std::endl; 
        } break;

        case OpType::SOFTMAX_EXT: {
            auto x = op->prev[0]->tensor();
            auto mask = op->prev[1]->tensor();
            auto out = op->output();
            auto [scale, max_bias] = op->get_params<SoftmaxExtParams>();
            os << "SOFTMAX_EXT: src " << static_cast<int>(x->m_backend) <<  " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            os << "mask " << static_cast<int>(mask->m_backend) << " shape is ";
            for (auto &&p : mask->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(out->m_backend) << " shape is ";
            for (auto &&p : out->m_shape) {
                os << p << " ";
            }
            os << "scale " << scale << " max_bias " << max_bias << std::endl;
        } break;

        case OpType::GET_MASK: {
            auto out = op->output();
            auto [mask, pos] = op->get_params<GetMaskParams>();
            os << "GET_MASK: dst " << static_cast<int>(out->m_backend) <<  " shape is ";
            for (auto &&p : out->m_shape) {
                os << p << " ";
            }
            os << "pos " << pos[0] << " pos size is " << pos.size() << std::endl;
        } break;

        case OpType::TRANSPOSE: {
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            os << "TRANSPOSE: src " << static_cast<int>(x->m_backend) <<  " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(out->m_backend) << " shape is ";
            for (auto &&p : out->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        default:
            os << "Unknown OpType: " << static_cast<int>(op->op) << std::endl;
        }
    }
}

#if defined(POWERSERVE_WITH_CUDA)
void Executor::run_with_backend() {
    // auto &model_id{m_graph.m_model_id};
    plan();
    // shed_op_to_backend();
    for (auto op : m_graph.ops) {
        if (op->compute_backend == TensorBackend::GGML_GPU) {
            run_forward_gpu(op);
        } else {
            POWERSERVE_ABORT("Unknown backend: {}", static_cast<int>(op->compute_backend));
        }
    }
}

void Executor::run_forward_gpu(std::shared_ptr<OpNode> op) {
    // forward compute to gpu backend by case
    switch (op->op) {
    case OpType::ADD: {
        // get two input tensor and output tensor, check backend, then call add on GPU backend
        auto src0 = op->prev[0]->tensor();
        auto src1 = op->prev[1]->tensor();
        auto out  = op->output();
        POWERSERVE_ASSERT(src0->m_backend == src1->m_backend and src0->m_backend == TensorBackend::GGML_GPU and out->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->add(out, src0, src1);
    } break;

    case OpType::MAT_MUL: {
        // get two input tensor and output tensor, check backend, then call matmul on GPU backend
        auto src0 = op->prev[0]->tensor();
        auto src1 = op->prev[1]->tensor();
        auto out  = op->output();
        POWERSERVE_ASSERT(src0->m_backend == src1->m_backend and src0->m_backend == TensorBackend::GGML_GPU and out->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->matmul(out, src0, src1);
    } break;

    case OpType::RMS_NORM: {
        // get input tensor, weight tensor and output tensor, check backend, then call rmsnorm on GPU backend
        auto x      = op->prev[0]->tensor();
        auto weight = op->prev[1]->tensor();
        auto out    = op->output();
        auto [eps]  = op->get_params<RMSNormParams>();
        POWERSERVE_ASSERT(x->m_backend == TensorBackend::GGML_GPU and x->m_backend == TensorBackend::GGML_GPU and 
                 out->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->rmsnorm(out, x, weight, eps);
    } break;

    case OpType::SILU_HADAMARD: {
        // get gate tensor, up tensor and output tensor, check backend, then call silu_hadamard on GPU backend
        auto gate = op->prev[0]->tensor();
        auto up   = op->prev[1]->tensor();
        auto out  = op->output();
        POWERSERVE_ASSERT(gate->m_backend == TensorBackend::GGML_GPU and up->m_backend == TensorBackend::GGML_GPU and 
                 out->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->silu_and_mul(out, gate, up);
    } break;

    case OpType::ROPE: {
        // get input tensor, output tensor, pos and rope config, check backend, then call rope on GPU backend
        auto src             = op->prev[0]->tensor();
        auto out             = op->output();
        auto [pos, rope_cfg] = op->get_params<RopeParams>();
        POWERSERVE_ASSERT(src->m_backend == TensorBackend::GGML_GPU and out->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->rope(out, src, pos, rope_cfg);
    } break;

    case OpType::SOFTMAX: {
        // get input tensor and output tensor, check backend, then call softmax on GPU backend
        auto src   = op->prev[0]->tensor();
        auto out = op->output();
        POWERSERVE_ASSERT(src->m_backend == TensorBackend::GGML_GPU and out->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->softmax(out, src, nullptr, 1.0, 0.0);
    } break;

    case OpType::COPY: {
        // get input tensor and output tensor, check backend, then call copy on GPU backend
        auto dst = op->prev[0]->tensor();
        auto src = op->prev[1]->tensor();
        POWERSERVE_ASSERT(src->m_backend == TensorBackend::GGML_GPU && dst->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->copy(dst, src);
    } break;

    case OpType::PRINT: {
        // get input tensor and size, check backend, then call print on GPU backend
        auto x    = op->prev[0]->tensor();
        auto size = op->get_params<PrintParams>().size;
        POWERSERVE_ASSERT(x->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->print(x, size);
    } break;

    case OpType::GET_EMBEDDING: {
        // get weight tensor, output tensor and tokens, check backend, then call get_embedding on GPU backend
        auto weight   = op->prev[0]->tensor();
        auto out      = op->output();
        auto [tokens] = op->get_params<GetEmbeddingParams>();
        POWERSERVE_ASSERT(out->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->get_embedding(out, weight, tokens);
    } break;

    case OpType::ADD_CACHE: {
        // get k tensor, v tensor, L, pos and head_id, check backend, then call add_cache on GPU backend
        auto k                 = op->prev[0]->tensor();
        auto v                 = op->prev[1]->tensor();
        auto [L, pos, head_id] = op->get_params<AddCacheParams>();
        POWERSERVE_ASSERT(k->m_backend == v->m_backend and k->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->append_kv_cache(k, L, pos.size(), true);
        m_platform.ggml_cuda_backend->append_kv_cache(v, L, pos.size(), false);
    } break;

    case OpType::PERMUTE: {
        // get input tensor, output tensor and axes, check backend, then call permute on GPU backend
        auto x      = op->prev[0]->tensor();
        auto out    = op->output();
        auto [axes] = op->get_params<PermuteParams>();
        POWERSERVE_ASSERT(x->m_backend == TensorBackend::GGML_GPU and out->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->permute(out, x, axes);
    } break;

    case OpType::CONT: {
        // get input tensor and output tensor, check backend, then call cont on GPU backend
        auto x   = op->prev[0]->tensor();
        auto out = op->output();
        POWERSERVE_ASSERT(x->m_backend == TensorBackend::GGML_GPU and out->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->cont(out, x);
    } break;

    case OpType::VIEW: {
        // get output tensor, stride and offset, check backend, then set stride and data on GPU backend
        auto out = op->output();
        auto [stride, offset] = op->get_params<ViewParams>();
        POWERSERVE_ASSERT(out->m_backend == TensorBackend::GGML_GPU);
        out->get<ggml_cuda::Buffer_CUDA>().m_stride = stride;
        out->get<ggml_cuda::Buffer_CUDA>().m_data_cuda = (char *)out->get<ggml_cuda::Buffer_CUDA>().m_data_cuda + offset;
    } break;

    case OpType::SOFTMAX_EXT: {
        // get output tensor, input tensor, mask tensor, scale and max_bias, check backend, then call softmax_ext on GPU backend
        auto out               = op->output();
        auto x                 = op->prev[0]->tensor();
        auto mask              = op->prev[1]->tensor();
        auto [scale, max_bias] = op->get_params<SoftmaxExtParams>();
        POWERSERVE_ASSERT(x->m_backend == TensorBackend::GGML_GPU and mask->m_backend == TensorBackend::GGML_GPU and 
                 out->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->softmax(out, x, mask, scale, max_bias);
    } break;

    case OpType::GET_MASK: {
        // get output tensor, mask tensor and pos, check backend, then set mask on GPU backend
        auto out         = op->output();
        auto [mask, pos] = op->get_params<GetMaskParams>();
        POWERSERVE_ASSERT(out->m_dtype == DataType::FP32 and out->m_backend == TensorBackend::GGML_GPU);
        auto n_kv       = out->m_shape[0];
        auto batch_size = out->m_shape[1];
        m_platform.ggml_cuda_backend->get_mask(out, pos, n_kv, batch_size);
    } break;

    case OpType::TRANSPOSE: {
        // get input tensor and output tensor, check backend, then call transpose on GPU backend
        auto x   = op->prev[0]->tensor();
        auto out = op->output();
        POWERSERVE_ASSERT(x->m_backend == TensorBackend::GGML_GPU and out->m_backend == TensorBackend::GGML_GPU);
        m_platform.ggml_cuda_backend->transpose(out, x);
    } break;

    default: 
        POWERSERVE_ABORT("Unknown OpType: {}", static_cast<int>(op->op));
    } // end of switch statement
}
#endif
} // namespace powerserve
