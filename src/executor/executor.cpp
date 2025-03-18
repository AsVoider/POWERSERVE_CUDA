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
        case OpType::TRANSPOSE: {
            op->compute_backend = op->output()->m_backend;
        } break;

        case OpType::COPY:
        case OpType::PRINT:
        case OpType::ADD_CACHE: {
            op->compute_backend = op->prev[0]->tensor()->m_backend;
        } break;

        default:
            POWERSERVE_ASSERT(false and "op not implemented");
        }
    }
}

void Executor::allocate_buffer_with_backend() {
    for (size_t device{0UL}; device < m_graph.backend_size.size(); ++device) {
        auto device_size = m_graph.backend_size[device];
        if (device_size == 0 or device == static_cast<size_t>(TensorBackend::GGML_CPU)) {
            continue;
        }
        Platform::buffer_interfaces.at(static_cast<TensorBackend>(device)).alloc_total(device_size + m_graph.pos_size);
    }

    for (auto &op : m_graph.ops) {
        auto out{op->output()};
        switch (op->op) {
        case OpType::ADD:
        case OpType::MAT_MUL:
        case OpType::RMS_NORM:
        case OpType::SILU_HADAMARD:
        case OpType::SOFTMAX:
        case OpType::GET_EMBEDDING:
        case OpType::CONT:
        case OpType::SOFTMAX_EXT:
        case OpType::GET_MASK: {
            out->m_data = Platform::buffer_interfaces.at(out->m_backend).create_buffer(out->m_shape, sizeof(float));
        } break;

        case OpType::ROPE: {
            auto src{op->prev[0]->tensor_view()};
            src->m_data = Platform::buffer_interfaces.at(src->m_backend)
                              .create_buffer_view(*src->parent->m_data, src->m_shape, sizeof(float), 0UL);
            // auto out{op->output()};
            out->m_data = Platform::buffer_interfaces.at(out->m_backend).create_buffer(out->m_shape, sizeof(float));
        } break;

        case OpType::PERMUTE: {
            auto x{op->prev[0]->tensor()};
            // auto out{op->output()};
            auto [axes]{op->get_params<PermuteParams>()};
            out->m_data = Platform::buffer_interfaces.at(out->m_backend)
                              .create_buffer_view(*x->m_data, out->m_shape, sizeof(float), 0UL);
            auto &x_stride{x->m_data->m_stride};
            Stride new_stride{};
            new_stride[axes[0]]   = x_stride[0];
            new_stride[axes[1]]   = x_stride[1];
            new_stride[axes[2]]   = x_stride[2];
            new_stride[axes[3]]   = x_stride[3];
            out->m_data->m_stride = std::move(new_stride);
        } break;

        case OpType::VIEW: {
            auto x{op->prev[0]->tensor()};
            // auto out{op->output()};
            auto [stride, offset]{op->get_params<ViewParams>()};
            out->m_data = Platform::buffer_interfaces.at(out->m_backend)
                              .create_buffer_view(*x->m_data, out->m_shape, sizeof(float), offset);
            out->m_data->m_stride = std::move(stride);
        } break;

        case OpType::TRANSPOSE: {
            auto x{op->prev[0]->tensor()};
            // auto out{op->output()};
            out->m_data = Platform::buffer_interfaces.at(out->m_backend)
                              .create_buffer_view(*x->m_data, out->m_shape, sizeof(float), 0UL);
            auto &x_stride{x->m_data->m_stride};
            Stride new_stride{x_stride[1], x_stride[0], x_stride[2], x_stride[3]};
            out->m_data->m_stride = std::move(new_stride);
        } break;

        case OpType::COPY:
        case OpType::PRINT:
        case OpType::ADD_CACHE:
            break;
        default:
            POWERSERVE_ASSERT(false and "op not implemented");
        }
    }
}

// #if defined(POWERSERVE_WITH_QNN)
//         case OpType::QNN_FORWARD: {
//             auto x     = op->prev[0]->tensor();
//             auto out   = op->output();
//             auto pos   = op->get_params<QNNForwardParams>().pos;
//             auto &mask = op->get_params<QNNForwardParams>().mask;
//             m_platform.qnn_backend->forward(m_graph.m_model_id, out, x, pos, mask);
//         } break;
//         case OpType::QNN_FORWARD_VL: {
//             auto x                  = op->prev[0]->tensor();
//             auto out                = op->output();
//             auto pos                = op->get_params<QNNForwardVLParams>().pos;
//             auto &mask              = op->get_params<QNNForwardVLParams>().mask;
//             auto &pixel_values_list = op->get_params<QNNForwardVLParams>().pixel_values_list;
//             auto &img_infos         = op->get_params<QNNForwardVLParams>().img_infos;
//             m_platform.qnn_backend->forward(m_graph.m_model_id, out, x, pixel_values_list, img_infos, pos, mask);
//             pixel_values_list.clear();
//             img_infos.clear();
//         } break;
// #endif

void Executor::print_graph(std::ostream &os) {
    os << "total tensor num is " << m_graph.tensors.size() << std::endl;

    for (auto op : m_graph.ops) {
        switch (op->op) {
        case OpType::ADD: {
            auto a = op->prev[0]->tensor();
            auto b = op->prev[1]->tensor();
            auto c = op->output();
            os << "ADD: src0 " << static_cast<int>(a->m_backend) << " type is " << static_cast<int>(a->m_dtype)
               << " shape is ";
            for (auto &&p : a->m_shape) {
                os << p << " ";
            }
            os << "src1 " << static_cast<int>(b->m_backend) << " type is " << static_cast<int>(b->m_dtype)
               << " shape is ";
            for (auto &&p : b->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(c->m_backend) << " type is " << static_cast<int>(c->m_dtype)
               << " shape is ";
            for (auto &&p : c->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::MAT_MUL: {
            auto a = op->prev[0]->tensor();
            auto b = op->prev[1]->tensor();
            auto c = op->output();
            os << "MAT_MUL: src0 " << static_cast<int>(a->m_backend) << " type is " << static_cast<int>(a->m_dtype)
               << " shape is ";
            for (auto &&p : a->m_shape) {
                os << p << " ";
            }
            os << "src1 " << static_cast<int>(b->m_backend) << " type is " << static_cast<int>(b->m_dtype)
               << " shape is ";
            for (auto &&p : b->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(c->m_backend) << " type is " << static_cast<int>(c->m_dtype)
               << " shape is ";
            for (auto &&p : c->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::RMS_NORM: {
            auto x      = op->prev[0]->tensor();
            auto weight = op->prev[1]->tensor();
            auto c      = op->output();
            os << "RMS_NORM: src " << static_cast<int>(x->m_backend) << " type is " << static_cast<int>(x->m_dtype)
               << " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            if (weight != nullptr) {
                os << "weight " << static_cast<int>(weight->m_backend) << " type is "
                   << static_cast<int>(weight->m_dtype) << " shape is ";
                for (auto &&p : weight->m_shape) {
                    os << p << " ";
                }
            }
            os << "dst " << static_cast<int>(c->m_backend) << " type is " << static_cast<int>(c->m_dtype)
               << " shape is ";
            for (auto &&p : c->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::SILU_HADAMARD: {
            auto gate = op->prev[0]->tensor();
            auto up   = op->prev[1]->tensor();
            auto c    = op->output();
            os << "SILU_HADAMARD: gate " << static_cast<int>(gate->m_backend) << " type is "
               << static_cast<int>(gate->m_dtype) << " shape is ";
            for (auto &&p : gate->m_shape) {
                os << p << " ";
            }
            os << "up " << static_cast<int>(up->m_backend) << " type is " << static_cast<int>(up->m_dtype)
               << " shape is ";
            for (auto &&p : up->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(c->m_backend) << " type is " << static_cast<int>(c->m_dtype)
               << " shape is ";
            for (auto &&p : c->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::ROPE: {
            auto src             = op->prev[0]->tensor();
            auto rope_factors    = op->prev[1]->tensor();
            auto c               = op->output();
            auto [pos, rope_cfg] = op->get_params<RopeParams>();
            os << "ROPE: src " << static_cast<int>(src->m_backend) << " type is " << static_cast<int>(src->m_dtype)
               << " shape is ";
            for (auto &&p : src->m_shape) {
                os << p << " ";
            }

            if (rope_factors not_eq nullptr) {
                os << "rope_factors " << static_cast<int>(rope_factors->m_backend) << " type is "
                   << static_cast<int>(rope_factors->m_dtype) << " shape is ";
                for (auto &&p : rope_factors->m_shape) {
                    os << p << " ";
                }
            }

            os << "dst " << static_cast<int>(c->m_backend) << " type is " << static_cast<int>(c->m_dtype)
               << " shape is ";
            for (auto &&p : c->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::SOFTMAX: {
            auto x = op->prev[0]->tensor();
            auto c = op->output();
            os << "SOFTMAX: src " << static_cast<int>(x->m_backend) << " type is " << static_cast<int>(x->m_dtype)
               << " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(c->m_backend) << " type is " << static_cast<int>(c->m_dtype)
               << " shape is ";
            for (auto &&p : c->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::COPY: {
            auto dst = op->prev[0]->tensor();
            auto src = op->prev[1]->tensor();
            os << "COPY: src " << static_cast<int>(src->m_backend) << " type is " << static_cast<int>(src->m_dtype)
               << " shape is ";
            for (auto &&p : src->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(dst->m_backend) << " type is " << static_cast<int>(dst->m_dtype)
               << " shape is ";
            for (auto &&p : dst->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::PRINT: {
            auto x      = op->prev[0]->tensor();
            auto [size] = op->get_params<PrintParams>();
            os << "PRINT: src " << static_cast<int>(x->m_backend) << " type is " << static_cast<int>(x->m_dtype)
               << " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            os << "size " << size << std::endl;
        } break;

        case OpType::GET_EMBEDDING: {
            auto weight   = op->prev[0]->tensor();
            auto out      = op->output();
            auto [tokens] = op->get_params<GetEmbeddingParams>();
            os << "GET_EMBEDDING: weight " << static_cast<int>(weight->m_backend) << " type is "
               << static_cast<int>(weight->m_dtype) << " shape is ";
            for (auto &&p : weight->m_shape) {
                os << p << " ";
            }
            os << "out " << static_cast<int>(out->m_backend) << " type is " << static_cast<int>(out->m_dtype)
               << " shape is ";
            for (auto &&p : out->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::ADD_CACHE: {
            auto k                 = op->prev[0]->tensor();
            auto v                 = op->prev[1]->tensor();
            auto [L, pos, head_id] = op->get_params<AddCacheParams>();
            os << "ADD_CACHE: k " << static_cast<int>(k->m_backend) << " type is " << static_cast<int>(k->m_dtype)
               << " shape is ";
            for (auto &&p : k->m_shape) {
                os << p << " ";
            }
            os << "v " << static_cast<int>(v->m_backend) << " type is " << static_cast<int>(v->m_dtype) << " shape is ";
            for (auto &&p : v->m_shape) {
                os << p << " ";
            }
            os << "L " << L << " pos " << pos[0] << " head_id " << head_id << std::endl;
        } break;

        case OpType::PERMUTE: {
            auto x      = op->prev[0]->tensor();
            auto out    = op->output();
            auto [axes] = op->get_params<PermuteParams>();
            os << "PERMUTE: src " << static_cast<int>(x->m_backend) << " type is " << static_cast<int>(x->m_dtype)
               << " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            os << "out " << static_cast<int>(out->m_backend) << " type is " << static_cast<int>(out->m_dtype)
               << " shape is ";
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
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            os << "CONT: src " << static_cast<int>(x->m_backend) << " type is " << static_cast<int>(x->m_dtype)
               << " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            os << "out " << static_cast<int>(out->m_backend) << " type is " << static_cast<int>(out->m_dtype)
               << " shape is ";
            for (auto &&p : out->m_shape) {
                os << p << " ";
            }
            os << std::endl;
        } break;

        case OpType::VIEW: {
            auto out              = op->output();
            auto [stride, offset] = op->get_params<ViewParams>();
            os << "VIEW: dst " << static_cast<int>(out->m_backend) << " type is " << static_cast<int>(out->m_dtype);
            os << " stride is ";
            for (auto &&p : stride) {
                os << p << " ";
            }
            os << "offset is " << offset << std::endl;
        } break;

        case OpType::SOFTMAX_EXT: {
            auto x                 = op->prev[0]->tensor();
            auto mask              = op->prev[1]->tensor();
            auto out               = op->output();
            auto [scale, max_bias] = op->get_params<SoftmaxExtParams>();
            os << "SOFTMAX_EXT: src " << static_cast<int>(x->m_backend) << " type is " << static_cast<int>(x->m_dtype)
               << " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            os << "mask " << static_cast<int>(mask->m_backend) << " type is " << static_cast<int>(mask->m_dtype)
               << " shape is ";
            for (auto &&p : mask->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(out->m_backend) << " type is " << static_cast<int>(out->m_dtype)
               << " shape is ";
            for (auto &&p : out->m_shape) {
                os << p << " ";
            }
            os << "scale " << scale << " max_bias " << max_bias << std::endl;
        } break;

        case OpType::GET_MASK: {
            auto out         = op->output();
            auto [mask, pos] = op->get_params<GetMaskParams>();
            os << "GET_MASK: dst " << static_cast<int>(out->m_backend) << " type is " << static_cast<int>(out->m_dtype)
               << " shape is ";
            for (auto &&p : out->m_shape) {
                os << p << " ";
            }
            os << "pos " << pos[0] << " pos size is " << pos.size() << std::endl;
        } break;

        case OpType::TRANSPOSE: {
            auto x   = op->prev[0]->tensor();
            auto out = op->output();
            os << "TRANSPOSE: src " << static_cast<int>(x->m_backend) << " type is " << static_cast<int>(x->m_dtype)
               << " shape is ";
            for (auto &&p : x->m_shape) {
                os << p << " ";
            }
            os << "dst " << static_cast<int>(out->m_backend) << " type is " << static_cast<int>(out->m_dtype)
               << " shape is ";
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

void Executor::split_graph() {
    size_t first_or_end{0};
    while (first_or_end < m_graph.ops.size()) {
        graph_splits.emplace_back(std::make_unique<GraphSplit>(m_graph.ops, first_or_end));
        auto &graph_split = graph_splits.back();
        graph_split->set_backend(m_platform.backends[m_graph.m_model_id][graph_split->graph_backend].get());
    }
}

void Executor::run_with_backend() {
    auto &model_id{m_graph.m_model_id};
    POWERSERVE_UNUSED(model_id);

    for (auto &graph_split : graph_splits) {
        graph_split->run_graph_compute();
    }
}

} // namespace powerserve
