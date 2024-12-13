#include "graph_interface.hpp"

#include "causal_lm.hpp"

namespace smart::qnn {

GraphInterface::GraphInterface(CausalLM &parent, const Path &model_path) :
    m_parent(parent),
    m_model_config(parent.m_model_config),
    m_context_binary(m_parent.load_context_binary(model_path)) {}

// GraphInterface::GraphInterface(GraphInterface &&other) noexcept :
//     m_parent(other.m_parent),
//     m_model_config(m_parent.m_model_config),
//     m_context_binary(other.m_context_binary) {}

void GraphInterface::execute() const {
    m_graph->execute();
}

Embedding::Embedding(CausalLM &parent, EmbeddingConfig &info) :
    GraphInterface(parent, info.model_path),
    m_config(info) {
    m_graph = std::make_unique<Graph>(*m_context_binary.m_context, info.graph_name);
    m_graph->set_n_hvx_threads(m_parent.m_config.n_hvx_threads);
    setup_tensors();
}

// Embedding::Embedding(Embedding &&other) noexcept : GraphInterface(std::move(other)), m_config(other.m_config) {
//     m_sibling_embedding       = other.m_sibling_embedding;
//     other.m_sibling_embedding = nullptr;
//     m_tensors                 = std::move(other.m_tensors);
//     m_buffers                 = std::move(other.m_buffers);
//     m_graph                   = std::move(other.m_graph);
// }

auto Embedding::io_tensor_size() const -> size_t {
    size_t size = 0;

    size += m_tensors.x->size();
    size += m_tensors.out->size();
    return size;
}

auto Embedding::input_buffer() -> void * {
    SMART_ASSERT(m_buffers.x->m_type == (Qnn_DataType_t)m_config.input_type);
    return (void *)m_buffers.x->m_data;
}

auto Embedding::output_buffer() const -> const float * {
    SMART_ASSERT(m_buffers.out->m_type == QNN_DATATYPE_FLOAT_32);
    return (const float *)m_buffers.out->m_data;
}

void Embedding::setup_tensors() {
    if (m_config.input_dim == 1) {
        m_tensors.x =
            m_graph->get_tensor(m_config.input_name)->check({m_config.batch_size}, (Qnn_DataType_t)m_config.input_type);
    } else {
        m_tensors.x = m_graph->get_tensor(m_config.input_name)
                          ->check({m_config.batch_size, m_config.input_dim}, (Qnn_DataType_t)m_config.input_type);
    }
    m_tensors.out = m_graph->get_tensor(m_config.output_name)
                        ->check({m_config.batch_size, m_config.output_dim}, QNN_DATATYPE_FLOAT_32);
}

void Embedding::setup_buffers() {
    if (m_sibling_embedding != nullptr) {
        m_buffers = m_sibling_embedding->m_buffers;
    }
    auto setup = [&](std::shared_ptr<SharedBuffer> &buffer, Tensor *tensor) {
        if (!buffer) {
            buffer = std::make_shared<SharedBuffer>(
                *m_context_binary.m_context, *m_context_binary.m_alloc, tensor->type(), tensor->n_elements()
            );
        }
        tensor->setup_shared_buffer(*buffer);
    };

    setup(m_buffers.x, m_tensors.x);
    setup(m_buffers.out, m_tensors.out);
}

ModelChunk::ModelChunk(CausalLM &parent, ChunkConfig &info) : GraphInterface(parent, info.model_path), m_config(info) {
    m_graph = std::make_unique<Graph>(*m_context_binary.m_context, m_config.graph_name);
    m_graph->set_n_hvx_threads(m_parent.m_config.n_hvx_threads);

    setup_tensors();
}

// ModelChunk::ModelChunk(ModelChunk &&other) noexcept : GraphInterface(std::move(other)), m_config(other.m_config) {
//     m_sibling_chunk       = other.m_sibling_chunk;
//     other.m_sibling_chunk = nullptr;
//     m_tensors             = std::move(other.m_tensors);
//     m_buffers             = std::move(other.m_buffers);
//     m_graph               = std::move(other.m_graph);
// }

auto ModelChunk::n_layers() const -> size_t {
    return m_config.end_layer_id - m_config.start_layer_id;
}

auto ModelChunk::io_tensor_size() const -> size_t {
    size_t size = 0;

    size += m_tensors.x->size();
    size += m_tensors.attn_bias->size();
    size += m_tensors.rope_embed_sin->size();
    size += m_tensors.rope_embed_cos->size();
    size += m_tensors.out->size();

    for (size_t i = 0; i < n_layers(); i++) {
        for (size_t j = 0; j < m_model_config->n_kv_heads; j++) {
            size += m_tensors.kvs[i].keys[j]->size();
            size += m_tensors.kvs[i].values[j]->size();
        }
    }

    return size;
}

auto ModelChunk::kv_cache_size() const -> size_t {
    size_t size = 0;

    for (size_t i = 0; i < n_layers(); i++) {
        for (size_t j = 0; j < m_model_config->n_kv_heads; j++) {
            size += m_tensors.caches[i].keys_t[j]->size();
            size += m_tensors.caches[i].values[j]->size();
        }
    }

    return size;
}

auto ModelChunk::input_buffer() -> void * {
    SMART_ASSERT(m_buffers.x->m_type == QNN_DATATYPE_FLOAT_32);
    return (void *)m_buffers.x->m_data;
}

auto ModelChunk::output_buffer() const -> const float * {
    SMART_ASSERT(m_buffers.out->m_type == QNN_DATATYPE_FLOAT_32);
    return (const float *)m_buffers.out->m_data;
}

void ModelChunk::setup_tensors() {
    auto &llm_config = m_model_config;
    auto head_dim    = llm_config->head_size;
    auto dim         = llm_config->dim;
    auto n_kv_heads  = llm_config->n_kv_heads;
    m_tensors.x      = m_graph->get_tensor("x")->check({m_config.batch_size, dim}, QNN_DATATYPE_FLOAT_32);

    m_tensors.attn_bias =
        m_graph->get_tensor("attn_bias")->check({m_config.batch_size, m_config.context_size}, QNN_DATATYPE_FLOAT_16);

    m_tensors.rope_embed_cos =
        m_graph->get_tensor("rope_embed_cos")->check({m_config.batch_size, head_dim / 2}, QNN_DATATYPE_FLOAT_32);

    m_tensors.rope_embed_sin =
        m_graph->get_tensor("rope_embed_sin")->check({m_config.batch_size, head_dim / 2}, QNN_DATATYPE_FLOAT_32);

    m_tensors.caches.resize(n_layers());
    for (size_t i = 0; i < n_layers(); i++) {
        auto &cache = m_tensors.caches[i];

        cache.keys_t.resize(n_kv_heads);
        cache.values.resize(n_kv_heads);
        for (size_t j = 0; j < n_kv_heads; j++) {
            cache.keys_t[j] =
                m_graph->get_tensor(fmt::format("layer_{}_key_t_cache_{}", m_config.start_layer_id + i, j))
                    ->check({head_dim, m_config.cache_size}, kv_type);

            cache.values[j] =
                m_graph->get_tensor(fmt::format("layer_{}_value_cache_{}", m_config.start_layer_id + i, j))
                    ->check({m_config.cache_size, head_dim}, kv_type);
        }
    }

    m_tensors.out = m_graph->get_tensor("out")->check({m_config.batch_size, dim}, QNN_DATATYPE_FLOAT_32);

    m_tensors.kvs.resize(n_layers());
    for (size_t i = 0; i < n_layers(); i++) {
        auto &kv = m_tensors.kvs[i];

        kv.keys.resize(n_kv_heads);
        kv.values.resize(n_kv_heads);
        for (size_t j = 0; j < n_kv_heads; j++) {
            kv.keys[j] = m_graph->get_tensor(fmt::format("layer_{}_key_{}", m_config.start_layer_id + i, j))
                             ->check({m_config.batch_size, head_dim}, QNN_DATATYPE_FLOAT_16);
            kv.values[j] = m_graph->get_tensor(fmt::format("layer_{}_value_{}", m_config.start_layer_id + i, j))
                               ->check({m_config.batch_size, head_dim}, QNN_DATATYPE_FLOAT_16);
        }
    }
}

void ModelChunk::initialize(KVCacheInterface &kv_cache) {
    setup_buffers();

    if (!m_sibling_chunk) {
        load_kv(kv_cache);
    }

    // Initialize attn_bias with mask values
    auto attn_bias = (__fp16 *)m_buffers.attn_bias->m_data;
    std::fill(attn_bias, attn_bias + m_tensors.attn_bias->n_elements(), m_parent.m_config.attention_mask_value);
}

void ModelChunk::setup_buffers() {
    auto &llm_config = m_model_config;
    auto n_kv_heads  = llm_config->n_kv_heads;

    // Share buffers with sibling chunk
    if (m_sibling_chunk) {
        m_buffers = m_sibling_chunk->m_buffers;
    }

    auto setup = [&](std::shared_ptr<SharedBuffer> &buffer, Tensor *tensor) {
        if (!buffer) {
            buffer = std::make_shared<SharedBuffer>(
                *m_context_binary.m_context, *m_context_binary.m_alloc, tensor->type(), tensor->n_elements()
            );
        }

        tensor->setup_shared_buffer(*buffer);
    };

    setup(m_buffers.x, m_tensors.x);
    setup(m_buffers.attn_bias, m_tensors.attn_bias);
    setup(m_buffers.rope_embed_cos, m_tensors.rope_embed_cos);
    setup(m_buffers.rope_embed_sin, m_tensors.rope_embed_sin);
    setup(m_buffers.out, m_tensors.out);

    m_buffers.kvs.resize(n_layers());
    m_buffers.caches.resize(n_layers());
    for (size_t i = 0; i < n_layers(); i++) {
        m_buffers.kvs[i].keys.resize(n_kv_heads);
        m_buffers.caches[i].keys_t.resize(n_kv_heads);
        m_buffers.kvs[i].values.resize(n_kv_heads);
        m_buffers.caches[i].values.resize(n_kv_heads);

        for (size_t j = 0; j < n_kv_heads; j++) {
            setup(m_buffers.kvs[i].keys[j], m_tensors.kvs[i].keys[j]);
            setup(m_buffers.kvs[i].values[j], m_tensors.kvs[i].values[j]);
            setup(m_buffers.caches[i].keys_t[j], m_tensors.caches[i].keys_t[j]);
            setup(m_buffers.caches[i].values[j], m_tensors.caches[i].values[j]);
        }
    }
}

void ModelChunk::load_kv(KVCacheInterface &kv_cache) {
    auto &llm_config = m_model_config;
    auto head_dim    = llm_config->head_size;
    auto n_kv_heads  = llm_config->n_kv_heads;

    auto load = [&](const std::string &kv_type, size_t layer_id, size_t head_id) {
        auto layer_id_arg = fmt::arg("layer_id", layer_id);
        auto kv_type_arg  = fmt::arg("kv_type", kv_type);
        auto head_id_arg  = fmt::arg("head_id", head_id);
        auto path =
            m_parent.m_model_folder /
            fmt::vformat(m_config.kv_path_format, fmt::make_format_args(layer_id_arg, kv_type_arg, head_id_arg));

        size_t n_elements = m_config.kv_size * head_dim;
        auto data         = read_binary_file<float>(path, n_elements);

        std::vector<__fp16> fp16_data(n_elements);
        for (size_t i = 0; i < n_elements; i++) {
            fp16_data[i] = data[i];
        }

        for (size_t i = 0; i < m_config.kv_size; i++) {
            KVView entry;

            if (kv_type == "key") {
                entry = kv_cache.key_entry({.layer_id = layer_id, .head_id = head_id, .index = i});
            } else {
                entry = kv_cache.value_entry({.layer_id = layer_id, .head_id = head_id, .index = i});
            }

            entry.copy_from({
                .n_elements   = head_dim,
                .element_size = kv_element_size,
                .stride       = kv_element_size,
                .data         = fp16_data.data() + i * head_dim,
            });
        }
    };

    for (size_t i = 0; i < n_layers(); i++) {
        size_t layer_id = m_config.start_layer_id + i;
        for (size_t j = 0; j < n_kv_heads; j++) {
            load("key", layer_id, j);
            load("value", layer_id, j);
        }
    }
}

} // namespace smart::qnn
