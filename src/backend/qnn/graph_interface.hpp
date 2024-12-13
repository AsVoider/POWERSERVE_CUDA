#pragma once

#include "common.hpp"
#include "config.hpp"
#include "core/kv_cache.hpp"
#include "qnn.hpp"

namespace smart::qnn {

struct CausalLM;

struct GraphInterface {
    CausalLM &m_parent;
    const std::shared_ptr<LLMConfig> &m_model_config;
    ContextBinary &m_context_binary;
    std::unique_ptr<qnn::Graph> m_graph;

    GraphInterface(CausalLM &parent, const Path &model_path);
    virtual ~GraphInterface() = default;

    virtual auto io_tensor_size() const -> size_t       = 0;
    virtual auto input_buffer() -> void               * = 0;
    virtual auto output_buffer() const -> const float * = 0;

    virtual void setup_tensors() = 0;
    virtual void setup_buffers() = 0;

    void execute() const;
};

struct Embedding : GraphInterface {
    EmbeddingConfig &m_config;
    Embedding *m_sibling_embedding = nullptr;

    struct {
        qnn::Tensor *x;
        qnn::Tensor *out;
    } m_tensors;

    struct BufferSet {
        std::shared_ptr<SharedBuffer> x;
        std::shared_ptr<SharedBuffer> out;
    } m_buffers;

    Embedding(CausalLM &parent, EmbeddingConfig &info);
    virtual ~Embedding() override = default;
    auto io_tensor_size() const -> size_t override;
    auto input_buffer() -> void * override;
    auto output_buffer() const -> const float * override;

    void setup_tensors() override;
    void setup_buffers() override;
};

struct ModelChunk final : GraphInterface {
    static constexpr auto kv_type           = QNN_DATATYPE_FLOAT_16;
    static constexpr size_t kv_element_size = type_size(kv_type);

    ChunkConfig &m_config;
    ModelChunk *m_sibling_chunk = nullptr;

    struct {
        qnn::Tensor *x;
        qnn::Tensor *attn_bias;
        qnn::Tensor *rope_embed_cos;
        qnn::Tensor *rope_embed_sin;

        struct KVCache {
            std::vector<qnn::Tensor *> keys_t;
            std::vector<qnn::Tensor *> values;
        };

        std::vector<KVCache> caches;

        qnn::Tensor *out;

        struct KV {
            std::vector<qnn::Tensor *> keys;
            std::vector<qnn::Tensor *> values;
        };

        std::vector<KV> kvs;
    } m_tensors;

    struct {
        std::shared_ptr<SharedBuffer> x;
        std::shared_ptr<SharedBuffer> attn_bias;
        std::shared_ptr<SharedBuffer> rope_embed_cos;
        std::shared_ptr<SharedBuffer> rope_embed_sin;

        struct KVCache {
            std::vector<std::shared_ptr<SharedBuffer>> keys_t;
            std::vector<std::shared_ptr<SharedBuffer>> values;
        };

        std::vector<KVCache> caches;

        std::shared_ptr<SharedBuffer> out;

        struct KV {
            std::vector<std::shared_ptr<SharedBuffer>> keys;
            std::vector<std::shared_ptr<SharedBuffer>> values;
        };

        std::vector<KV> kvs;
    } m_buffers;

    ModelChunk(CausalLM &parent, ChunkConfig &info);
    virtual ~ModelChunk() override = default;
    auto n_layers() const -> size_t;
    auto io_tensor_size() const -> size_t override;
    auto kv_cache_size() const -> size_t;
    auto input_buffer() -> void * override;
    auto output_buffer() const -> const float * override;

    void setup_tensors() override;
    void initialize(KVCacheInterface &kv_cache);
    void setup_buffers() override;
    void load_kv(KVCacheInterface &kv_cache);
};

} // namespace smart::qnn
