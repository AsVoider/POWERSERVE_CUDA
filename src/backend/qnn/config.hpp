#pragma once

#include "common.hpp"
#include "core/config.hpp"

#include <nlohmann/json.hpp>

namespace smart::qnn {

const std::string QNN_WORKSPACE_DIR_NAME = "qnn-workspace";

struct QNNGraphConfig {
    std::string type;
    std::string graph_name;
    size_t batch_size;
    Path model_path;

    virtual ~QNNGraphConfig() = default;
};

struct ChunkConfig : QNNGraphConfig {
    size_t start_layer_id;
    size_t end_layer_id;
    size_t cache_size;
    size_t context_size;
    std::string kv_path_format;
    size_t kv_size;

    virtual ~ChunkConfig() override = default;
};

struct EmbeddingConfig : QNNGraphConfig {
    size_t input_type;
    size_t output_dim;
    size_t input_dim;
    std::string input_name;
    std::string output_name;
    virtual ~EmbeddingConfig() override = default;
};

struct QNNConfig {
    size_t n_hvx_threads;
    float attention_mask_value;
    const std::shared_ptr<LLMConfig> &model_config;
    std::vector<EmbeddingConfig> lm_heads;

    std::vector<ChunkConfig> chunks;

    QNNConfig(const Path &path, const std::shared_ptr<LLMConfig> &model_config) : model_config(model_config) {
        std::ifstream f(path);
        auto json = nlohmann::json::parse(f);
        {
            auto data = json.at("model_parameters");
            data.at("attention_mask_value").get_to(attention_mask_value);
        }
        {
            auto data = json.at("qnn_parameters");
            data.at("n_hvx_threads").get_to(n_hvx_threads);
        }

        {
            auto data_array = json.at("graphs");
            SMART_ASSERT(data_array.is_array());

            chunks.reserve(data_array.size());
            for (auto data : data_array) {
                ChunkConfig info;
                data.at("type").get_to(info.type);
                data.at("graph_name").get_to(info.graph_name);
                data.at("start_layer_id").get_to(info.start_layer_id);
                data.at("end_layer_id").get_to(info.end_layer_id);
                data.at("batch_size").get_to(info.batch_size);
                data.at("cache_size").get_to(info.cache_size);
                data.at("context_size").get_to(info.context_size);
                data.at("model_path").get_to(info.model_path);
                data.at("kv_path_format").get_to(info.kv_path_format);
                data.at("kv_size").get_to(info.kv_size);
                chunks.push_back(info);
            }
        }

        if (json.contains("embeddings")) {
            auto data_array = json.at("embeddings");
            lm_heads.reserve(data_array.size());
            for (auto data : data_array) {
                EmbeddingConfig info;
                data.at("graph_name").get_to(info.graph_name);
                data.at("batch_size").get_to(info.batch_size);
                data.at("model_path").get_to(info.model_path);
                data.at("input_type").get_to(info.input_type);
                data.at("input_dim").get_to(info.input_dim);
                data.at("output_dim").get_to(info.output_dim);
                data.at("input_name").get_to(info.input_name);
                data.at("output_name").get_to(info.output_name);
                lm_heads.push_back(info);
            }
        }
    }
};

} // namespace smart::qnn
