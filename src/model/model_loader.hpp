#include "model/internvl/internvl_model.hpp"
#include "model/llama/llama_model.hpp"
#include "model/model.hpp"

namespace smart {

static std::unique_ptr<Model> load_model(std::shared_ptr<Config> config) {
    config->main_model_config = std::make_shared<ModelConfig>(config->main_model_dir / MODEL_CONFIG_FILENAME);
    auto arch                 = config->main_model_config->arch;
    std::unique_ptr<Model> model;
    auto weight_path = config->main_model_dir / MODEL_WEIGHTS_FILENAME;
    if (arch == "llama" || arch == "qwen2") {
        model = std::make_unique<smart::LlamaModel>(weight_path, config->main_model_config);
    } else if (arch == "internvl") {
        model = std::make_unique<smart::InternVL>(weight_path, config->main_model_config);
    } else {
        fmt::print("Unknown model type\n");
    }

    return std::move(model);
}

} // namespace smart
