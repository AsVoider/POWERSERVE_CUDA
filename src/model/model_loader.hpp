#include "model/llama/llama_model.hpp"
#include "model/model.hpp"

namespace smart {

static std::unique_ptr<Model> load_model(std::shared_ptr<LLMConfig> &config, const Path &config_dir) {
    config = std::make_shared<LLMConfig>(config_dir / LLM_CONFIG_FILENAME);

    auto arch = config->arch;
    std::unique_ptr<Model> model;
    auto weight_path = config_dir / LLM_WEIGHTS_FILENAME;
    if (arch == "llama" || arch == "qwen2") {
        model = std::make_unique<LlamaModel>(weight_path, config);
    } else {
        fmt::print("Unknown model type\n");
        SMART_ASSERT(false);
    }
    return std::move(model);
}

} // namespace smart
