#include "model/internvl/internvl_model.hpp"
#include "model/llama/llama_model.hpp"
#include "model/model.hpp"
#include "model/qwen2/qwen2_model.hpp"

namespace smart {

static std::unique_ptr<Model> load_model(std::shared_ptr<ModelConfig> &config, const Path &model_dir) {
    config    = std::make_shared<ModelConfig>(model_dir / MODEL_CONFIG_FILENAME);
    auto arch = config->arch;
    std::unique_ptr<Model> model;
    auto weight_path = model_dir / MODEL_WEIGHTS_DIR / MODEL_WEIGHTS_FILENAME;
    if (arch == "llama") {
        model = std::make_unique<smart::LlamaModel>(weight_path, config);
    } else if (arch == "qwen2") {
        model = std::make_unique<smart::Qwen2Model>(weight_path, config);
    } else if (arch == "internvl") {
        model = std::make_unique<smart::InternVL>(weight_path, config);
    } else {
        SMART_ABORT("unknown model type: {}", arch);
    }

    SMART_LOG_INFO("Load model {} ...", arch);
    return model;
}

} // namespace smart
