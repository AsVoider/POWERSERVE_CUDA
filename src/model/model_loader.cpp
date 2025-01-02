#include "model/model_loader.hpp"

#include "model/internvl/internvl_model.hpp"
#include "model/llama/llama_model.hpp"
#include "model/qwen2/qwen2_model.hpp"

namespace smart {

auto load_model(const Path &model_dir, std::shared_ptr<ModelConfig> &out_config) -> std::shared_ptr<Model> {
    std::shared_ptr<Model> out_model;
    out_config = std::make_shared<ModelConfig>(model_dir / MODEL_CONFIG_FILENAME);

    auto arch        = out_config->arch;
    auto weight_path = model_dir / MODEL_WEIGHTS_DIR / MODEL_WEIGHTS_FILENAME;
    if (arch == "llama") {
        out_model = std::make_shared<LlamaModel>(weight_path, out_config);
    } else if (arch == "qwen2") {
        out_model = std::make_shared<Qwen2Model>(weight_path, out_config);
    } else if (arch == "internvl") {
        out_model = std::make_shared<InternVL>(weight_path, out_config);
    } else {
        SMART_ABORT("unknown model type: {}", arch);
    }

    SMART_LOG_INFO("Load model {} ...", arch);
    return out_model;
}

} // namespace smart
