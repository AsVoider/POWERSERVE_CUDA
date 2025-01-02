#include "model/model.hpp"

namespace smart {

auto load_model(const Path &model_dir, std::shared_ptr<ModelConfig> &out_config) -> std::shared_ptr<Model>;

} // namespace smart
