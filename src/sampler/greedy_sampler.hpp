#pragma once

#include "sampler.hpp"
#include <vector>

namespace smart {

class GreedySampler : public Sampler {
public:
	int sample(std::vector<float> &logits) override;
	GreedySampler() = default;
	~GreedySampler() = default;
};

} // namespace smart