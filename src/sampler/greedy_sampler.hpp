#pragma once

#include "sampler.hpp"

#include <vector>

namespace smart {

struct GreedySampler : Sampler {
	int sample(std::vector<float> &logits) override;
	GreedySampler()	 = default;
	~GreedySampler() = default;
};

} // namespace smart
