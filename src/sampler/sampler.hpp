#pragma once

#include <vector>
namespace smart {

struct Sampler {
	virtual int sample(std::vector<float> &logits) = 0;
	Sampler()									   = default;
	~Sampler()									   = default;
};

} // namespace smart
