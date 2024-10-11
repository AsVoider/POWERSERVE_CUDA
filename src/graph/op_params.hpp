#pragma once

#include "common.hpp"

namespace smart {

// Base class for op parameters
struct OpParams {
	virtual ~OpParams() = default;
};

struct MHAParams : OpParams {
	size_t layer_id_ = 0;
	MHAParams()		 = default;
	explicit MHAParams(size_t layer_id) : layer_id_(layer_id) {}
	~MHAParams() override = default;
};

struct CopyParams : OpParams {
	size_t off_	 = 0;
	CopyParams() = default;
	explicit CopyParams(size_t off) : off_(off) {}
	~CopyParams() override = default;
};

} // namespace smart
