#pragma once

#include "common.hpp"

namespace smart {

// Base class for op parameters
struct OpParams {
	virtual ~OpParams() = default;
};

struct MHAParams : OpParams {
	size_t layer_id = 0;
	MHAParams()		= default;
	explicit MHAParams(size_t layer_id_) : layer_id(layer_id_) {}
	~MHAParams() override = default;
};

struct CopyParams : OpParams {
	size_t off	 = 0;
	CopyParams() = default;
	explicit CopyParams(size_t off_) : off(off_) {}
	~CopyParams() override = default;
};

} // namespace smart
