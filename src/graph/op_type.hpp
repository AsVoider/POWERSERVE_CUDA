#pragma once

namespace smart {

enum class OpType {
	NONE = 0,

	ADD,
	MAT_MUL,
	RMS_NORM,
	SILU_HADAMARD,
	ROPE,
	SOFT_MAX,

	MHA,
};

}
