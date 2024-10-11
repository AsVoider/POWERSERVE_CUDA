#pragma once

namespace smart {

enum class OpType {
	NONE = 0,

	ADD,
	MAT_MUL,
	RMS_NORM,
	SILU_HADAMARD,
	ROPE,
	SOFTMAX,
	COPY,

	MHA,
};

}
