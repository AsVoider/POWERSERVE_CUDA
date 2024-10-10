#pragma once

#include "common.hpp"

namespace smart {

// Base class for op parameters
struct OpParams {};

struct MHAParams : OpParams {
    size_t layer_id;
};

struct CopyParams : OpParams {
    int64_t off;
};

}
