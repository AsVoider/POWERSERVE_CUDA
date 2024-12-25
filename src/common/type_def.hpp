#pragma once

#include "llama-vocab.h"

#include <filesystem>

namespace smart {

using Path  = std::filesystem::path;
using Token = llama_vocab::id;

} // namespace smart
