#include "ggml.h"

#include <memory>

namespace powerserve::ggml {

class cpu_mempool;

extern std::unique_ptr<cpu_mempool> default_mempool;

constexpr size_t PAGE_SIZE = 4096;

class cpu_mempool {
    using Alloc_ = std::allocator<char>;

    static constexpr size_t PADDING_SIZE = static_cast<size_t>(PAGE_SIZE / sizeof(char));

public:
    char *ptr{nullptr};
    size_t offset{0UL};
    size_t total_size{0UL};

public:
    cpu_mempool() = default;
    ~cpu_mempool();
    auto init_total(size_t size) -> void;
    auto reset() -> int;
    auto allocate(size_t size) -> void *;
};

static void default_alloc_total(size_t size) {
    default_mempool->init_total(size);
}

} // namespace powerserve::ggml
