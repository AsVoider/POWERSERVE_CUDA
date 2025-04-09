#include "backend/ggml/interface.hpp"

namespace powerserve::ggml {

std::unique_ptr<cpu_mempool> default_mempool{std::make_unique<cpu_mempool>()};

cpu_mempool::~cpu_mempool() {
    if (ptr not_eq nullptr) {
        free(ptr);
    }
}

auto cpu_mempool::init_total(size_t size) -> void {
    if (size == 0) [[unlikely]] {
        exit(1);
    }

    size = (size + PADDING_SIZE - 1) / PADDING_SIZE * PADDING_SIZE;

    if (total_size > 2 * size or total_size < size) {
        reset();
    }

    if (size <= total_size) {
        offset = 0;
        return;
    }

    ptr        = static_cast<char *>(malloc(size));
    total_size = size;
}

auto cpu_mempool::reset() -> int {
    if (ptr not_eq nullptr) {
        free(ptr);
        ptr        = nullptr;
        offset     = 0;
        total_size = 0;
    }

    return 0;
}

auto cpu_mempool::allocate(size_t size) -> void * {
    if (ptr == nullptr) [[unlikely]] {
        exit(1);
    }

    if (size % 256 not_eq 0UL or size + offset > total_size) {
        exit(1);
    }

    void *ret{static_cast<void *>(ptr + offset)};
    offset += size;
    return ret;
}

} // namespace powerserve::ggml
