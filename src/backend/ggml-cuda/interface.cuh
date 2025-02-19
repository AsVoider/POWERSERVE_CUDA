#include "ggml.h"

#include <functional>
#include <iostream>


namespace smart::ggml_cuda {

class cuda_context_warp;

using op_interface = std::function<void (cuda_context_warp &, ggml_tensor *dst)>;

class cuda_context_warp {
public:
    int device_count{0};
    void *ctx{nullptr};

    explicit cuda_context_warp();

    static auto malloc_cuda_buffer(void **ptr, size_t size) -> int;
    static auto free_cuda_buffer(void *ptr) -> int;
    // static inline auto malloc_host_buffer(void **ptr, size_t size) -> void const;
    // static inline auto free_host_buffer(void *ptr);
    static inline auto device_sync() -> int;

    template <int type>
    static inline auto copy_memory(void *dst, void *src, size_t size) -> int {
        if constexpr (type == 0) {
            return copy_memory_host_to_host(dst, src, size);
        } else if constexpr (type == 1) {
            return copy_memory_host_to_device(dst, src, size);
        } else if constexpr (type == 2) {
            return copy_memory_device_to_host(dst, src, size);
        } else if constexpr (type == 3) {
            return copy_memory_device_to_device(dst, src, size);
        } else {
            exit(1);
        }
    }

    template <int type>
    static inline auto copy_memory_async(void *dst, void *src, size_t size, void *context = nullptr) -> int {
        if constexpr (type == 0) {
            return copy_memory_host_to_host_async(dst, src, size, context);
        } else if constexpr (type == 1) {
            return copy_memory_host_to_device_async(dst, src, size, context);
        } else if constexpr (type == 2) {
            return copy_memory_device_to_host_async(dst, src, size, context);
        } else if constexpr (type == 3) {
            return copy_memory_device_to_device_async(dst, src, size, context);
        } else {
            exit(1);
        }
    }

private:
    static auto copy_memory_host_to_host(void *dst, void *src, size_t size) -> int;
    static auto copy_memory_host_to_device(void *dst, void *src, size_t size) -> int;
    static auto copy_memory_device_to_host(void *dst, void *src, size_t size) -> int;
    static auto copy_memory_device_to_device(void *dst, void *src, size_t size) -> int;

    static auto copy_memory_host_to_host_async(void *dst, void *src, size_t size, void *context = nullptr) -> int;
    static auto copy_memory_host_to_device_async(void *dst, void *src, size_t size, void *context = nullptr) -> int;
    static auto copy_memory_device_to_host_async(void *dst, void *src, size_t size, void *context = nullptr) -> int;
    static auto copy_memory_device_to_device_async(void *dst, void *src, size_t size, void *context = nullptr) -> int;
};

class op_interfaces {
public:
    static op_interface op_get_embedding;
    static op_interface op_mat_mul;
    static op_interface op_rms_norm;
    static op_interface op_softmax;
    static op_interface op_rope;
    static op_interface op_add;
    static op_interface op_cont;
    static op_interface op_copy;
};

} // namespace smart::ggml_cuda
