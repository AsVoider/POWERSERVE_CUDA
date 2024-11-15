#pragma once

#include "common.hpp"

namespace smart {

struct KVPosition {
    size_t layer_id = 0;
    size_t head_id  = 0;
    size_t index    = 0;
};

struct KVView {
    size_t n_elements   = 0;
    size_t element_size = 0;
    size_t stride       = 0;
    void *data          = nullptr;

    bool is_contiguous() const {
        return element_size == stride;
    }

    ALWAYS_INLINE void copy_from(KVView other) {
        SMART_ASSERT(n_elements == other.n_elements);
        SMART_ASSERT(element_size == other.element_size);

        if (is_contiguous() && other.is_contiguous()) {
            memcpy(data, other.data, n_elements * element_size);
        } else if (other.is_contiguous() && element_size == 2) {
            auto src = (uint16_t *)other.data;
            auto dst = (uint8_t *)data;

            #pragma unroll(4)
            for (size_t i = 0; i < n_elements; i++) {
                *(uint16_t *)dst = *src;
                src++;
                dst += stride;
            }
        } else if (element_size == 2) {
            auto src = (uint8_t *)other.data;
            auto dst = (uint8_t *)data;

            #pragma unroll(4)
            for (size_t i = 0; i < n_elements; i++) {
                *(uint16_t *)dst = *(uint16_t *)src;
                src += other.stride;
                dst += stride;
            }
        } else {
            SMART_ASSERT(false);
        }
    }
};

struct KVInterfaceBase {
    auto get_key(KVPosition token_pos) const -> KVView;
    auto get_value(KVPosition token_pos) const -> KVView;
    auto key_entry(KVPosition cache_pos) const -> KVView;
    auto value_entry(KVPosition cache_pos) const -> KVView;
    void set_mask(size_t cache_index, bool mask);
};

struct KVCacheInterface {
    virtual ~KVCacheInterface() = default;

    virtual auto key_entry(KVPosition cache_pos) const -> KVView      = 0;
    virtual auto value_entry(KVPosition cache_pos) const -> KVView    = 0;
    virtual void copy(size_t dst_cache_index, size_t src_token_index) = 0;
    virtual void move(size_t dst_cache_index, size_t src_cache_index) = 0;
    virtual void advance(size_t n_tokens)                             = 0;
    virtual void rollback(size_t n_tokens)                            = 0;
    virtual void truncate(size_t n_tokens)                            = 0;
    virtual void save(size_t n_tokens)                                = 0;
};

template <typename KVInterface>
struct KVCache final : KVCacheInterface {
    const size_t n_layers   = 0;
    const size_t n_kv_heads = 0;
    const size_t n_ctx      = 0;
    size_t position         = 0; // current recived kv

    template <typename... Args>
    KVCache(size_t n_layers, size_t n_kv_heads, size_t n_ctx, Args &&...args) :
        n_layers(n_layers),
        n_kv_heads(n_kv_heads),
        n_ctx(n_ctx),
        interface(std::forward<Args>(args)...) {}

    auto key_entry(KVPosition cache_pos) const -> KVView override {
        return interface.key_entry(cache_pos);
    }

    auto value_entry(KVPosition cache_pos) const -> KVView override {
        return interface.value_entry(cache_pos);
    }

    void copy(size_t dst_cache_index, size_t src_token_index) override {
        for (size_t i = 0; i < n_layers; i++) {
            for (size_t j = 0; j < n_kv_heads; j++) {
                auto src_key = interface.get_key({.layer_id = i, .head_id = j, .index = src_token_index});
                auto dst_key = interface.key_entry({.layer_id = i, .head_id = j, .index = dst_cache_index});
                dst_key.copy_from(src_key);

                auto src_value = interface.get_value({.layer_id = i, .head_id = j, .index = src_token_index});
                auto dst_value = interface.value_entry({.layer_id = i, .head_id = j, .index = dst_cache_index});
                dst_value.copy_from(src_value);
            }
        }
    }

    void move(size_t dst_cache_index, size_t src_cache_index) override {
        for (size_t i = 0; i < n_layers; i++) {
            for (size_t j = 0; j < n_kv_heads; j++) {
                auto src_key = interface.key_entry({.layer_id = i, .head_id = j, .index = src_cache_index});
                auto dst_key = interface.key_entry({.layer_id = i, .head_id = j, .index = dst_cache_index});
                dst_key.copy_from(src_key);

                auto src_value = interface.value_entry({.layer_id = i, .head_id = j, .index = src_cache_index});
                auto dst_value = interface.value_entry({.layer_id = i, .head_id = j, .index = dst_cache_index});
                dst_value.copy_from(src_value);
            }
        }
    }

    void advance(size_t n_tokens) override {
        SMART_ASSERT(position + n_tokens <= n_ctx);
        for (size_t i = 0; i < n_tokens; i++) {
            interface.set_mask(position + i, false);
        }
        position += n_tokens;
    }

    void rollback(size_t n_tokens) override {
        SMART_ASSERT(position >= n_tokens);
        position -= n_tokens;
        for (size_t i = 0; i < n_tokens; i++) {
            interface.set_mask(position + i, true);
        }
    }

    void truncate(size_t n_tokens) override {
        if (n_tokens < position) {
            rollback(position - n_tokens);
        }
    }

    void save(size_t n_tokens) override { // copy temp buffer's kvcache into real cache
        SMART_ASSERT(position + n_tokens <= n_ctx);
        for (size_t i = 0; i < n_tokens; i++) {
            copy(position + i, i);
        }
    }

private:
    KVInterface interface;
};

} // namespace smart
