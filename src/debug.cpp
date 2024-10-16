#include "debug.hpp"

#include <fmt/core.h>

namespace smart {

void debug_meta_info(gguf_context *gguf_ctx, ggml_context *ggml_ctx) {
    {
        fmt::println("version     : {:10}", gguf_get_version(gguf_ctx));
        fmt::println("n_kv        : {:10}", gguf_get_n_kv(gguf_ctx));
        fmt::println("n_tensors   : {:10}", gguf_get_n_tensors(gguf_ctx));
        fmt::println("alignment   : {:10}", gguf_get_alignment(gguf_ctx));
        fmt::println("meta size   : {:10}", gguf_get_meta_size(gguf_ctx));
        fmt::println("data offset : {:10}", gguf_get_data_offset(gguf_ctx));
    }

    {
        for (auto i = 0; i < gguf_get_n_kv(gguf_ctx); i++) {
            auto key      = gguf_get_key(gguf_ctx, i);
            auto v_type   = gguf_get_kv_type(gguf_ctx, i);
            auto type_str = gguf_type_name(v_type);
            fmt::println("{:40}: {:4}", key, type_str);
        }
    }

    {
        for (auto i = 0; i < gguf_get_n_tensors(gguf_ctx); i++) {
            auto name   = gguf_get_tensor_name(gguf_ctx, i);
            auto t_type = gguf_get_tensor_type(gguf_ctx, i);
            fmt::println("{:40}: {:6}: {:10}", name, ggml_type_name(t_type), gguf_get_tensor_offset(gguf_ctx, i));
        }
    }

    {
        fmt::println("GGML used mem        : {:10}", ggml_used_mem(ggml_ctx));
        fmt::println("GGML no alloc        : {:10}", ggml_get_no_alloc(ggml_ctx));
        fmt::println("GGML mem buffer      : {:10}", ggml_get_mem_buffer(ggml_ctx));
        fmt::println("GGML mem size        : {:10}", ggml_get_mem_size(ggml_ctx));
        fmt::println("GGML max tensor size : {:10}", ggml_get_max_tensor_size(ggml_ctx));
    }
}

void debug_tensors_info(gguf_context *gguf_ctx, ggml_context *ggml_ctx) {
    for (auto i = 0; i < gguf_get_n_tensors(gguf_ctx); i++) {
        auto t = ggml_get_tensor(ggml_ctx, gguf_get_tensor_name(gguf_ctx, i));

        fmt::println(
            "{:40}|{:>5}|({:5},{:5},{:1},{:1})|{:10}|{:4}|{:4}|{:10}",
            ggml_get_name(t),
            ggml_type_name(t->type),
            t->ne[0],
            t->ne[1],
            t->ne[2],
            t->ne[3],
            ggml_get_data(t),
            ggml_type_size(t->type),
            ggml_blck_size(t->type),
            ggml_row_size(t->type, ggml_nelements(t)) // ne * ggml_type_size / ggml_blk_size (bytes)
        );
    }
}

void debug_weight_info(std::string name, OpTensor *opt) {
    auto out = fmt::format("{:5}|{:15}", ggml_type_name(opt->type), opt->data);
    fmt::println("{:15}: {}", name, out);
}

} // namespace smart
