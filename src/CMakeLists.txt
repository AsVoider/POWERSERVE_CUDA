add_library(smart_serving STATIC
    "llama_tokenizer.cpp"
    "generate.hpp" "generate.cpp"
    "model.hpp" "model.cpp"
    "sampler.hpp" "sampler.cpp"
    "tensor.hpp"
    "debug.hpp" "debug.cpp"
)
target_link_libraries(smart_serving PUBLIC fmt::fmt ggml llama_tokenizer)
target_include_directories(smart_serving PUBLIC .)
