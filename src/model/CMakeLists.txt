add_subdirectory(llama)
add_subdirectory(internvl)
add_subdirectory(qwen2)
add_subdirectory(module)

target_sources(smart_serving PRIVATE
    "model_loader.cpp"
)
