target_sources(smart_serving PRIVATE
    "internvl_model.cpp"
)
target_link_libraries(smart_serving PRIVATE stb_headers)
target_link_libraries(smart_serving PRIVATE xtensor xtl)
