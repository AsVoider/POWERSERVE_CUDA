file(GLOB_RECURSE storage_source *.cpp)
target_sources(smart_serving PRIVATE
    ${storage_source}
)
