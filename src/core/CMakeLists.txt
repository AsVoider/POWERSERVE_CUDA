target_sources(smart_serving PRIVATE
    "config.cpp"
    "perf.cpp"
    "perfetto_trace.cpp"
    "thread_pool.cpp"
    "timer.cpp"
)

if (SMART_WITH_PERFETTO)
    target_compile_definitions(smart_serving PUBLIC SMART_WITH_PERFETTO)
endif()
