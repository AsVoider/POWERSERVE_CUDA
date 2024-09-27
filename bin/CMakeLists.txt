add_executable(llama_tokenize "llama_tokenize.cpp")
target_link_libraries(llama_tokenize PRIVATE smart_serving CLI11::CLI11)

add_executable(run "run.cpp")
target_link_libraries(run PRIVATE smart_serving CLI11::CLI11)

add_executable(run_q "run_q.cpp")
target_link_libraries(run_q PRIVATE smart_serving CLI11::CLI11)

add_executable(run_q_ggml "run_q_ggml.cpp")
target_link_libraries(run_q_ggml PRIVATE smart_serving CLI11::CLI11 ggml)

add_executable(run_q_ggml2 "run_q_ggml_newapi.cpp")
target_link_libraries(run_q_ggml2 PRIVATE smart_serving CLI11::CLI11 ggml)

# add_executable(main "main.cpp")
# target_link_libraries(main PRIVATE smart_serving CLI11::CLI11)