add_executable(llama_tokenize "llama_tokenize.cpp")
target_link_libraries(llama_tokenize PRIVATE smart_serving CLI11::CLI11)

add_executable(run "run.cpp")
target_link_libraries(run PRIVATE smart_serving CLI11::CLI11)
