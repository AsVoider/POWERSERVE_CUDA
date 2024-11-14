#!/bin/bash

# cmake --build build_android --config RelWithDebInfo --parallel 12

# 使用 scp 命令传输文件
# scp -P 8022 -r ./build_android/bin/speculative u0_a334@192.168.61.65:/data/data/com.termux/files/home

sleep 1

# 在远程 ssh 中执行命令
ssh -p 8022 u0_a334@192.168.61.65 "export LD_LIBRARY_PATH=/vendor/lib64 && sudo ./run \
        --file-path 3_1/llama_3.1_8b_q4_0.gguf --vocab-path 3_1/Llama-3.1-Instruct-vocab.gguf \
        --qnn-path ./libdir32_i --config-path 3_1/model_config.json \
        --prompt \"<|start_header_id|>user<|end_header_id|>How do you complete the following paragraph: One day, I was in a store and I saw a woman with a baby?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\" --steps 512"
