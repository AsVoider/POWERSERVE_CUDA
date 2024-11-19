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

# server 测试
# ssh -p 8022 u0_a334@192.168.61.65 "export LD_LIBRARY_PATH=/vendor/lib64 && sudo ./server --host 192.168.61.65 --port 18080 --file-path 3_1/llama_3.1_8b_q4_0.gguf --vocab-path 3_1/Llama-3.1-Instruct-vocab.gguf --qnn-path ./libdir32_i --config-path 3_1/model_config.json"
# export LD_LIBRARY_PATH=/vendor/lib64 && sudo ./run --file-path 3_1/llama_3.1_8b_q4_0.gguf --vocab-path 3_1/Llama-3.1-Instruct-vocab.gguf --qnn-path ./libdir32_i --config-path 3_1/model_config.json --prompt "<|start_header_id|>user<|end_header_id|>HThe following is an example of multiple choice question on abstract algebra:\n\nFind all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\nWe will give one another question on abstract algebra to you. You can think carefully and step by step, but do not offer any explanation. Please answer the following question as the same format in the previous example:\n\nFind the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>Answer:" --steps 200
# export LD_LIBRARY_PATH=/vendor/lib64 && sudo ./run --file-path 3_1/llama_3.1_8b_q4_0.gguf --vocab-path 3_1/Llama-3.1-Instruct-vocab.gguf --config-path 3_1/model_config.json --prompt "The following is an example of multiple choice question on abstract algebra:\n\nFind all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\nWe will give one another question on abstract algebra to you. You can think carefully and step by step, but do not offer any explanation. Please answer the following question as the same format in the previous example:\n\nFind the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer:" --steps 200
# ./build/bin/run --file-path ~/Downloads/Meta-Llama-3.1-8B-Instruct/llama3.1-q4_0.gguf   --vocab-path ~/Downloads/Meta-Llama-3.1-8B-Instruct/llama3.1-vocab.gguf  --config-path ../models/Meta-Llama-3.1-8B-Instruct/llama3.1.json --steps 16 --prompt "<|start_header_id|>user<|end_header_id|>The following is an example of multiple choice question on abstract algebra:\n\nFind all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\nWe will give one another question on abstract algebra to you. You can think carefully and step by step, but do not offer any explanation. Please answer the following question as the same format in the previous example:\n\nFind the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\nA. 0\nB. 4\nC. 2\nD. 6\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>Answer:"
