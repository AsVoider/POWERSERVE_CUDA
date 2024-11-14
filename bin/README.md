# `llama_tokenize.cpp`

首先准备vocab数据：

```bash
python tools/generate_llama_vocab.py /data/llama_3.1_8b -o /data/llama_3.1_8b_vocab.gguf
```

编译：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo --target llama_tokenize
```

运行：

```bash
./build/bin/llama_tokenize --vocab-path ~/Downloads/llama_3.1_8b_vocab.gguf --text "Penguins are superbly adapted to aquatic life."
```

参考输出：

```
/home/riteme/Downloads/llama_3.1_8b_vocab.gguf
#vocab: 128256
BOS token: 128000
[128000, 47, 56458, 527, 33689, 398, 30464, 311, 72491, 2324, 13]
<|begin_of_text|>Penguins are superbly adapted to aquatic life.
```

# how to run `run_qnn`
1. set the right values in config.json and push it to the device
2. push the necessary qnn libs and model context binary files to the device
3. run: sudo ./run_qnn --file-path path/to/model_gguf_file --vocab-path path/to/vocab_gguf_file --qnn_path path/to/qnn_folder --prompt "Once upon a time" --steps 64