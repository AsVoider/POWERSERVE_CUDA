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

# how to run `server`
- request:
```
apt install libboost-all-dev libasio-dev
```