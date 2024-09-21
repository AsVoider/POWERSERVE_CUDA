# `src/main.cpp`

`thirdparty`目录下包含了完整的ggml库和llama.cpp的tokenizer代码。

首先需要准备vocab数据。这里复用llama.cpp的`convert_hf_to_gguf.py`：

```bash
python convert_hf_to_gguf.py /data/llama_3.1_8b --outtype f16 --vocab-only --outfile /data/llama_3.1_8b_vocab.gguf
```

编译：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo --target main
```

运行：

```bash
./build/src/main --vocab-path ~/Downloads/llama_3.1_8b_vocab.gguf --text "Penguins are superbly adapted to aquatic life."
```

参考输出：

```
/home/riteme/Downloads/llama_3.1_8b_vocab.gguf
#vocab: 128256
BOS token: 128000
[128000, 47, 56458, 527, 33689, 398, 30464, 311, 72491, 2324, 13]
<|begin_of_text|>Penguins are superbly adapted to aquatic life.
```
