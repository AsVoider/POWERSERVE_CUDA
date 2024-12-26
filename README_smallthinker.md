## 初始化仓库

```bash
pip install -r requirements.txt
git submodule update --init --recursive
```

## 编译

```bash
cmake \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-34 \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_OPENMP=OFF \
    -DSMART_WITH_QNN=ON \
    -S . -B build_android
cmake --build build_android -j8
```

## 导出3B模型

```bash
python ./tools/convert_hf_to_gguf/convert_hf_to_gguf.py /data/smallthinker_3b_20241220 --outtype q8_0 --outfile /data/smallthinker_3b_q8_0.gguf
python ./tools/convert_hf_to_gguf/convert_hf_to_gguf.py /data/smallthinker_3b_20241220 --outtype q8_0 --vocab-only --outfile /data/smallthinker_3b_q8_0_vocab.gguf

cd tools/qnn_converter
python converter.py \
    --model-folder /data/smallthinker_3b_20241220 \
    --model-name smallthinker_3b \
    --system-prompt-file qwen_system_prompt.txt \
    --prompt-file lab_intro_llama.md \
    --batch-size 1 128 \
    --artifact-name smallthinker_3b \
    --n-model-chunks 2 \
    --max-n-tokens 1024 \
    --n-threads 8 \
    --soc 8gen4

rm -rf smallthinker_3b
mkdir smallthinker_3b
mv output smallthinker_3b/qnn-workspace

cp /data/smallthinker_3b_q8_0.gguf smallthinker_3b/weights.gguf
cp /data/smallthinker_3b_q8_0_vocab.gguf smallthinker_3b/vocab.gguf
```
