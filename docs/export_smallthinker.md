## 导出3B模型

```bash
rm -r /data/qnn_converter

pushd tools/qnn_converter

time python converter.py \
    --model-folder /data/smallthinker_3b_20241220 \
    --model-name smallthinker_3b \
    --system-prompt-file ../../assets/system_prompts/qwen2.txt \
    --prompt-file ../../assets/calibration_data/strawberry_qwen2.txt \
    --batch-size 16 128 \
    --artifact-name smallthinker_3b \
    --n-model-chunks 2 \
    --max-n-tokens 896 \
    --n-threads 8 \
    --soc 8gen4 \
    --build-folder /data/qnn_converter

rm -rf smallthinker_3b_qnn
mv output smallthinker_3b_qnn
popd
python ./tools/gguf_export.py \
    --model-id smallthinker_3b \
    -m /data/smallthinker_3b_20241220 \
    --qnn-path tools/qnn_converter/smallthinker_3b_qnn \
    -o /data/smallthinker_3b
```

## 导出0.5B模型

```bash
rm -r /data/qnn_converter

pushd tools/qnn_converter

time python converter.py \
    --model-folder /data/smallthinker_500m_20241222 \
    --model-name smallthinker_500m \
    --system-prompt-file ../../assets/system_prompts/qwen2.txt \
    --prompt-file ../../assets/calibration_data/strawberry_qwen2.txt \
    --batch-size 1 128 \
    --artifact-name smallthinker_500m \
    --n-model-chunks 1 \
    --max-n-tokens 896 \
    --n-threads 8 \
    --soc 8gen4 \
    --build-folder /data/qnn_converter

rm -rf smallthinker_500m_qnn
mv output smallthinker_500m_qnn
popd
python ./tools/gguf_export.py \
    --model-id smallthinker_500m \
    -m /data/smallthinker_500m_20241222 \
    --qnn-path tools/qnn_converter/smallthinker_500m_qnn \
    -o /data/smallthinker_500m
```

## 编译项目

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-35 \
    -DSMART_WITH_QNN=ON
    # -DSMART_ENABLE_ASAN=OFF \
    # -DSMART_ENABLE_UBSAN=ON
    # -DSMART_WITH_PERFETTO=ON
time cmake --build build
```

## 组装模型

单独运行3B模型：

```bash
./smartserving create \
    -m /data/smallthinker_3b \
    -o /data/smallthinker \
    --exe-path build/out
```

单独运行0.5B模型：

```bash
./smartserving create \
    -m /data/smallthinker_500m \
    -o /data/smallthinker \
    --exe-path build/out
```

投机推理：

```bash
./smartserving create \
    -m /data/smallthinker_3b \
    -d /data/smallthinker_500m \
    -o /data/smallthinker \
    --exe-path build/out
```

## 运行

上传模型到手机：

```bash
rsync -avzP ipads:/ssd/smallthinker/ ~/Downloads/smallthinker/
rsync -avzP ~/Downloads/smallthinker/ 8gen4:~/smallthinker/
rsync -avzP assets/prompts/*.txt 8gen4:~/
```

```bash
export ASAN_OPTIONS=abort_on_error=1
export UBSAN_OPTIONS=print_stacktrace=1
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64

sudo cp smallthinker/bin/smart-{run,speculative} .

# 不开投机推理
sudo ./smart-run --work-folder smallthinker --prompt-file strawberry_qwen2.txt -n 1536

# 开投机推理
sudo ./smart-speculative --work-folder smallthinker --prompt-file strawberry_qwen2.txt -n 1536
```
