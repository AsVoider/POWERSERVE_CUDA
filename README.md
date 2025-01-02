# SmartServing

## Support models
| model    | CPU      | NPU      |Note      |
|----------|----------|----------|----------|
| LLaMA 3.1 🦙🦙🦙  | ✔️    | ✔️    |    |
| LLaMA 3.2 🦙🦙🦙  | ✔️    | ✔️    |   |
| Qwen2  | ✔️[Need test]    | ✔️    |    |
| Phi3  | ✔️[Need test]    |     |    |
| InternVL2 (1B, 2B, 8B) |     |✔️    |    |

## Prerequisites

```bash
pip install -r requirements.txt
git submodule update --init --recursive
```

## Build

###  x86

```bash
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build -S .
cmake --build build
```

### Android aarch64 with QNN

```bash
cmake \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-34 \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DSMART_WITH_QNN=ON \
    -S . -B build
cmake --build build
```

## Build
- Edit in 2024.12.04

### Build for Linux cpu
```
cmake -B build -S ./ -D CMAKE_EXPORT_COMPILE_COMMANDS=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j12
```

### Build for Andorid cpu
```
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DSMART_WITH_QNN=OFF -S . -B build_android
cmake --build build_android -j12
```

### Build for Andorid qnn
```
cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DSMART_WITH_QNN=ON -S . -B build_android
cmake --build build_android -j12
```

## Run
- Edit in 2024.12.25

### Generate config file
```
./build/tools/gguf_config_to_json/config-generator --file-path ./qwen2-q4_0.gguf  --target-path ./qwen2.config
```

### Run for cpu
```
./build/bin/run --work-folder /path/to/work/folder --prompt 'One day,' --no-qnn
```

### Run for Andorid qnn
- Prepare qnn models and shared libraries in workspace directory, includes:
```
config.json  llama3_1_8b_0.bin  llama3_1_8b_2.bin  llama3_1_8b_4.bin  llama3_1_8b_6.bin  lm_head.bin
kv           llama3_1_8b_1.bin  llama3_1_8b_3.bin  llama3_1_8b_5.bin  llama3_1_8b_7.bin
libQnnHtp.so  libQnnHtpV75.so  libQnnHtpV75Skel.so  libQnnHtpV75Stub.so  libQnnSystem.so
```
```
./build/bin/run --work-folder /path/to/work/folder --prompt 'One day,'
```

### Run for mmlu test
- Run server
```
# cpu
./build/bin/server --work-folder /path/to/work/folder --host 0.0.0.0 --port 18080 --no-qnn
# qnn
export LD_LIBRARY_PATH=/vendor/lib64 && sudo -E ./build/bin/server --work-folder /path/to/work/folder --host 0.0.0.0 --port 18080
```
- Run client
```
cd ./tools/mmlu
pip install requests pandas
python ./mmlu_test.py --host 0.0.0.0 --port 18080 -s 1
```

### Run for ppl
```
# cpu
./build/tools/perpelxity/perpelxity_test --work-folder /path/to/work/folder --batch-size 32 --no-qnn
# npu
export LD_LIBRARY_PATH=/vendor/lib64 && sudo -E ./build/tools/perpelxity/perpelxity_test --work-folder /path/to/work/folder --batch-size 32
```


# 准备环境和运行【CPU】
- 准备qnn models（batch-sizes最多指定两个值）：
```
cd smartserving/tools/qnn_converter

python converter.py \
    --model-folder Llama-3.2-1B-Instruct \
    --model-name llama3_2_1b \
    --system-prompt-file system_prompt_llama.txt \ 
    --prompt-file lab_intro_llama.md \
    --batch-sizes 1 128 \
    --artifact-name llama3_2_1b \
    --soc 8gen3
```
- 准备gguf models
```
cd smartserving
python ./tools/gguf_export.py -m hf_model --qnn-path tools/qnn_converter/output -o ./model_dir
```
- 准备workspace + 运行程序【cpu】
```
cd smartserving
./smartserving create -m ./model_dir/  --exe-path /home/zwb/SS/smartserving/build_x86_64/out -o proj
./proj/bin/smart-run -d ./proj --no-qnn
```
- 准备workspace + 运行程序【qnn】
```
cd smartserving
./smartserving create -m ./model_dir/  --exe-path /home/zwb/SS/smartserving/build_aarch64/out -o proj
# 将proj传输到qnn运行设备上
export LD_LIBRARY_PATH=/vendor/lib64 && sudo -E ./proj/bin/smart-run -d ./proj
```

# Performance
- QNN: 8gen3 phone + n_predicts = 256 + n_prompts = 1652
- CPU: n_threads = 8 + n_predicts = 128 + n_prompts = 95

| model    | CPU(Prefill / Decode)     | NPU(Prefill / Decode)      |Note      |
|----------|----------|----------|----------|
| LLaMA 3.1-8b-q4_0  | 19.88 / 6.75 tokens/s    | 559.81 / 11.21 tokens/s    |  |
| LLaMA 3.2-1b-q4_0  | 127.86 / 38.04 tokens/s   | 1764.96 / 57.53 tokens/s    |  |
| Qwen2  | TODO   | TODO    |  |


# Known Issues

## Model Conversion

1. **When exporting model to onnx**: RuntimeError: The serialized model is larger than the 2GiB limit imposed by the protobuf library. Therefore the output file must be a file path, so that the ONNX external data can be written to the same directory. Please specify the output file name.

    > The version of pytorch should be less than **2.5.1**. Please reinstall pytorch like:
    > ```shell
    > pip install pytorch==2.4.1
    > ```
    
## Execution

1. **When inferencing with QNN**: Failed to open lib /vendor/lib64/libcdsprpc.so: dlopen failed: library "/vendor/lib64/libcdsprpc.so" needed or dlopened by "/data/data/com.termux/files/home/workspace/qnn/llama-3.2-1b-instruct/bin/smart-run" is not accessible for the namespace "(default)

    > Use `export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64` before executing the program.
    >
    > Because `libcdsprpc.so` depends on `/system/lib64/libbinder.so` instead of `/vendor/lib64/libbinder.so`. If the linker searches and links the `/vendor/lib64` at first, it may find `/vendor/lib64/libbinder.so` which does not contain corresponding function definitions.
