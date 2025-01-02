# PowerServe

## Support models
| model    | CPU      | NPU      |Note      |
|----------|----------|----------|----------|
| LLaMA 3.1 🦙🦙🦙  | ✔️    | ✔️    |    |
| LLaMA 3.2 🦙🦙🦙  | ✔️    | ✔️    |   |
| Qwen2  | ✔️[Need test]    | ✔️    |    |
| Phi3  | ✔️[Need test]    |     |    |
| InternVL2 (1B, 2B, 8B) |     |✔️    |    |


## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Model Preparation](#model-preparation)
3. [Compile PowerServe](#compile-powerserve)
4. [Prepare PowerServe Workspace](#prepare-powerserve-workspace)
5. [Execution](#execution)
6. [Known Issues](#known-issues)


## Prerequisites

```bash
pip install -r requirements.txt
git submodule update --init --recursive
```

To deploy on aarch64 with Qualcomm NPU using QNN, [**NDK**](https://developer.android.google.cn/ndk/downloads) and [**QNN**](https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/linux_setup.html) are required to be installed.

```shell
export NDK=<path-to-ndk>
export QNN_SDK_ROOT=<path-to-QNN>
```


## Model Preparation

For CPU-only execution, only `Models For CPU` is required. For NPU execution, both `Models For CPU` and `Models For NPU` is required.

Take llama3.1-8b-instruct model as example, the structure of model folder:
```shell
-- models                       # Level-1 dir, where server search different models.
    -- llama3.1-8b-instruct         # Level-2 dir, where CLI search for runtime configurations
        -- hparams.json                 # Hyper params, containing #threads, #batch_size and sampler configurations.
        -- workspace.json               # The defintion of model workspace structure, where main model and target model(if exist) is determined.
        -- bin                          # The binaries for execution
            -- smart-config-generator
            -- smart-perplexity-test
            -- smart-run
            -- smart-server
        -- model_dir                    # The model weights of GGUF and QNN
            -- model.json               #
            -- vocab.gguf               # The vocab table of model
            -- ggml                     # GGUF model binaries
                -- weights.gguf
            -- qnn                      # QNN model bianries
                -- kv
                    -- *.raw
                    -- ...
                -- config.json          # The information of QNN models and QNN backend configurations
                -- llama3_1_8b_0.bin
                -- llama3_1_8b_1.bin
                -- llama3_1_8b_2.bin
                -- llama3_1_8b_3.bin
                -- lmhead.bin
        -- qnn_libs                     # Dependent libraries of QNN
            -- libQNNSystem.so
            -- libQNNHtp.so
            -- libQNNHtpV79.so
            -- libQNNHtpV79Skel.so
            -- libQNNHtpV79Stub.so

    -- qwen2_7b_instruct            # Level-2 dir of another model
        -- ...

```

### Convert Models For CPU

```shell
# Under the root directory of PowerServe
python ./tools/gguf_export.py -m <hf-model> -o models/llama3.1-8b-instruct
```


### Convert Models For NPU

If you just want to run PowerServe on CPUs, this step can be skipped. More details please refer to [QNN Model Conversion](./tools/qnn_converter/README.md)

```shell
# Under the root directory of PowerServe
cd smartserving/tools/qnn_converter

# This may take a long time...
python converter.py                                 \
    --model-folder Llama-3.1-8B-Instruct            \
    --model-name llama3_1_8b                        \
    --system-prompt-file system_prompt_llama.txt    \
    --prompt-file lab_intro_llama.md                \
    --batch-sizes 1 128                             \
    --artifact-name llama3_1_8b                     \
    --n-model-chunk 4                               \
    --output-folder ./llama3.1-8b-QNN               \
    --build-folder ./llama3.1-8b-QNN-tmp            \
    --soc 8gen4

```
Convert GGUF models and integrate them with QNN models

```shell
# Under the root directory of PowerServe
python ./tools/gguf_export.py -m <hf-model> --qnn-path tools/qnn_converter/llama3.1-8b-QNN -o ./llama3.1-8b-instruct-model
```

## Compile PowerServe

The options of platform and ABI vary when deploying on different devices. DO CARE about the configuration.

### Build for Linux cpu
```shell
# Under the root directory of PowerServe
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Build for Andorid cpu
```shell
# Under the root directory of PowerServe
cmake -B build                                                      \
    -DCMAKE_BUILD_TYPE=Release                                      \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a                                         \
    -DANDROID_PLATFORM=android-35                                   \
    -DGGML_OPENMP=OFF                                               \
    -DSMART_WITH_QNN=OFF

cmake --build build
```

### Build for Andorid qnn
```shell
# Under the root directory of PowerServe
cmake -B build                                                      \
    -DCMAKE_BUILD_TYPE=Release                                      \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a                                         \
    -DANDROID_PLATFORM=android-35                                   \
    -DGGML_OPENMP=OFF                                               \
    -DSMART_WITH_QNN=ON

cmake --build build
```


## Prepare PowerServe Workspace

```shell
# Under the root directory of PowerServe
mkdir -p models

# Generate PowerServe Workspace
./smartserving create -m ./llama3.1-8b-instruct-model --exe-path ./build -o ./models/llama3.1-8b-instruct
```

## Execution

### CLI
More details please refer to [CLI App](./app/run/README.md)

For pure CPU execution
```shell
# Under the root directory of PowerServe
./models/llama3.1-8b-instruct/bin/smart-run --work-folder ./models/llama3.1-8b-instruct --prompt "Once upon a time, there was a little girl named Lucy" --no-qnn
```
For NPU execution
```shell
# Under the root directory of PowerServe
export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64 && ./models/llama3.1-8b-instruct/bin/smart-run --work-folder ./models/llama3.1-8b-instruct --prompt "Once upon a time, there was a little girl named Lucy"
```

### Server
More details please refer to [Server App](./app/server/README.md)
```shell
# Under the root directory of PowerServe
./models/llama3.1-8b-instruct/bin/smart-server --model-folder ./models --host <ip-addr> --port <port>
```


## Performance
- QNN: 8gen3 phone + n_predicts = 256 + n_prompts = 1652
- CPU: n_threads = 8 + n_predicts = 128 + n_prompts = 95

| model    | CPU(Prefill / Decode)     | NPU(Prefill / Decode)      |Note      |
|----------|----------|----------|----------|
| LLaMA 3.1-8b-q4_0  | 19.88 / 6.75 tokens/s    | 559.81 / 11.21 tokens/s    |  |
| LLaMA 3.2-1b-q4_0  | 127.86 / 38.04 tokens/s   | 1764.96 / 57.53 tokens/s    |  |
| Qwen2  | TODO   | TODO    |  |


## Known Issues

### Model Conversion

1. **When exporting model to onnx**: RuntimeError: The serialized model is larger than the 2GiB limit imposed by the protobuf library. Therefore the output file must be a file path, so that the ONNX external data can be written to the same directory. Please specify the output file name.

    > The version of pytorch should be less than **2.5.1**. Please reinstall pytorch like:
    > ```shell
    > pip install pytorch==2.4.1
    > ```

### Execution

1. **When inferencing with QNN**: Failed to open lib /vendor/lib64/libcdsprpc.so: dlopen failed: library "/vendor/lib64/libcdsprpc.so" needed or dlopened by "/data/data/com.termux/files/home/workspace/qnn/llama-3.2-1b-instruct/bin/smart-run" is not accessible for the namespace "(default)

    > Use `export LD_LIBRARY_PATH=/system/lib64:/vendor/lib64` before executing the program.
    >
    > Because `libcdsprpc.so` depends on `/system/lib64/libbinder.so` instead of `/vendor/lib64/libbinder.so`. If the linker searches the `/vendor/lib64` at first, it may find and links `/vendor/lib64/libbinder.so` which does not contain corresponding function definitions.
