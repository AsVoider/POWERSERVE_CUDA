# SmartServing

## Support models
| model    | CPU      | NPU      |Note      |
|----------|----------|----------|----------|
| LLaMA 3.1 🦙🦙🦙  | ✔️    | ✔️    |    |
| LLaMA 3.2 🦙🦙🦙  | ✔️    | ✔️    |   |
| Qwen2  | ✔️    | ✔️    |    |
| Phi3  | ✔️    |     |    |
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
    -S . -B build_android
cmake --build build_android
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
./build/tools/gguf_config_to_json/config_generator --file-path ./qwen2-q4_0.gguf  --target-path ./qwen2.config
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

# cmdline-tools
- export gguf models
```
python ./tools/gguf_export.py -m /path/to/hf_model -o ./gguf-out [--qnn-path <qnn_path>]
```

- generate work-space
```
smartserving create -m ./models/ -o ./proj --exe-path ./gguf-out/bin/<target-arch[x86_64, aarch64]>
```

[optional] Modify Config by manual or cli
- run with config
```
smartserving run -d ./proj
```
- run server with config
```
smartserving server -d ./proj
```
