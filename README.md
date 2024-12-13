# SmartServing

[项目日志](https://ipads.se.sjtu.edu.cn:1312/smartserving/smartserving/-/wikis/%E9%A1%B9%E7%9B%AE%E6%97%A5%E5%BF%97)

```bash
pip install -r requirements.txt
git submodule update --init --recursive
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
- Edit in 2024.12.04

### Generate config file
```
./build/tools/gguf_config_to_json/config_generator --file-path ./qwen2-q4_0.gguf  --target-path ./qwen2.config
```

### Run for cpu
```
./build/bin/run --file-path ./llama3.1-q4_0.gguf  --vocab-path ./llama3.1-vocab.gguf --config-path ./llama3.1.json --steps 4 --prompt 'One day,'
```

### Run for Andorid qnn
- Prepare qnn models and shared libraries in workspace directory, includes:
```
config.json  llama3_1_8b_0.bin  llama3_1_8b_2.bin  llama3_1_8b_4.bin  llama3_1_8b_6.bin  lm_head.bin
kv           llama3_1_8b_1.bin  llama3_1_8b_3.bin  llama3_1_8b_5.bin  llama3_1_8b_7.bin
libQnnHtp.so  libQnnHtpV75.so  libQnnHtpV75Skel.so  libQnnHtpV75Stub.so  libQnnSystem.so
```
```
./build/bin/run --file-path ./llama3.1-q4_0.gguf  --vocab-path ./llama3.1-vocab.gguf --config-path .llama3.1.json --steps 4 --prompt 'One day,' --qnn-path ./workspace
```

### Run for mmlu test
- Run server
```
# cpu
./build/bin/server --file-path ./llama3.1-q4_0.gguf  --vocab-path ./llama3.1-vocab.gguf  --config-path ./llama3.1.json --host 0.0.0.0 --port 18080
# qnn
export LD_LIBRARY_PATH=/vendor/lib64 && sudo -E ./build/bin/server --file-path ./llama3.1-q4_0.gguf  --vocab-path ./llama3.1-vocab.gguf  --config-path ./llama3.1.json --host 0.0.0.0 --port 18080 --qnn-path ./workspace
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
./build/tools/perpelxity/perpelxity_test --file-path ./llama3.1-q4_0.gguf --vocab-path ./llama3.1-vocab.gguf --config-path ./llama3.1.json --prompt-file ./prompt.txt --batch-size 32 --n-threads 12
# npu
export LD_LIBRARY_PATH=/vendor/lib64 && sudo -E ./build/tools/perpelxity/perpelxity_test --file-path ./llama3.1-q4_0.gguf --vocab-path ./llama3.1-vocab.gguf --config-path ./llama3.1.json --prompt-file ./prompt.txt --batch-size 32 --qnn-path ./workspace
```

# cmdline-tools
- generate config (support gguf, safetensors)
```
smartserving create -m ./models/ -o ./proj.config
```
- generate qnn models
```
smartserving create -m ./models/ --qnn-out ./qnn_models/ -o ./proj.config
```
[optional] Modify Config by manual or cli
- run with config
```
smartserving run -c ./proj.config
```
- run server with config
```
smartserving server -c ./proj.config
```