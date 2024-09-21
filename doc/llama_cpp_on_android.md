# 在Android上运行llama.cpp

首先clone llama.cpp的仓库：

```bash
git clone https://github.com/ggerganov/llama.cpp.git
```

## 准备NDK的Docker/Podman镜像

也可以直接把NDK装到host系统上。这里用Docker/Podman是为了确保在各种发行版上都能跑。另外，因为DockerHub被墙了，所以这里用可以挂代理的Podman来pull Ubuntu镜像。

`./llama.cpp`换成llama.cpp的仓库目录，`~/Downloads`可以换成别的目录，之后NDK要下载到这个目录里。

```bash
export https_proxy=http://ipads:ipads123@202.120.40.82:11235
sudo -E podman run -v $(realpath ./llama.cpp):/code -v $(realpath ~/Downloads):/data -dit --name ndk ubuntu:22.04
```

进入镜像：

```bash
sudo podman exec -it ndk bash
```

准备环境：

```bash
apt update
apt upgrade
apt install sudo vim cmake unzip git python-is-python3 python3-pip build-essential
pip install -r /code/requirements/requirements-convert_hf_to_gguf.txt
git config --global --add safe.directory /code
```

在<https://developer.android.com/ndk/downloads>下载NDK。我下载的是`android-ndk-r27b-linux.zip`。压缩包放到`~/Downloads`下，然后在容器内用`unzip`命令解压：

```bash
unzip /data/android-ndk-r27b-linux.zip -d /data
```

使用Vim编辑容器内的`~/.bashrc`，在文件末尾添加如下内容：

```bash
export NDK=/data/android-ndk-r27b
```

重新加载`.bashrc`：

```bash
source ~/.bashrc
```

## 编译llama.cpp

```bash
cd /code
cmake \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-34 \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_OPENMP=OFF \
    -S . -B build_android
cmake --build build_android --config RelWithDebInfo --target llama-cli
```

## 生成GGUF模型

先前往<https://huggingface.co/meta-llama/Meta-Llama-3.1-8B>申请访问权限。切记国籍不能填China。

在容器内下载Llama3.1的HuggingFace权重：

```bash
pip install "huggingface_hub[cli]"
huggingface-cli login
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --cache-dir /data/hf_cache --resume-download meta-llama/Meta-Llama-3.1-8B --exclude original/ --local-dir /data/llama_3.1_8b
```

转换并量化模型：

```bash
python convert_hf_to_gguf.py /data/llama_3.1_8b --outtype f16 --outfile /data/llama_3.1_8b.gguf
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build_native
cmake --build build_native --config Release --target llama-quantize
./build_native/bin/llama-quantize --pure /data/llama_3.1_8b.gguf /data/llama_3.1_8b_q4_0.gguf q4_0
```

## 在Android上运行

先将`./llama.cpp/build_android/bin/llama-cli`和`~/Downloads/llama_3.1_8b_q4_0.gguf`推到手机上。

- 如果用`adb shell`运行，则用`adb push`推文件。
- 如果用Termux内的shell运行，则用`ssh`连接，用`scp`/`rsync`来推文件。

运行llama.cpp：

```bash
./llama-cli --no-warmup -c 1024 -m llama_3.1_8b_q4_0.gguf -s 42 -p "Penguins are superbly adapted to aquatic life." -t 6 -n 128
```

参考输出：

```
Penguins are superbly adapted to aquatic life. Their solid, streamlined bodies, supported by fully waterproof feathers, help them glide gracefully through the water. They have flipper-like wings, webbed feet, and a wedge-shaped tail that helps them swim and steer. Their feathers also protect them from the cold of the Arctic and Antarctic.
Penguins feed mainly on fish. They have sharp eyesight and can see prey swimming under the water surface. Their method of hunting is similar to that of a seal: the penguin swims underwater toward its prey, takes it in its bill, and swallows it head-first. Penguins are gregarious animals. They live in colonies and are very social

llama_perf_sampler_print:    sampling time =      12.19 ms /   139 runs   (    0.09 ms per token, 11406.53 tokens per second)
llama_perf_context_print:        load time =    1845.01 ms
llama_perf_context_print: prompt eval time =    1327.69 ms /    11 tokens (  120.70 ms per token,     8.29 tokens per second)
llama_perf_context_print:        eval time =   22479.08 ms /   127 runs   (  177.00 ms per token,     5.65 tokens per second)
llama_perf_context_print:       total time =   24355.12 ms /   138 tokens
```
