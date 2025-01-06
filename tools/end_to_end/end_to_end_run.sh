#!/bin/bash

# 如果不是x86-64架构，则退出
if [ "$(uname -m)" != "x86_64" ]; then
    echo "This script is only supported on x86-64 architecture"
    exit 1
fi

echo "Installing necessary packages"
apt-get update
apt-get install -y wget unzip cmake git git-lfs python3

echo "Downloading models from huggingface"
# 把这个模型下载到/models目录下
mkdir -p /models
cd /models
# Now it's private, so need to login first
# After it's public, we can use the following command without login
echo "Now it's private, so need to login first. After it's public, we can git clone without login"
echo "You may have to wait for a while, about 1 to 10 minutes according to your network speed"

# 如果已经下载过，先删掉目录再下载
if [ -d "/models/Llama-3.2-1B-PowerServe-QNN" ]; then
    rm -rf /models/Llama-3.2-1B-PowerServe-QNN
fi
git clone https://huggingface.co/PowerInfer/Llama-3.2-1B-PowerServe-QNN

# 先检测是否有NDK环境变量
echo "Setting up NDK environment variable"
if [ -z "$ANDROID_NDK" ]; then
    echo "NDK not found"
else
    echo "NDK found at $ANDROID_NDK"
fi

bash /lib/qnn/unzip_qnn.sh
source /qnn/bin/envsetup.sh

# 寻找QNN_SDK_ROOT
if [ -z "$QNN_SDK_ROOT" ]; then
    echo "QNN_SDK_ROOT not found"
    exit 1
else
    echo "QNN_SDK_ROOT found at $QNN_SDK_ROOT"
fi

cd /code

echo "Creating build directory for Android"
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DPOWERSERVE_WITH_QNN=ON -S . -B build_android

# 构建项目
echo "Building project for Android"
cmake --build build_android --config RelWithDebInfo --parallel 12 --target all

./powerserve create -m /models/Llama-3.2-1B-PowerServe-QNN --exe-path /code/build_android/out

rm -rf /qnn
