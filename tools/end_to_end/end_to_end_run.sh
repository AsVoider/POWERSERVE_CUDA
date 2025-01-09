#!/bin/bash

function clean() {
    set +e
    rm -rf /qnn
}

set -e
trap clean EXIT

echo "Please choose your SoC chip"
echo "1. Qualcomm Snapdragon 8Gen3"
echo "2. Qualcomm Snapdragon 8Gen4 (also known as Snapdragon 8 elite)"
read -p "Enter your choice(only a number): " choice

soc_name=""
if [ $choice -eq 1 ]; then
    soc_name="8G3"
elif [ $choice -eq 2 ]; then
    soc_name="8G4"
else
    echo "Other chips are not supported yet, please wait for the update"
    exit 1
fi

echo "Do you want to enable speculation? (yes/no)"
read -p "Enter your choice(only yes or no): " speculation_enabled

model=""
if [ "$speculation_enabled" == "yes" ]; then
    echo "Which speculation model do you need?"
    echo "1. Llama-3.1 8B (use Llama-3.2 1B as draft model)"
    echo "2. SmallThinker-3B (use SmallThinker-0.5B as draft model)"
    read -p "Enter your choice(only a number): " main_model_choice
    case $main_model_choice in
        1)
            model1="Llama-3.1-8B-PowerServe-Speculate-QNN29-${soc_name}"
            model2="Llama-3.2-1B-PowerServe-QNN29-${soc_name}"
            ;;
        2)
            model1="SmallThinker-3B-PowerServe-Speculate-QNN29-${soc_name}"
            model2="SmallThinker-0.5B-PowerServe-QNN29-${soc_name}"
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
else
    echo "Which model parameters do you need?"
    echo "1. Llama-3.2 1B"
    echo "2. Llama-3.1 8B"
    echo "3. SmallThinker 0.5B"
    echo "4. SmallThinker 3B"
    read -p "Enter your choice(only a number): " param_choice

    case $param_choice in
        1)
            model="Llama-3.2-1B-PowerServe-QNN29-${soc_name}"
            ;;
        2)
            model="Llama-3.1-8B-PowerServe-QNN29-${soc_name}"
            ;;
        3)
            model="SmallThinker-0.5B-PowerServe-QNN29-${soc_name}"
            ;;
        4)
            model="SmallThinker-3B-PowerServe-QNN29-${soc_name}"
            ;;
        *)
            echo "Invalid choice"
            exit 1
            ;;
    esac
fi

echo "Debug checking: soc_name=$soc_name, speculation_enabled=$speculation_enabled, model1=$model1, model2=$model2, model=$model"

# Check if the model exists
# TODO: after open_sourced, remove this check.
OPEN_SOURCE="NO"

if [ "$OPEN_SOURCE" == "YES" ]; then
    if [ "$speculation_enabled" == "yes" ]; then
        for model in "$model1" "$model2"; do
            link="https://huggingface.co/PowerInfer/${model}"
            if ! curl --output /dev/null --silent --head --fail "$link"; then
                echo "Model $model does not exist"
                exit 1
            fi
        done
    else
        link="https://huggingface.co/PowerInfer/${model}"
        if ! curl --output /dev/null --silent --head --fail "$link"; then
            echo "Model $model does not exist"
            exit 1
        fi
    fi
fi

echo "Downloading models from huggingface"

mkdir -p /models
cd /models

echo "Now we are downloading the models from huggingface"
echo "You may have to wait for a while, about 1 to 10 minutes according to your network speed"

if [ "$speculation_enabled" == "yes" ]; then
    for model in "$model1" "$model2"; do
        echo "Downloading model $model"
        if [ -d "/models/${model}" ]; then
            rm -rf "/models/${model}"
        fi
        git clone "https://huggingface.co/PowerInfer/${model}"
    done
else
    echo "Downloading model $model"
    if [ -d "/models/${model}" ]; then
        rm -rf "/models/${model}"
    fi
    git clone "https://huggingface.co/PowerInfer/${model}"
fi

echo "Setting up NDK environment variable"
if [ -z "$ANDROID_NDK" ]; then
    echo "NDK not found"
    exit 1
else
    echo "NDK found at $ANDROID_NDK"
fi

bash /lib/qnn/unzip_qnn.sh
source /qnn/bin/envsetup.sh

if [ -z "$QNN_SDK_ROOT" ]; then
    echo "QNN_SDK_ROOT not found"
    exit 1
else
    echo "QNN_SDK_ROOT found at $QNN_SDK_ROOT"
fi

cd /code

echo "Creating build directory for Android"
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DPOWERSERVE_WITH_QNN=ON -S . -B build_android

echo "Building project for Android"
cmake --build build_android --config RelWithDebInfo --parallel 12 --target all

if [ "$speculation_enabled" == "yes" ]; then
    ./powerserve create -m "/models/${model1}" -d "/models/${model2}" --exe-path /code/build_android/out
else
    ./powerserve create -m "/models/${model}" --exe-path /code/build_android/out
fi
