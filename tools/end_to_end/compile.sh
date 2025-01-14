#!/bin/bash

function clean() {
    set +e
    rm -rf /qnn
}

set -e
trap clean EXIT

# Default values
prompt="In recent years, the landscape of artificial intelligence has been significantly transformed by the advent of large language models (LLMs). Traditionally, these models have been deployed in cloud environments due to their computational demands. However, the emergence of on-edge LLMs is reshaping how AI can be utilized at the edge of networks, offering numerous advantages in terms of latency, privacy, and accessibility."
speculation_enabled="no"
cpu_only="no"
model_name=""
soc_name=$(cat tmpfile)

supported_model_list=("smallthinker-3b" "llama-3.1-8b" "llama-3.2-1b")
supported_speculation_model_list=("smallthinker-3b" "llama-3.1-8b")
model_repo_name_list=("SmallThinker-3B-PowerServe-QNN29" "SmallThinker-0.5B-PowerServe-QNN29" "Llama-3.1-8B-PowerServe-QNN29" "Llama-3.2-1B-PowerServe-QNN29")

# Parse command-line options
while getopts ":p:sc" opt; do
    case ${opt} in
        p )
            prompt=$OPTARG
            ;;
        s )
            speculation_enabled="yes"
            ;;
        c )
            cpu_only="yes"
            ;;
        \? )
            echo -e "\033[31mInvalid option: -$OPTARG\033[0m" 1>&2
            exit 1
            ;;
        : )
            echo -e "\033[31mInvalid option: -$OPTARG requires an argument\033[0m" 1>&2
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

model_name=$1
# echo -e "\033[32mModel name is $model_name\033[0m"
# echo -e "\033[32mSpeculation enabled: $speculation_enabled\033[0m"
# echo -e "\033[32mCPU only: $cpu_only\033[0m"
# echo -e "\033[32mPrompt: $prompt\033[0m"


model=""
# the input is the format in the supported_model_list
# now we convert it to the format in the model_repo_name_list
# if speculation_enabled then we need to set model1 and model2
if [ -z "$model_name" ]; then
    echo -e "\033[31mModel name is required.\033[0m"
    exit 1
fi

if [[ ! " ${supported_model_list[@]} " =~ " ${model_name} " ]]; then
    echo -e "\033[31mModel $model_name is not supported.\033[0m"
    echo -e "\033[31mSupported models: ${supported_model_list[@]}\033[0m"
    exit 1
fi

if [ "$speculation_enabled" == "yes" ] && [[ ! " ${supported_speculation_model_list[@]} " =~ " ${model_name} " ]]; then
    echo -e "\033[31mSpeculation is not supported for model $model_name.\033[0m"
    echo -e "\033[31mSupported speculation models: ${supported_speculation_model_list[@]}\033[0m"
    exit 1
fi

# Convert the model name to the format in the model_repo_name_list
if [ "$model_name" == "smallthinker-3b" ]; then
    model="SmallThinker-3B-PowerServe-QNN29-${soc_name}"
elif [ "$model_name" == "smallthinker-0.5b" ]; then
    model="SmallThinker-0.5B-PowerServe-QNN29-${soc_name}"
elif [ "$model_name" == "llama-3.1-8b" ]; then
    model="Llama-3.1-8B-PowerServe-QNN29-${soc_name}"
elif [ "$model_name" == "llama-3.2-1b" ]; then
    model="Llama-3.2-1B-PowerServe-QNN29-${soc_name}"
fi

if [ "$speculation_enabled" == "yes" ]; then
    model1=$model
    if [ "$model_name" == "smallthinker-3b" ]; then
        model2="SmallThinker-0.5B-PowerServe-QNN29-${soc_name}"
    elif [ "$model_name" == "llama-3.1-8b" ]; then
        model2="Llama-3.2-1B-PowerServe-QNN29-${soc_name}"
    fi
fi

# echo -e "\033[32mModel name is $model\033[0m"
# FORDEBUG
# echo -e "\033[32mModel name is $model\033[0m"
# echo model1 model2
# echo -e "\033[32mModel1 name is $model1\033[0m"
# echo -e "\033[32mModel2 name is $model2\033[0m"
if [ -z "$model2" ]; then
    echo -e "\033[32mModel Repo   : $model\033[0m."
else
    echo -e "\033[32mModel Repo   : Use $model1 and $model2 for speculation.\033[0m"
fi

model_dir="models"

bash /lib/qnn/unzip_qnn.sh
source /qnn/bin/envsetup.sh
ANDROID_NDK=/ndk
echo -e "\033[32mSetting up NDK environment variable\033[0m"
if [ -z "$ANDROID_NDK" ]; then
    echo -e "\033[31mNDK not found\033[0m"
    exit 1
else
    echo -e "\033[32mNDK found at $ANDROID_NDK\033[0m"
fi

if [ -z "$QNN_SDK_ROOT" ]; then
    echo -e "\033[31mQNN_SDK_ROOT not found\033[0m"
    exit 1
else
    echo -e "\033[32mQNN_SDK_ROOT found at $QNN_SDK_ROOT\033[0m"
fi

cd /code

echo -e "\033[32mCreating build directory for Android\033[0m"
cmake -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DPOWERSERVE_WITH_QNN=ON -S . -B build_android

echo -e "\033[32mBuilding project for Android\033[0m"
cmake --build build_android --config RelWithDebInfo --parallel 12 --target all

if [ "$speculation_enabled" == "yes" ]; then
    ./powerserve create -m "${model_dir}/${model1}" -d "${model_dir}/${model2}" --exe-path /code/build_android/out
else
    ./powerserve create -m "${model_dir}/${model}" --exe-path /code/build_android/out
fi
