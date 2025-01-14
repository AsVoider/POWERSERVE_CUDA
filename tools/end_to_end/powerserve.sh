#!/bin/bash

clean() {
    set +e
    rm -rf tmpfile
}

set -e
trap clean EXIT

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "\033[31m$1 could not be found. Please install it.\033[0m"
        exit 1
    fi
}

# Check necessary commands
check_command "docker"

# Check whether now locates at .../PowerServe
if [ ! -d "tools" ]; then
    echo -e "\033[31mPlease run this script from the root directory of PowerServe.\033[0m"
    exit 1
fi

# Default values
prompt="In recent years, the landscape of artificial intelligence has been significantly transformed by the advent of large language models (LLMs). Traditionally, these models have been deployed in cloud environments due to their computational demands. However, the emergence of on-edge LLMs is reshaping how AI can be utilized at the edge of networks, offering numerous advantages in terms of latency, privacy, and accessibility."
speculation_enabled="no"
cpu_only="no"
model_name=""

supported_model_list=("smallthinker-3b" "llama-3.1-8b" "llama-3.2-1b")
supported_speculation_model_list=("smallthinker-3b" "llama-3.1-8b")

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

# Get the model name
model_name=$1
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

# Output configuration
echo -e "\033[32mModel        : $model_name\033[0m"
echo -e "\033[32mPrompt       : $prompt\033[0m"
if [ "$cpu_only" == "yes" ]; then
    echo -e "\033[32mCPU only     : $cpu_only\033[0m"
fi
echo -e "\033[32mSpeculation  : $speculation_enabled\033[0m"

./tools/end_to_end/get_soc.sh

if [ $? -ne 0 ]; then
    echo -e "\033[31mFailed to get SoC information.\033[0m"
    exit 1
fi

# download command
download_command="./tools/end_to_end/download.sh -p \"$prompt\""
if [ "$speculation_enabled" == "yes" ]; then
    download_command+=" -s"
fi
if [ "$cpu_only" == "yes" ]; then
    download_command+=" -c"
fi
download_command+=" $model_name"

# run download command directly
eval "$download_command"

if [ $? -ne 0 ]; then
    echo -e "\033[31mFailed to download the model.\033[0m"
    exit 1
fi

# Compile command
shell_command="./tools/end_to_end/compile.sh -p \"$prompt\""
if [ "$speculation_enabled" == "yes" ]; then
    shell_command+=" -s"
fi
if [ "$cpu_only" == "yes" ]; then
    shell_command+=" -c"
fi
shell_command+=" $model_name"

# Run Docker command
sudo docker run --platform linux/amd64 --rm --name powerserve_container -v "$(pwd):/code" -w /code -e https_proxy="$https_proxy" -e http_proxy="$http_proxy" -e socks_proxy="$socks_proxy" --network host -it santoxin/mobile-build:v1.1 /bin/bash -c "$shell_command"

if [ $? -ne 0 ]; then
    echo -e "\033[31mFailed to compile the model.\033[0m"
    exit 1
fi

deploy_command="./tools/end_to_end/deploy_to_phone.sh -p \"$prompt\""
if [ "$speculation_enabled" == "yes" ]; then
    deploy_command+=" -s"
fi
if [ "$cpu_only" == "yes" ]; then
    deploy_command+=" -c"
fi

# run deploy command directly
eval "$deploy_command"

# Check the result
if [ $? -eq 0 ]; then
    echo -e "\033[32mProcess completed\033[0m"
else
    echo -e "\033[31mProcess Aborted\033[0m"
fi
