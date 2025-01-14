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

# Check if the model exists
OPEN_SOURCE="YES"

echo -e "\033[32mChecking the connection with GitHub and Hugging Face\033[0m"

if [ "$OPEN_SOURCE" == "YES" ]; then
    # check connection with github
    github_link="https://github.com/"
    if ! curl --output /dev/null --silent --head --fail --max-time 10 "$github_link"; then
        echo -e "\033[31mYour connection with GitHub is not okay. (You may need a proxy?)\033[0m"
        exit 1
    fi

    if [ "$speculation_enabled" == "yes" ]; then
        for model in "$model1" "$model2"; do
            link="https://huggingface.co/PowerServe/${model}"
            if ! curl --output /dev/null --silent --head --fail --max-time 10 "$link"; then
                echo -e "\033[31mYour connection with Hugging Face is not okay, or Model $model does not exist. (You may need a proxy?)\033[0m"
                exit 1
            fi
        done
    else
        link="https://huggingface.co/PowerServe/${model}"
        if ! curl --output /dev/null --silent --head --fail --max-time 10 "$link"; then
            echo -e "\033[31mYour connection with Hugging Face is not okay, or Model $model does not exist. (You may need a proxy?)\033[0m"
            exit 1
        fi
    fi
fi

echo -e "\033[36mDownloading the submodule from GitHub\033[0m"
# git config --global --add safe.directory '/code'
# git config --global --add safe.directory '*'
git submodule update --init --recursive

echo -e "\033[36mDownloading models from huggingface\033[0m"


model_dir="models"
mkdir -p $model_dir
cd $model_dir

echo -e "\033[36mNow we are downloading the models from huggingface.\033[0m"
echo -e "\033[33mYou may have to wait for a while, about 2 to 10 minutes according to your network speed.\033[0m"
echo -e "\033[33mWait patiently. :)\033[0m"

if [ "$speculation_enabled" == "yes" ]; then
    for model in "$model1" "$model2"; do
        echo -e "\033[32mDownloading model $model\033[0m"
        if [ -d "${model}" ]; then
            rm -rf "${model}"
        fi
        git clone "https://huggingface.co/PowerServe/${model}"
    done
else
    echo -e "\033[32mDownloading model $model\033[0m"
    if [ -d "${model}" ]; then
        rm -rf "${model}"
    fi
    git clone "https://huggingface.co/PowerServe/${model}"
fi

cd ..
