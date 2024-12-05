#!/bin/bash
# test-ppl.sh /data/data/com.termux/files/home/CI u0_a342@192.168.60.173 8022

DEVICE_ROOT=$1
DEVICE_URL=$2
DEVICE_PORT=$3

QNN_PATH="${DEVICE_ROOT}/qnn_models/2_27/llama3.1-8b-2"
MODEL_PATH="${DEVICE_ROOT}/models/llama3.1-8b-instr"
SDK_PATH="${DEVICE_ROOT}/qnn_sdk/2.27/v75"
WORKSPACE_PATH="${DEVICE_ROOT}/workspace"
BIN_PATH="${DEVICE_ROOT}/bin"
PROMPT_PATH="${DEVICE_ROOT}/prompts/service_lab.txt"

function help() {
    echo "Usage: $0 <device_root> <device_url> <device_port>"
    exit 1
}

if [ $# -lt 3 ]; then
    help
fi

set -e

source .gitlab/common.sh
clean_workspace ${DEVICE_ROOT} ${DEVICE_URL} ${DEVICE_PORT} ${WORKSPACE_PATH} ${BIN_PATH} ${QNN_PATH} ${SDK_PATH}

set -x

echo '>>>>>>>>>>>> Test ppl. <<<<<<<<<<<<';
ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    LD_LIBRARY_PATH=/vendor/lib64 sudo ${BIN_PATH}/perpelxity_test \
    --qnn-path ${WORKSPACE_PATH} \
    --file-path ${MODEL_PATH}/q4_0.gguf \
    --vocab-path ${MODEL_PATH}/vocab.gguf \
    --config-path ${MODEL_PATH}/model_config.json \
    --prompt-file ${PROMPT_PATH} \
    --batch-size 32
"
echo '>>>>>>>>>>>> Test ppl over. <<<<<<<<<<<<';
