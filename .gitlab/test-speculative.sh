#!/bin/bash
# test-llama3.1.sh /data/data/com.termux/files/home/CI u0_a342@192.168.60.173 8022

DEVICE_ROOT=$1
DEVICE_URL=$2
DEVICE_PORT=$3

THREADS_NUM=$4
if [ "${THREADS_NUM}" == "" ]; then
    THREADS_NUM=4
fi

STEPS=$5
if [ "${STEPS}" == "" ]; then
    STEPS=1000
fi

USE_QNN=$6
if [ "${USE_QNN}" == "" ]; then
    USE_QNN=1
fi

TARGET_QNN_PATH="${DEVICE_ROOT}/qnn_models/2_28/3_1"
DRAFT_QNN_PATH="${DEVICE_ROOT}/qnn_models/2_28/3_2"
TARGET_MODEL_PATH="${DEVICE_ROOT}/models/llama3.1-8b-instr"
DRAFT_MODEL_PATH="${DEVICE_ROOT}/models/llama3.2-1b-instr"
WORKSPACE_PATH="${DEVICE_ROOT}/workspace"
BIN_PATH="${DEVICE_ROOT}/bin"

function help() {
    echo "Usage: \$0 <device_root> <device_url> <device_port> [-] [threads_num] [steps] [use_qnn]"
    exit 1
}

function print_args() {
    echo "- DEVICE_ROOT: ${DEVICE_ROOT}"
    echo "- DEVICE_URL: ${DEVICE_URL}"
    echo "- DEVICE_PORT: ${DEVICE_PORT}"
    echo "- THREADS_NUM: ${THREADS_NUM}"
    echo "- STEPS: ${STEPS}"
    echo "- TARGET_MODEL_PATH: ${TARGET_MODEL_PATH}"
    echo "- DRAFT_MODEL_PATH: ${DRAFT_MODEL_PATH}"
    echo "- TARGET_QNN_PATH: ${TARGET_QNN_PATH}"
    echo "- DRAFT_QNN_PATH: ${DRAFT_QNN_PATH}"
    echo "- WORKSPACE_PATH: ${WORKSPACE_PATH}"
    echo "- BIN_PATH: ${BIN_PATH}"
}

if [ $# -lt 3 ]; then
    help
fi

print_args

set -e

source .gitlab/common.sh

# Execute commands on the remote device
ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    export LD_LIBRARY_PATH=/vendor/lib64
    if [ \"${USE_QNN}\" == \"1\" ]; then
        echo '>>>>>>>>>>>> Run test with Speculative-QNN. <<<<<<<<<<<<';
        sudo -E ${BIN_PATH}/speculative \
        --target-path ${TARGET_QNN_PATH}/llama_3.1_8b_q4_0.gguf \
        --draft-path ${DRAFT_MODEL_PATH}/llama_3.2_1b_q4_0.gguf \
        --vocab-path ${TARGET_MODEL_PATH}/vocab.gguf \
        --target-config-path ${TARGET_QNN_PATH}/model_config.json \
        --draft-config-path ${DRAFT_QNN_PATH}/model_config.json \
        --target-qnn-path ${TARGET_QNN_PATH} \
        --draft-qnn-path ${DRAFT_QNN_PATH}
        echo '>>>>>>>>>>>> Run over. <<<<<<<<<<<<';
    fi"

set +x
