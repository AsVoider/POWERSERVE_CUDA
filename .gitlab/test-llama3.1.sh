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
    STEPS=32
fi

USE_QNN=$6
if [ "${USE_QNN}" == "" ]; then
    USE_QNN=0
fi

QNN_PATH="${DEVICE_ROOT}/qnn_models/2_27/llama3.1-8b-2"
MODEL_PATH="${DEVICE_ROOT}/models/llama3.1-8b-instr"
SDK_PATH="${DEVICE_ROOT}/qnn_sdk/2.27/v75"
WORKSPACE_PATH="${DEVICE_ROOT}/workspace"
BIN_PATH="${DEVICE_ROOT}/bin"

function help() {
    echo "Usage: $0 <device_root> <device_url> <device_port> [-] [threads_num] [steps] [use_qnn]"
    exit 1
}

function print_args() {
    echo "- DEVICE_ROOT: ${DEVICE_ROOT}"
    echo "- DEVICE_URL: ${DEVICE_URL}"
    echo "- DEVICE_PORT: ${DEVICE_PORT}"
    echo "- THREADS_NUM: ${THREADS_NUM}"
    echo "- STEPS: ${STEPS}"
    echo "- QNN_PATH: ${QNN_PATH}"
    echo "- MODEL_PATH: ${MODEL_PATH}"
    echo "- SDK_PATH: ${SDK_PATH}"
    echo "- WORKSPACE_PATH: ${WORKSPACE_PATH}"
    echo "- BIN_PATH: ${BIN_PATH}"
}

if [ $# -lt 3 ]; then
    help
fi

print_args

set -e

source .gitlab/common.sh

if [ "${USE_QNN}" == "1" ]; then
    clean_workspace ${DEVICE_ROOT} ${DEVICE_URL} ${DEVICE_PORT} ${WORKSPACE_PATH} ${BIN_PATH} ${QNN_PATH} ${SDK_PATH}
fi

ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    echo '>>>>>>>>>>>> Generate configuration. <<<<<<<<<<<<';
    ${BIN_PATH}/config_generator \
    --file-path ${MODEL_PATH}/q4_0.gguf \
    --target-path ${MODEL_PATH}/model_config.json
    echo '>>>>>>>>>>>> Generate over. <<<<<<<<<<<<';
"

set -x

if [ "${USE_QNN}" == "1" ]; then
    ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
        echo '>>>>>>>>>>>> Run test. <<<<<<<<<<<<';
        export LD_LIBRARY_PATH=/vendor/lib64 && sudo -E ${BIN_PATH}/run \
        --file-path ${MODEL_PATH}/q4_0.gguf \
        --vocab-path ${MODEL_PATH}/vocab.gguf \
        --config-path ${MODEL_PATH}/model_config.json \
        --prompt \"One day,\" --steps ${STEPS} \
        --n-threads ${THREADS_NUM} \
        --qnn-path ${WORKSPACE_PATH}
        echo '>>>>>>>>>>>> Run over. <<<<<<<<<<<<';
    "
else
    ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
        echo '>>>>>>>>>>>> Run test. <<<<<<<<<<<<';
        ${BIN_PATH}/run \
        --file-path ${MODEL_PATH}/q4_0.gguf \
        --vocab-path ${MODEL_PATH}/vocab.gguf \
        --config-path ${MODEL_PATH}/model_config.json \
        --prompt \"One day,\" --steps ${STEPS} \
        --n-threads ${THREADS_NUM}
        echo '>>>>>>>>>>>> Run over. <<<<<<<<<<<<';
    "
fi

set +x
