#!/bin/bash
# test-decode.sh /data/data/com.termux/files/home/CI u0_a342@192.168.60.173 8022 smart-llama3.1-8b / smart-qwen2-7b

DEVICE_ROOT=$1
DEVICE_URL=$2
DEVICE_PORT=$3

TARGET=$4
if [ "${TARGET}" == "" ]; then
    TARGET="smart-llama3.1-8b"
fi

USE_QNN=$5
if [ "${USE_QNN}" == "" ]; then
    USE_QNN=0
fi

STEPS=$6
if [ "${STEPS}" == "" ]; then
    STEPS=32
fi

THREADS_NUM=$7
if [ "${THREADS_NUM}" == "" ]; then
    THREADS_NUM=4
fi

PROMPT_FILE=$8
if [ "${PROMPT_FILE}" == "" ]; then
    PROMPT_FILE="hello.txt"
fi

CONFIG_PATH="${DEVICE_ROOT}/${TARGET}"

function help() {
    echo "Usage: $0 <device_root> <device_url> <device_port> [-] [target] [use_qnn] [steps] [threads_num] [prompt_file]"
    exit 1
}

function clean() {
    ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
        ${DEVICE_ROOT}/smartserving params load -c ${CONFIG_PATH} -f ./params.old;
        ${DEVICE_ROOT}/smartserving params get -c ${CONFIG_PATH};
    "
}

if [ $# -lt 3 ]; then
    help
fi

set -e
trap clean EXIT

ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    ${DEVICE_ROOT}/smartserving params store -c ${CONFIG_PATH} -f ./params.old;
    ${DEVICE_ROOT}/smartserving params set -c ${CONFIG_PATH} -e n_predicts=${STEPS} n_threads=${THREADS_NUM} prompt_file=${PROMPT_FILE};
    ${DEVICE_ROOT}/smartserving params get -c ${CONFIG_PATH};
"

ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    echo '>>>>>>>>>>>> Run test. <<<<<<<<<<<<';
"

set -x
if [ "${USE_QNN}" == "1" ]; then
    ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
        ${DEVICE_ROOT}/smartserving run -c ${CONFIG_PATH} --use-qnn;
    "
else
    ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
        ${DEVICE_ROOT}/smartserving run -c ${CONFIG_PATH};
    "
fi
set +x

ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    echo '>>>>>>>>>>>> Run over. <<<<<<<<<<<<';
"
