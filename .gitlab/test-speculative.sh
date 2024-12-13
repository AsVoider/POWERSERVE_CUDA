#!/bin/bash
# test-speculative.sh /data/data/com.termux/files/home/CI u0_a342@192.168.60.173 8022 smart-llama3.1-8b-spec

DEVICE_ROOT=$1
DEVICE_URL=$2
DEVICE_PORT=$3

TARGET=$4
if [ "${TARGET}" == "" ]; then
    TARGET="smart-llama3.1-8b-spec"
fi

USE_QNN=$5
if [ "${USE_QNN}" == "" ]; then
    USE_QNN=1
fi

STEPS=$6
if [ "${STEPS}" == "" ]; then
    STEPS=500
fi

CONFIG_PATH="${DEVICE_ROOT}/${TARGET}"
PROMPT_FILE="math.txt"

function help() {
    echo "Usage: \$0 <device_root> <device_url> <device_port> [-] [target] [use_qnn] [steps]"
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
    ${DEVICE_ROOT}/smartserving params set -c ${CONFIG_PATH} -e n_predicts=${STEPS} prompt_file=${PROMPT_FILE};
    ${DEVICE_ROOT}/smartserving params get -c ${CONFIG_PATH};
"

ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    echo '>>>>>>>>>>>> Run test. <<<<<<<<<<<<';
"

set -x
if [ "${USE_QNN}" == "1" ]; then
ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
        ${DEVICE_ROOT}/smartserving speculate -c ${CONFIG_PATH} --use-qnn;
    "
else
    echo "No support"
fi
set +x

ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    echo '>>>>>>>>>>>> Run over. <<<<<<<<<<<<';
"
