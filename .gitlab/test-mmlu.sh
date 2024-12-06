#!/bin/bash
# test-mmlu.sh /data/data/com.termux/files/home/CI u0_a342@192.168.60.173 8022

DEVICE_ROOT=$1
DEVICE_USER=$2
DEVICE_HOST=$3
DEVICE_PORT=$4
CONTAINER_NAME=$5

SERVER_HOST=${DEVICE_HOST}
SERVER_PORT="18080"
DEVICE_URL="${DEVICE_USER}@${DEVICE_HOST}"

QNN_PATH="${DEVICE_ROOT}/qnn_models/2_27/llama3.1-8b-2"
MODEL_PATH="${DEVICE_ROOT}/models/llama3.1-8b-instr"
SDK_PATH="${DEVICE_ROOT}/qnn_sdk/2.27/v75"
WORKSPACE_PATH="${DEVICE_ROOT}/workspace"
BIN_PATH="${DEVICE_ROOT}/bin"

function help() {
    echo "Usage: $0 <device_root> <device_host> <device_user> <device_port> <mmlu_client_container_name>"
    exit 1
}

function clean() {
    set +x
    temp_disable_errexit try_twice 10 ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
        echo '>>>>>>>>>>>> Stop server. <<<<<<<<<<<<';
        sudo ps -e -o comm= | grep 'server' |xargs -n 1 echo;
        sudo pkill ${BIN_PATH}/server;
        sleep 3;
        echo '>>>>>>>>>>>> Stop server over. <<<<<<<<<<<<';
        sudo ps -e -o comm= | grep 'server' |xargs -n 1 echo
    "
}

if [ $# -lt 5 ]; then
    help
fi

set -e

source .gitlab/common.sh
clean_workspace ${DEVICE_ROOT} ${DEVICE_URL} ${DEVICE_PORT} ${WORKSPACE_PATH} ${BIN_PATH} ${QNN_PATH} ${SDK_PATH}

trap clean EXIT

set -x

echo '>>>>>>>>>>>> Start server. <<<<<<<<<<<<';
ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    LD_LIBRARY_PATH=/vendor/lib64 sudo ${BIN_PATH}/server \
    --host ${SERVER_HOST} \
    --port ${SERVER_PORT} \
    --qnn-path ${WORKSPACE_PATH} \
    --file-path ${MODEL_PATH}/q4_0.gguf \
    --vocab-path ${MODEL_PATH}/vocab.gguf \
    --config-path ${MODEL_PATH}/model_config.json >/dev/null 2>&1
" &
echo '>>>>>>>>>>>> Start server over. <<<<<<<<<<<<';

sleep 10

echo '>>>>>>>>>>>> Test mmlu. <<<<<<<<<<<<';
sudo podman exec -it ${CONTAINER_NAME} bash -c -i "
    cd /code/tools/mmlu;
    python ./mmlu_test.py --host ${SERVER_HOST} --port ${SERVER_PORT} -s 1
"
echo '>>>>>>>>>>>> Test mmlu over. <<<<<<<<<<<<';

set +x
