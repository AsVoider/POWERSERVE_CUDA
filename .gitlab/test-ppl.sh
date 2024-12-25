#!/bin/bash

DEVICE_ROOT=$1
DEVICE_URL=$2
DEVICE_PORT=$3

TARGET=$4
if [ "${TARGET}" == "" ]; then
    TARGET="smart-llama3.1-8b"
fi

CONFIG_PATH="${DEVICE_ROOT}/${TARGET}"
PROMPT_FILE="wikitext-2-small.csv"


function help() {
    echo "Usage: $0 <device_root> <device_url> <device_port> - [target]"
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
source .gitlab/common.sh

ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    ${DEVICE_ROOT}/smartserving params store -c ${CONFIG_PATH} -f ./params.old;
    ${DEVICE_ROOT}/smartserving params set -c ${CONFIG_PATH} -e prompt_file=${PROMPT_FILE};
    ${DEVICE_ROOT}/smartserving params get -c ${CONFIG_PATH};
"

echo '>>>>>>>>>>>> Test ppl. <<<<<<<<<<<<';
set -x
ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    ${DEVICE_ROOT}/smartserving ppl -c ${CONFIG_PATH};
"
set +x
echo '>>>>>>>>>>>> Test ppl over. <<<<<<<<<<<<';
