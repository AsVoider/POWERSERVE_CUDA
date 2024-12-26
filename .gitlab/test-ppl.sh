#!/bin/bash

DEVICE_ROOT=$1
DEVICE_URL=$2
DEVICE_PORT=$3

TARGET=$4
if [ "${TARGET}" == "" ]; then
    TARGET="smart-llama3.1-8b"
fi

WORK_FOLDER="${DEVICE_ROOT}/${TARGET}"
PROMPT_FILE="wikitext-2-small.csv"


function help() {
    echo "Usage: $0 <device_root> <device_url> <device_port> - [target]"
    exit 1
}

function clean() {
    ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
        ${DEVICE_ROOT}/smartserving hparams load -d ${WORK_FOLDER} -f ./hparams.old;
        ${DEVICE_ROOT}/smartserving hparams get -d ${WORK_FOLDER};
    "
}

if [ $# -lt 3 ]; then
    help
fi

set -e
trap clean EXIT
source .gitlab/common.sh

ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    ${DEVICE_ROOT}/smartserving hparams store -d ${WORK_FOLDER} -f ./hparams.old;
    ${DEVICE_ROOT}/smartserving hparams set -d ${WORK_FOLDER} -e prompt_file=${PROMPT_FILE};
    ${DEVICE_ROOT}/smartserving hparams get -d ${WORK_FOLDER};
"

echo '>>>>>>>>>>>> Test ppl. <<<<<<<<<<<<';
set -x
ssh -o StrictHostKeyChecking=no -p ${DEVICE_PORT} ${DEVICE_URL} "
    ${DEVICE_ROOT}/smartserving ppl -d ${WORK_FOLDER};
"
set +x
echo '>>>>>>>>>>>> Test ppl over. <<<<<<<<<<<<';
