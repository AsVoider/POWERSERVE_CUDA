#!/bin/bash

if [ -z "$QNN_SDK_ROOT" ]; then
    echo "[ERROR] QNN_SDK_ROOT unset" 
    exit
fi

if [ -z "$SMART_SERVING_ROOT" ]; then
    echo "[ERROR] SMART_SERVING_ROOT unset" 
    exit
fi

if [ -z "$1" ]; then
    echo "Usage: $0 <model-name> <model-path>"
    exit
fi
if [ -z "$2" ]; then
    echo "Usage: $0 <model-name> <model-path>"
    exit
fi

MODEL_NAME=$1
MODEL_PATH=$2
QNN_OUTPUT_PATH=$MODEL_PATH\_qnn
QNN_LMHEAD_TMP_PATH=$QNN_OUTPUT_PATH/lmhead_tmp
QNN_MODEL_TMP_PATH=$QNN_OUTPUT_PATH/model_tmp

SMART_SERVING_SCRITPS=$SMART_SERVING_ROOT/tools/qnn_converter
SYSTEM_PROMPT=$SMART_SERVING_SCRITPS/system_prompt.txt
PROMPT_FILE=$SMART_SERVING_SCRITPS/lab_intro.md

LMHEAD_CHUNK=1

# Configurable parameters
THREAD_NUM=24
BATCH_SIZE=128
BATCH_PARAM=batch_$BATCH_SIZE
MAX_TOKEN_NUM=1000
MODEL_CHUNK=4

echo "Model Name       : $MODEL_NAME"
echo "Model Path       : $MODEL_PATH"
echo "Output Path      : $QNN_OUTPUT_PATH"
echo "SmartServing Path: $SMART_SERVING_ROOT"

echo "#Thread          : $THREAD_NUM"
echo "#Batch           : $BATCH_SIZE"
echo "#Token           : $MAX_TOKEN_NUM"

mkdir -p $QNN_OUTPUT_PATH
mkdir -p $QNN_LMHEAD_TMP_PATH
mkdir -p $QNN_MODEL_TMP_PATH


echo "-------------------------- [Convert LMHead] --------------------------"
echo ""
echo ""

python $SMART_SERVING_SCRITPS/llama_model_lmhead.py \
    --n-threads             $THREAD_NUM         \
    --model-folder          $MODEL_PATH         \
    --model-name            $MODEL_NAME         \
    --graph-name            $BATCH_PARAM        \
    --system-prompt-file    $SYSTEM_PROMPT      \
    --prompt-file           $PROMPT_FILE        \
    --output-folder         $QNN_LMHEAD_TMP_PATH\
    --max-n-tokens          $MAX_TOKEN_NUM      \
    --n-model-chunks        $LMHEAD_CHUNK

if [ -z "$?" ]; then
    echo "[ERROR] ret: $?"
    exit
fi

echo "-------------------------- [Export Dynamic Libs] --------------------------"
echo ""
echo ""

python $SMART_SERVING_SCRITPS/build_all_layers.py \
    --build-folder      $QNN_LMHEAD_TMP_PATH\
    --batch-sizes       $BATCH_SIZE         \
    --n-model-chunks    1                   \
    --artifact-name     lm_head             \
    --graph-names       $BATCH_PARAM        \
    --embedding

if [ -z "$?" ]; then
    echo "[ERROR] ret: $?"
    exit
fi

echo "-------------------------- [Generate Binary] --------------------------"
echo ""
echo ""

rm -rf $QNN_LMHEAD_TMP_PATH/output_embedding/batch_*/data
rm -rf $QNN_LMHEAD_TMP_PATH/output_embedding/batch_*/onnx_model/*.bin

python $SMART_SERVING_SCRITPS/generate_bin.py                   \
    --output-folder     $QNN_LMHEAD_TMP_PATH/output             \
    --model-folder      $QNN_LMHEAD_TMP_PATH/output_embedding   \
    --artifact-name     lm_head                                 \
    --graph-name        $BATCH_PARAM

if [ -z "$?" ]; then
    echo "[ERROR] ret: $?"
    exit
fi

mv $QNN_LMHEAD_TMP_PATH/output/lm_head.bin $QNN_OUTPUT_PATH

echo "-------------------------- [Convert Model] --------------------------"
echo ""
echo ""

python $SMART_SERVING_SCRITPS/llama_model.py            \
    --n-threads                 $THREAD_NUM             \
    --model-folder              $MODEL_PATH             \
    --model-name                $MODEL_NAME             \
    --graph-name                $BATCH_PARAM            \
    --system-prompt-file        $SYSTEM_PROMPT          \
    --prompt-file               $PROMPT_FILE            \
    --output-folder             $QNN_MODEL_TMP_PATH     \
    --max-n-tokens              $MAX_TOKEN_NUM          \
    --n-model-chunks            $MODEL_CHUNK

if [ -z "$?" ]; then
    echo "[ERROR] ret: $?"
    exit
fi

echo "-------------------------- [Export Dynamic Libs] --------------------------"
echo ""
echo ""

python $SMART_SERVING_SCRITPS/build_all_layers.py       \
    --build-folder              $QNN_MODEL_TMP_PATH     \
    --batch-sizes               $BATCH_SIZE             \
    --n-model-chunks            $MODEL_CHUNK            \
    --artifact-name             $MODEL_NAME             \
    --graph-names               $BATCH_PARAM

if [ -z "$?" ]; then
    echo "[ERROR] ret: $?"
    exit
echo ""
fi

rm -rf $QNN_MODEL_TMP_PATH/model_chunk_*/batch_*/data
rm -rf $QNN_MODEL_TMP_PATH/model_chunk_*/batch_*/onnx_model/*.bin


echo "-------------------------- [Generate Binary] --------------------------"
echo ""
echo ""

for((i=0;i<MODEL_CHUNK;i++));  
do   
python $SMART_SERVING_SCRITPS/generate_bin.py           \
    --output-folder $QNN_MODEL_TMP_PATH/output          \
    --model-folder  $QNN_MODEL_TMP_PATH/model_chunk_$i  \
    --artifact-name $MODEL_NAME\_$i                     \
    --graph-name    $BATCH_PARAM
done  

if [ -z "$?" ]; then
    echo "[ERROR] ret: $?"
    exit
fi

mkdir -p $QNN_OUTPUT_PATH/$BATCH_PARAM
mkdir -p $QNN_OUTPUT_PATH/$BATCH_PARAM/kv

mv $QNN_MODEL_TMP_PATH/output/*.bin $QNN_OUTPUT_PATH
mv $QNN_MODEL_TMP_PATH/model_chunk_*/$BATCH_PARAM/kv/* $QNN_OUTPUT_PATH/$BATCH_PARAM/kv
cp $SMART_SERVING_SCRITPS/config.json $QNN_OUTPUT_PATH


echo "[WARNING] DO notice the $QNN_OUTPUT_PATH/config.json which may be wrong"
echo ""
