#!/bin/bash

if [ -z "$QNN_SDK_ROOT" ]; then
    echo "[ERROR] QNN_SDK_ROOT hasn't been set"
    exit
fi

if [ -z "$SMART_SERVING_ROOT" ]; then
    echo "[ERROR] SMART_SERVING_ROOT hasn't been set"
    exit
fi

if [ -z "$NDK" ]; then
    echo "[ERROR] NDK hasn't been set"
    exit
fi

if [ -z "$ADB_ARTIFACTS_PATH" ]; then
    echo "[ERROR] ADB_ARTIFACTS_PATH hasn't been set"
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
GGUF_OUTPUT_PATH=$MODEL_PATH\_gguf

BATCH_SIZE=128
BATCH_PARAM=batch_$BATCH_SIZE

HEXAGON_VERSION=79

mkdir -p $ADB_ARTIFACTS_PATH

ADB_MODEL_PATH=$ADB_ARTIFACTS_PATH/$MODEL_NAME
ADB_WORKSPACE_PATH=$ADB_MODEL_PATH/qnn-workspace
ADB_BINARY_PATH=$ADB_ARTIFACTS_PATH/bin/aarch64

mkdir -p $ADB_MODEL_PATH
mkdir -p $ADB_WORKSPACE_PATH
mkdir -p $ADB_BINARY_PATH

# Prepare QNN env
echo "-------------------------- [Prepare QNN] --------------------------"
echo ""
echo ""

cp -f $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so   $ADB_WORKSPACE_PATH
cp -f $QNN_SDK_ROOT/lib/aarch64-android/libQnnSystem.so   $ADB_WORKSPACE_PATH
cp -f $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV$HEXAGON_VERSION\Stub.so   $ADB_WORKSPACE_PATH

cp -f $QNN_SDK_ROOT/lib/hexagon-v$HEXAGON_VERSION\/unsigned/libQnnHtpV$HEXAGON_VERSION\.so $ADB_WORKSPACE_PATH
cp -f $QNN_SDK_ROOT/lib/hexagon-v$HEXAGON_VERSION\/unsigned/libQnnHtpV$HEXAGON_VERSION\Skel.so $ADB_WORKSPACE_PATH

# Prepare model
echo "-------------------------- [Prepare Model] --------------------------"
echo ""
echo ""

Q4_MODEL_PATH=$GGUF_OUTPUT_PATH/$MODEL_NAME.q4.gguf
MODEL_VOCAB_PATH=$GGUF_OUTPUT_PATH/$MODEL_NAME.vocab.gguf

# Extract vocab
# python3 $SMART_SERVING_ROOT/tools/generate_llama_vocab.py $MODEL_PATH --o $MODEL_VOCAB_PATH

## copy gguf models
cp -f $Q4_MODEL_PATH    $ADB_MODEL_PATH/weights.gguf
cp -f $MODEL_VOCAB_PATH $ADB_MODEL_PATH/vocab.gguf

## copy QNN models
cp -f $QNN_OUTPUT_PATH/*.bin         $ADB_WORKSPACE_PATH
cp -rf $QNN_OUTPUT_PATH/$BATCH_PARAM $ADB_WORKSPACE_PATH

cp -f $QNN_OUTPUT_PATH/config.json   $ADB_WORKSPACE_PATH

# Prepare smartserving
echo "-------------------------- [Prepare SmartServing] --------------------------"
echo ""
echo ""

cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake           \
        -S $SMART_SERVING_ROOT                                                  \
        -B $SMART_SERVING_ROOT/build_android                                    \
        -DANDROID_ABI=arm64-v8a                                                 \
        -DANDROID_PLATFORM=android-34                                           \
        -DCMAKE_BUILD_TYPE=Release                                              \
        -DBUILD_SHARED_LIBS=OFF                                                 \
        -DGGML_OPENMP=OFF                                                       \
        -DSMART_ENABLE_ASAN=OFF                                                 \
        -DSMART_WITH_QNN=ON                                                    

cmake --build $SMART_SERVING_ROOT/build_android -j 16

cp -f $SMART_SERVING_ROOT/build_android/out/smart-* $ADB_BINARY_PATH
cp -f $SMART_SERVING_ROOT/smartserving              $ADB_ARTIFACTS_PATH


# Prepare config json
echo "-------------------------- [Prepare Config] --------------------------"
echo ""
echo ""

cmake -DCMAKE_BUILD_TYPE=Release        \
        -S $SMART_SERVING_ROOT          \
        -B $SMART_SERVING_ROOT/build

cmake --build $SMART_SERVING_ROOT/build -j 16

$SMART_SERVING_ROOT/build/out/smart-config_generator \
    --file-path $Q4_MODEL_PATH                       \
    --target-path $ADB_MODEL_PATH/llm.json
