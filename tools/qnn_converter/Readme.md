# Script for exporting Qnn model chunk
## 配置QNN环境
echo $QNN_SDK_ROOT
/home/zwb/SS/qnn-sdk/2.28
echo $PYTHONPATH  
/home/zwb/SS/qnn-sdk/2.28/lib/python/

# ==========================================
# llama3.2-1b 记得替换掉里面的路径

## 创建一个目录放导出的模型
mkdir /data1/qnn-workspace

## lmhead
python llama_model_lmhead.py \
    --n-threads 24 \
    --model-folder /data1/models/Llama-3.2-1B \
    --model-name llama3_2_1b \
    --graph-name batch_128 \
    --system-prompt-file /home/zwb/SS/llama_tf_to_qnn_bin/system_prompt.txt \
    --prompt-file /home/zwb/SS/llama_tf_to_qnn_bin/lab_intro.md \
    --output-folder /data1/tmp_lmhead \
    --max-n-tokens 1000 \
    --n-model-chunks 1

## 导出so文件
python build_all_layers.py \
    --build-folder /data1/tmp_lmhead \
    --batch-sizes 128 \
    --n-model-chunks 1 \
    --artifact-name lm_head \
    --graph-names batch_128 \
    --embedding

## 这些文件在导出so文件之后就不需要了，可以删除，防止占用很多磁盘空间
rm -rf /data1/tmp_lmhead/output_embedding/batch_*/data
rm -rf /data1/tmp_lmhead/output_embedding/batch_*/onnx_model/*.bin

python generate_bin.py \
    --output-folder /data1/tmp_lmhead/output \
    --model-folder /data1/tmp_lmhead/output_embedding \
    --artifact-name lm_head \
    --graph-name batch_128

mv /data1/tmp_lmhead/output/lm_head.bin /data1/qnn-workspace

## model
python llama_model.py \
    --n-threads 24 \
    --model-folder /data1/models/Llama-3.2-1B \
    --model-name llama3_2_1b \
    --graph-name batch_128 \
    --system-prompt-file /home/zwb/SS/llama_tf_to_qnn_bin/system_prompt.txt \
    --prompt-file /home/zwb/SS/llama_tf_to_qnn_bin/lab_intro.md \
    --output-folder /data1/tmp_model \
    --max-n-tokens 1000 \
    --n-model-chunks 4

python build_all_layers.py \
    --build-folder /data1/tmp_model \
    --batch-sizes 128 \
    --n-model-chunks 4 \
    --artifact-name llama3_2_1b \
    --graph-names batch_128

rm -rf /data1/tmp_model/model_chunk_*/batch_*/data
rm -rf /data1/tmp_model/model_chunk_*/batch_*/onnx_model/*.bin

## 每个chunk都要生成一个bin
python generate_bin.py \
    --output-folder /data1/tmp_model/output \
    --model-folder /data1/tmp_model/model_chunk_0 \
    --artifact-name llama3_2_1b_0 \
    --graph-name batch_128
;
python generate_bin.py \
    --output-folder /data1/tmp_model/output \
    --model-folder /data1/tmp_model/model_chunk_1 \
    --artifact-name llama3_2_1b_1 \
    --graph-name batch_128
;
python generate_bin.py \
    --output-folder /data1/tmp_model/output \
    --model-folder /data1/tmp_model/model_chunk_2 \
    --artifact-name llama3_2_1b_2 \
    --graph-name batch_128
;
python generate_bin.py \
    --output-folder /data1/tmp_model/output \
    --model-folder /data1/tmp_model/model_chunk_3 \
    --artifact-name llama3_2_1b_3 \
    --graph-name batch_128

mv /data1/tmp_model/output/*.bin /data1/qnn-workspace
mv /data1/tmp_model/model_chunk_*/batch_128/kv /data1/qnn-workspace/batch_128
cp ./config.json /data1/qnn-workspace