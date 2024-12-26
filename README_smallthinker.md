## 初始化仓库

```bash
pip install -r requirements.txt
git submodule update --init --recursive
```

## 导出3B模型

```bash
pushd tools/qnn_converter

python converter.py \
    --model-folder /data/smallthinker_3b_20241220 \
    --model-name smallthinker_3b \
    --system-prompt-file qwen_system_prompt.txt \
    --prompt-file qwen_service_lab_intro.md \
    --batch-size 1 \
    --artifact-name smallthinker_3b \
    --n-model-chunks 2 \
    --max-n-tokens 1024 \
    --n-threads 8 \
    --soc 8gen4 \
    --fp16-lm-head

rm -rf smallthinker_3b_qnn
mv output smallthinker_3b_qnn

popd

python ./tools/gguf_export.py -m /data/smallthinker_3b_20241220 --qnn-path tools/qnn_converter/smallthinker_3b_qnn -o /data/smallthinker_3b
```

## 运行

sync.sh

```bash
cd /data/local/tmp
sudo cp ~/smallthinker/bin/smart-run .
sudo ./smart-run --work-folder ~/smallthinker --prompt-file ~/prompt_1.txt
```
