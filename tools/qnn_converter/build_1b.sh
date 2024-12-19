for i in {1,128};do
python llama_model.py \
    --n-threads 24 \
    --model-folder /Llama-3.2-1B-Instruct \
    --model-name llama3_2_1b \
    --graph-name batch_${i} \
    --system-prompt-file system_prompt_i.txt \
    --prompt-file lab_intro.md \
    --output-folder /117_3_2_fp16 \
    --max-n-tokens 1000 \
    --n-model-chunks 1

python build_all_layers.py \
--build-folder /117_3_2_fp16 \
--batch-sizes ${i} \
--n-model-chunks 1 \
--artifact-name llama3_2_1b \
--graph-names batch_${i}

rm -rf /117_3_2/m*/batch_${i}/data
rm -rf /117_3_2/m*/batch_${i}/o*/*.bin
done




for i in {1,128};do
python llama_model_lmhead.py \
    --n-threads 24 \
    --model-folder /Llama-3.2-1B-Instruct \
    --model-name llama3_2_1b \
    --graph-name batch_${i} \
    --system-prompt-file system_prompt_i.txt \
    --prompt-file lab_intro.md \
    --output-folder /117_3_2_e \
    --max-n-tokens 1000 \
    --n-model-chunks 1

python build_all_layers.py \
--build-folder /117_3_2_e \
--batch-sizes ${i} \
--n-model-chunks 1 \
--artifact-name lm_head \
--graph-names batch_${i} \
--embedding


rm -rf /117_3_2_e/*/batch_${i}/data
rm -rf /117_3_2_e/*/batch_${i}/o*/*.bin
done