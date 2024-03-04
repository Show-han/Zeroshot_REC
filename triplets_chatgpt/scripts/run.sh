#!/bin/bash

ROOT_PATH="path/to/your/Zeroshot_REC"
DATA_BASE_PATH="path/to/reclip_preprocess/directory"
PYTHON_SCRIPT="triplets_chatgpt/run.py"
OUTPUT_BASE_PATH="triplets/"

cd ROOT_PATH

declare -a file_list=(
    'refcocog_test.jsonl'
    'refcocog_val.jsonl'
    'refcoco_testa.jsonl'
    'refcoco+_testa.jsonl'
    'refcoco_testb.jsonl'
    'refcoco+_testb.jsonl'
    'refcoco_val.jsonl'
    'refcoco+_val.jsonl'
)

for file in "${file_list[@]}"
do
    input_path="${DATA_BASE_PATH}/${file}"
    output_path="${OUTPUT_BASE_PATH}/gpt_${file}"

    python "${PYTHON_SCRIPT}" "${input_path}" "${output_path}"
    sleep 60
done

echo "finished!"