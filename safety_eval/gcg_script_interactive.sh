#!/bin/bash

module load anaconda/2024.02-py311
module load cudnn/cuda-12.1_8.9

conda activate /your_env_path/halos

hf_cache=your_hf_cache_dir
wand_api_key=your_wandb_key
wdir=your_working_dir
hf_key=your_hugging_face_key

export TRANSFORMERS_CACHE=${hf_cache}/hub
export HF_HOME=${hf_cache}/
export HF_DATASETS_CACHE=${hf_cache}/datasets
export WANDB_API_KEY=${wand_api_key}
export HUGGING_FACE_API_KEY=${hf_key}

huggingface-cli login --token $HUGGING_FACE_API_KEY

model_path="../data/models/bco_llama-3.2-1b_5e-05_5/FINAL"
model_name="meta-llama/Llama-3.2-1B"
dataset="../dataset_generation/data/advbench/harmful_behaviors.csv"
column="goal" # "goal" for adv_bench, None for MaliciousInstruct
max_epochs=20


if [ "$model_path" = "base" ]; then
    model_config=${model_name}
else
    model_config=$(echo "$model_path" | awk -F'/' '{print $(NF-1)}')
fi

start_idx=0
end_idx=520 #520 for adv_bench

# Define the log file path
LOG_PATH="./log/gcg/${data_name}/${model_config}.log"

# Create the directory if it doesn't exist
mkdir -p "$(dirname "$LOG_PATH")"

python jailbreak_gcg.py \
        --model_name ${model_name} \
        --model_path ${model_path} \
        --dataset ${dataset} \
        --column ${column} \
        --idx ${start_idx} ${end_idx} \
        --model_judge "cais/HarmBench-Llama-2-13b-cls" \
        --max_epochs ${max_epochs} \
        --output_dir "./results/gcg/${data_name}/${model_config}.csv" 2>&1 | tee ${LOG_PATH}