#!/bin/bash
#SBATCH -A standby
#SBATCH --gpus-per-node=2
#SBATCH -C A100
#SBATCH --time=04:00:00
#SBATCH --job-name kl
#SBATCH --output ../log/%x_%j.out
#SBATCH --error ../log/%x_%j.err

module load anaconda/2024.02-py311
module load cudnn/cuda-12.1_8.9

conda activate /depot/qfsong/LLM/env/halos

export TRANSFORMERS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/hub
export HF_HOME=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/
export HF_DATASETS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/datasets
export WANDB_API_KEY=8e4a0bf8aa276a6a441763aab7441d43ed309205
export HUGGING_FACE_API_KEY=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

huggingface-cli login --token $HUGGING_FACE_API_KEY

data_name="adv_bench"
dataset="../dataset_generation/data/advbench/harmful_behaviors.csv"
column="goal" # "goal" for adv_bench, None for MaliciousInstruct
model_name=google/gemma-2-2b
model_path="../data/models/kto_gemma2-2b_5e-05_5/FINAL" # "base", for original model

# Extract the model configuration
if [ "$model_path" = "base" ]; then
    model_config=${model_name}
else
    model_config=$(echo "$model_path" | awk -F'/' '{print $(NF-1)}')
fi

start_idx=0
end_idx=520 #520 for adv_bench


# Define the log file path
LOG_PATH="./log/clean/${data_name}/${model_config}.log"

# Create the directory if it doesn't exist
mkdir -p "$(dirname "$LOG_PATH")"

python compliance.py \
        --model_name ${model_name} \
        --model_path ${model_path} \
        --dataset ${dataset} \
        --column ${column} \
        --idx ${start_idx} ${end_idx} \
        --model_judge "cais/HarmBench-Llama-2-13b-cls" \
        --output "./results/clean/${data_name}/${model_config}.csv" 2>&1 | tee ${LOG_PATH}