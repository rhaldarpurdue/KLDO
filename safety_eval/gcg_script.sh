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
model_name=meta-llama/Llama-3.2-1B
model_path="base" #"../

