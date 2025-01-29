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

data_name="adv_bench"
dataset="../dataset_generation/data/advbench/harmful_behaviors.csv"
column="goal" # "goal" for adv_bench, None for MaliciousInstruct
model_name=meta-llama/Llama-3.2-1B
model_path="base" #"../

