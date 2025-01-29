#!/bin/bash
#SBATCH -A standby
#SBATCH --gpus-per-node=1
#SBATCH -C A100
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name kto_pythia_alpaca_generate
#SBATCH --output ./log/%x_%j.out
#SBATCH --error ./log/%x_%j.err
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16

MODEL_PATH="./data/models/kto_mistral_5e-05_5_F/FINAL"
OUTPUT_FILE="./outputs/alpacaeval/kto_mistral_5e-05_5_F/generation.json"
MODEL_NAME=mistralai/Mistral-7B-v0.1 #google/gemma-2-2b #Qwen/Qwen2.5-1.5B #mistralai/Mistral-7B-v0.1 #ContextualAI/archangel_sft_llama7b #meta-llama/Llama-3.2-1B #


echo "Running on node: $(hostname)"
echo "Machine Rank: $SLURM_PROCID"

module load anaconda/2024.02-py311
module load cudnn/cuda-12.1_8.9

conda activate /your_env_path/halos

hf_cache=your_hf_cache_dir
wand_api_key=your_wandb_key
wdir=your_working_dir
hf_key=your_hugging_face_key
openai_key=your_openai_key

export TRANSFORMERS_CACHE=${hf_cache}/hub
export HF_HOME=${hf_cache}/
export HF_DATASETS_CACHE=${hf_cache}/datasets
export WANDB_API_KEY=${wand_api_key}
export HUGGING_FACE_API_KEY=${hf_key}

huggingface-cli login --token $HUGGING_FACE_API_KEY

echo "Machine Rank: $SLURM_PROCID"

# AlpacaEval
# Count the number of GPUs, default to 0 if nvidia-smi is not available
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l) || GPU_COUNT=0
# OPENAI_API_KEY needs to be set
export OPENAI_API_KEY=${openai_key}
time=$(date "+%Y%m%d-%H%M%S")
echo "Start time: $time"
nvidia-smi
python -m eval.sample_for_alpacaeval --model_path "$MODEL_PATH" --model_name "$MODEL_NAME" --gpu_count $GPU_COUNT --output_file "$OUTPUT_FILE" --google
time=$(date "+%Y%m%d-%H%M%S")
echo "Start time: $time"



