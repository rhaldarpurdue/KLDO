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

MODEL_PATH="./data/models/kto_pythia_5e-05_5_F/FINAL"
OUTPUT_FILE="./outputs/alpacaeval/kto_pythia_5e-05_5_F/generation.json"
MODEL_NAME=EleutherAI/pythia-1b #google/gemma-2-2b #Qwen/Qwen2.5-1.5B #mistralai/Mistral-7B-v0.1 #ContextualAI/archangel_sft_llama7b #meta-llama/Llama-3.2-1B #


echo "Running on node: $(hostname)"
echo "Machine Rank: $SLURM_PROCID"

module load anaconda/2024.02-py311
module load cudnn/cuda-12.1_8.9

conda activate /depot/qfsong/LLM/env/halos

export TRANSFORMERS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/hub
export HF_HOME=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/
export HF_DATASETS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/datasets
export WANDB_API_KEY=8e4a0bf8aa276a6a441763aab7441d43ed309205
export HUGGING_FACE_API_KEY=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

huggingface-cli login --token $HUGGING_FACE_API_KEY

echo "Machine Rank: $SLURM_PROCID"

# AlpacaEval
# Count the number of GPUs, default to 0 if nvidia-smi is not available
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l) || GPU_COUNT=0
# OPENAI_API_KEY needs to be set
export OPENAI_API_KEY=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
time=$(date "+%Y%m%d-%H%M%S")
echo "Start time: $time"
nvidia-smi
python -m eval.sample_for_alpacaeval --model_path "$MODEL_PATH" --model_name "$MODEL_NAME" --gpu_count $GPU_COUNT --output_file "$OUTPUT_FILE" --google
time=$(date "+%Y%m%d-%H%M%S")
echo "Start time: $time"



