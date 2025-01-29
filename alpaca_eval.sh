#!/bin/bash


MODEL_PATH="./data/models/dpo_mistral_5e-05_5/FINAL"
OUTPUT_FILE="./outputs/alpacaeval/kto_mistral_5e-05_5/generation.json"
OUT_PATH="./outputs/alpacaeval/kl-shift_mistral_5e-05_5_F/kto_mistral_5e-05_5"
REFERENCE_OUTPUTS="./outputs/alpacaeval/kl-shift_mistral_5e-05_5_F/generation2.json"


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

# accelerate launch -m lm_eval --model hf \
#   --model_args pretrained="$MODEL_PATH",tokenizer="$TOKENIZER_PATH",parallelize=True \
#   --tasks arc_easy \
#   --batch_size 4


# AlpacaEval
# Count the number of GPUs, default to 0 if nvidia-smi is not available
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l) || GPU_COUNT=0
# OPENAI_API_KEY needs to be set
export OPENAI_API_KEY=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
time=$(date "+%Y%m%d-%H%M%S")
echo "Start time: $time"
# time=$(date "+%Y%m%d-%H%M%S")
# echo "Generation End time: $time"
alpaca_eval evaluate --is_overwrite_leaderboard=True --model_outputs="$OUTPUT_FILE" --reference_outputs="$REFERENCE_OUTPUTS" --output_path="$OUT_PATH"
time=$(date "+%Y%m%d-%H%M%S")
echo "Evaluation End time: $time"

