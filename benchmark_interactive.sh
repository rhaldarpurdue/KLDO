#!/bin/bash

MODEL_PATH=./data/models/kl-ma_llama-3.2-1b_5e-05_5/FINAL
MODEL_NAME=meta-llama/Llama-3.2-1B


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
OUTPUT_FILE="outputs/alpacaeval/$(echo "$MODEL_PATH" | tr '/' '_').json"
# OPENAI_API_KEY needs to be set
python -m eval.sample_for_alpacaeval --model_path "$MODEL_PATH" --model_name "$MODEL_NAME" --gpu_count $GPU_COUNT --output_file "$OUTPUT_FILE"
alpaca_eval evaluate --is_overwrite_leaderboard=True --model_outputs="$OUTPUT_FILE"
