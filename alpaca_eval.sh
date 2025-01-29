#!/bin/bash


MODEL_PATH="./data/models/kto_mistral_5e-05_5/FINAL"
OUTPUT_FILE="./outputs/alpacaeval/kto_mistral_5e-05_5/generation.json"
OUT_PATH="./outputs/alpacaeval/kl-shift_mistral_5e-05_5_F/kto_mistral_5e-05_5"
REFERENCE_OUTPUTS="./outputs/alpacaeval/kl-shift_mistral_5e-05_5_F/generation2.json"


echo "Running on node: $(hostname)"
echo "Machine Rank: $SLURM_PROCID"

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

echo "Machine Rank: $SLURM_PROCID"

# accelerate launch -m lm_eval --model hf \
#   --model_args pretrained="$MODEL_PATH",tokenizer="$TOKENIZER_PATH",parallelize=True \
#   --tasks arc_easy \
#   --batch_size 4


# AlpacaEval
# Count the number of GPUs, default to 0 if nvidia-smi is not available
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l) || GPU_COUNT=0
# OPENAI_API_KEY needs to be set
export OPENAI_API_KEY=your_openai_key # replace your openAI_api key
time=$(date "+%Y%m%d-%H%M%S")
echo "Start time: $time"
# time=$(date "+%Y%m%d-%H%M%S")
# echo "Generation End time: $time"
alpaca_eval evaluate --is_overwrite_leaderboard=True --model_outputs="$OUTPUT_FILE" --reference_outputs="$REFERENCE_OUTPUTS" --output_path="$OUT_PATH"
time=$(date "+%Y%m%d-%H%M%S")
echo "Evaluation End time: $time"

