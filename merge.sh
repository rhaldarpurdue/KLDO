#!/bin/bash

module load anaconda/2024.02-py311 
module load cudnn/cuda-12.1_8.9 
conda activate /depot/qfsong/LLM/env/halos

base=google/gemma-2-2b    #EleutherAI/pythia-1b #mistralai/Mistral-7B-v0.1  #google/gemma-2-2b #Qwen/Qwen2.5-1.5B  #meta-llama/Llama-3.2-1B #ContextualAI/archangel_sft_llama7b 
lora=./data/models/hdo_gemma2-2b_5e-05_5_F/FINAL
out=./data/models/hdo_gemma2-2b_5e-05_5_F/FINAL/merged.pt

export TRANSFORMERS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/hub
export HF_HOME=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/
export HF_DATASETS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/datasets
export HUGGING_FACE_API_KEY=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

huggingface-cli login --token $HUGGING_FACE_API_KEY

python merge.py --base_model_path ${base} \
    --adapter_path ${lora} \
    --merged_model_path ${out}