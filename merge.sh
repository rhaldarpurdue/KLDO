module load anaconda/2024.02-py311 
module load cudnn/cuda-12.1_8.9 
conda activate /depot/qfsong/LLM/env/halos

base=ContextualAI/archangel_sft_llama7b
lora=./dpo_llama7b_sft_5e-05_5/step-5000/
out=./dpo_llama7b_sft_5e-05_5/FINAL/merged.pt

export TRANSFORMERS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/hub
export HF_HOME=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/
export HF_DATASETS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/datasets

python merge.py --base_model_path ${base} \
    --adapter_path ${lora} \
    --merged_model_path ${out}