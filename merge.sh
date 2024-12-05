module load anaconda/2024.02-py311 
module load cudnn/cuda-12.1_8.9 
conda activate /depot/qfsong/LLM/env/halos

base=Qwen/Qwen2.5-1.5B
lora=./data/models/kl-ma-NoSCALE_qwen_5e-05_5/FINAL/
out=./data/models/kl-ma-NoSCALE_qwen_5e-05_5/FINAL/merged.pt

export TRANSFORMERS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/hub
export HF_HOME=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/
export HF_DATASETS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/datasets
export HUGGING_FACE_API_KEY=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

huggingface-cli login --token $HUGGING_FACE_API_KEY

python merge.py --base_model_path ${base} \
    --adapter_path ${lora} \
    --merged_model_path ${out}