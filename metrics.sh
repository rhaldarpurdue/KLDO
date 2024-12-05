module load anaconda/2024.02-py311 
module load cudnn/cuda-12.1_8.9 
conda activate /depot/qfsong/LLM/env/halos

export TRANSFORMERS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/hub
export HF_HOME=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/
export HF_DATASETS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/datasets
export HUGGING_FACE_API_KEY=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

huggingface-cli login --token $HUGGING_FACE_API_KEY

base_model=Qwen/Qwen2.5-1.5B
base_path=./data/models
alignments=("kl-ma_qwen_5e-05_5" "dpo_qwen_5e-05_5" "kto_qwen_5e-05_5" "kl-ma-NoSCALE_qwen_5e-05_5" "bco_qwen_5e-05_5")
anchors=("./dataset_generation/benign.txt" "./dataset_generation/harmful.txt")
output_dir=../experiments/anchor_train

python metrics.py --num_samples 2000\
    --viz \
    --base_model ${base_model}\
    --base_path ${base_path}\
    --alignments "${alignments[@]}"\
    --anchors "${anchors[@]}" \
    --output_dir ${output_dir}