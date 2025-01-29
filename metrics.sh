module load anaconda/2024.02-py311 
module load cudnn/cuda-12.1_8.9 
conda activate /depot/qfsong/LLM/env/halos

export TRANSFORMERS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/hub
export HF_HOME=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/
export HF_DATASETS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/datasets
export HUGGING_FACE_API_KEY=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

huggingface-cli login --token $HUGGING_FACE_API_KEY

base_model=ContextualAI/archangel_sft_llama7b #EleutherAI/pythia-1b  #mistralai/Mistral-7B-v0.1  #google/gemma-2-2b #Qwen/Qwen2.5-1.5B  #meta-llama/Llama-3.2-1B #ContextualAI/archangel_sft_llama7b 
base_path=./data/models
name=llama7b_sft_5e-05_5
alignments=(hdo_${name}_F kl-ma_${name}_F dpo_${name} kto_${name} bco_${name}_F)
anchors=("./dataset_generation/benign.txt" "./dataset_generation/harmful.txt")
output_dir=../experiments/final_paper_plots

python metrics.py --num_samples 2000\
    --viz \
    --base_model ${base_model}\
    --base_path ${base_path}\
    --alignments "${alignments[@]}"\
    --anchors "${anchors[@]}" \
    --output_dir ${output_dir}