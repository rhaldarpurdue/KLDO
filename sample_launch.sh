#!/bin/bash
#SBATCH -A standby
#SBATCH --gpus-per-node=2
#SBATCH -C A100-80GB
#SBATCH --time=04:00:00
#SBATCH --job-name Training
#SBATCH --output ./log/%x_%j.out
#SBATCH --error ./log/%x_%j.err
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16

module load anaconda/2024.02-py311 
module load cudnn/cuda-12.1_8.9 

conda activate /env_path/halos

loss=kl # choose between loss configs from ./config/loss/  
datasets=[cr] #[pref] #[shp,hh,oasst]
model=mistral #llama7b_sft #llama-3.2-1b #mistral #qwen #gemma2-2b #pythia #Can also define custom config in ./config/model/
lr=5e-05
epochs=5
cache=./data/models/
batch_size=8
optimizer=AdamW
gradient_accumulation=4
#type=ma #ma, biased, f-div
exp_name=${loss}_${model}_${lr}_${epochs}_${gradient_accumulation}

export TRANSFORMERS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/hub
export HF_HOME=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/
export HF_DATASETS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/datasets
export WANDB_API_KEY=8e4a0bf8aa276a6a441763aab7441d43ed309205
export HUGGING_FACE_API_KEY=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

huggingface-cli login --token $HUGGING_FACE_API_KEY

cd /depot/qfsong/LLM/scratch/rhaldar/HALOs
accelerate launch \
    --config_file accelerate_config/fsdp_2gpu.yaml \
    --main_process_port 29500 \
    launch.py loss=${loss} model=${model} datasets=${datasets} exp_name=${exp_name}\
	++cache_dir=${cache} \
    ++global_epochs=${epochs} \
	++lr=${lr} \
    ++model.use_peft=true \
    ++model.batch_size=${batch_size} \
    ++optimizer=${optimizer} \
    ++model.gradient_accumulation_steps=${gradient_accumulation} \
    ++intermediate_checkpoints=false