#!/bin/bash
#SBATCH -A standby
#SBATCH --gpus-per-node=2
#SBATCH -C A100
#SBATCH --time=04:00:00
#SBATCH --job-name kl
#SBATCH --output ./log/%x_%j.out
#SBATCH --error ./log/%x_%j.err

module load anaconda/2024.02-py311 
module load cudnn/cuda-12.1_8.9 

conda activate /depot/qfsong/LLM/env/halos

loss=kl
datasets=[kl] #[shp,hh,oasst]
model=llama7b_sft
lr=5e-05
epochs=5
#exp_name=${loss}_${model}_${lr}_${epochs}
cache=./data/models
batch_size=8
optimizer=AdamW
gradient_accumulation=4
type=ma #ma, biased, f-div
exp_name=${loss}-${type}_${model}_${lr}_${epochs}

export TRANSFORMERS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/hub
export HF_HOME=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/
export HF_DATASETS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/datasets
export WANDB_API_KEY=8e4a0bf8aa276a6a441763aab7441d43ed309205
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
    ++loss.type=${type} \
    ++loss.normalize=true \
    ++intermediate_checkpoints=true