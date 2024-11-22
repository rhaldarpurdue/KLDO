#!/bin/bash
#SBATCH --job-name=kto
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH -C A100
#SBATCH --time=04:00:00
#SBATCH --output=./log/%x_%j.out
#SBATCH --error=./log/%x_%j.err

# Function to find an available port
find_free_port() {
    local port
    while true; do
        # Generate a random port number between 20000 and 65000
        port=$(shuf -i 29500-29510 -n 1)
        # Check if the port is in use
        if ! netstat -tuln | grep -q ":$port "; then
            echo "$port"
            break
        fi
    done
}

# Function to initialize the environment and print diagnostic information
# very important that this is run within srun for training to work!!!
init_env() {
    # Load necessary modules (adjust as needed for your system)
    module load anaconda/2024.02-py311 
    module load cudnn/cuda-12.1_8.9 

    conda activate /depot/qfsong/LLM/env/halos

    echo "Running on node: $(hostname)"
    echo "Machine Rank: $SLURM_PROCID"
    
    export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
    export MASTER_PORT=$(find_free_port | tr -d '\n')
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1

    export TRANSFORMERS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/hub
    export HF_HOME=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/
    export HF_DATASETS_CACHE=/depot/qfsong/LLM/scratch/rhaldar/hf_cache/datasets
    export WANDB_API_KEY=8e4a0bf8aa276a6a441763aab7441d43ed309205

    cd /depot/qfsong/LLM/scratch/rhaldar/HALOs
    
    echo "Master node: $MASTER_ADDR"
    echo "Number of nodes: $SLURM_JOB_NUM_NODES"
    echo "GPUs per node: $SLURM_GPUS_PER_NODE"
}

export -f find_free_port
export -f init_env

loss=kto
datasets=[kl] #[shp,hh,oasst]
model=llama7b_sft
lr=5e-05
epochs=5
exp_name=${loss}_${model}_${lr}_${epochs}
cache=./data/models
batch_size=8
optimizer=AdamW
gradient_accumulation=4




# Run the training script using srun
srun --jobid=$SLURM_JOB_ID --nodes=$SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "
init_env
accelerate launch \
    --config_file accelerate_config/fsdp_2x2gpu.yaml \
    --machine_rank \$SLURM_PROCID \
    --main_process_ip \$MASTER_ADDR \
    --main_process_port \$MASTER_PORT \
    launch.py loss=${loss} model=${model} datasets=${datasets} exp_name=${exp_name}\
	++cache_dir=${cache} \
    ++global_epochs=${epochs} \
	++lr=${lr} \
    ++model.use_peft=true \
    ++model.batch_size=${batch_size} \
    ++optimizer=${optimizer} \
    ++model.gradient_accumulation_steps=${gradient_accumulation}
"
