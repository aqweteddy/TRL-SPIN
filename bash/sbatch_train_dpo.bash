#!/bin/bash

#SBATCH --job-name=dpo_4096
#SBATCH --account=GOV112004
#SBATCH --partition=gpNCHC_H100
#SBATCH --output=O-%x.log
#SBATCH --nodes=2                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:8                # number of GPUs per node
#SBATCH --cpus-per-task=32         # number of cores per tasks

######################
### Set enviroment ###
######################
export GPUS_PER_NODE=8
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################
export DATASET="{'path':'json','data_files':'output_data/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e4_spin-b8.3p1b3-mtfull_hf/round_0.jsonl'}"
export MODEL=/home/u3844240/checkpoints/ft/llama-3-8b_p1_tv-e-llama_b8.3-patch1_e0-s1900/
export OUTPUT_MODEL_PATH="/home/u3844240/checkpoints/ft/llama3-8b_cp-p1_tv-llama3-emb-ft-b8.3patch1_spinmt-dpo-b8.3patch2-kto_pair-rand-4096/"

export LAUNCHER="accelerate launch \
    --config_file config/deepspeed_zero3.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --rdzv_backend c10d \
    "
# --rdzv_backend c10d \

export PYTHON_FILE="script/spin/run_dpo.py"


export SCRIPT_ARGS=" \
    --beta 0.1 \
    --bf16 True \
    --learning_rate 5e-7 \
    --warmup_ratio 0.1 \
    --loss_type kto_pair \
    --attn_implementation flash_attention_2 \
    --optim rmsprop \
    --model_name_or_path $MODEL \
    --save_only_model \
    --dataset_name $DATASET \
    --dataloader_pin_memory False \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --save_strategy epoch \
    --max_prompt_length 2048 \
    --max_length 4096 \
    --num_train_epochs 5 \
    --use_peft False \
    --report_to wandb \
    --gradient_checkpointing True \
    --dataset_num_proc 1 \
    --remove_unused_columns False \
    --output_dir $OUTPUT_MODEL_PATH"

# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $SCRIPT_ARGS" 
srun $CMD
#echo $CMD