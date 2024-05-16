#!/bin/bash

#SBATCH --job-name=kto-tv
#SBATCH --account=GOV112004
#SBATCH --partition=gpNCHC_H100
#SBATCH --output=O-%x.log
#SBATCH --nodes=2 # number of nodes
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

export LAUNCHER="accelerate launch \
    --config_file config/deepspeed_zero2.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_PROCID \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --rdzv_backend c10d \
    "
# --rdzv_backend c10d \
export DATASET="{'path':'json','data_files':'output_data/llama3-8b_cp-p1_tv-llama3-emb_spin-b8.3p3b1_hf/round_part00.jsonl'}"
export MODEL=/home/u3844240/checkpoints/chat_vector/llama-3-p1_tv-e-llama/
# export MODEL=/home/u3844240/checkpoints/ft/llama-3-8b_p1_tv-e-llama_b8.3-patch1_e0-s1900/
export OUTPUT_MODEL_PATH="/home/u3844240/checkpoints/ft/llama3-8b_cp-p1_tv-llama3-emb_spin-kto-b8.3p3b1/"

export PYTHON_FILE="script/spin/run_kto.py"

export SCRIPT_ARGS=" \
    --bf16 True \
    --learning_rate 1e-5 \
    --lr_scheduler_type=cosine \
    --warmup_ratio 0.1 \
    --attn_implementation flash_attention_2 \
    --optim rmsprop \
    --model_name_or_path $MODEL \
    --save_only_model \
    --dataset $DATASET \
    --dataloader_pin_memory False \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --save_strategy epoch \
    --max_length 8192 \
    --max_prompt_length 2048 \
    --num_train_epochs 3 \
    --use_peft False \
    --report_to wandb \
    --gradient_checkpointing True \
    --dataset_num_proc 16 \
    --remove_unused_columns False \
    --output_dir $OUTPUT_MODEL_PATH"

    # --max_prompt_length 6000 \

    # --neftune_noise_alpha  10 \
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="$LAUNCHER $PYTHON_FILE $SCRIPT_ARGS" 
srun $CMD
#echo $CMD