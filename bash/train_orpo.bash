#!/bin/bash
accelerate launch \
    --config_file config/deepspeed_zero2.yaml \
    --num_processes 8 \
    script/spin/run_orpo.py \
    --beta 0.1 \
    --bf16 True \
    --learning_rate 8e-6 \
    --warmup_steps 150 \
    --attn_implementation flash_attention_2 \
    --optim rmsprop \
    --model_name_or_path $1 \
    --save_only_model \
    --dataset output_data/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e1_spin-b8.3p1b2-mtfull_hf/round_0.jsonl \
    --dataloader_pin_memory False \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --save_strategy epoch \
    --max_length 8192 \
    --max_prompt_length 7000 \
    --num_train_epochs 2 \
    --use_peft False \
    --report_to wandb \
    --gradient_checkpointing True \
    --dataset_num_proc 8 \
    --remove_unused_columns False \
    --output_dir $2 