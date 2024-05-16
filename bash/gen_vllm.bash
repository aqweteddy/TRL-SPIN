#!/bin/bash

DATA_PATH=output_data/llama3-8b_cp-p1_tv-llama3-emb_spin-b8.3p3b1_hf/round_part00.jsonl
MODEL_PATH=/home/u3844240/checkpoints/ft/llama-3-8b_p1_tv-e-llama_b8.3-patch1_e0-s1900/
OUTPUT_PATH=/home/u3844240/trl/output_data/llama3-8b_cp-p1_tv-llama3-emb_spin-b8.3p3b1_hf/
ROUND=0

CUDA_VISIBLE_DEVICES=0 python script/spin/generate_vllm.py "{'path':'json','data_files':'$DATA_PATH','split':'train[0%:12%]'}" --output_path $OUTPUT_PATH/round${ROUND}_1.jsonl --model_path $MODEL_PATH &
CUDA_VISIBLE_DEVICES=1 python script/spin/generate_vllm.py "{'path':'json','data_files':'$DATA_PATH','split':'train[12%:25%]'}" --output_path $OUTPUT_PATH/round${ROUND}_2.jsonl --model_path $MODEL_PATH&
CUDA_VISIBLE_DEVICES=2 python script/spin/generate_vllm.py "{'path':'json','data_files':'$DATA_PATH','split':'train[25%:37%]'}" --output_path $OUTPUT_PATH/round${ROUND}_3.jsonl --model_path $MODEL_PATH&
CUDA_VISIBLE_DEVICES=3 python script/spin/generate_vllm.py "{'path':'json','data_files':'$DATA_PATH','split':'train[37%:50%]'}" --output_path $OUTPUT_PATH/round${ROUND}_5.jsonl --model_path $MODEL_PATH&
CUDA_VISIBLE_DEVICES=4 python script/spin/generate_vllm.py "{'path':'json','data_files':'$DATA_PATH','split':'train[50%:63%]'}" --output_path $OUTPUT_PATH/round${ROUND}_6.jsonl --model_path $MODEL_PATH&
CUDA_VISIBLE_DEVICES=5 python script/spin/generate_vllm.py "{'path':'json','data_files':'$DATA_PATH','split':'train[63%:75%]'}" --output_path $OUTPUT_PATH/round${ROUND}_7.jsonl --model_path $MODEL_PATH&
CUDA_VISIBLE_DEVICES=6 python script/spin/generate_vllm.py "{'path':'json','data_files':'$DATA_PATH','split':'train[75%:88%]'}" --output_path $OUTPUT_PATH/round${ROUND}_8.jsonl --model_path $MODEL_PATH&
CUDA_VISIBLE_DEVICES=7 python script/spin/generate_vllm.py "{'path':'json','data_files':'$DATA_PATH','split':'train[88%:]'}" --output_path $OUTPUT_PATH/round${ROUND}_4.jsonl --model_path $MODEL_PATH&
wait