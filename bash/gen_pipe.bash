#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python script/spin/generate_pipeline.py "{'path':'json','data_files':'data/b8.3-patch2_beta2.jsonl','split':'train[0%:12%]'}" --output_path output_data/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e1_spin-b8.3p1b2/round0_1.jsonl --model_path /home/u3844240/checkpoints/ft/llama-3-8b_p1_tv-e-llama_b8.3-patch1_e0-s1900/ &
CUDA_VISIBLE_DEVICES=1 python script/spin/generate_pipeline.py "{'path':'json','data_files':'data/b8.3-patch2_beta2.jsonl','split':'train[12%:25%]'}" --output_path output_data/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e1_spin-b8.3p1b2/round0_2.jsonl --model_path /home/u3844240/checkpoints/ft/llama-3-8b_p1_tv-e-llama_b8.3-patch1_e0-s1900/&
CUDA_VISIBLE_DEVICES=2 python script/spin/generate_pipeline.py "{'path':'json','data_files':'data/b8.3-patch2_beta2.jsonl','split':'train[25%:37%]'}" --output_path output_data/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e1_spin-b8.3p1b2/round0_3.jsonl --model_path /home/u3844240/checkpoints/ft/llama-3-8b_p1_tv-e-llama_b8.3-patch1_e0-s1900/&
CUDA_VISIBLE_DEVICES=3 python script/spin/generate_pipeline.py "{'path':'json','data_files':'data/b8.3-patch2_beta2.jsonl','split':'train[37%:50%]'}" --output_path output_data/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e1_spin-b8.3p1b2/round0_5.jsonl --model_path /home/u3844240/checkpoints/ft/llama-3-8b_p1_tv-e-llama_b8.3-patch1_e0-s1900/&
CUDA_VISIBLE_DEVICES=4 python script/spin/generate_pipeline.py "{'path':'json','data_files':'data/b8.3-patch2_beta2.jsonl','split':'train[50%:63%]'}" --output_path output_data/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e1_spin-b8.3p1b2/round0_6.jsonl --model_path /home/u3844240/checkpoints/ft/llama-3-8b_p1_tv-e-llama_b8.3-patch1_e0-s1900/&
CUDA_VISIBLE_DEVICES=5 python script/spin/generate_pipeline.py "{'path':'json','data_files':'data/b8.3-patch2_beta2.jsonl','split':'train[63%:75%]'}" --output_path output_data/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e1_spin-b8.3p1b2/round0_7.jsonl --model_path /home/u3844240/checkpoints/ft/llama-3-8b_p1_tv-e-llama_b8.3-patch1_e0-s1900/&
CUDA_VISIBLE_DEVICES=6 python script/spin/generate_pipeline.py "{'path':'json','data_files':'data/b8.3-patch2_beta2.jsonl','split':'train[75%:88%]'}" --output_path output_data/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e1_spin-b8.3p1b2/round0_8.jsonl --model_path /home/u3844240/checkpoints/ft/llama-3-8b_p1_tv-e-llama_b8.3-patch1_e0-s1900/&
CUDA_VISIBLE_DEVICES=7 python script/spin/generate_pipeline.py "{'path':'json','data_files':'data/b8.3-patch2_beta2.jsonl','split':'train[88%:]'}" --output_path output_data/llama3-8b_cp-p1_tv-llama3-emb_ft-b8.3patch1e1_spin-b8.3p1b2/round0_4.jsonl --model_path /home/u3844240/checkpoints/ft/llama-3-8b_p1_tv-e-llama_b8.3-patch1_e0-s1900/&
wait