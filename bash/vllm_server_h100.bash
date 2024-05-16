SINGULARITY=(
    singularity run
    --nv
    --bind $HF_HOME:$HF_HOME
    --bind ./:/workspace
    --pwd /workspace
    "/home/u3844240/checkpoints/singularity/vllm-0.3.3+cu118.sif"
)

ARGS=(
    --model "/home/u3844240/checkpoints/ft/llama-3-8b_p1_tv-e-llama_b8.3-patch1_e0-s1900/ "
    --tokenizer-mode auto
    --dtype bfloat16
    --tensor-parallel-size 8
    --pipeline-parallel-size 1
    --seed 42
    --served-model-name llama3-8b
)

${SINGULARITY[@]} python -m vllm.entrypoints.openai.api_server ${ARGS[@]}