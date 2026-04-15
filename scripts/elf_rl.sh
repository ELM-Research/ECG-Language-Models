#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7
NPROC=4
SYSTEM_PROMPT="src/dataloaders/system_prompts/system_prompt.txt"

COMMON_FLAGS=(
    --data_representation signal
    --batch_size 4
    --ref_global_bs 32
    --optimizer muon
    --beta1 0.9
    --beta2 0.95
    --epochs 10
    --muon_adamw_lr_ratio 0.1
    --lr_schedule cosine
    --grad_clip 1.0
    --llm_input_len 2048
    --system_prompt "$SYSTEM_PROMPT"
    --lr 1e-3
    --weight_decay 5e-2
    --llm qwen2.5-0.5b-instruct
    --data rl-ecg-r1
    --warmup 36793
    --torch_compile
    --distributed
    --wandb
)


uv run torchrun --standalone --nproc_per_node=$NPROC \
    src/main_trainer.py \
    "${COMMON_FLAGS[@]}" \
    --num_encoder_tokens 50 \
    --elm patch_elf \
    --train_phase rl \
    --rl_algo sapo \
    --rl_group_size 4
