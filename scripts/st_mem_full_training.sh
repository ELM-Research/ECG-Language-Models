#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=8
SYSTEM_PROMPT="src/dataloaders/system_prompts/system_prompt.txt"


COMMON_FLAGS=(
    --data_representation signal
    --elm mlp_llava
    --encoder st_mem
    --beta1 0.9
    --beta2 0.95
    --grad_clip 1.0
    --llm_input_len 2048
    --num_encoder_tokens 50 \
    --parallel_strategy ddp
    --system_prompt "$SYSTEM_PROMPT"
    --llm qwen2.5-3b-instruct
    --gradient_checkpointing
    --wandb
)


# uv run torchrun --standalone --nproc_per_node=$NPROC \
#     src/main_trainer.py \
#     "${COMMON_FLAGS[@]}" \
#     --train_phase pretrain \
#     --data pretrain-agh10 \
#     --epochs 30 \
#     --update connector \
#     --optimizer adamw \
#     --lr 5e-4 \
#     --lr_schedule constant \
#     --weight_decay 0.01 \
#     --warmup 108 \
#     --batch_size 8 \
#     --encoder_ckpt ../ecg_encoder/src/runs/pretrain/st_mem/0/checkpoints/epoch_best.pt \
#     --grad_accum_steps 2 \
#     --num_workers 16 \
#     --torch_compile \
#     --ref_global_bs 128

# uv run torchrun --standalone --nproc_per_node=$NPROC \
#     src/main_trainer.py \
#     "${COMMON_FLAGS[@]}" \
#   --train_phase pretrain \
#   --data pretrain-agh9 pretrain-agh8 \
#   --update encoder connector llm \
#   --optimizer muon \
#   --lr 1e-3 \
#   --muon_adamw_lr_ratio 0.1 \
#   --weight_decay 0.05 \
#   --lr_schedule cosine \
#   --batch_size 8 \
#   --grad_accum_steps 4 \
#   --ref_global_bs 256 \
#   --num_workers 16 \
#   --epochs 5 \
#   --warmup 2355 \
#   --torch_compile \
#   --elm_ckpt src/runs/mlp_llava_qwen2.5-1.5b-instruct_st_mem/pretrain-agh10/0/checkpoints/epoch_best.pt


# uv run torchrun --standalone --nproc_per_node=$NPROC \
#     src/main_trainer.py \
#     "${COMMON_FLAGS[@]}" \
#   --train_phase sft \
#   --data ecg-qa-mimic-iv-ecg-250-2500 pretrain-mimic-250-2500 ecg-instruct-45k-250-2500 \
#   --llm qwen2.5-1.5b-instruct \
#   --update connector llm \
#   --optimizer adamw \
#   --lr 1e-4 \
#   --lr_schedule cosine \
#   --weight_decay 0.01 \
#   --batch_size 8 \
#   --grad_accum_steps 2 \
#   --ref_global_bs 128 \
#   --llm_input_len 2048 \
#   --epochs 3 \
#   --num_workers 16 \
#   --torch_compile \
#   --elm_ckpt src/runs/mlp_llava_qwen2.5-1.5b-instruct_st_mem/pretrain-agh9_pretrain-agh8/0/checkpoints/epoch_best.pt \
#   --warmup 18836



# uv run torchrun --standalone --nproc_per_node=$NPROC \
#     src/main_trainer.py \
#     "${COMMON_FLAGS[@]}" \
#   --train_phase sft \
#   --data ecg-qa-mimic-iv-ecg-250-2500 pretrain-mimic-250-2500 ecg-instruct-45k-250-2500 ecg-grounding-250-2500 ecg-instruct-ecg-r1 base-ecg-r1 ecg-qa-cot \
#   --update encoder connector llm \
#   --optimizer muon \
#   --lr 2e-4 \
#   --muon_adamw_lr_ratio 0.05 \
#   --lr_schedule cosine \
#   --weight_decay 0.05 \
#   --batch_size 8 \
#   --torch_compile \
#   --grad_accum_steps 4 \
#   --ref_global_bs 256 \
#   --llm_input_len 2048 \
#   --num_workers 16 \
#   --elm_ckpt src/runs/mlp_llava_qwen2.5-1.5b-instruct_st_mem/ecg-qa-mimic-iv-ecg-250-2500_pretrain-mimic-250-2500_ecg-instruct-45k-250-2500/0/checkpoints/epoch_best.pt \
#   --epochs 3 \
#   --warmup 15779


uv run torchrun --standalone --nproc_per_node=$NPROC \
    src/main_trainer.py \
    "${COMMON_FLAGS[@]}" \
  --train_phase rl \
  --data rl-ecg-r1 \
  --update connector llm \
  --optimizer muon \
  --lr 5e-5 \
  --muon_adamw_lr_ratio 0.1 \
  --lr_schedule cosine \
  --weight_decay 0.01 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --ref_global_bs 64 \
  --llm_input_len 2048 \
  --epochs 3 \
  --rl_algo sapo \
  --rl_group_size 4 \
  --rl_max_new_tokens 384 \
  --rl_temperature 0.8 \
  --rl_top_p 0.95 \
  --rl_tau_pos 1.0 \
  --rl_tau_neg 1.05 \
  --rl_loss_agg_mode seq-mean-token-mean \
  --elm_ckpt src/runs/mlp_llava_qwen2.5-1.5b-instruct_st_mem/ecg-qa-mimic-iv-ecg-250-2500_pretrain-mimic-250-2500_ecg-instruct-45k-250-2500_ecg-grounding-250-2500_ecg-instruct-ecg-r1_base-ecg-r1_ecg-qa-cot/0/checkpoints/epoch_best.pt \
  --num_workers 8