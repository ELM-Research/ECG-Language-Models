#!/bin/bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC=8
SYSTEM_PROMPT="src/dataloaders/system_prompts/system_prompt_think.txt"


COMMON_FLAGS=(
    --data_representation signal
    --elm mlp_llava
    --encoder siglip-ecg
    --beta1 0.9
    --beta2 0.95
    --grad_clip 1.0
    --llm_input_len 1024
    --num_encoder_tokens 100
    --distributed
    --system_prompt "$SYSTEM_PROMPT"
    --llm qwen3.5-2b-base
    --gradient_checkpointing
    --wandb
)

# uv run torchrun --standalone --nproc_per_node=$NPROC \
#     src/main_trainer.py \
#     "${COMMON_FLAGS[@]}" \
#   --train_phase pretrain \
#   --data pretrain-stage2 \
#   --update encoder connector llm \
#   --optimizer muon \
#   --lr 1e-3 \
#   --muon_adamw_lr_ratio 0.1 \
#   --weight_decay 0.05 \
#   --lr_schedule cosine \
#   --batch_size 4 \
#   --grad_accum_steps 8 \
#   --ref_global_bs 256 \
#   --num_workers 16 \
#   --epochs 3

# qwen3-0.6b  qwen3.5-2b
# qwen 3.5 4b max batch size qwen3.5 models no torch compile

# qwen3-0.6b only produces
# [transformers] Model config: bos_token_id must be `None` or an integer within the vocabulary (between 0 and 31999), got 49406. This may result in unexpected behavior.
# [transformers] Model config: eos_token_id must be `None` or an integer within the vocabulary (between 0 and 31999), got 49407. This may result in unexpected behavior.

# uv run torchrun --standalone --nproc_per_node=$NPROC \
#     src/main_trainer.py \
#     "${COMMON_FLAGS[@]}" \
#     --train_phase pretrain \
#     --data pretrain-stage1 \
#     --epochs 10 \
#     --update connector \
#     --optimizer adamw \
#     --lr 5e-4 \
#     --lr_schedule constant \
#     --weight_decay 0.01 \
#     --batch_size 8 \
#     --encoder_ckpt ../ecg_encoder/src/runs/pretrain/siglip2-base-patch16-naflex/2/checkpoints/epoch_best.pt \
#     --grad_accum_steps 2 \
#     --num_workers 16 \
#     --ref_global_bs 128

# uv run torchrun --standalone --nproc_per_node=$NPROC \
#     src/main_trainer.py \
#     "${COMMON_FLAGS[@]}" \
#   --train_phase pretrain \
#   --data pretrain-stage2 \
#   --update connector llm \
#   --optimizer muon \
#   --lr 1e-3 \
#   --muon_adamw_lr_ratio 0.1 \
#   --weight_decay 0.05 \
#   --lr_schedule cosine \
#   --batch_size 4 \
#   --grad_accum_steps 8 \
#   --ref_global_bs 256 \
#   --num_workers 16 \
#   --epochs 3 \
#   --elm_ckpt src/runs/mlp_llava_qwen3.5-2b-base_siglip-ecg/pretrain-stage1/0/checkpoints/epoch_best.pt


uv run torchrun --standalone --nproc_per_node=$NPROC \
    src/main_trainer.py \
    "${COMMON_FLAGS[@]}" \
  --train_phase sft \
  --data sft-stage1-noaug sft-stage1-aug \
  --update connector llm \
  --optimizer adamw \
  --lr 1e-4 \
  --lr_schedule cosine \
  --weight_decay 0.01 \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --ref_global_bs 128 \
  --epochs 3 \
  --num_workers 16 \
  --elm_ckpt src/runs/mlp_llava_qwen3.5-2b-base_siglip-ecg/pretrain-stage2/0/checkpoints/epoch_best.pt

### RESTSRT HERE QWEN 2.5 to 3.5 correction
uv run torchrun --standalone --nproc_per_node=$NPROC \
    src/main_trainer.py \
    "${COMMON_FLAGS[@]}" \
  --train_phase sft \
  --data sft-stage2-ptbxl-noaug sft-stage2-ptbxl-aug \
  --update encoder connector llm \
  --optimizer muon \
  --lr 2e-4 \
  --muon_adamw_lr_ratio 0.05 \
  --lr_schedule cosine \
  --weight_decay 0.05 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --ref_global_bs 256 \
  --num_workers 16 \
  --elm_ckpt src/runs/mlp_llava_qwen3.5-2b-base_siglip-ecg/sft-stage1-noaug_sft-stage1-aug/1/checkpoints/epoch_best.pt \
  --epochs 3


  uv run torchrun --standalone --nproc_per_node=$NPROC \
    src/main_trainer.py \
    "${COMMON_FLAGS[@]}" \
  --train_phase sft \
  --data sft-stage2-mimic-noaug sft-stage2-mimic-aug \
  --update encoder connector llm \
  --optimizer muon \
  --lr 2e-4 \
  --muon_adamw_lr_ratio 0.05 \
  --lr_schedule cosine \
  --weight_decay 0.05 \
  --batch_size 4 \
  --grad_accum_steps 8 \
  --ref_global_bs 256 \
  --num_workers 16 \
  --elm_ckpt src/runs/mlp_llava_qwen3.5-2b-base_siglip-ecg/sft-stage1-noaug_sft-stage1-aug/1/checkpoints/epoch_best.pt \
  --epochs 3

# uv run torchrun --standalone --nproc_per_node=$NPROC \
#     src/main_trainer.py \
#     "${COMMON_FLAGS[@]}" \
#   --train_phase rl \
#   --data rl-ecg-r1 \
#   --update connector llm \
#   --optimizer muon \
#   --lr 5e-5 \
#   --muon_adamw_lr_ratio 0.1 \
#   --lr_schedule cosine \
#   --weight_decay 0.01 \
#   --batch_size 1 \
#   --grad_accum_steps 8 \
#   --ref_global_bs 64 \
#   --epochs 3 \
#   --rl_algo sapo \
#   --rl_group_size 16 \
#   --rl_max_new_tokens 1024 \
#   --rl_temperature 0.8 \
#   --rl_top_p 0.95 \
#   --rl_tau_pos 1.0 \
#   --rl_tau_neg 1.05 \
#   --rl_loss_agg_mode seq-mean-token-mean \
#   --elm_ckpt src/runs/mlp_llava_qwen2.5-1.5b-instruct_st_mem/ecg-grounding-250-2500_ecg-instruct-ecg-r1_base-ecg-r1_ecg-qa-cot/1/checkpoints/epoch_best.pt \
#   --num_workers 8