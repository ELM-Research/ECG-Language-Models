#!/usr/bin/env bash
set -euo pipefail

# CUDA_VISIBLE_DEVICES=0,1,2,4,6,7 uv run torchrun \
#   --standalone \
#   --nproc-per-node=6 \
#   --module elm.train \
#   --config "src/elm/config/experiment/pretrain_stage1.yaml"

# CUDA_VISIBLE_DEVICES=0,1,2,4,6,7 uv run torchrun \
#   --standalone \
#   --nproc-per-node=6 \
#   --module elm.train \
#   --config "src/elm/config/experiment/pretrain_stage2.yaml"


CUDA_VISIBLE_DEVICES=0,1,2,4,6,7 uv run torchrun \
  --standalone \
  --nproc-per-node=6 \
  --module elm.train \
  --config "src/elm/config/experiment/sft_stage2.yaml"
