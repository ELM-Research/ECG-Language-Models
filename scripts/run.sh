#!/usr/bin/env bash
set -euo pipefail

config="${1:-src/elm/config/experiment/pretrain_stage1.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
num_gpus="${#gpu_ids[@]}"

uv run torchrun \
  --standalone \
  --nproc-per-node="${num_gpus}" \
  --module elm.train \
  --config "${config}"
