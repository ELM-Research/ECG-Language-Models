#!/usr/bin/env bash
set -euo pipefail

GPU=3,5
DATASET=mimic_iv_ecg
BENCHMARK_DATA=./ecg-reasoning-benchmark/data
ECG_DATA=../../data/mimic_iv
CONFIG=src/elm/config/experiment/ecg_reasoning.yaml
CHECKPOINT=/p01/whan/refactor/ECG-Language-Models/src/runs/sft_stage2/0/checkpoints/epoch_best
RESULTS=./results/mimic_orah_4b_base_no_rl_no_data
EVALUATION=./results/ecg_reasoning_evaluation
JUDGE=google/gemini-2.5-flash

CUDA_VISIBLE_DEVICES=$GPU uv run scripts/run_ecg_reasoning_bench.py "$BENCHMARK_DATA" \
    --dataset "$DATASET" \
    --ecg-base-dir "$ECG_DATA" \
    --output-dir "$RESULTS" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT"

# uv run scripts/openrouter_eval.py "$RESULTS" \
#     --dataset "$DATASET" \
#     --model ecglm \
#     --evaluator openrouter \
#     --openrouter-model "$JUDGE" \
#     --save-dir "$EVALUATION"
