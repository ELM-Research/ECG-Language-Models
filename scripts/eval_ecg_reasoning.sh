#!/usr/bin/env bash
# Two-step ECG-Reasoning-Benchmark pipeline for an ELM trained in this repo.
set -euo pipefail

DATASET=mimic_iv_ecg
RESULTS=./mimic_inference_results_3b
CKPT=src/runs/mlp_llava_qwen2.5-3b-instruct_st_mem/rl-ecg-r1/0/checkpoints/epoch_best.pt

# Step 1: run our model and curate per-sample responses under $RESULTS/ecglm/$DATASET/<dx>/*.json
CUDA_VISIBLE_DEVICES=4 uv run scripts/run_ecg_reasoning_bench.py ./ecg-reasoning-benchmark/data \
    --dataset "$DATASET" \
    --ecg-base-dir ../data/mimic_iv/ \
    --output-dir "$RESULTS" \
    --enable-condensed-chat \
    --llm qwen2.5-3b-instruct \
    --encoder st_mem \
    --elm mlp_llava \
    --num-encoder-tokens 50 \
    --explicit-thinking \
    --elm-ckpt "$CKPT"

# Step 2: score the saved responses with an OpenRouter LLM judge
OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}" \
uv run scripts/openrouter_eval.py "$RESULTS" \
    --dataset "$DATASET" \
    --model ecglm \
    --evaluator openrouter \
    --openrouter-model google/gemini-2.5-flash \
    --save-dir ./mimic_eval_results_3b