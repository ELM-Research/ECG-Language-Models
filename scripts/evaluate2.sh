#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5 uv run python -m elm.evaluate \
--config src/elm/config/experiment/evaluate_sft_stage2_cot.yaml
