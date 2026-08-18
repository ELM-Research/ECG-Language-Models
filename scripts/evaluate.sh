#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=3 uv run python -m elm.evaluate \
--config src/elm/config/experiment/evaluate_rl.yaml
