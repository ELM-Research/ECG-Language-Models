CUDA_VISIBLE_DEVICES=3 uv run scripts/erb_minimal.py \
  --erb-dir ./ecg-reasoning-benchmark \
  --llm qwen2.5-1.5b-instruct \
  --encoder st_mem \
  --elm mlp_llava \
  --elm-ckpt src/runs/mlp_llava_qwen2.5-1.5b-instruct_st_mem/rl-ecg-r1/0/checkpoints/epoch_best.pt \
  -- ./ecg-reasoning-benchmark/data \
  --dataset ptbxl \
  --ecg-base-dir ../data/ptb_xl/ \
  --output-dir ./results
