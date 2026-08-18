CUDA_VISIBLE_DEVICES=3,5 uv run torchrun \
--standalone --nproc-per-node=6 scripts/profile_training.py \
--config src/elm/config/experiment/sft_stage1.yaml --steps 10