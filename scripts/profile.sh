CUDA_VISIBLE_DEVICES=0,1,2,4,6,7 uv run torchrun \
--standalone --nproc-per-node=6 scripts/profile_training.py \
--config src/elm/config/experiment/sft_stage1.yaml --steps 10