CUDA_VISIBLE_DEVICES=3,5 uv run torchrun \
--standalone \
--nproc-per-node=2 \
--module elm.train \
--config src/elm/config/experiment/sft_stage2.yaml
