# uv run python3 -m elm.test \
# --config src/elm/config/experiment/run_single_pretrain.yaml

# uv run python3 -m elm.test \
# --config src/elm/config/experiment/run_single_sft.yaml

# uv run python3 -m elm.test \
# --config src/elm/config/experiment/run_single_rl.yaml

# CUDA_VISIBLE_DEVICES=3 uv run python3 -m elm.train \
# --config src/elm/config/experiment/run_single_sft.yaml


CUDA_VISIBLE_DEVICES=3,5 uv run torchrun \
--standalone \
--nproc-per-node=2 \
--module elm.train \
--config src/elm/config/experiment/run_multi_sft.yaml
