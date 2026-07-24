uv run python3 -m elm.train \
--config src/elm/config/experiment/run_single_pretrain.yaml


uv run torchrun \
    --standalone \
    --nproc-per-node=8 \
    --module elm.train \
    --config src/elm/config/experiment/run_multi_pretrain.yaml
