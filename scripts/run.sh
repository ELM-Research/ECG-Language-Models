# uv run python3 -m elm.train \
# --config src/elm/config/experiment/run_test.yaml


torchrun --nnodes=1 --standalone --nproc_per_node=8 uv run python3 -m elm.train \
--config src/elm/config/experiment/run_test.yaml

