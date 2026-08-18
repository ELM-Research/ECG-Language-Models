CUDA_VISIBLE_DEVICES=0,1,2,4,6,7 uv run torchrun \
--standalone --nproc-per-node=6 scripts/profile_training.py \
--config src/elm/config/experiment/rl.yaml --steps 10


CUDA_VISIBLE_DEVICES=0,1,2,4,6,7 uv run torchrun \
--standalone --nproc-per-node=6 scripts/profile_training.py \
--config src/elm/config/experiment/sft_stage1.yaml --steps 10


CUDA_VISIBLE_DEVICES=0,1,2,4,6,7 uv run torchrun \
--standalone --nproc-per-node=6 scripts/profile_training.py \
--config src/elm/config/experiment/sft_stage2.yaml --steps 10


CUDA_VISIBLE_DEVICES=0,1,2,4,6,7 uv run torchrun \
--standalone --nproc-per-node=6 scripts/profile_training.py \
--config src/elm/config/experiment/pretrain_stage1.yaml --steps 10



CUDA_VISIBLE_DEVICES=0,1,2,4,6,7 uv run torchrun \
--standalone --nproc-per-node=6 scripts/profile_training.py \
--config src/elm/config/experiment/pretrain_stage2.yaml --steps 10