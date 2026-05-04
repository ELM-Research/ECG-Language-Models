# uv run src/main_trainer.py \
# --data_representation signal \
# --data ecg-qa-ptbxl-250-2500 \
# --llm llama-3.2-3b-instruct \
# --encoder merl \
# --encoder_ckpt ../ecg_encoder/src/runs/pretrain/merl/0/checkpoints/epoch_best.pt \
# --elm llava \
# --dev

# uv run src/main_trainer.py \
# --data_representation signal \
# --data ecg-qa-ptbxl-250-2500 \
# --llm llama-3.2-3b-instruct \
# --encoder mtae \
# --elm llava \
# --encoder_ckpt ../ecg_encoder/src/runs/pretrain/mtae/0/checkpoints/epoch_best.pt \
# --dev

# uv run src/main_trainer.py \
# --data_representation signal \
# --data ecg-qa-ptbxl-250-2500 \
# --llm llama-3.2-3b-instruct \
# --encoder mlae \
# --elm llava \
# --encoder_ckpt ../ecg_encoder/src/runs/pretrain/mlae/0/checkpoints/epoch_best.pt


# uv run src/main_trainer.py \
# --data_representation signal \
# --data ecg-qa-ptbxl-250-2500 \
# --llm llama-3.2-3b-instruct \
# --encoder st_mem \
# --elm llava \
# --dev

# uv run src/main_trainer.py \
# --data_representation rgb \
# --data ecg-qa-ptbxl-250-2500 \
# --llm llama-3.2-3b-instruct \
# --encoder clip-vit-base-patch32 \
# --elm llava \
# --dev

# uv run src/main_trainer.py \
# --data_representation stacked_signal \
# --data ecg-qa-ptbxl-250-2500 \
# --llm llama-3.2-3b-instruct \
# --encoder vit-base-patch16-224-in21k \
# --elm llava \
# --dev

# uv run src/main_trainer.py \
# --data_representation symbolic \
# --data ecg-qa-ptbxl-250-2500 \
# --llm llama-3.2-3b-instruct \
# --ecg_tokenizer src/dataloaders/data_representation/bpe/ecg_byte_tokenizer_10000.pkl \
# --dev

export CUDA_VISIBLE_DEVICES=4,5,6,7
uv run torchrun --standalone --nproc_per_node=4 \
src/main_trainer.py \
--data_representation signal \
--data ecg-qa-ptbxl-250-2500 \
--llm llama-3.2-1b-instruct \
--elm fuyu \
--batch_size 24 \
--ref_global_bs 96 \
--optimizer adamw \
--lr 1e-2 \
--lr_schedule cosine \
--weight_decay 5e-2 \
--beta1 0.9 \
--beta2 0.95 \
--llm_input_len 1024 \
--epochs 10 \
--grad_clip 1.0 \
--num_workers 16 \
--parallel_strategy ddp \
--peft \
--torch_compile \
--wandb


export CUDA_VISIBLE_DEVICES=4,5,6,7
uv run torchrun --standalone --nproc_per_node=4 \
src/main_trainer.py \
--data_representation signal \
--data ecg-qa-ptbxl-250-2500 \
--llm llama-3.2-1b-instruct \
--elm fuyu \
--batch_size 24 \
--ref_global_bs 96 \
--optimizer adamw \
--lr 1e-3 \
--lr_schedule cosine \
--weight_decay 5e-2 \
--beta1 0.9 \
--beta2 0.95 \
--llm_input_len 1024 \
--epochs 10 \
--grad_clip 1.0 \
--num_workers 16 \
--parallel_strategy ddp \
--peft \
--torch_compile \
--wandb

export CUDA_VISIBLE_DEVICES=4,5,6,7
uv run torchrun --standalone --nproc_per_node=4 \
src/main_trainer.py \
--data_representation signal \
--data ecg-qa-ptbxl-250-2500 \
--llm llama-3.2-1b-instruct \
--elm fuyu \
--batch_size 24 \
--ref_global_bs 96 \
--optimizer adamw \
--lr 1e-2 \
--lr_schedule cosine \
--weight_decay 5e-2 \
--beta1 0.9 \
--beta2 0.95 \
--llm_input_len 1024 \
--epochs 10 \
--num_workers 16 \
--parallel_strategy ddp \
--peft \
--torch_compile \
--wandb

# uv run src/main_trainer.py \
# --data_representation signal \
# --data ecg-qa-mimic-iv-ecg-250-2500 \
# --llm llama-3.2-3b-instruct \
# --elm fuyu \
# --dev