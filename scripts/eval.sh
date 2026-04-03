# Single GPU, batched
CUDA_VISIBLE_DEVICES=0 uv run src/main_evaluator.py \
--data_representation signal \
--data ecg-qa-ptbxl-250-2500 \
--llm llama-3.2-1b-instruct \
--elm llava \
--peft \
--encoder st_mem \
--num_workers 4 \
--eval_batch_size 8 \
--system_prompt src/dataloaders/system_prompts/system_prompt.txt \
--elm_ckpt src/runs/llama-3.2-1b-instruct_st_mem/ecg-instruct-45k-250-2500/4/checkpoints/epoch_best.pt

# Multi-GPU, distributed + batched (uncomment to use)
# CUDA_VISIBLE_DEVICES=0,1 uv run -m torch.distributed.run \
# --nproc_per_node=2 \
# src/main_evaluator.py \
# --distributed \
# --data_representation signal \
# --data ecg-qa-ptbxl-250-2500 \
# --llm llama-3.2-1b-instruct \
# --elm llava \
# --peft \
# --encoder st_mem \
# --num_workers 4 \
# --eval_batch_size 8 \
# --system_prompt src/dataloaders/system_prompts/system_prompt.txt \
# --elm_ckpt src/runs/llama-3.2-1b-instruct_st_mem/ecg-instruct-45k-250-2500/4/checkpoints/epoch_best.pt
