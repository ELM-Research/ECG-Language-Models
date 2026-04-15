# 8xH100 Training Configurations (Pretrain → SFT → RL)

This plan is tailored to the current repo arguments (`src/configs/config.py`) and your dataset token counts.
Warmup is intentionally omitted from configs since you will set it as **10% of total optimizer steps**.

## Assumptions used for step estimates

- GPUs: `8 x H100`
- Distributed launch: `torchrun --nproc_per_node=8 --distributed`
- Sequence length: `--llm_input_len 2048`
- Approximate optimizer steps:

```
steps ~= ceil((dataset_tokens * epochs) / (global_batch_size * llm_input_len))
```

These are planning estimates; exact step counts can differ slightly due to filtering/padding/last-batch behavior.

---

## 1) Pretraining plan (with projection alignment first)

Dataset token counts:
- `pretrain-agh10`: 302,372
- `pretrain-agh9`: 3,976,704
- `pretrain-agh8`: 9,071,869
- `agh9 + agh8`: 13,048,573

### Stage P1 — Projection alignment only (agh10)

Goal: quickly align connector/projector to ECG latent space before opening full model updates.

- Data: `pretrain-agh10`
- Update scope: `--update connector`
- Suggested optimizer: `adamw`
- Suggested LR: `2e-4`
- Global batch size target: `128` (e.g., `batch_size=8`, `grad_accum_steps=2`, world=8)
- Epochs: `6`
- Estimated steps: `~7`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 src/main_trainer.py \
  --distributed \
  --train_phase pretrain \
  --data_representation signal \
  --data pretrain-agh10 \
  --llm qwen2.5-1.5b-instruct \
  --encoder merl \
  --elm llava \
  --update connector \
  --optimizer adamw \
  --lr 2e-4 \
  --lr_schedule cosine \
  --weight_decay 0.01 \
  --beta1 0.9 --beta2 0.95 \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --ref_global_bs 128 \
  --llm_input_len 2048 \
  --epochs 6 \
  --grad_clip 1.0 \
  --num_workers 16 \
  --wandb
```

### Stage P2 — Full ELM pretraining (agh9 + agh8)

Goal: train encoder + connector + LLM jointly after connector is aligned.

- Data: `pretrain-agh9 pretrain-agh8`
- Update scope: `--update encoder connector llm`
- Suggested optimizer: `muon`
- Suggested LR: `1e-3`
- Global batch size target: `256` (e.g., `batch_size=8`, `grad_accum_steps=4`, world=8)
- Epochs: `3`
- Estimated steps: `~75`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 src/main_trainer.py \
  --distributed \
  --train_phase pretrain \
  --data_representation signal \
  --data pretrain-agh9 pretrain-agh8 \
  --llm qwen2.5-1.5b-instruct \
  --encoder merl \
  --elm llava \
  --update encoder connector llm \
  --optimizer muon \
  --lr 1e-3 \
  --muon_adamw_lr_ratio 0.1 \
  --lr_schedule cosine \
  --weight_decay 0.05 \
  --beta1 0.9 --beta2 0.95 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --ref_global_bs 256 \
  --llm_input_len 2048 \
  --epochs 3 \
  --grad_clip 1.0 \
  --torch_compile \
  --num_workers 16 \
  --wandb
```

---

## 2) SFT plan (parallel curriculum similar to pretrain)

SFT total tokens: `395,717,230`

### Stage S1 — Alignment SFT on cleaner/core instruction sets

Use smaller/more structured sets first to stabilize instruction following before massive synthetic expansion.

- Data:
  - `ecg-qa-mimic-iv-ecg-250-2500`
  - `pretrain-mimic-250-2500`
  - `ecg-instruct-45k-250-2500`
- Tokens: `37,323,930`
- Update scope: `--update connector llm` (or `connector`-only for first few hundred steps if unstable)
- Optimizer: `adamw`
- LR: `1e-4`
- Global batch size: `128`
- Epochs: `3`
- Estimated steps: `~427`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 src/main_trainer.py \
  --distributed \
  --train_phase sft \
  --data_representation signal \
  --data ecg-qa-mimic-iv-ecg-250-2500 pretrain-mimic-250-2500 ecg-instruct-45k-250-2500 \
  --llm qwen2.5-1.5b-instruct \
  --encoder merl \
  --elm llava \
  --update connector llm \
  --optimizer adamw \
  --lr 1e-4 \
  --lr_schedule cosine \
  --weight_decay 0.01 \
  --beta1 0.9 --beta2 0.95 \
  --batch_size 8 \
  --grad_accum_steps 2 \
  --ref_global_bs 128 \
  --llm_input_len 2048 \
  --epochs 3 \
  --grad_clip 1.0 \
  --num_workers 16 \
  --wandb
```

### Stage S2 — Full SFT on all datasets

- Data:
  - `ecg-qa-mimic-iv-ecg-250-2500`
  - `pretrain-mimic-250-2500`
  - `ecg-instruct-45k-250-2500`
  - `ecg-grounding-250-2500`
  - `ecg-instruct-ecg-r1`
  - `base-ecg-r1`
  - `ecg-qa-cot`
- Tokens: `395,717,230`
- Update scope: `--update encoder connector llm`
- Optimizer: `muon`
- LR: `5e-4` (lower than pretrain to avoid catastrophic drift)
- Global batch size: `256`
- Epochs: `2`
- Estimated steps: `~1509`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 src/main_trainer.py \
  --distributed \
  --train_phase sft \
  --data_representation signal \
  --data ecg-qa-mimic-iv-ecg-250-2500 pretrain-mimic-250-2500 ecg-instruct-45k-250-2500 ecg-grounding-250-2500 ecg-instruct-ecg-r1 base-ecg-r1 ecg-qa-cot \
  --llm qwen2.5-1.5b-instruct \
  --encoder merl \
  --elm llava \
  --update encoder connector llm \
  --optimizer muon \
  --lr 5e-4 \
  --muon_adamw_lr_ratio 0.1 \
  --lr_schedule cosine \
  --weight_decay 0.05 \
  --beta1 0.9 --beta2 0.95 \
  --batch_size 8 \
  --grad_accum_steps 4 \
  --ref_global_bs 256 \
  --llm_input_len 2048 \
  --epochs 2 \
  --grad_clip 1.0 \
  --torch_compile \
  --num_workers 16 \
  --wandb
```

---

## 3) RL configuration (focus)

RL dataset tokens: `3,338,968` (`rl-ecg-r1` only).

For RL in this repo, effective per-step token use is dominated by generated rollouts, not only prompt tokens. So planning by epoch is still valid, but runtime/memory depends heavily on `--rl_group_size` and `--rl_max_new_tokens`.

### Recommended RL setup (stable on 8xH100)

- Data: `rl-ecg-r1`
- Train phase: `--train_phase rl`
- Algorithm: `--rl_algo sapo`
- `--rl_group_size 4`
- `--rl_max_new_tokens 256` (safer than 512 for throughput)
- `--rl_temperature 0.8`
- `--rl_top_p 0.95`
- `--rl_tau_pos 1.0 --rl_tau_neg 1.05`
- Update scope: `--update connector llm` (recommended initial RL); optionally unlock encoder late
- Optimizer: `muon`
- LR: `1e-4`
- Global batch size: `64` (e.g., `batch_size=2`, `grad_accum_steps=4`, world=8)
- Epochs: `6`
- Estimated prompt-token steps baseline: `~153`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run torchrun --standalone --nproc_per_node=8 src/main_trainer.py \
  --distributed \
  --train_phase rl \
  --data_representation signal \
  --data rl-ecg-r1 \
  --llm qwen2.5-1.5b-instruct \
  --encoder merl \
  --elm llava \
  --update connector llm \
  --optimizer muon \
  --lr 1e-4 \
  --muon_adamw_lr_ratio 0.1 \
  --lr_schedule cosine \
  --weight_decay 0.02 \
  --beta1 0.9 --beta2 0.95 \
  --batch_size 2 \
  --grad_accum_steps 4 \
  --ref_global_bs 64 \
  --llm_input_len 2048 \
  --epochs 6 \
  --grad_clip 1.0 \
  --rl_algo sapo \
  --rl_group_size 4 \
  --rl_max_new_tokens 256 \
  --rl_temperature 0.8 \
  --rl_top_p 0.95 \
  --rl_tau_pos 1.0 \
  --rl_tau_neg 1.05 \
  --rl_loss_agg_mode seq-mean-token-mean \
  --torch_compile \
  --num_workers 8 \
  --wandb
```

### Optional RL Stage R2 (if reward plateaus)

Run an additional short phase unlocking encoder with lower LR:

- `--update encoder connector llm`
- `--lr 5e-5`
- `--epochs 2`

This is usually enough to improve grounding without overfitting the RL dataset.

---

## 4) End-to-end sequence you requested

1. **Pretrain P1**: `pretrain-agh10`, connector only.
2. **Pretrain P2**: `pretrain-agh9 + pretrain-agh8`, full model updates.
3. **SFT S1**: smaller/core sets, connector+llm.
4. **SFT S2**: all SFT sets, full model updates.
5. **RL R1**: `rl-ecg-r1`, connector+llm with SAPO.
6. **RL R2 (optional)**: short full-model RL polish.

This gives you the same alignment-first philosophy for both pretraining and SFT, and keeps RL conservative/stable with one dataset.
