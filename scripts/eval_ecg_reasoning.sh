CUDA_VISIBLE_DEVICES=4 uv run scripts/erb_minimal.py \
--erb-dir ./ecg-reasoning-benchmark \
--llm qwen2.5-3b-instruct \
--encoder st_mem \
--elm mlp_llava \
--elm-ckpt src/runs/mlp_llava_qwen2.5-3b-instruct_st_mem/rl-ecg-r1/0/checkpoints/epoch_best.pt \
-- ./ecg-reasoning-benchmark/data \
--dataset mimic_iv_ecg \
--ecg-base-dir ../data/mimic_iv/ \
--system-prompt src/dataloaders/system_prompts/system_prompt_think.txt \
--output-dir ./mimic_inference_results_3b \
--enable-condensed-chat



uv run ecg-reasoning-benchmark/evaluation.py ./mimic_inference_results_3b \
--dataset mimic_iv_ecg \
--model ecglm \
--evaluator gemini \
--gemini-model gemini-3-flash-preview \
--use-cache \
--save-cache \
--load-cache \
--save-cache-interval 1 \
--save-dir ./mimic_eval_results_3b