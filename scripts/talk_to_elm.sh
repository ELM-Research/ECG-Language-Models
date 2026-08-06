
CUDA_VISIBLE_DEVICES=0 \
uv run torchrun --standalone --nproc_per_node=1 \
src/main_chat.py \
--system_prompt src/dataloaders/system_prompts/system_prompt_original.txt \
--llm qwen3.5-0.8b \
--elm mlp_llava \
--encoder siglip-ecg \
--elm_ckpt ./src/runs/mlp_llava_qwen3.5-0.8b_siglip-ecg/ecg-qa-ptbxl-250-2500/1/checkpoints/epoch_best.pt \
--num_encoder_tokens 25


#data/ptb_xl/preprocessed_2500/records500_05000_05725_hr_0.npy
#Sinus rhythm. Normal ECG.

#data/ptb_xl/preprocessed_2500/records500_21000_21555_hr_0.npy
#Sinus rhythm with left ventricular pattern; moderate voltage criteria suggesting left ventricular hypertrophy; ST/T abnormalities possibly indicating anterior or lateral ischemia or left ventricular overload. Unconfirmed report.

#data/ptb_xl/preprocessed_2500/records500_18000_18812_hr_0.npy
#Atrial fibrillation/flutter with left-sided QRS/T abnormality; anteroseptal and inferior infarction; possible anterolateral ischemia or left ventricular overload. Unconfirmed findings.
