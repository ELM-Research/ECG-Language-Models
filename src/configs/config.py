import argparse
from configs.constants import Mode


def get_args(mode: Mode) -> argparse.Namespace:
    if mode not in {"train", "eval", "inference", "post_train", "analyze"}:
        raise ValueError(f"invalid mode: {mode}")

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    parser.add_argument("--dev", action="store_true", default=None, help="Development mode")
    if mode in {"train", "eval", "inference", "post_train"}:
        parser.add_argument("--ecg_tokenizer", type=str, default="None", help="Path to ECG Tokenizer")
        parser.add_argument("--augment_ecg", action="store_true", default=None, help="Choose whether you want to augment your ECG")
        parser.add_argument("--augment_rgb", action="store_true", default=None, help="Choose whether you want to augment your ECG Image")
        parser.add_argument("--segment_len", type=int, default=2500, help="ECG Segment Length")
        parser.add_argument("--data_representation", type=str, default=None, choices=["signal", "rgb", "symbolic", "stacked_signal"],
                            help="Please choose the representation of data you want to input into the neural network.")
        parser.add_argument("--perturb", type=str, default=None, choices=["noise", "zeros", "only_text"],
                            help="Please choose the perturbation you want to apply into the neural network.")
        parser.add_argument(
            "--data",
            type=str,
            nargs="+",
        )
        parser.add_argument("--scratch", action="store_true", default=False, help="Use randomly initialized LLM instead of pretrained weights")
        parser.add_argument("--leads", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], help="leads to use")
        parser.add_argument("--data_subset", type=float, default=None, help="Subset of data to use (between 0 and 1)")
        parser.add_argument("--encoder", type=str, default=None, help="Neural Network Encoder Model")
        parser.add_argument("--llm", type=str, default=None, help="Large Language Model")
        parser.add_argument("--elm", type=str, default=None, help="ECG Language Model")
        parser.add_argument("--peft", action="store_true", default=None, help="Use PEFT")
        parser.add_argument("--save_step", action="store_true", default=None, help="Save step wise")
        parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
        parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
        parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
        parser.add_argument("--encoder_ckpt", type=str, default=None, help="Path to the encoder checkpoint")
        parser.add_argument("--elm_ckpt", type=str, default=None, help="Path to the LLM checkpoint")
        parser.add_argument("--attention_type", type=str, default="sdpa", help="Attention Type")
        parser.add_argument("--num_encoder_tokens", type=int, default=1, help="Number of encoder tokens")
        parser.add_argument("--update", type=str, nargs="+", default=["connector", "llm"],
                            choices=["encoder", "connector", "llm"], help="Components to update (default: connector llm)")
        parser.add_argument("--output_hidden_states", action="store_true", default=False, help="Output hidden states")
        parser.add_argument("--system_prompt", type=str, default=None, help="Path to System Prompt")
        parser.add_argument("--fold", type=str, default="1", help="Data Fold Number")
        parser.add_argument("--num_workers", type=int, default=0, help="Please choose the num works for the dataloader")
        parser.add_argument("--wandb", action="store_true", default=None, help="Enable logging")
        parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
        parser.add_argument("--distributed", action="store_true", default=None, help="Enable distributed training")
        parser.add_argument("--torch_compile", action="store_true", default=None,
                            help="Torch compile the model (should really only be used during pretraining or large finetuning.)")
        parser.add_argument("--train_phase", type=str, default="sft", choices=["pretrain", "sft", "rl"],
                            help="Training phase: pretrain (raw text + bos/signal/eos, no chat template), sft (chat template), rl (sft + think/answer special tokens)")
        parser.add_argument("--llm_input_len", type=int, default=2048, help="LLM Input Sequence Length")
        parser.add_argument("--min_ecg_tokens_len", type=int, default=512, help="Minimum ECG token length to consider")
        parser.add_argument("--norm_eps", type=float, default=1e-6, help="Please choose the normalization epsilon")
    if mode == "train":
        parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "muon"], help="Optimizer type")
        parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
        parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
        parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
        parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
        parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
        parser.add_argument("--patience_delta", type=float, default=0.1, help="Delta for early stopping")
        parser.add_argument("--early_stopping", action="store_true", default=False, help="Enable early stopping")
        parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for optimizer")
        parser.add_argument("--beta2", type=float, default=0.99, help="Beta2 for optimizer")
        parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon for optimizer")
        parser.add_argument("--muon_momentum", type=float, default=0.95, help="Muon momentum")
        parser.add_argument("--muon_nesterov", action="store_true", default=True, help="Nesterov momentum for Muon")
        parser.add_argument("--muon_ns_steps", type=int, default=5, help="Newton-Schulz iteration steps")
        parser.add_argument("--muon_adamw_lr_ratio", type=float, default=0.015, help="AdamW LR as fraction of Muon LR")
        parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine", "inv_sqrt"], help="LR schedule after warmup")
        parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Min LR as fraction of peak LR (for cosine schedule)")
        parser.add_argument("--warmup", type=int, default=500, help="Warmup steps")
        parser.add_argument("--ref_global_bs", type=int, default=None)
        parser.add_argument("--grad_accum_steps", type=int, default=1)
        parser.add_argument("--grad_clip", type=float, default=0.0, help="Max gradient norm for clipping (0 to disable)")
        parser.add_argument("--scale_wd", type=str, default="none", choices=["none", "inv_sqrt", "inv_linear"])
        parser.add_argument("--resume_ckpt", type=str, default=None, help="Full training resume: restores model, optimizer, and LR schedule state")
        # RL (train_phase=rl) — agnostic policy-gradient pipeline
        parser.add_argument("--rl_algo", type=str, default="sapo", choices=["sapo"], help="RL policy-loss algorithm")
        parser.add_argument("--rl_group_size", type=int, default=4, help="G: rollouts per prompt for group-relative advantage")
        parser.add_argument("--rl_max_new_tokens", type=int, default=512, help="Max new tokens per rollout")
        parser.add_argument("--rl_temperature", type=float, default=1.0, help="Sampling temperature")
        parser.add_argument("--rl_top_p", type=float, default=1.0, help="Nucleus sampling p")
        parser.add_argument("--rl_tau_pos", type=float, default=1.0, help="SAPO temperature for positive advantages")
        parser.add_argument("--rl_tau_neg", type=float, default=1.05, help="SAPO temperature for negative advantages")
        parser.add_argument("--rl_loss_agg_mode", type=str, default="seq-mean-token-mean",
                            choices=["token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm", "seq-mean-token-mean"])

    if mode == "analyze":
        parser.add_argument("--json_dirs", type = str, nargs="+", default = None)
        parser.add_argument("--json_paths", type = str, nargs="+", default = None)
        parser.add_argument("--output_dir", type=str, default=".")
        parser.add_argument("--ckpt_type", type = str, default = "best")
        parser.add_argument("--stepwise_dirs", type=str, nargs="+", default=None)
        parser.add_argument("--checkpoint_dirs", type=str, nargs="+", default=None)
    return parser.parse_args()
