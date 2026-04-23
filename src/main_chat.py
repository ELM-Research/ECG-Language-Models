"""
Interactive chat interface for trained ECG Language Models (ELMs).

Loads a trained ELM checkpoint and allows multi-turn conversation
with optional ECG signal input (.npy files).

Usage:
    uv run src/main_chat.py \
        --llm llama-3.2-1b-instruct \
        --encoder merl \
        --elm llava \
        --encoder_ckpt path/to/encoder.pt \
        --elm_ckpt path/to/elm_checkpoint.pt \
        --system_prompt src/dataloaders/system_prompts/system_prompt.txt \
        --data_representation signal
"""

import numpy as np
import torch

from configs.config import get_args
from configs.constants import HF_LLMS, SIGNAL_TOKEN_PLACEHOLDER
from utils.chat_template_manager import get_conv_template
from utils.gpu_manager import GPUSetup
from utils.seed_manager import set_seed
from elms.build_elm import BuildELM
from runners.helper import batch_to_device
from transformers import AutoTokenizer


def build_tokenizer(args):
    """Build and modify the LLM tokenizer (mirrors DatasetMixer.build_llm_tokenizer)."""
    llm_tokenizer = AutoTokenizer.from_pretrained(HF_LLMS[args.llm]["tokenizer"])
    if getattr(llm_tokenizer, "pad_token", None) is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    tokens_to_add = HF_LLMS[args.llm]["tokens_to_add"]
    tokens_to_add["additional_special_tokens"].append(SIGNAL_TOKEN_PLACEHOLDER)
    llm_tokenizer.add_special_tokens(tokens_to_add)
    return llm_tokenizer


def build_chat_template(args):
    """Build the chat template with the system prompt."""
    chat_template = get_conv_template(HF_LLMS[args.llm]["chat_template"])
    if HF_LLMS[args.llm]["system_prompt"] and args.system_prompt:
        with open(args.system_prompt, encoding="utf-8") as f:
            chat_template.set_system_message(f.read())
    return chat_template


def load_ecg_signal(ecg_path, args):
    """Load and normalize an ECG .npy file."""
    ecg_np = np.load(ecg_path, allow_pickle=True)
    if isinstance(ecg_np, np.ndarray) and ecg_np.dtype == object:
        ecg_np = ecg_np.item()
    if isinstance(ecg_np, dict):
        ecg_signal = ecg_np["ecg"][args.leads]
    else:
        ecg_signal = ecg_np[args.leads] if ecg_np.ndim == 2 else ecg_np
    # Normalize to [0, 1]
    eps = 1e-6
    min_val = np.min(ecg_signal)
    max_val = np.max(ecg_signal)
    ecg_signal = (ecg_signal - min_val) / (max_val - min_val + eps)
    ecg_signal = np.clip(ecg_signal, 0, 1)
    return torch.tensor(ecg_signal, dtype=torch.float32)


def prepare_generation_input(prompt_str, llm_tokenizer, ecg_tensor, args, device):
    """Tokenize the prompt and prepare the generation batch."""
    input_ids = llm_tokenizer.encode(prompt_str, add_special_tokens=False)
    attention_mask = [1] * len(input_ids)

    signal_token_id = llm_tokenizer.convert_tokens_to_ids(SIGNAL_TOKEN_PLACEHOLDER)
    signal_indices = [i for i, x in enumerate(input_ids) if x == signal_token_id]
    if not signal_indices:
        signal_indices = [-1]

    needs_signal = args.elm in ("llava", "base_elf", "patch_elf", "conv_elf")

    gen_batch = {
        "elm_input_ids": torch.tensor(input_ids, dtype=torch.int64).unsqueeze(0),
        "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32).unsqueeze(0),
    }

    if needs_signal and ecg_tensor is not None:
        gen_batch["encoder_tokenizer_out"] = {"ecg_signal": ecg_tensor.unsqueeze(0)}
        gen_batch["signal_id_indices"] = torch.tensor(signal_indices, dtype=torch.int64).unsqueeze(0)
    elif needs_signal:
        # No ECG loaded — set dummy signal indices so the model doesn't crash
        gen_batch["signal_id_indices"] = torch.tensor([-1], dtype=torch.int64).unsqueeze(0)

    gen_batch = {k: batch_to_device(v, device) for k, v in gen_batch.items()}
    return gen_batch, input_ids


def decode_response(input_ids, generated_ids, llm_tokenizer, args):
    """Extract the generated response text from output token ids."""
    generated_ids = generated_ids[0].cpu().tolist()
    # Slice off the prompt prefix if the model echoes it
    K = len(input_ids)
    if len(generated_ids) >= K and generated_ids[:K] == input_ids:
        cont = generated_ids[K:]
    else:
        cont = generated_ids

    # Trim at EOS
    wt = HF_LLMS[args.llm]["watch_tokens"]
    eos = set(wt["eos_token"].keys() if isinstance(wt["eos_token"], dict) else wt["eos_token"])
    fe = wt.get("final_eos_token", ())
    final_eos = set(fe.keys() if isinstance(fe, dict) else fe)
    stop_ids = eos | final_eos
    cut = next((i for i, t in enumerate(cont) if t in stop_ids), len(cont))
    cont = cont[:cut]

    return llm_tokenizer.decode(cont, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()


def print_banner():
    print("=" * 60)
    print("  ELM Chat Interface")
    print("=" * 60)
    print()
    print("Commands:")
    print("  /ecg <path>   Load an ECG signal (.npy file)")
    print("  /clear        Clear conversation history")
    print("  /quit         Exit")
    print()


def main():
    mode = "inference"
    args = get_args(mode)
    args.mode = mode
    if not args.data:
        args.data = ["ecg-qa-ptbxl-250-2500"]
    set_seed(args.seed)

    print("Loading tokenizer...")
    llm_tokenizer = build_tokenizer(args)

    print("Loading model...")
    build_elm = BuildELM(args)
    elm_components = build_elm.build_elm(llm_tokenizer)
    gpu_setup = GPUSetup(args)
    elm = gpu_setup.setup_gpu(elm_components["elm"], elm_components["find_unused_parameters"])
    elm.eval()
    device = next(elm.parameters()).device
    print(f"Model loaded on {device}.")

    chat_template = build_chat_template(args)
    ecg_tensor = None
    ecg_path_display = None
    needs_signal = args.elm in ("llava", "base_elm", "patch_elm", "base_elf", "patch_elf")

    signal_placeholder = "".join([SIGNAL_TOKEN_PLACEHOLDER] * args.num_encoder_tokens) + "\n"

    print_banner()

    conversation_messages = []
    turn_count = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "/quit":
            print("Exiting.")
            break

        if user_input.lower() == "/clear":
            conversation_messages = []
            turn_count = 0
            ecg_tensor = None
            ecg_path_display = None
            print("Conversation cleared.\n")
            continue

        if user_input.lower().startswith("/ecg "):
            path = user_input[5:].strip()
            try:
                ecg_tensor = load_ecg_signal(path, args)
                ecg_path_display = path
                print(f"ECG loaded: {path} (shape: {tuple(ecg_tensor.shape)})\n")
            except Exception as e:
                print(f"Error loading ECG: {e}\n")
            continue

        # Build the message with signal placeholder on first user turn
        message = user_input
        if turn_count == 0 and needs_signal:
            message = f"{signal_placeholder}{message}"

        conversation_messages.append({"role": "user", "content": message})
        turn_count += 1

        # Rebuild the full prompt from conversation history
        prompt = chat_template.copy()
        for msg in conversation_messages:
            role = prompt.roles[0] if msg["role"] == "user" else prompt.roles[1]
            prompt.append_message(role, msg["content"])
        # Add empty assistant turn to signal the model should generate
        prompt.append_message(prompt.roles[1], None)
        prompt_str = prompt.get_prompt()
        if getattr(args, "explicit_thinking", False):
            prompt_str += "<think>\n"

        # Generate
        with torch.no_grad():
            gen_batch, input_ids = prepare_generation_input(
                prompt_str, llm_tokenizer, ecg_tensor, args, device
            )
            gen_out = elm.generate(**gen_batch, max_new_tokens = 2048)
            response = decode_response(input_ids, gen_out, llm_tokenizer, args)

        stored = f"<think>\n{response}" if getattr(args, "explicit_thinking", False) else response
        conversation_messages.append({"role": "assistant", "content": stored})

        if ecg_path_display and turn_count == 1:
            print(f"[ECG: {ecg_path_display}]")
        print(f"ELM: {response}\n")


if __name__ == "__main__":
    main()