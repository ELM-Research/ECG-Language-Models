"""Convert one legacy Orah checkpoint to a self-contained Hugging Face repository."""

import argparse

import torch
from huggingface_hub import HfApi
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from elm.model import Orah
from elm.utils.constants import ECG_TOKEN_PLACEHOLDER


PREFIXES = {
    "llm.llm.": "llm.",
    "encoder.vision_encoder.vision_model.": "encoder.vision_model.",
    "projection.projection.": "connector.",
}


def convert_state_dict(checkpoint):
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    converted = {}
    for key, value in state.items():
        key = key.removeprefix("module.").removeprefix("_orig_mod.")
        for old, new in PREFIXES.items():
            if key.startswith(old):
                key = new + key[len(old):]
                break
        converted[key] = value
    return converted


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--llm", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--extra-token", action="append", default=[])
    parser.add_argument("--repo-id")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--num-ecg-tokens", type=int, default=100)
    parser.add_argument("--segment-length", type=int, default=2500)
    parser.add_argument("--patch-size", type=int, default=25)
    parser.add_argument("--num-leads", type=int, default=12)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.llm)
    tokenizer.add_tokens([ECG_TOKEN_PLACEHOLDER, *args.extra_token])
    llm = AutoModelForCausalLM.from_pretrained(args.llm)
    llm.resize_token_embeddings(len(tokenizer))
    model = Orah.from_components(
        llm, AutoModel.from_pretrained(args.encoder),
        tokenizer.convert_tokens_to_ids(ECG_TOKEN_PLACEHOLDER),
        num_ecg_tokens=args.num_ecg_tokens,
        segment_length=args.segment_length,
        patch_size=args.patch_size,
        num_leads=args.num_leads,
    )
    missing, unexpected = model.load_state_dict(
        convert_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)),
        strict=False,
    )
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    if args.repo_id:
        api = HfApi()
        api.create_repo(args.repo_id, private=args.private, exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, folder_path=args.output)


if __name__ == "__main__":
    main()
