#!/usr/bin/env python3
"""Minimal bridge: run ECG-Language-Models on ECG-Reasoning-Benchmark inference.py."""

import argparse
import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from elms.build_elm import BuildELM
from main_chat import build_chat_template, build_tokenizer, decode_response, prepare_generation_input
from configs.constants import SIGNAL_TOKEN_PLACEHOLDER

def _pick_option(text: str, options: list[str]) -> str:
    norm = lambda s: " ".join(s.lower().split())
    t = norm(text)
    exact = {norm(o): o for o in options}
    if t in exact:
        return exact[t]
    for o in options:
        if norm(o) in t:
            return o
    if options and options == ["Yes", "No"]:
        if "yes" in t:
            return "Yes"
        if "no" in t:
            return "No"
    return options[0]


def _fit_signal_len(ecg: torch.Tensor | None, target_len: int | None) -> torch.Tensor | None:
    if ecg is None or target_len is None or ecg.shape[-1] == target_len:
        return ecg
    if ecg.shape[-1] > target_len:
        return ecg[..., :target_len]
    return torch.nn.functional.pad(ecg, (0, target_len - ecg.shape[-1]))


class BaseModel:
    ecg_modality_base = "signal"

    @classmethod
    def build_model(cls, model_variant=None):
        raise NotImplementedError

    def get_response(self, conversation, **kwargs) -> str:
        raise NotImplementedError

    @classmethod
    def require_base64_image(cls):
        return False


class ECGLMModel(BaseModel):
    ecg_modality_base = "signal"

    def __init__(self, **kwargs):
        cfg = SimpleNamespace(
            llm=kwargs["llm"],
            encoder=kwargs["encoder"],
            elm=kwargs["elm"],
            encoder_ckpt=kwargs.get("encoder_ckpt"),
            elm_ckpt=kwargs["elm_ckpt"],
            num_encoder_tokens=kwargs.get("num_encoder_tokens", 1),
            system_prompt=kwargs.get("system_prompt"),
            segment_len = kwargs.get("segment_len"),
            update = kwargs.get("update"),
            perturb = kwargs.get("perturb"),
            data_representation="signal",
            attention_type="sdpa",
            scratch=False,
            peft=False,
            dev=False,
            device=kwargs.get("device"),
            leads=list(range(12)),
            norm_eps=1e-6,
            explicit_thinking=False,
        )
        self.args = cfg
        self.tokenizer = build_tokenizer(cfg)
        self.prompt_template = build_chat_template(cfg)
        self.elm = BuildELM(cfg).build_elm(self.tokenizer)["elm"].eval()
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.elm.to(self.device)

    @classmethod
    def build_model(cls, **kwargs):
        return cls(**kwargs)

    def get_response(self, conversation, **kwargs) -> str:
        prompt = self.prompt_template.copy()
        ecg = None
        first_user = True
        for turn in conversation.conversation:
            if turn["role"] == "system":
                continue
            if turn["role"] == "user":
                msg = f"Question: {turn['question']}\nOptions:\n" + "\n".join(f"- {o}" for o in turn["options"])
                msg += "\nAnswer with exactly one option text."
                if first_user:
                    sig = f"{SIGNAL_TOKEN_PLACEHOLDER} " * self.args.num_encoder_tokens
                    msg = f"{sig.strip()}\n{msg}"
                    ecg = _fit_signal_len(turn.get("signal"), self.args.segment_len)
                    first_user = False
                prompt.append_message(prompt.roles[0], msg)
            else:
                prompt.append_message(prompt.roles[1], turn["text"])
        prompt.append_message(prompt.roles[1], None)
        prompt_str = prompt.get_prompt()
        with torch.no_grad():
            batch, in_ids = prepare_generation_input(prompt_str, self.tokenizer, ecg, self.args, self.device)
            out = self.elm.generate(**batch, max_new_tokens=kwargs.get("max_new_tokens", 1024))
        text = decode_response(in_ids, out, self.tokenizer, self.args)
        return _pick_option(text, conversation.conversation[-1]["options"])

    @classmethod
    def require_base64_image(cls):
        return False


def install_models_shim(default_model_kwargs=None):
    import types

    default_model_kwargs = default_model_kwargs or {}
    shim = types.ModuleType("models")

    def get_model_name(model):
        return "ecglm"

    def build_model(model_name: str, **kwargs):
        if model_name != "ecglm":
            raise ValueError(f"Only model='ecglm' is supported by this bridge, got: {model_name}")
        merged_kwargs = {**default_model_kwargs, **kwargs}
        return ECGLMModel.build_model(**merged_kwargs)

    shim.BaseModel = BaseModel
    shim.build_model = build_model
    shim.get_model_name = get_model_name
    sys.modules["models"] = shim


def install_erb_utils_module(erb_dir: str):
    utils_path = os.path.join(erb_dir, "utils.py")
    spec = importlib.util.spec_from_file_location("utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load ERB utils module from: {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["utils"] = module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--erb-dir", default=os.path.join(REPO_ROOT, "ecg-reasoning-benchmark"))
    parser.add_argument("--llm", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--elm", required=True)
    parser.add_argument("--elm-ckpt", required=True)
    parser.add_argument("--encoder-ckpt")
    parser.add_argument("--system-prompt", default=os.path.join(SRC_DIR, "dataloaders/system_prompts/system_prompt_think.txt"))
    parser.add_argument("--device")
    parser.add_argument("--segment_len", type=int, default=2500, help="ECG Segment Length")
    parser.add_argument("--num-encoder-tokens", type=int, default=1)
    parser.add_argument("--update", type=str, nargs="+", default=["connector", "llm"],
                            choices=["encoder", "connector", "llm"], help="Components to update (default: connector llm)")
    parser.add_argument("--perturb", type=str, default=None, choices=["noise", "zeros", "only_text"],
                            help="Please choose the perturbation you want to apply into the neural network.")
    args, passthrough = parser.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    default_model_kwargs = {
        "llm": args.llm,
        "encoder": args.encoder,
        "elm": args.elm,
        "elm_ckpt": args.elm_ckpt,
        "encoder_ckpt": args.encoder_ckpt,
        "system_prompt": args.system_prompt,
        "device": args.device,
        "num_encoder_tokens": args.num_encoder_tokens,
        "segment_len": args.segment_len,
        "update": args.update,
        "perturb": args.perturb,
    }

    if args.erb_dir not in sys.path:
        sys.path.insert(0, args.erb_dir)
    install_models_shim(default_model_kwargs=default_model_kwargs)
    install_erb_utils_module(args.erb_dir)
    from inference import get_parser, main as erb_main  # pylint: disable=import-error

    erb_args = get_parser().parse_args(passthrough)
    erb_args.model = "ecglm"
    for k in ["llm", "encoder", "elm", "elm_ckpt", "encoder_ckpt", "system_prompt",
              "device", "num_encoder_tokens", "segment_len", "update", "perturb"]:
        setattr(erb_args, k, getattr(args, k))
    erb_main(erb_args)


if __name__ == "__main__":
    main()
