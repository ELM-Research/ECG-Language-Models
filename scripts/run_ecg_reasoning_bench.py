import argparse
import os
import re
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(REPO, "src"), os.path.join(REPO, "ecg-reasoning-benchmark")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs.constants import HF_LLMS, SIGNAL_TOKEN_PLACEHOLDER  # noqa: E402
from elms.build_elm import BuildELM  # noqa: E402
from main_chat import build_chat_template, build_tokenizer, prepare_generation_input  # noqa: E402
from ecg_reasoning_benchmark.inference import get_parser  # noqa: E402
from ecg_reasoning_benchmark.inference import main as run_inference  # noqa: E402
from ecg_reasoning_benchmark.models import BaseModel, register_model  # noqa: E402

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _id_set(entry) -> set[int]:
    """Token ids from a watch_tokens entry (an ``{id: str}`` dict or an id iterable)."""
    return set(entry.keys() if isinstance(entry, dict) else entry)


def extract_answer(text: str) -> str:
    """Return the final answer from a plain or ``<think>``/``<answer>`` RL response."""
    if "</think>" in text and "<think>" not in text:
        text = "<think>\n" + text  # explicit-thinking opener was consumed as the prompt prefix
    answer, think = _ANSWER.search(text), _THINK.search(text)
    text = answer.group(1) if answer else (text[think.end():] if think else text)
    return re.sub(r"</?(?:think|answer)>", "", text).strip()


@register_model("ecglm")
class ELM(BaseModel):
    ecg_modality_base = "signal"

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = build_tokenizer(args)
        self.template = build_chat_template(args)
        self.elm = BuildELM(args).build_elm(self.tokenizer)["elm"].to(self.device).eval()
        self.placeholder = SIGNAL_TOKEN_PLACEHOLDER * args.num_encoder_tokens + "\n"
        wt = HF_LLMS[args.llm]["watch_tokens"]
        self.stop_ids = _id_set(wt["eos_token"]) | _id_set(wt.get("final_eos_token", ()))

    @classmethod
    def build_model(cls, **kwargs) -> "ELM":
        args = argparse.Namespace(**kwargs)
        args.leads = list(range(12))
        args.attention_type = "sdpa"
        args.norm_eps = 1e-6
        args.scratch = args.peft = args.dev = args.distributed = args.gradient_checkpointing = False
        return cls(args)

    def _prepare_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """Match training preprocessing: decimate 500->250 Hz, fit segment_len, min-max to [0, 1]."""
        seg = self.args.segment_len
        signal = signal.to(torch.float32)
        signal = signal[:, :: max(1, signal.shape[-1] // seg)][:, :seg]
        lo, hi = signal.min(), signal.max()
        signal = ((signal - lo) / (hi - lo + self.args.norm_eps)).clamp(0, 1)
        if signal.shape[-1] < seg:  # only trips if a signal is shorter than segment_len; pad baseline
            signal = torch.nn.functional.pad(signal, (0, seg - signal.shape[-1]))
        return signal.flatten() if self.args.elm == "base_elf" else signal

    def _decode(self, prompt_ids: list[int], output: torch.Tensor) -> str:
        cont = output[0].cpu().tolist()
        if cont[: len(prompt_ids)] == prompt_ids:  # defensive: strip an echoed prompt
            cont = cont[len(prompt_ids):]
        cut = next((i for i, t in enumerate(cont) if t in self.stop_ids), len(cont))
        return self.tokenizer.decode(cont[:cut], skip_special_tokens=False, clean_up_tokenization_spaces=True)

    def get_response(self, conversation, enable_condensed_chat: bool = False, verbose: bool = False, **kwargs) -> str:
        turns = conversation.get_turns_for_prompt()
        prompt, signal = self.template.copy(), None
        for i, turn in enumerate(turns):
            if turn.get("role") == "model":
                prompt.append_message(prompt.roles[1], turn["text"])
                continue
            message = f"Question: {turn['question']}"
            if not enable_condensed_chat or i == len(turns) - 1:  # condensed chat: options on the live turn only
                message += "\nOptions:\n" + "\n".join(f"- {o}" for o in turn["options"])
                message += "\nRespond with exactly one of the options."
            if "signal" in turn:  # the ECG rides on the first (initial-diagnostic) user turn
                signal = self._prepare_signal(turn["signal"])
                message = self.placeholder + message
            prompt.append_message(prompt.roles[0], message)
        prompt.append_message(prompt.roles[1], None)

        prompt_str = prompt.get_prompt()
        if self.args.explicit_thinking:
            prompt_str += "<think>\n"
        with torch.no_grad():
            batch, prompt_ids = prepare_generation_input(prompt_str, self.tokenizer, signal, self.args, self.device)
            output = self.elm.generate(**batch, max_new_tokens=self.args.max_new_tokens)
        answer = extract_answer(self._decode(prompt_ids, output))
        if verbose:
            print(f"Q: {turns[-1]['question']}\nA: {answer}\n")
        return answer


def main() -> None:
    parser = get_parser()
    parser.add_argument("--llm", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--elm", required=True)
    parser.add_argument("--elm-ckpt", required=True)
    parser.add_argument("--encoder-ckpt", default=None)
    parser.add_argument("--system-prompt", default=os.path.join(REPO, "src/dataloaders/system_prompts/system_prompt_think.txt"))
    parser.add_argument("--num-encoder-tokens", type=int, default=50)
    parser.add_argument("--segment-len", type=int, default=2500)
    parser.add_argument("--update", nargs="+", default=["connector", "llm"], choices=["encoder", "connector", "llm"])
    parser.add_argument("--perturb", default=None, choices=["noise", "zeros", "only_text"])
    parser.add_argument("--train-phase", default="sft", choices=["pretrain", "sft", "rl"])
    parser.add_argument("--explicit-thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    args.model = "ecglm"
    run_inference(args)


if __name__ == "__main__":
    main()