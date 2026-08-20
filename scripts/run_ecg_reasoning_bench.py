import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ecg-reasoning-benchmark"))

from ecg_reasoning_benchmark.inference import get_parser, main as run_benchmark
from ecg_reasoning_benchmark.models import BaseModel, register_model
from scipy import interpolate

from elm.config.load import load_config
from elm.data.build import DataBuilder
from elm.data.modality.text import chat_prompt
from elm.evaluation.evaluator import generate_response
from elm.model import build_model
from elm.utils.constants import ECG_TOKEN_PLACEHOLDER
from elm.utils.parallelism import configure_runtime, setup_model
from elm.utils.seed import set_seed

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_ONLY = (
    "Your response must be **ONLY** the full text of the selected option. Do not include any "
    "uncertainty, explanation, reasoning, or extra words."
)


def _format_question(turn: dict, include_options: bool) -> str:
    text = f"Question: {turn['question']}\n\n"
    if not include_options:
        return text
    intro = (
        "This question may have multiple correct answers from the following options:"
        if "select all possible leads" in turn["question"].lower()
        else "This question has one of the following options as the correct answer:"
    )
    return text + intro + "\n" + "".join(f"- {option}\n" for option in turn["options"]) + _ANSWER_ONLY


def _resample_ecg(ecg, orig_fs=500, target_fs=250):
    duration = len(ecg) / orig_fs
    source = np.linspace(0, duration, len(ecg), endpoint=True)
    target = np.linspace(0, duration, int(len(ecg) * target_fs / orig_fs), endpoint=True)
    return interpolate.interp1d(
        source, ecg, kind="cubic", axis=0, bounds_error=False, fill_value="extrapolate"
    )(target)


def _extract_answer(text: str) -> str:
    if "</think>" in text and "<think>" not in text:
        text = "<think>\n" + text
    answer, think = _ANSWER.search(text), _THINK.search(text)
    text = answer.group(1) if answer else (text[think.end():] if think else text)
    return re.sub(r"</?(?:think|answer)>", "", text).strip()


@register_model("ecglm")
class ELM(BaseModel):
    ecg_modality_base = "signal"

    def __init__(self, config: dict):
        configure_runtime(config)
        set_seed(config["seed"])
        self.config = config
        self.tokenizer = DataBuilder(config, training=False).build_llm_tokenizer()
        self.model = setup_model(build_model(config, self.tokenizer), config["gpu"]).eval()
        with open(config["system_prompt_path"], encoding="utf-8") as file:
            self.system_prompt = file.read()

    @classmethod
    def build_model(cls, config: str, checkpoint=None, **_):
        config = load_config(config)
        if checkpoint:
            config["model"]["checkpoint"] = checkpoint
        return cls(config)

    def _prepare_signal(self, signal: torch.Tensor) -> torch.Tensor:
        signal = torch.from_numpy(_resample_ecg(signal.T.cpu().numpy()).T).to(torch.float32)
        length = self.config["segment_length"]
        signal = signal[:, :length]
        return torch.nn.functional.pad(signal, (0, length - signal.shape[-1]))

    def get_response(self, conversation, enable_condensed_chat=False, verbose=False, **_) -> str:
        turns = conversation.get_turns_for_prompt()
        messages = [{"role": "system", "content": self.system_prompt}]
        ecg = None
        for index, turn in enumerate(turns):
            if turn["role"] == "model":
                messages.append({"role": "assistant", "content": turn["text"]})
                continue
            content = _format_question(turn, not enable_condensed_chat or index == len(turns) - 1)
            if "signal" in turn:
                ecg = self._prepare_signal(turn["signal"])
                content = ECG_TOKEN_PLACEHOLDER * self.config["model"]["num_ecg_tokens"] + "\n" + content
            messages.append({"role": "user", "content": content})

        prompt = chat_prompt(self.tokenizer, messages, self.config["explicit_thinking"])
        with torch.no_grad():
            output = generate_response(
                self.model,
                self.tokenizer.encode(prompt, add_special_tokens=False),
                ecg,
                self.tokenizer,
                self.config["evaluation"],
            )
        answer = _extract_answer(output)
        if verbose:
            print(f"Q: {turns[-1]['question']}\nA: {answer}\n")
        return answer


def main() -> None:
    parser = get_parser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    args.model = "ecglm"
    run_benchmark(args)


if __name__ == "__main__":
    main()
