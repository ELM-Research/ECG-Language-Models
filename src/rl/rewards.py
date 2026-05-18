"""Reward for ECG RL: binary + graded tag formatting + multi-label answer F1.

The opening <think> is not rewarded: in explicit mode it is a fixed prompt
prefix (never generated, SFT-masked); in non-explicit mode a distilled base
emits it for free. So only the closing </think> and the <answer> block are
scored, identically for both modes.

answer_reward uses F1 (2|p∩g| / (|p|+|g|)), not recall: recall is maxed by
emitting every plausible label, which F1 penalizes via precision.
"""
import re

_FMT = re.compile(r"^\s*[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$")
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TAGS = ("</think>", "<answer>", "</answer>")


def _labels(text: str) -> set:
    m = _ANSWER.search(text)
    body = (m.group(1) if m else text).strip().lower()
    return {x.strip() for x in body.split(";") if x.strip()}


def format_reward(text: str) -> float:
    return 1.0 if _FMT.fullmatch(text) else 0.0


def tag_count_reward(text: str) -> float:
    return sum(text.count(t) == 1 for t in _TAGS) / len(_TAGS)


def answer_reward(text: str, gt: str) -> float:
    p, g = _labels(text), _labels(gt)
    return 2 * len(p & g) / max(len(p) + len(g), 1)


def compute_reward(text: str, gt: str) -> float:
    return format_reward(text) + tag_count_reward(text) + answer_reward(text, gt)
