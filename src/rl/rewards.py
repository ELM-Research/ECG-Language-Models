"""Minimal reward functions for ECG RL: format (think/answer tags) + answer overlap."""
import re

_FORMAT_RE = re.compile(r"^\s*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    return (m.group(1) if m else text).strip().lower()


def format_reward(text: str) -> float:
    return 1.0 if _FORMAT_RE.fullmatch(text) else 0.0


def answer_reward(text: str, gt: str) -> float:
    r = {x.strip() for x in _extract_answer(text).split(";") if x.strip()}
    g = {x.strip() for x in _extract_answer(gt).split(";") if x.strip()}
    return len(r & g) / max(len(g), 1)


def compute_reward(text: str, gt: str) -> float:
    return format_reward(text) + answer_reward(text, gt)
