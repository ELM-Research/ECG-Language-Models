"""Minimal reward functions for ECG RL: format (strict + graded) + answer F1."""
import re

_FORMAT_RE = re.compile(r"^\s*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TAGS = ("<think>", "</think>", "<answer>", "</answer>")


def _labels(text: str) -> set:
    m = _ANSWER_RE.search(text)
    body = (m.group(1) if m else text).strip().lower()
    return {x.strip() for x in body.split(";") if x.strip()}


def format_reward(text: str) -> float:
    return 1.0 if _FORMAT_RE.fullmatch(text) else 0.0


def tag_count_reward(text: str) -> float:
    return 0.25 * sum(text.count(t) == 1 for t in _TAGS)


def answer_reward(text: str, gt: str) -> float:
    p, g = _labels(text), _labels(gt)
    return 2 * len(p & g) / max(len(p) + len(g), 1)


def compute_reward(text: str, gt: str) -> float:
    return format_reward(text) + tag_count_reward(text) + answer_reward(text, gt)
