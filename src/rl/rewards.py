"""Minimal reward functions for ECG RL: format (think/answer tags) + answer overlap."""
import re

_FORMAT_RE = re.compile(r"^\s*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$")
_FORMAT_RE_THINK_PREFIX = re.compile(r"^\s*[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$")
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TAG_RE = re.compile(r"</?think>|</?answer>")
_TAGS = ("<think>", "</think>", "<answer>", "</answer>")


def _extract_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    return (m.group(1) if m else text).strip().lower()


def format_reward(text: str, has_think_prefix: bool = False) -> float:
    pat = _FORMAT_RE_THINK_PREFIX if has_think_prefix else _FORMAT_RE
    return 1.0 if pat.fullmatch(text) else 0.0


def tag_count_reward(text: str, has_think_prefix: bool = False) -> float:
    tags = _TAG_RE.findall(text)
    target = _TAGS[1:] if has_think_prefix else _TAGS
    return max(0.0, 1.0 - abs(len(tags) - len(target)) / len(target)) * len(set(tags) & set(target)) / len(target)


def answer_reward(text: str, gt: str) -> float:
    p = {x.strip() for x in _extract_answer(text).split(";") if x.strip()}
    g = {x.strip() for x in _extract_answer(gt).split(";") if x.strip()}
    return 2 * len(p & g) / max(len(p) + len(g), 1)


def compute_reward(text: str, gt: str, has_think_prefix: bool = False) -> float:
    return format_reward(text, has_think_prefix) + tag_count_reward(text, has_think_prefix) + answer_reward(text, gt)
