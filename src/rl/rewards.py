"""Reward for ECG RL: binary + graded tag formatting + multi-label answer F1.

The opening <think> is scored only in non-explicit mode. With
explicit_thinking the opener is a fixed prompt prefix (consumed before
generation, SFT-masked) so the model never emits it; without it the model
must produce <think> itself and is rewarded for doing so. The closing
</think> and the <answer> block are always scored.

answer_reward uses F1 (2|p∩g| / (|p|+|g|)), not recall: recall is maxed by
emitting every plausible label, which F1 penalizes via precision.
"""
from __future__ import annotations

import re

_FMT_X = re.compile(r"^\s*[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$")
_FMT_NX = re.compile(r"^\s*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$")
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TAGS_X = ("</think>", "<answer>", "</answer>")
_TAGS_NX = ("<think>",) + _TAGS_X


def _labels(text: str) -> set[str]:
    m = _ANSWER.search(text)
    body = (m.group(1) if m else text).strip().lower()
    return {x.strip() for x in body.split(";") if x.strip()}


def format_reward(text: str, explicit_thinking: bool) -> float:
    return 1.0 if (_FMT_X if explicit_thinking else _FMT_NX).fullmatch(text) else 0.0


def tag_count_reward(text: str, explicit_thinking: bool) -> float:
    tags = _TAGS_X if explicit_thinking else _TAGS_NX
    return sum(text.count(t) == 1 for t in tags) / len(tags)


def answer_reward(text: str, gt: str) -> float:
    p, g = _labels(text), _labels(gt)
    return 2 * len(p & g) / max(len(p) + len(g), 1)


def reward_components(text: str, gt: str, explicit_thinking: bool = True) -> dict[str, float]:
    """Return decomposed reward terms for logging and debugging."""
    return {
        "format": format_reward(text, explicit_thinking),
        "tag_count": tag_count_reward(text, explicit_thinking),
        "answer": answer_reward(text, gt),
    }


def compute_reward(text: str, gt: str, explicit_thinking: bool = True) -> float:
    return sum(reward_components(text, gt, explicit_thinking).values())
