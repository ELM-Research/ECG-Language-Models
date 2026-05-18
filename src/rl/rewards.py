"""Reward for ECG RL: correct <think>/<answer> tag formatting only.

Explicit thinking injects "<think>\n" as a fixed prompt prefix, so the
generation starts mid-thought and must emit only </think> then
<answer>...</answer> (a stray <think> means the opener was doubled).
Otherwise the model must produce the opening <think> tag itself.
"""
import re

_STRICT = re.compile(r"^\s*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$")
_EXPLICIT = re.compile(r"^\s*(?:(?!<think>)[\s\S])*?</think>\s*<answer>[\s\S]*?</answer>\s*$")


def compute_reward(text: str, explicit: bool) -> float:
    return 1.0 if (_EXPLICIT if explicit else _STRICT).fullmatch(text) else 0.0
