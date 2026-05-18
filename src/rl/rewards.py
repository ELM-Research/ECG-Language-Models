"""Reward for ECG RL: correct </think> close + <answer> tag formatting.

The opening <think> is not rewarded. In explicit mode it is a fixed
prompt prefix (never generated, and SFT masks it); in non-explicit mode
a distilled base produces it for free. So the reward only checks that the
generation closes thinking and emits a well-formed answer block, which
works identically for both modes.
"""
import re

_FMT = re.compile(r"^\s*[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$")


def compute_reward(text: str) -> float:
    return 1.0 if _FMT.fullmatch(text) else 0.0
