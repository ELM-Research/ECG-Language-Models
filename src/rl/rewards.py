"""Reward for ECG RL: binary + graded tag formatting + answer score.

The opening <think> is scored only in non-explicit mode. With
explicit_thinking the opener is a fixed prompt prefix (consumed before
generation, SFT-masked) so the model never emits it; without it the model
must produce <think> itself and is rewarded for doing so. The closing
</think> and the <answer> block are always scored.

answer_reward blends a dense label-set F1 with an exact-match bonus:
0.5*F1 + 0.5*exact, in [0, 1]. The F1 term keeps a gradient toward
partially-correct answers (group-relative advantage is flat when every
sample misses, so a pure exact term gives no signal there), while the
exact term adds a distinct peak so the optimum is the precise answer, not
the recall-maxing "emit every plausible label". F1 is order/duplicate
invariant; the exact term is order- and spacing-sensitive.
"""
import re

_FMT_X = re.compile(r"^\s*[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$")
_FMT_NX = re.compile(r"^\s*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$")
_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_TAGS_X = ("</think>", "<answer>", "</answer>")
_TAGS_NX = ("<think>",) + _TAGS_X


def _answer_body(text: str) -> str:
    m = _ANSWER.search(text)
    return (m.group(1) if m else text).strip().lower()


def _labels(text: str) -> set:
    return {x.strip() for x in _answer_body(text).split(";") if x.strip()}


def format_reward(text: str, explicit_thinking: bool) -> float:
    return 1.0 if (_FMT_X if explicit_thinking else _FMT_NX).fullmatch(text) else 0.0


def tag_count_reward(text: str, explicit_thinking: bool) -> float:
    tags = _TAGS_X if explicit_thinking else _TAGS_NX
    return sum(text.count(t) == 1 for t in tags) / len(tags)


def answer_reward(text: str, gt: str) -> float:
    p, g = _labels(text), _labels(gt)
    f1 = 2 * len(p & g) / max(len(p) + len(g), 1)
    exact = 1.0 if _answer_body(text) == _answer_body(gt) else 0.0
    return 0.5 * f1 + 0.5 * exact


def compute_reward(text: str, gt: str, explicit_thinking: bool = True) -> float:
    return (format_reward(text, explicit_thinking)
            + tag_count_reward(text, explicit_thinking)
            + answer_reward(text, gt))