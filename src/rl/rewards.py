import re

_FMT_X = re.compile(r"^\s*[\s\S]*?</think>[\s\S]*?<answer>[\s\S]*?</answer>\s*$")
_FMT_NX = re.compile(r"^\s*<think>[\s\S]*?</think>[\s\S]*?<answer>[\s\S]*?</answer>\s*$")
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
