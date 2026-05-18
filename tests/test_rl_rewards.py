import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rl.rewards import compute_reward, reward_components


def test_non_explicit_rewards_start_and_end_thinking_and_answer_tags():
    gt = "<think>reference reasoning</think><answer>afib; pvc</answer>"
    text = "<think>sample reasoning</think><answer>afib; pvc</answer>"

    comp = reward_components(text, gt, explicit_thinking=False)

    assert comp == {"format": 1.0, "tag_count": 1.0, "answer": 1.0}
    assert compute_reward(text, gt, explicit_thinking=False) == 3.0


def test_explicit_rewards_end_thinking_and_answer_tags_without_start_think():
    gt = "reference reasoning</think><answer>afib; pvc</answer>"
    text = "sample reasoning</think><answer>afib</answer>"

    comp = reward_components(text, gt, explicit_thinking=True)

    assert comp["format"] == 1.0
    assert comp["tag_count"] == 1.0
    assert comp["answer"] == 2 / 3


def test_thinking_content_is_not_semantically_scored():
    gt = "reference reasoning</think><answer>afib</answer>"
    text = "completely different reasoning</think><answer>afib</answer>"

    assert reward_components(text, gt, explicit_thinking=True)["answer"] == 1.0
    assert compute_reward(text, gt, explicit_thinking=True) == 3.0


def test_missing_special_tags_get_partial_tag_reward_only():
    gt = "<think>reference reasoning</think><answer>afib</answer>"
    text = "<think>sample reasoning</think>afib"

    comp = reward_components(text, gt, explicit_thinking=False)

    assert comp["format"] == 0.0
    assert comp["tag_count"] == 0.5
    assert comp["answer"] == 0.0
