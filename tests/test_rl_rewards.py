import re

from rl.rewards import compute_reward, format_reward


def test_regression_legacy_format_rejects_summary_but_current_format_accepts_it():
    response = (
        "<think>ECG reasoning</think>\n\n"
        "The ECG demonstrates sinus rhythm.\n\n"
        "<answer>Sinus rhythm</answer>"
    )

    legacy_format = re.compile(
        r"^\s*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*$"
    )

    assert legacy_format.fullmatch(response) is None

    assert format_reward(response, explicit_thinking=False) == 1.0
    assert compute_reward(response, response, explicit_thinking=False) == 3.0


def test_explicit_thinking_accepts_clinical_summary_between_sections():
    response = (
        "ECG reasoning</think>\n\n"
        "The ECG demonstrates sinus rhythm.\n\n"
        "<answer>Sinus rhythm</answer>"
    )

    assert format_reward(response, explicit_thinking=True) == 1.0


def test_format_reward_still_accepts_response_without_summary():
    response = "<think>ECG reasoning</think>\n<answer>Sinus rhythm</answer>"

    assert format_reward(response, explicit_thinking=False) == 1.0


def test_format_reward_rejects_missing_think_close():
    response = "<think>ECG reasoning\n<answer>Sinus rhythm</answer>"

    assert format_reward(response, explicit_thinking=False) == 0.0


def test_format_reward_rejects_text_after_answer():
    response = (
        "<think>ECG reasoning</think>\n"
        "Clinical summary\n"
        "<answer>Sinus rhythm</answer> trailing text"
    )

    assert format_reward(response, explicit_thinking=False) == 0.0
