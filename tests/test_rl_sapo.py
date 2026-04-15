import torch

from src.rl.rewards import format_reward
from src.rl.sapo.sapo_loss import compute_policy_loss_sapo


def test_format_reward_requires_clean_think_answer_structure():
    good = "<think>a</think>\n<answer>x;y</answer>"
    bad_prefix = "note\n<think>a</think><answer>x;y</answer>"
    bad_suffix = "<think>a</think><answer>x;y</answer>\nextra"
    assert format_reward(good) == 1.0
    assert format_reward(bad_prefix) == 0.0
    assert format_reward(bad_suffix) == 0.0


def test_sapo_raises_for_non_positive_temperature():
    t = torch.zeros((1, 2), dtype=torch.float32)
    ones = torch.ones_like(t)
    try:
        compute_policy_loss_sapo(
            old_log_prob=t,
            log_prob=t,
            advantages=ones,
            response_mask=ones,
            tau_pos=0.0,
            tau_neg=1.0,
            global_batch_size=1,
        )
        assert False, "expected ValueError"
    except ValueError:
        assert True
