from types import SimpleNamespace

import torch

import rl.rollout as rollout
from rl.rollout import _log_prob_at_response, rollout_group
from rl.rewards import compute_reward, reward_components
from rl.sapo.sapo_loss import compute_policy_loss_sapo


class Policy(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits)
        self.generation_kwargs = {}

    def forward(self, elm_input_ids, **_):
        return SimpleNamespace(logits=self.logits.expand(elm_input_ids.shape[0], -1, -1))

    def generate(self, elm_input_ids, **kwargs):
        self.generation_kwargs = kwargs
        return torch.full((elm_input_ids.shape[0], 1), 2, device=elm_input_ids.device)


def test_sapo_on_policy_gradient_matches_unclipped_objective():
    log_prob = torch.zeros(1, 2, dtype=torch.double, requires_grad=True)
    advantages = torch.tensor([[2.0, -3.0]], dtype=torch.double)
    loss, _ = compute_policy_loss_sapo(
        log_prob.detach(), log_prob, advantages, torch.ones_like(log_prob),
        tau_pos=0.5, tau_neg=2.0,
    )
    loss.backward()
    torch.testing.assert_close(log_prob.grad, -advantages / 2)


def test_reward_components_preserve_total_in_both_thinking_modes():
    cases = [
        ("<think>x</think><answer>a; b</answer>", False),
        ("x</think><answer>a; b</answer>", True),
    ]
    for text, explicit in cases:
        parts = reward_components(text, text, explicit)
        assert parts == {"format": 1.0, "tag_count": 1.0, "answer": 1.0}
        assert compute_reward(text, text, explicit) == sum(parts.values())


def test_policy_scoring_matches_top_p_sampling_distribution():
    logits = torch.zeros(4, 4)
    logits[1] = torch.tensor([4.0, 3.0, 2.0, 1.0])
    logits[2] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    policy = Policy(logits)
    ids = torch.tensor([[0, 0, 1, 3]])
    log_prob = _log_prob_at_response(policy, ids, torch.ones_like(ids), None, {}, 2, 1.0, 0.7)
    torch.testing.assert_close(log_prob, torch.tensor([[-1.3132616, -0.3132617]]))
    assert log_prob.requires_grad


def test_rollout_disables_implicit_top_k(monkeypatch):
    monkeypatch.setitem(
        rollout.HF_LLMS, "test", {"watch_tokens": {"eos_token": {2: "eos"}}},
    )
    policy = Policy(torch.zeros(3, 4))
    batch = {
        "elm_input_ids": torch.tensor([[1, 1, 3]]),
        "elm_attention_mask": torch.ones(1, 3),
        "elm_labels": torch.tensor([[-100, -100, 3]]),
        "signal_id_indices": torch.tensor([[0]]),
        "encoder_tokenizer_out": {"x": torch.ones(1, 1)},
    }
    tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda *_args, **_kwargs: "")
    args = SimpleNamespace(
        llm="test", rl_group_size=2, rl_max_new_tokens=1, rl_temperature=1.0,
        rl_top_p=1.0, explicit_thinking=False,
    )
    rollout_group(policy, batch, 0, tokenizer, args)
    assert policy.generation_kwargs["top_k"] == 0
