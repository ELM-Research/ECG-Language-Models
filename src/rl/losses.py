from __future__ import annotations

from typing import Any

import torch

from rl.sapo.sapo_loss import compute_policy_loss_sapo


def _shift_to_response(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if x.shape == target.shape:
        return x
    if x.ndim == target.ndim and x.shape[-1] == target.shape[-1] + 1:
        return x[..., 1:]
    raise ValueError(f"Incompatible shape {tuple(x.shape)} for target {tuple(target.shape)}")


def _expand_advantages(advantages: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if advantages.shape == target.shape:
        return advantages
    if advantages.ndim == target.ndim - 1 and advantages.shape == target.shape[:-1]:
        return advantages.unsqueeze(-1).expand_as(target)
    raise ValueError(f"Incompatible advantages shape {tuple(advantages.shape)} for {tuple(target.shape)}")


def _selected_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits, shift_labels = logits[:, :-1], labels[:, 1:]
    mask = shift_labels.ne(-100).float()
    safe_labels = shift_labels.masked_fill(shift_labels.eq(-100), 0)
    selected = torch.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return selected, mask


def compute_rl_loss(batch: dict[str, Any], model_out: Any, args: Any) -> tuple[torch.Tensor, dict[str, float]]:
    if args.rl_loss != "sapo":
        raise ValueError(f"Unsupported rl_loss={args.rl_loss}")
    log_prob, auto_mask = _selected_log_probs(model_out.logits, batch["elm_labels"])
    response_mask = batch.get("response_mask", auto_mask).float()
    old_log_prob = _shift_to_response(batch["old_log_prob"], log_prob)
    advantages = _expand_advantages(batch["advantages"], log_prob)
    if "response_mask" in batch:
        response_mask = _shift_to_response(response_mask, log_prob)
    return compute_policy_loss_sapo(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        rollout_is_weights=batch.get("rollout_is_weights"),
        loss_agg_mode=args.rl_loss_agg_mode,
        tau_pos=args.sapo_tau_pos,
        tau_neg=args.sapo_tau_neg,
        global_batch_size=args.batch_size * max(1, int(getattr(args, "world_size", 1))),
        dp_size=max(1, int(getattr(args, "world_size", 1))),
    )
