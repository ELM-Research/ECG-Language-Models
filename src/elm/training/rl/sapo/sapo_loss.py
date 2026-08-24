# Soft Adaptive Policy Adaptation https://arxiv.org/abs/2511.20347
import torch

def compute_policy_loss_sapo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    tau_pos: float = 1.0,
    tau_neg: float = 1.05,
    global_batch_size: int | None = None,
    dp_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    tau_pos = torch.as_tensor(tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(tau_neg, dtype=advantages.dtype, device=advantages.device)
    log_ratio = (log_prob - old_log_prob).clamp(min=-20.0, max=20.0)
    ratio = log_ratio.exp()
    taus = torch.where(advantages > 0, tau_pos, tau_neg)
    gate_probs = torch.sigmoid(taus * (ratio - 1.0))
    gates = gate_probs * (4.0 / taus)
    pg_losses = -gates * advantages
    token_counts = response_mask.sum(dim=-1)
    sequence_losses = (pg_losses * response_mask).sum(dim=-1) / (token_counts + 1e-8)
    global_batch_size = (token_counts > 0).sum() if global_batch_size is None else global_batch_size
    loss = sequence_losses.sum() * dp_size / global_batch_size
    return loss, ((ratio - 1.0 - log_ratio) * response_mask).sum()
