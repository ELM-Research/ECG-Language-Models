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
) -> torch.Tensor:
    """Compute the sequence-mean/token-mean SAPO policy loss."""
    if tau_pos <= 0 or tau_neg <= 0:
        raise ValueError(f"tau_pos and tau_neg must be > 0, got tau_pos={tau_pos}, tau_neg={tau_neg}")
    tau_pos = torch.as_tensor(tau_pos, dtype=advantages.dtype, device=advantages.device)
    tau_neg = torch.as_tensor(tau_neg, dtype=advantages.dtype, device=advantages.device)

    negative_approx_kl = (log_prob - old_log_prob).clamp(min=-20.0, max=20.0)
    ratio = negative_approx_kl.exp()

    taus = torch.where(advantages > 0, tau_pos, tau_neg)
    gate_probs = torch.sigmoid(taus * (ratio - 1.0))
    gates = gate_probs * (4.0 / taus)
    pg_losses = -gates * advantages
    token_counts = response_mask.sum(dim=-1)
    sequence_losses = (pg_losses * response_mask).sum(dim=-1) / (token_counts + 1e-8)
    global_batch_size = (token_counts > 0).sum() if global_batch_size is None else global_batch_size
    return sequence_losses.sum() * dp_size / global_batch_size
