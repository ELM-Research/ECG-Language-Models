"""Agnostic RL loss dispatcher. Each loss takes
(old_log_prob, log_prob, advantages, response_mask, **algo_kwargs, global_batch_size)
and returns (loss, metrics_dict)."""
from rl.sapo.sapo_loss import compute_policy_loss_sapo

RL_LOSSES = {"sapo": compute_policy_loss_sapo}


def get_rl_loss(name: str):
    if name not in RL_LOSSES:
        raise ValueError(f"Unknown RL loss '{name}'. Available: {list(RL_LOSSES)}")
    return RL_LOSSES[name]


def get_loss_kwargs(name: str, args) -> dict:
    if name == "sapo":
        return {"tau_pos": args.rl_tau_pos, "tau_neg": args.rl_tau_neg,
                "loss_agg_mode": args.rl_loss_agg_mode}
    return {}
