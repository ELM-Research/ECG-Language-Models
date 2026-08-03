"""Group rollout: per-prompt, sample G trajectories, score, build tensors for policy-loss computation."""
import torch

from elm.configs.constants import HF_LLMS
from elm.training.rl.rewards import reward_components


def _unwrap(m):
    m = getattr(m, "_orig_mod", m)
    return m.module if hasattr(m, "module") else m


def _eos_set(llm_name: str) -> set:
    wt = HF_LLMS[llm_name]["watch_tokens"]
    eos = set(wt["eos_token"].keys() if isinstance(wt["eos_token"], dict) else wt["eos_token"])
    fe = wt.get("final_eos_token", ())
    return eos | set(fe.keys() if isinstance(fe, dict) else fe)


def _trim_mask(new_tokens: torch.Tensor, eos_ids: set, pad_id: int | None = None) -> torch.Tensor:
    """Return (G, L) mask = 1 up to and including first EOS per row, 0 after.

    Also zeros pad positions so a row that never emits an EOS (ran to
    max_new_tokens) cannot leak right-padding into the policy loss.
    """
    G, L = new_tokens.shape
    mask = torch.ones(G, L, dtype=torch.float32, device=new_tokens.device)
    toks = new_tokens.tolist()
    for i in range(G):
        for j in range(L):
            if toks[i][j] in eos_ids:
                mask[i, j + 1:] = 0
                break
    if pad_id is not None and pad_id not in eos_ids:
        mask *= (new_tokens != pad_id).float()
    return mask


def _decode_for_reward(tokenizer, ids: torch.Tensor, strip_ids: set) -> str:
    kept = [int(t) for t in ids.tolist() if int(t) not in strip_ids]
    return tokenizer.decode(kept, skip_special_tokens=False).strip()


def _log_prob_at_response(model, ids, attn, ecg, pL: int, temperature: float) -> torch.Tensor:
    was_training = model.training
    model.eval()
    try:
        out = model(input_ids=ids, attention_mask=attn, ecg_values=ecg)
    finally:
        model.train(was_training)
    targets = ids[:, pL:]
    logits = out.logits[:, pL - 1:-1, :] / temperature
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def rollout_group(model, batch: dict, item_idx: int, tokenizer, args) -> dict:
    """Sample G responses for one prompt, compute rewards, advantages, and old log-probs."""
    base = _unwrap(model)
    device = batch["input_ids"].device
    G = args.rl_group_size

    eos_ids = _eos_set(args.llm)
    strip_ids = eos_ids | {int(tokenizer.pad_token_id)}

    labels = batch["labels"][item_idx]
    nz = (labels != -100).nonzero(as_tuple=True)[0]
    if nz.numel() == 0:
        raise ValueError("No response tokens found (labels all -100).")
    rs = nz[0].item()
    gt_text = _decode_for_reward(tokenizer, labels[nz], strip_ids)

    prompt_ids = batch["input_ids"][item_idx, :rs]
    prompt_attn = batch["attention_mask"][item_idx, :rs]
    pL = prompt_ids.shape[0]

    pb = {
        "input_ids": prompt_ids.unsqueeze(0).expand(G, -1).contiguous(),
        "attention_mask": prompt_attn.unsqueeze(0).expand(G, -1).contiguous(),
        "ecg_values": batch["ecg_values"][item_idx:item_idx + 1].expand(
            G, *batch["ecg_values"].shape[1:]).contiguous(),
    }

    was_training = base.training
    try:
        base.eval()
        with torch.no_grad():
            gen = base.generate(**pb, max_new_tokens=args.rl_max_new_tokens,
                                do_sample=True, temperature=args.rl_temperature,
                                top_p=args.rl_top_p, top_k=0)

        new_tokens = gen[:, pL:] if gen.shape[1] > pL and torch.equal(gen[0, :pL], prompt_ids) else gen
        if new_tokens.shape[1] == 0:                                 # pathological: nothing generated
            new_tokens = torch.full((G, 1), int(tokenizer.pad_token_id), dtype=torch.long, device=device)

        resp_mask = _trim_mask(new_tokens, eos_ids, int(tokenizer.pad_token_id))  # (G, gen_len)

        reward_parts = [
            reward_components(_decode_for_reward(tokenizer, new_tokens[i][resp_mask[i].bool()], strip_ids),
                              gt_text, getattr(args, "explicit_thinking", False))
            for i in range(G)
        ]
        rewards = torch.tensor([sum(parts.values()) for parts in reward_parts], dtype=torch.float32, device=device)
        reward_std = rewards.std(unbiased=False)
        # All G samples scored identically: group-relative advantage is pure
        # 1e-6-scaled noise. Flag so the trainer can skip this prompt.
        degenerate = bool(reward_std < 1e-6)
        adv = ((rewards - rewards.mean()) / (reward_std + 1e-6)).unsqueeze(1).expand_as(resp_mask)

        full_ids = torch.cat([pb["input_ids"], new_tokens], dim=1)
        full_attn = torch.cat([pb["attention_mask"], resp_mask], dim=1)

        with torch.no_grad():
            old_lp = _log_prob_at_response(
                base, full_ids, full_attn, pb["ecg_values"], pL, args.rl_temperature)
    finally:
        if was_training:
            base.train()

    return {
        "full_ids": full_ids, "full_attn": full_attn,
        "ecg_values": pb["ecg_values"],
        "resp_mask": resp_mask, "advantages": adv, "old_log_prob": old_lp, "pL": pL,
        "mean_reward": rewards.mean().item(), "degenerate": degenerate,
        "mean_reward_components": {k: sum(parts[k] for parts in reward_parts) / G for k in reward_parts[0]},
        "temperature": args.rl_temperature,
    }


def current_log_prob(model, ro: dict) -> torch.Tensor:
    """Log-prob of rollout under the current (post-update) policy (keeps DDP graph)."""
    return _log_prob_at_response(model, ro["full_ids"], ro["full_attn"], ro["ecg_values"],
                                 ro["pL"], ro["temperature"])
