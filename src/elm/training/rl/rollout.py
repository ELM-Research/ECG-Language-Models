"""Group rollout: per-prompt, sample G trajectories, score, build tensors for policy-loss computation."""
import torch

from elm.training.rl.rewards import reward_components


def _eos_set(model) -> set:
    generation_model = getattr(model, "language_model", model)
    eos = generation_model.generation_config.eos_token_id
    return {eos} if isinstance(eos, int) else set(eos or ())


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


def rollout_group(
    model,
    batch: dict,
    item_idx: int,
    tokenizer,
    config: dict,
    explicit_thinking: bool,
) -> dict:
    """Sample G responses for one prompt, compute rewards, advantages, and old log-probs."""
    device = batch["input_ids"].device
    group_size = config["group_size"]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("RL requires a tokenizer pad token")

    eos_ids = _eos_set(model)
    strip_ids = eos_ids | {int(pad_id)}

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
        "input_ids": prompt_ids.unsqueeze(0).expand(group_size, -1).contiguous(),
        "attention_mask": prompt_attn.unsqueeze(0).expand(group_size, -1).contiguous(),
        "ecg_values": batch["ecg_values"][item_idx:item_idx + 1].expand(
            group_size, *batch["ecg_values"].shape[1:]).contiguous(),
    }

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            gen = model.generate(
                **pb,
                max_new_tokens=config["max_new_tokens"],
                do_sample=True,
            )

        new_tokens = gen[:, pL:] if gen.shape[1] > pL and torch.equal(gen[0, :pL], prompt_ids) else gen
        if new_tokens.shape[1] == 0:                                 # pathological: nothing generated
            new_tokens = torch.full((group_size, 1), pad_id, dtype=torch.long, device=device)

        resp_mask = _trim_mask(new_tokens, eos_ids, pad_id)

        rewards = torch.tensor([
            sum(reward_components(
                _decode_for_reward(tokenizer, new_tokens[i][resp_mask[i].bool()], strip_ids),
                gt_text,
                explicit_thinking,
            ).values())
            for i in range(group_size)
        ], dtype=torch.float32, device=device)
        reward_std = rewards.std(unbiased=False)
        # All G samples scored identically: group-relative advantage is pure
        # 1e-6-scaled noise. Flag so the trainer can skip this prompt.
        degenerate = bool(reward_std < 1e-6)
        adv = ((rewards - rewards.mean()) / (reward_std + 1e-6)).unsqueeze(1).expand_as(resp_mask)

        full_ids = torch.cat([pb["input_ids"], new_tokens], dim=1)
        full_attn = torch.cat([pb["attention_mask"], resp_mask], dim=1)

        with torch.no_grad():
            old_lp = _log_prob_at_response(
                model, full_ids, full_attn, pb["ecg_values"], pL, config["temperature"])
    finally:
        if was_training:
            model.train()

    return {
        "full_ids": full_ids, "full_attn": full_attn,
        "ecg_values": pb["ecg_values"],
        "response_mask": resp_mask, "advantages": adv, "old_log_prob": old_lp, "pL": pL,
        "mean_reward": rewards.mean().item(), "degenerate": degenerate,
        "temperature": config["temperature"],
    }


def current_log_prob(model, ro: dict) -> torch.Tensor:
    return _log_prob_at_response(model, ro["full_ids"], ro["full_attn"], ro["ecg_values"], ro["pL"], ro["temperature"])
