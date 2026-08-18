import torch

from elm.training.rl.rewards import reward_components


def eos_set(model) -> set:
    generation_model = getattr(model, "language_model", model)
    eos = generation_model.generation_config.eos_token_id
    return {eos} if isinstance(eos, int) else set(eos or ())


def trim_mask(new_tokens: torch.Tensor, eos_ids: set, pad_id: int | None = None) -> torch.Tensor:
    is_eos = torch.zeros_like(new_tokens, dtype=torch.bool)
    for eos_id in eos_ids:
        is_eos |= new_tokens == eos_id
    mask = is_eos.cumsum(dim=1) - is_eos.long() == 0
    if pad_id is not None and pad_id not in eos_ids:
        mask &= new_tokens != pad_id
    return mask


def _decode_for_reward(tokenizer, ids: torch.Tensor, strip_ids: set) -> str:
    kept = [int(t) for t in ids.tolist() if int(t) not in strip_ids]
    return tokenizer.decode(kept, skip_special_tokens=False).strip()


def final_response_range(labels: torch.Tensor) -> tuple[int, int]:
    indices = labels.ne(-100).nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        raise ValueError("No response tokens found (labels all -100)")
    gaps = (indices[1:] != indices[:-1] + 1).nonzero(as_tuple=True)[0]
    start = indices[gaps[-1] + 1] if gaps.numel() else indices[0]
    return start.item(), indices[-1].item() + 1


def log_prob_at_response(model, ids, attn, ecg, pL: int, temperature: float) -> torch.Tensor:
    targets = ids[:, pL:]
    was_training = model.training
    model.eval()
    try:
        out = model(
            input_ids=ids,
            attention_mask=attn,
            ecg_values=ecg,
            logits_to_keep=targets.shape[1] + 1,
            use_cache=False,
        )
    finally:
        model.train(was_training)
    logits = out.logits[:, -targets.shape[1] - 1:-1].float() / temperature
    selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return selected - logits.logsumexp(dim=-1)


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

    eos_ids = eos_set(model)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids.add(im_end_id)
    strip_ids = eos_ids | {int(pad_id)}

    labels = batch["labels"][item_idx]
    rs, re = final_response_range(labels)
    if labels[re - 1].item() != im_end_id:
        raise ValueError("RL target was truncated before <|im_end|>")
    gt_text = _decode_for_reward(tokenizer, labels[rs:re], strip_ids)

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
                temperature=config["temperature"],
                eos_token_id=sorted(eos_ids),
                pad_token_id=pad_id,
            )

        includes_prompt = gen.shape[1] >= pL and torch.equal(gen[0, :pL], prompt_ids)
        new_tokens = gen[:, pL:] if includes_prompt else gen
        if new_tokens.shape[1] == 0:
            new_tokens = torch.full((group_size, 1), pad_id, dtype=torch.long, device=device)

        resp_mask = trim_mask(new_tokens, eos_ids, pad_id)

        components = [
            reward_components(
                _decode_for_reward(tokenizer, new_tokens[i][resp_mask[i].bool()], strip_ids),
                gt_text,
                explicit_thinking,
            )
            for i in range(group_size)
        ]
        rewards = torch.tensor([sum(x.values()) for x in components], dtype=torch.float32, device=device)
        reward_std = rewards.std(unbiased=False)
        # All G samples scored identically: group-relative advantage is pure
        # 1e-6-scaled noise. Flag so the trainer can skip this prompt.
        degenerate = bool(reward_std < 1e-6)
        adv = ((rewards - rewards.mean()) / (reward_std + 1e-6)).unsqueeze(1).expand_as(resp_mask)

        full_ids = torch.cat([pb["input_ids"], new_tokens], dim=1)
        full_attn = torch.cat([pb["attention_mask"], resp_mask], dim=1)

        with torch.no_grad():
            old_lp = log_prob_at_response(
                model, full_ids, full_attn, pb["ecg_values"], pL, config["temperature"])
    finally:
        if was_training:
            model.train()

    return {
        "full_ids": full_ids, "full_attn": full_attn,
        "ecg_values": pb["ecg_values"],
        "response_mask": resp_mask, "advantages": adv, "old_log_prob": old_lp, "pL": pL,
        "rewards": {name: sum(x[name] for x in components) / group_size for name in components[0]},
        "degenerate": degenerate,
        "temperature": config["temperature"],
    }


def current_log_prob(model, ro: dict) -> torch.Tensor:
    return log_prob_at_response(model, ro["full_ids"], ro["full_attn"], ro["ecg_values"], ro["pL"], ro["temperature"])
