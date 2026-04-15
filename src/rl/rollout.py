"""Group rollout: per-prompt, sample G trajectories, score, build tensors for policy-loss computation."""
import torch

from configs.constants import HF_LLMS
from rl.rewards import compute_reward


def _unwrap(m):
    m = getattr(m, "_orig_mod", m)                      # torch.compile
    return m.module if hasattr(m, "module") else m      # DDP


def _eos_set(llm_name: str) -> set:
    wt = HF_LLMS[llm_name]["watch_tokens"]
    eos = set(wt["eos_token"].keys() if isinstance(wt["eos_token"], dict) else wt["eos_token"])
    fe = wt.get("final_eos_token", ())
    return eos | set(fe.keys() if isinstance(fe, dict) else fe)


def _trim_mask(new_tokens: torch.Tensor, eos_ids: set) -> torch.Tensor:
    """Return (G, L) mask = 1 up to and including first EOS per row, 0 after."""
    G, L = new_tokens.shape
    mask = torch.ones(G, L, dtype=torch.float32, device=new_tokens.device)
    toks = new_tokens.tolist()
    for i in range(G):
        for j in range(L):
            if toks[i][j] in eos_ids:
                mask[i, j + 1:] = 0
                break
    return mask


def _expand_enc(enc_out: dict, idx: int, G: int) -> dict:
    out = {}
    for k, v in enc_out.items():
        if isinstance(v, torch.Tensor):
            out[k] = v[idx:idx + 1].expand(G, *v.shape[1:]).contiguous()
        else:
            out[k] = v
    return out


def _log_prob_at_response(model, ids, attn, sig_idx, enc_out, pL: int) -> torch.Tensor:
    """Forward pass → log π(y_t | ...) for t in [pL, total_len). Returns (G, gen_len)."""
    out = model(elm_input_ids=ids, elm_attention_mask=attn, elm_labels=None,
                signal_id_indices=sig_idx, encoder_tokenizer_out=enc_out)
    logits = out.logits[:, pL - 1:-1, :]                       # predictions for positions [pL, total_len)
    targets = ids[:, pL:]
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def rollout_group(model, batch: dict, item_idx: int, tokenizer, args) -> dict:
    """Sample G responses for one prompt, compute rewards, advantages, and old log-probs."""
    base = _unwrap(model)
    device = batch["elm_input_ids"].device
    G = args.rl_group_size

    labels = batch["elm_labels"][item_idx]
    nz = (labels != -100).nonzero(as_tuple=True)[0]
    if nz.numel() == 0:
        raise ValueError("No response tokens found (labels all -100).")
    rs, re_ = nz[0].item(), nz[-1].item() + 1

    prompt_ids = batch["elm_input_ids"][item_idx, :rs]
    prompt_attn = batch["elm_attention_mask"][item_idx, :rs]
    gt_text = tokenizer.decode(batch["elm_input_ids"][item_idx, rs:re_],
                               skip_special_tokens=True).strip()
    pL = prompt_ids.shape[0]

    pb = {
        "elm_input_ids": prompt_ids.unsqueeze(0).expand(G, -1).contiguous(),
        "elm_attention_mask": prompt_attn.unsqueeze(0).expand(G, -1).contiguous(),
        "signal_id_indices": batch["signal_id_indices"][item_idx].unsqueeze(0).expand(G, -1).contiguous(),
        "encoder_tokenizer_out": _expand_enc(batch["encoder_tokenizer_out"], item_idx, G),
    }

    was_training = base.training
    base.eval()
    try:
        with torch.no_grad():
            gen = base.generate(**pb, max_new_tokens=args.rl_max_new_tokens,
                                do_sample=True, temperature=args.rl_temperature, top_p=args.rl_top_p)

            new_tokens = gen[:, pL:] if gen.shape[1] > pL and torch.equal(gen[0, :pL], prompt_ids) else gen
            if new_tokens.shape[1] == 0:                                 # pathological: nothing generated
                new_tokens = torch.full((G, 1), int(tokenizer.pad_token_id), dtype=torch.long, device=device)

            eos_ids = _eos_set(args.llm)
            resp_mask = _trim_mask(new_tokens, eos_ids)                  # (G, gen_len)

            rewards = torch.tensor(
                [compute_reward(tokenizer.decode(new_tokens[i][resp_mask[i].bool()],
                                                 skip_special_tokens=True).strip(), gt_text)
                 for i in range(G)], dtype=torch.float32, device=device)
            adv = ((rewards - rewards.mean()) / (rewards.std() + 1e-6)).unsqueeze(1).expand_as(resp_mask)

            full_ids = torch.cat([pb["elm_input_ids"], new_tokens], dim=1)
            full_attn = torch.cat([pb["elm_attention_mask"], resp_mask], dim=1)
            # old_log_prob must match the behavioral distribution used by generate() (eval mode).
            old_lp = _log_prob_at_response(base, full_ids, full_attn,
                                           pb["signal_id_indices"], pb["encoder_tokenizer_out"], pL)
    finally:
        if was_training:
            base.train()

    return {
        "full_ids": full_ids, "full_attn": full_attn,
        "sig_idx": pb["signal_id_indices"], "enc_out": pb["encoder_tokenizer_out"],
        "resp_mask": resp_mask, "advantages": adv, "old_log_prob": old_lp, "pL": pL,
        "mean_reward": rewards.mean().item(),
    }


def current_log_prob(model, ro: dict) -> torch.Tensor:
    """Log-prob of rollout under the current (post-update) policy (keeps DDP graph)."""
    return _log_prob_at_response(model, ro["full_ids"], ro["full_attn"],
                                 ro["sig_idx"], ro["enc_out"], ro["pL"])
