import functools
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import Counter
import string

from utils.gpu_manager import is_main, get_world_size, get_rank

from runners.helper import batch_to_device

def calculate_acc(references, hypotheses):
    return np.mean([ref == hyp for ref, hyp in zip(references, hypotheses)])

def evaluate_strings(references, hypotheses):
    if len(references) != len(hypotheses):
        raise ValueError("The number of references and hypotheses must be the same.")
    valid_pairs = [(ref, hyp) for ref, hyp in zip(references, hypotheses) if ref and hyp]
    if not valid_pairs:
        return {
            "ACC": 0.0,
            "F1": 0.0,
        }
    valid_refs, valid_hyps = zip(*valid_pairs)
    return {
        "ACC": calculate_acc(valid_refs, valid_hyps),
        "F1": calculate_f1(valid_refs, valid_hyps),
    }
def _normalize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

def _token_f1(ref, hyp):
    ref_tokens = _normalize(ref).split()
    hyp_tokens = _normalize(hyp).split()
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0
    common = Counter(ref_tokens) & Counter(hyp_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(hyp_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

def calculate_f1(references, hypotheses):
    return np.mean([_token_f1(ref, hyp) for ref, hyp in zip(references, hypotheses)])

def compute_classification_metrics(references, hypotheses):
    valid_classes = set(references)

    other_raw = [h for h in hypotheses if h not in valid_classes]
    other_counts = dict(Counter(other_raw))

    hypotheses = [h if h in valid_classes else "Other" for h in hypotheses]
    has_other = "Other" not in valid_classes and any(h == "Other" for h in hypotheses)

    row_classes = sorted(valid_classes)
    col_classes = row_classes + (["Other"] if has_other else [])
    cm = Counter((r, h) for r, h in zip(references, hypotheses))

    per_class_acc = {}
    for c in row_classes:
        total = sum(1 for r in references if r == c)
        per_class_acc[c] = cm[(c, c)] / total if total > 0 else 0.0

    confusion_matrix = {c: {p: cm[(c, p)] for p in col_classes} for c in row_classes}
    return per_class_acc, confusion_matrix, other_counts

def save_other_outputs_histogram_png(other_counts, path, top_k=20):
    if not other_counts:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = Counter(other_counts).most_common(top_k)
    labels, counts = zip(*items)
    fig_h = max(3, 0.45 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(labels))
    ax.barh(y, counts)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Top {min(top_k, len(labels))} 'Other' Outputs")
    for i, c in enumerate(counts):
        ax.text(c, i, f" {c}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved other-output histogram to {path}")

def print_classification_metrics(per_class_acc, confusion_matrix):
    row_classes = list(confusion_matrix.keys())
    col_classes = list(next(iter(confusion_matrix.values())).keys())
    print("\n=== Per-Class Accuracy ===")
    for c in row_classes:
        total = sum(confusion_matrix[c].values())
        correct = confusion_matrix[c][c]
        print(f"  {c}: {per_class_acc[c]:.4f} ({correct}/{total})")
    w = max(6, max(len(c) for c in col_classes) + 2)
    print("\n=== Confusion Matrix (rows=true, cols=predicted) ===")
    header = " " * w + "".join(f"{c:>{w}}" for c in col_classes)
    print(header)
    for c in row_classes:
        row = f"{c:>{w}}" + "".join(f"{confusion_matrix[c][p]:>{w}}" for p in col_classes)
        print(row)

def save_confusion_matrix_png(confusion_matrix, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row_classes = list(confusion_matrix.keys())
    col_classes = list(next(iter(confusion_matrix.values())).keys())
    matrix = np.array([[confusion_matrix[r][c] for c in col_classes] for r in row_classes])
    row_sums = matrix.sum(axis=1, keepdims=True)
    normed = np.divide(matrix, row_sums, where=row_sums != 0, out=np.zeros_like(matrix, dtype=float))

    n_rows, n_cols = len(row_classes), len(col_classes)
    fig, ax = plt.subplots(figsize=(max(4, n_cols * 1.5), max(4, n_rows * 1.5)))
    ax.imshow(normed, cmap="Blues", vmin=0, vmax=1)
    for i, r in enumerate(row_classes):
        for j, c in enumerate(col_classes):
            ax.text(j, i, f"{matrix[i, j]}\n({normed[i, j]:.1%})", ha="center", va="center",
                    color="white" if normed[i, j] > 0.5 else "black", fontsize=10)
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_classes)
    ax.set_yticklabels(row_classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {path}")

def run_statistical_analysis(all_seeds_results):
    metrics = list(all_seeds_results[0]["metrics"].keys())
    statistical_results = {}

    for metric in metrics:
        metric_values = [result["metrics"][metric] for result in all_seeds_results]

        if isinstance(metric_values[0], dict):
            statistical_results[metric] = {}
            for sub_metric in metric_values[0].keys():
                if isinstance(metric_values[0][sub_metric], list):
                    mean = np.mean(metric_values[0][sub_metric]) * 100
                    values = [np.mean(result["metrics"][metric][sub_metric]) * 100 for result in all_seeds_results]
                else:
                    mean = np.mean(metric_values[0][sub_metric]) * 100
                    values = [np.mean(result["metrics"][metric][sub_metric]) * 100 for result in all_seeds_results]

                mean = np.mean(values)
                std = np.std(values, ddof=1)
                confidence = 0.95
                degrees_of_freedom = len(values) - 1
                t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
                margin_of_error = t_value * (std / np.sqrt(len(values)))
                conf_interval = (mean - margin_of_error, mean + margin_of_error)
                statistical_results[metric][sub_metric] = {
                    "mean": mean,
                    "std": std,
                    "conf_interval": conf_interval,
                }
        else:
            values = [np.mean(result["metrics"][metric]) * 100 for result in all_seeds_results]
            mean = np.mean(values)
            std = np.std(values, ddof=1)

            confidence = 0.95
            degrees_of_freedom = len(values) - 1
            t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
            margin_of_error = t_value * (std / np.sqrt(len(values)))

            conf_interval = (mean - margin_of_error, mean + margin_of_error)

            statistical_results[metric] = {
                "mean": mean,
                "std": std,
                "conf_interval": conf_interval,
            }

    return statistical_results

def index_nested(encoder_tokenizer_out, batch):
    return {k: index_nested(v, batch) if isinstance(v, dict) else v[batch:batch+1] for k, v in encoder_tokenizer_out.items()}

def save_incorrect_predictions_histogram_png(references, hypotheses, path, top_k=10):
    incorrect = [h for r, h in zip(references, hypotheses) if r != h]
    if not incorrect:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = Counter(incorrect).most_common(top_k)
    labels, counts = zip(*items)
    labels = [l[:80] + "\u2026" if len(l) > 80 else l for l in labels]
    fig_h = max(3, 0.45 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y = np.arange(len(labels))
    ax.barh(y, counts)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title(f"Top {min(top_k, len(labels))} Incorrect Predictions")
    for i, c in enumerate(counts):
        ax.text(c, i, f" {c}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved incorrect-predictions histogram to {path}")


# ---------------------------------------------------------------------------
# Phase 1 helpers: flatten all (sample, turn) pairs on CPU
# ---------------------------------------------------------------------------

def _extract_encoder_out_item(encoder_tokenizer_out, b):
    """Extract single-sample encoder outputs from a batched dict (batch dim squeezed)."""
    out = {}
    for k, v in encoder_tokenizer_out.items():
        if isinstance(v, dict):
            out[k] = _extract_encoder_out_item(v, b)
        else:
            out[k] = v[b]  # remove batch dim
    return out


def flatten_eval_turns(dataloader, args, needs_signal_injection):
    """Iterate through the dataset (batch_size=1) and produce a flat list of turn-level work items."""
    dataset = dataloader.dataset
    turns = []
    global_idx = 0

    progress = tqdm(
        dataloader,
        desc="Flattening turns",
        disable=not is_main(),
        leave=False,
    )

    for batch_idx, batch in enumerate(progress):
        B = batch["elm_input_ids"].shape[0]
        for b in range(B):
            full_ids = batch["elm_input_ids"][b].tolist()
            full_attn = batch["elm_attention_mask"][b].tolist()

            if needs_signal_injection:
                signal_indices = batch["signal_id_indices"][b]
                encoder_out_item = _extract_encoder_out_item(batch["encoder_tokenizer_out"], b)

            ranges = dataset.get_response_ranges(full_ids)
            gt_texts = dataset.get_ground_truth_responses(full_ids, ranges)

            if getattr(args, "dev", False) and is_main():
                print(f"\n--- Batch {batch_idx}, Sample {b} ---")
                print(f"Total turns: {len(ranges)}")
                dataset.assert_range_alignment(full_ids, ranges)

            for turn_idx, ((s, _), gt) in enumerate(zip(ranges, gt_texts)):
                sub_ids = full_ids[:s]
                sub_attn = full_attn[:s]

                turn = {
                    "global_idx": global_idx,
                    "sample_idx": batch_idx * B + b,
                    "turn_idx": turn_idx,
                    "prefix_ids": sub_ids,
                    "prefix_attn": sub_attn,
                    "gt_text": gt,
                }

                if needs_signal_injection:
                    masked_indices = signal_indices.clone()
                    masked_indices[masked_indices >= len(sub_ids)] = -1
                    turn["signal_id_indices"] = masked_indices
                    turn["encoder_tokenizer_out"] = encoder_out_item

                turns.append(turn)
                global_idx += 1

    if is_main():
        print(f"Flattened {global_idx} turns from {len(dataloader.dataset)} samples")
    return turns


# ---------------------------------------------------------------------------
# Phase 2 helpers: batched generation
# ---------------------------------------------------------------------------

class TurnDataset(Dataset):
    """Wraps the flat turn list for DataLoader batching."""
    def __init__(self, turns):
        self.turns = turns

    def __len__(self):
        return len(self.turns)

    def __getitem__(self, idx):
        return self.turns[idx]


def _stack_encoder_out(items):
    """Stack per-sample encoder outputs along a new batch dimension."""
    if not items:
        return {}
    keys = items[0].keys()
    out = {}
    for k in keys:
        vals = [item[k] for item in items]
        if isinstance(vals[0], dict):
            out[k] = _stack_encoder_out(vals)
        elif isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], np.ndarray):
            out[k] = torch.from_numpy(np.stack(vals, axis=0))
        else:
            out[k] = vals
    return out


def eval_collate_fn(batch, pad_token_id):
    """Collate turn dicts into a left-padded batch for generation."""
    max_len = max(len(item["prefix_ids"]) for item in batch)

    all_input_ids = []
    all_attn_masks = []
    all_prefix_lens = []
    all_global_idxs = []
    all_gt_texts = []
    all_original_prefix_ids = []
    has_signal = "signal_id_indices" in batch[0]
    all_signal_indices = []
    all_encoder_outs = []

    for item in batch:
        prefix_ids = item["prefix_ids"]
        prefix_attn = item["prefix_attn"]
        prefix_len = len(prefix_ids)
        pad_len = max_len - prefix_len

        # Left-pad input_ids and attention mask
        padded_ids = [pad_token_id] * pad_len + prefix_ids
        padded_attn = [0] * pad_len + prefix_attn

        all_input_ids.append(padded_ids)
        all_attn_masks.append(padded_attn)
        all_prefix_lens.append(prefix_len)
        all_global_idxs.append(item["global_idx"])
        all_gt_texts.append(item["gt_text"])
        all_original_prefix_ids.append(prefix_ids)

        if has_signal:
            # Shift signal indices right by pad_len (left-padding offset)
            indices = item["signal_id_indices"].clone()
            valid = indices >= 0
            indices[valid] += pad_len
            all_signal_indices.append(indices)
            all_encoder_outs.append(item["encoder_tokenizer_out"])

    out = {
        "elm_input_ids": torch.tensor(all_input_ids, dtype=torch.int64),
        "elm_attention_mask": torch.tensor(all_attn_masks, dtype=torch.float32),
        "prefix_len": all_prefix_lens,
        "global_idx": all_global_idxs,
        "gt_text": all_gt_texts,
        "original_prefix_ids": all_original_prefix_ids,
    }

    if has_signal:
        out["signal_id_indices"] = torch.stack(all_signal_indices, dim=0)
        out["encoder_tokenizer_out"] = _stack_encoder_out(all_encoder_outs)

    return out


# ---------------------------------------------------------------------------
# Main evaluate function (batched + distributed)
# ---------------------------------------------------------------------------

def evaluate(elm, dataloader, args, debug_file=None):
    elm.eval()
    needs_signal_injection = args.elm in ("llava", "base_elm", "patch_elm")
    dataset = dataloader.dataset
    device = next(elm.parameters()).device
    distributed = getattr(args, "distributed", False)

    # --- Phase 1: Flatten all turns (CPU, batch_size=1) ---
    turns = flatten_eval_turns(dataloader, args, needs_signal_injection)

    if not turns:
        return {
            "num_pairs": 0,
            "metrics": {"ACC": 0.0, "F1": 0.0},
            "prompts": [],
            "references": [],
            "hypotheses": [],
        }

    # --- Phase 2: Batched generation ---
    turn_dataset = TurnDataset(turns)
    pad_token_id = dataset.llm_tokenizer.pad_token_id
    collate = functools.partial(eval_collate_fn, pad_token_id=pad_token_id)

    if distributed:
        sampler = DistributedSampler(
            turn_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False,
            drop_last=False,
        )
    else:
        sampler = None

    eval_batch_size = getattr(args, "eval_batch_size", 1)
    gen_loader = DataLoader(
        turn_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate,
        num_workers=0,  # data already in memory
        pin_memory=False,
    )

    # Unwrap DDP for generate()
    gen_model = elm.module if hasattr(elm, "module") else elm

    local_results = []

    with torch.no_grad():
        for batch in tqdm(gen_loader, desc=f"Generating (bs={eval_batch_size})",
                          disable=not is_main(), leave=False):
            gen_batch = {
                "elm_input_ids": batch["elm_input_ids"].to(device),
                "elm_attention_mask": batch["elm_attention_mask"].to(device),
            }
            if needs_signal_injection:
                enc_out = batch_to_device(batch["encoder_tokenizer_out"], device)
                gen_batch["encoder_tokenizer_out"] = enc_out
                gen_batch["signal_id_indices"] = batch["signal_id_indices"].to(device)

            gen_out = gen_model.generate(**gen_batch)  # [B, output_seq_len]

            B = gen_out.shape[0]
            for i in range(B):
                gidx = batch["global_idx"][i]
                prefix_ids = batch["original_prefix_ids"][i]
                gt = batch["gt_text"][i]

                gen_ids = gen_out[i].cpu().tolist()
                gen_txt = dataset.get_generated_response_for_turn(prefix_ids, gen_ids)

                local_results.append((gidx, gt, gen_txt, prefix_ids))

    # --- Phase 3: Gather, deduplicate, reorder ---
    if distributed:
        all_results_nested = [None] * get_world_size()
        dist.all_gather_object(all_results_nested, local_results)
        all_results = [item for sublist in all_results_nested for item in sublist]
    else:
        all_results = local_results

    # Sort by global_idx for deterministic ordering
    all_results.sort(key=lambda x: x[0])

    # Deduplicate (DistributedSampler with drop_last=False pads with duplicates)
    seen = set()
    deduped = []
    for item in all_results:
        if item[0] not in seen:
            seen.add(item[0])
            deduped.append(item)
    all_results = deduped

    # Build final lists (only keep pairs where both gt and gen are non-empty)
    all_refs, all_hyps, all_prompts = [], [], []
    example_idx = 0
    for gidx, gt, gen_txt, prefix_ids in all_results:
        if gt and gen_txt:
            prompt_txt = dataset.llm_tokenizer.decode(prefix_ids, skip_special_tokens=True).strip()
            all_prompts.append(prompt_txt)
            all_refs.append(gt)
            all_hyps.append(gen_txt)

            if debug_file is not None and is_main():
                debug_file.write(f"{'='*80}\n")
                debug_file.write(f"EXAMPLE {example_idx} | global_idx={gidx}\n")
                debug_file.write(f"{'='*80}\n")
                debug_file.write(f"PROMPT ({len(prefix_ids)} tokens):\n{prompt_txt[:500]}\n\n")
                debug_file.write(f"GROUND TRUTH:\n{gt}\n\n")
                debug_file.write(f"GENERATED:\n{gen_txt}\n\n")
                debug_file.write(f"MATCH: {gt == gen_txt}\n\n")
                debug_file.flush()
            example_idx += 1

    results = evaluate_strings(all_refs, all_hyps)
    if is_main():
        print("\n=== N-Turn Evaluation (generated vs. gold response only) ===")
        print(f"Pairs: {len(all_refs)}")
        print(f"ACC: {results['ACC']:.4f}")
        print(f"F1:  {results['F1']:.4f}")

    out = {
        "num_pairs": len(all_refs),
        "metrics": results,
        "prompts": all_prompts,
        "references": all_refs,
        "hypotheses": all_hyps,
    }
    if any(d.startswith("ecg-comp") for d in args.data):
        per_class_acc, confusion_matrix, other_counts = compute_classification_metrics(all_refs, all_hyps)
        if is_main():
            print_classification_metrics(per_class_acc, confusion_matrix)
        results["per_class_acc"] = per_class_acc
        out["confusion_matrix"] = confusion_matrix
        out["other_output_counts"] = other_counts
    return out
