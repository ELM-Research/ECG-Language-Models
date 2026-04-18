import re
import numpy as np
import scipy.stats as stats
from tqdm import tqdm
import torch
from collections import Counter
import string
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from rouge_score.rouge_scorer import RougeScorer

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

def split_response(text):
    if "</think>" in text and "<think>" not in text:
        text = "<think>\n" + text  # explicit_thinking: opener was consumed as prompt prefix
    t, a = _THINK_RE.search(text), _ANSWER_RE.search(text)
    thinking = t.group(1).strip() if t else ""
    answer = (a.group(1) if a else text[t.end():] if t else text).strip()
    return thinking, answer

from utils.gpu_manager import is_main, train_dev_break

from runners.helper import batch_to_device

_rouge_scorer = RougeScorer(["rougeL"], use_stemmer=True)

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
            "BLEU-4": 0.0,
            "ROUGE-L": 0.0,
            "METEOR": 0.0,
            "BERTScore-F1": 0.0,
        }
    valid_refs, valid_hyps = zip(*valid_pairs)
    return {
        "ACC": calculate_acc(valid_refs, valid_hyps),
        "F1": calculate_f1(valid_refs, valid_hyps),
        "BLEU-4": calculate_bleu4(valid_refs, valid_hyps),
        "ROUGE-L": calculate_rouge_l(valid_refs, valid_hyps),
        "METEOR": calculate_meteor(valid_refs, valid_hyps),
        "BERTScore-F1": calculate_bertscore_f1(valid_refs, valid_hyps),
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

def calculate_bleu4(references, hypotheses):
    return corpus_bleu([[r.split()] for r in references], [h.split() for h in hypotheses],
                       weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)

def calculate_rouge_l(references, hypotheses):
    return np.mean([_rouge_scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)])

def calculate_meteor(references, hypotheses):
    return np.mean([nltk_meteor([r.split()], h.split()) for r, h in zip(references, hypotheses)])

def calculate_bertscore_f1(references, hypotheses):
    from bert_score import score as bert_score
    _, _, f1 = bert_score(hypotheses, references, lang="en", verbose=False)
    return f1.mean().item()

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

def evaluate(elm, dataloader, args):
    show_progress = is_main()
    elm.eval()
    needs_signal_injection = args.elm in ("mlp_llava", "linear_llava", "base_elf",
                                          "patch_elf", "conv_elf")
    progress = tqdm(
        dataloader,
        desc=f"LLM: {args.llm} ENCODER: {args.encoder}",
        disable=not show_progress,
        leave=False,
    )
    dataset = dataloader.dataset
    device = next(elm.parameters()).device
    all_refs, all_hyps, all_prompts = [], [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress):
            B = batch["elm_input_ids"].shape[0]
            for b in range(B):
                full_ids = batch["elm_input_ids"][b].tolist()
                full_attn = batch["elm_attention_mask"][b].tolist()
                if needs_signal_injection:
                    signal_indices = batch["signal_id_indices"][b]
                    full_encoder_tokenizer_out = index_nested(batch["encoder_tokenizer_out"], b)
                ranges = dataset.get_response_ranges(full_ids)
                gt_texts = dataset.get_ground_truth_responses(full_ids, ranges)
                if getattr(args, "dev", False):
                    print(f"\n--- Batch {batch_idx}, Sample {b} ---")
                    print(f"Total turns: {len(ranges)}")
                    dataset.assert_range_alignment(full_ids, ranges)
                for turn_idx, ((s, _), gt) in enumerate(zip(ranges, gt_texts)):
                    sub_ids = full_ids[:s]
                    sub_attn = full_attn[:s]
                    gen_batch = {
                        "elm_input_ids": torch.tensor(sub_ids, dtype=torch.int64).unsqueeze(0),
                        "elm_attention_mask": torch.tensor(sub_attn, dtype=torch.float32).unsqueeze(0),
                        "max_new_tokens": args.max_new_tokens
                    }
                    if needs_signal_injection:
                        gen_batch["encoder_tokenizer_out"] = full_encoder_tokenizer_out
                        truncated_len = len(sub_ids)
                        masked_indices = signal_indices.clone()
                        masked_indices[masked_indices >= truncated_len] = -1
                        gen_batch["signal_id_indices"] = masked_indices
                    gen_batch = {k: batch_to_device(v, device) for k, v in gen_batch.items()}
                    gen_out = elm.generate(**gen_batch)[0].cpu().tolist()
                    gen_txt = dataset.get_generated_response_for_turn(sub_ids, gen_out)
                    if getattr(args, "dev", False):
                        print(f"\nTurn {turn_idx + 1}:")
                        print(f"\nGround Truth:\n{gt}")
                        print(f"\nGenerated:\n{gen_txt}")
                        print("-" * 100)
                    if gt and gen_txt:
                        all_prompts.append(dataset.llm_tokenizer.decode(sub_ids, skip_special_tokens=True).strip())
                        all_refs.append(gt)
                        all_hyps.append(gen_txt)
            if train_dev_break(getattr(args, "dev", False), batch, 0):
                break
            # if batch_idx == 10:
            #     break
            # input()
    refs_t, refs_a = map(list, zip(*map(split_response, all_refs))) if all_refs else ([], [])
    hyps_t, hyps_a = map(list, zip(*map(split_response, all_hyps))) if all_hyps else ([], [])
    think_pairs = [(r, h) for r, h in zip(refs_t, hyps_t) if r and h]
    results = {"answer": evaluate_strings(refs_a, hyps_a)}
    if think_pairs:
        results["thinking"] = evaluate_strings(*map(list, zip(*think_pairs)))
    print("\n=== N-Turn Evaluation (generated vs. gold response only) ===")
    print(f"Pairs: {len(all_refs)} (thinking pairs: {len(think_pairs)})")
    for group, mdict in results.items():
        print(f"[{group}]")
        for k, v in mdict.items():
            print(f"  {k}: {v:.4f}")
    out = {
        "num_pairs": len(all_refs),
        "metrics": results,
        "prompts": all_prompts,
        "references": all_refs,
        "hypotheses": all_hyps,
    }
    if any(d.startswith("ecg-comp") for d in args.data):
        per_class_acc, confusion_matrix, other_counts = compute_classification_metrics(refs_a, hyps_a)
        print_classification_metrics(per_class_acc, confusion_matrix)
        results["per_class_acc"] = per_class_acc
        out["confusion_matrix"] = confusion_matrix
        out["other_output_counts"] = other_counts
    return out