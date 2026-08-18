import json
import string
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import matplotlib
import numpy as np
import scipy.stats as stats
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROUGE_SCORER = RougeScorer(["rougeL"], use_stemmer=True)


def split_response(text: str) -> tuple[str, str]:
    thinking = ""
    answer = text
    if "<think>" in text:
        _, _, thinking_and_answer = text.partition("<think>")
        thinking, closed, answer = thinking_and_answer.partition("</think>")
        if not closed:
            return thinking.strip(), ""
    elif "</think>" in text:
        thinking, _, answer = text.partition("</think>")

    if "<answer>" in answer:
        _, _, answer = answer.partition("<answer>")
    if "</answer>" in answer:
        answer, _, _ = answer.partition("</answer>")
    return thinking.strip(), answer.strip()


def normalize(text: str) -> str:
    return " ".join(text.lower().translate(str.maketrans("", "", string.punctuation)).split())


def token_f1(reference: str, hypothesis: str) -> float:
    reference_tokens = normalize(reference).split()
    hypothesis_tokens = normalize(hypothesis).split()
    if not reference_tokens or not hypothesis_tokens:
        return float(reference_tokens == hypothesis_tokens)
    overlap = sum((Counter(reference_tokens) & Counter(hypothesis_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(hypothesis_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_strings(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    pairs = [(reference, hypothesis) for reference, hypothesis in zip(references, hypotheses) if reference]
    if not pairs:
        return {name: 0.0 for name in ("accuracy", "f1", "bleu_4", "rouge_l", "meteor")}
    references, hypotheses = map(list, zip(*pairs))
    return {
        "accuracy": float(np.mean([reference == hypothesis for reference, hypothesis in pairs])),
        "f1": float(np.mean([token_f1(reference, hypothesis) for reference, hypothesis in pairs])),
        "bleu_4": float(corpus_bleu(
            [[reference.split()] for reference in references],
            [hypothesis.split() for hypothesis in hypotheses],
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        )),
        "rouge_l": float(np.mean([
            ROUGE_SCORER.score(reference, hypothesis)["rougeL"].fmeasure
            for reference, hypothesis in pairs
        ])),
        "meteor": float(np.mean([
            meteor_score([reference.split()], hypothesis.split())
            for reference, hypothesis in pairs
        ])),
    }


def classification_metrics(references: list[str], hypotheses: list[str]) -> tuple[dict, dict, dict]:
    classes = sorted(set(references))
    other_counts = Counter(hypothesis for hypothesis in hypotheses if hypothesis not in classes)
    other_label = "Other"
    while other_label in classes:
        other_label = f"_{other_label}"
    predictions = [hypothesis if hypothesis in classes else other_label for hypothesis in hypotheses]
    columns = classes + ([other_label] if other_counts else [])
    counts = Counter(zip(references, predictions))
    confusion = {reference: {prediction: counts[reference, prediction] for prediction in columns}
                 for reference in classes}
    accuracy = {
        label: counts[label, label] / max(sum(reference == label for reference in references), 1)
        for label in classes
    }
    return accuracy, confusion, dict(other_counts)


def response_ranges(labels: torch.Tensor) -> list[tuple[int, int]]:
    indices = labels.ne(-100).nonzero(as_tuple=True)[0].tolist()
    if not indices:
        return []

    ranges = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            ranges.append((start, previous + 1))
            start = index
        previous = index
    ranges.append((start, previous + 1))
    return ranges


def prepare_evaluation(batch: dict, tokenizer) -> list[dict]:
    jobs = []
    input_ids = batch["input_ids"][0]
    attention_mask = batch["attention_mask"][0]
    labels = batch["labels"][0]
    for start, end in response_ranges(labels):
        prompt_ids = input_ids[:start]
        job = {
            "input_ids": prompt_ids.unsqueeze(0),
            "attention_mask": attention_mask[:start].unsqueeze(0),
            "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=True).strip(),
            "reference": tokenizer.decode(labels[start:end], skip_special_tokens=True).strip(),
        }
        if "ecg_values" in batch:
            job["ecg_values"] = batch["ecg_values"]
        jobs.append(job)
    return jobs


def eos_ids(model, tokenizer) -> list[int]:
    generation_model = getattr(model, "language_model", model)
    configured = getattr(getattr(generation_model, "generation_config", None), "eos_token_id", None)
    ids = [configured] if isinstance(configured, int) else list(configured or [])
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
        ids.append(im_end)
    return sorted(set(ids))


def generate_response(model, job: dict, tokenizer, evaluation: dict) -> str:
    stop_ids = eos_ids(model, tokenizer)
    generation = {
        "input_ids": job["input_ids"],
        "attention_mask": job["attention_mask"],
        "max_new_tokens": evaluation["max_new_tokens"],
        "do_sample": evaluation["do_sample"],
    }
    if "ecg_values" in job:
        generation["ecg_values"] = job["ecg_values"]
    if stop_ids:
        generation["eos_token_id"] = stop_ids
    if tokenizer.pad_token_id is not None:
        generation["pad_token_id"] = tokenizer.pad_token_id
    if evaluation["do_sample"]:
        generation["temperature"] = evaluation.get("temperature", 1.0)

    output = model.generate(**generation)[0]
    prompt_length = job["input_ids"].shape[1]
    if output.shape[0] >= prompt_length and torch.equal(output[:prompt_length], job["input_ids"][0]):
        output = output[prompt_length:]
    stop = next((index for index, token in enumerate(output.tolist()) if token in stop_ids), len(output))
    return tokenizer.decode(output[:stop], skip_special_tokens=True).strip()


def evaluate(model, dataloader, tokenizer, config: dict) -> dict:
    model.eval()
    device = next(model.parameters()).device
    records = []
    progress = tqdm(dataloader, desc="Evaluation", leave=False)
    with torch.inference_mode():
        for batch_index, batch in enumerate(progress):
            batch = {key: value.to(device) for key, value in batch.items()}
            jobs = prepare_evaluation(batch, tokenizer)
            for job in jobs:
                hypothesis = generate_response(model, job, tokenizer, config["evaluation"])
                records.append({
                    "prompt": job["prompt"], "reference": job["reference"],
                    "hypothesis": hypothesis,
                })
            if config["development"] and batch_index == 0:
                break

    references = [record["reference"] for record in records]
    hypotheses = [record["hypothesis"] for record in records]
    reference_parts = [split_response(text) for text in references]
    hypothesis_parts = [split_response(text) for text in hypotheses]
    answer_references = [parts[1] for parts in reference_parts]
    answer_hypotheses = [parts[1] for parts in hypothesis_parts]
    metrics = {"answer": evaluate_strings(answer_references, answer_hypotheses)}

    thinking = [(reference[0], hypothesis[0]) for reference, hypothesis in zip(reference_parts, hypothesis_parts)
                if reference[0]]
    if thinking:
        thinking_references, thinking_hypotheses = zip(*thinking)
        metrics["thinking"] = evaluate_strings(list(thinking_references), list(thinking_hypotheses))

    result = {
        "num_pairs": len(records), "metrics": metrics, "predictions": records,
        "answer_references": answer_references, "answer_hypotheses": answer_hypotheses,
    }
    if config["training"]["training_stage"] == "pretrain" and answer_references:
        result["pretrain_breakdown"] = pretrain_breakdown(answer_references, answer_hypotheses)
    classification = config["evaluation"].get("classification")
    if classification is None:
        classification = any("ecg-comp" in name.lower() for name in config["data"]["data_names"])
    if classification and answer_references:
        accuracy, confusion, other = classification_metrics(answer_references, answer_hypotheses)
        metrics["per_class_accuracy"] = accuracy
        result.update(confusion_matrix=confusion, other_output_counts=other)
    print_metrics(result)
    return result


def pretrain_breakdown(references: list[str], hypotheses: list[str]) -> dict:
    def statements(text):
        return {part.strip() for part in text.split(";") if part.strip()}

    counts = Counter()
    missed = Counter()
    extra = Counter()
    for reference, hypothesis in zip(references, hypotheses):
        expected, predicted = statements(reference), statements(hypothesis)
        if not predicted:
            counts["other"] += 1
            continue
        if expected == predicted:
            counts["matched"] += 1
            continue
        missing, added = expected - predicted, predicted - expected
        counts["missed"] += bool(missing)
        counts["extra"] += bool(added)
        counts["both" if missing and added else "only_missed" if missing else "only_extra"] += 1
        missed.update(missing)
        extra.update(added)
    total = len(references)
    return {
        "total": total, "matched": counts["matched"], "other": counts["other"],
        "not_matched": total - counts["matched"] - counts["other"],
        "missed": counts["missed"], "extra": counts["extra"],
        "only_missed": counts["only_missed"], "only_extra": counts["only_extra"], "both": counts["both"],
        "top_missed": missed.most_common(15), "top_extra": extra.most_common(15),
    }


def run_statistical_analysis(results: list[dict]) -> dict:
    if not results:
        raise ValueError("At least one evaluation result is required")

    def summarize(values):
        if any(isinstance(value, Mapping) for value in values):
            keys = sorted({key for value in values if isinstance(value, Mapping) for key in value})
            return {key: summarize([value[key] for value in values if isinstance(value, Mapping) and key in value])
                    for key in keys}
        values = np.asarray(values, dtype=float)
        mean = float(values.mean())
        if len(values) == 1:
            return {"mean": mean, "std": 0.0, "confidence_interval": [mean, mean], "n": 1}
        std = float(values.std(ddof=1))
        margin = float(stats.t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
        return {"mean": mean, "std": std, "confidence_interval": [mean - margin, mean + margin], "n": len(values)}

    return summarize([result["metrics"] for result in results])


def print_metrics(result: dict) -> None:
    print(f"\nEvaluation pairs: {result['num_pairs']}")
    for group, metrics in result["metrics"].items():
        print(f"[{group}]")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")


def horizontal_bar(path: Path, items, title: str, top_k: int = 10) -> None:
    items = list(items)[:top_k]
    if not items:
        return
    labels, values = zip(*reversed(items))
    figure, axis = plt.subplots(figsize=(10, max(3, 0.45 * len(labels) + 1.5)))
    axis.barh([str(label)[:80] for label in labels], values)
    axis.set_xlabel("Count")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_confusion_matrix(confusion: dict, path: Path) -> None:
    rows = list(confusion)
    columns = list(next(iter(confusion.values())))
    matrix = np.array([[confusion[row][column] for column in columns] for row in rows])
    totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)
    figure, axis = plt.subplots(figsize=(max(4, len(columns) * 1.5), max(4, len(rows) * 1.5)))
    axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for row in range(len(rows)):
        for column in range(len(columns)):
            color = "white" if normalized[row, column] > 0.5 else "black"
            axis.text(column, row, f"{matrix[row, column]}\n({normalized[row, column]:.1%})",
                      ha="center", va="center", color=color)
    axis.set(xticks=range(len(columns)), yticks=range(len(rows)), xticklabels=columns, yticklabels=rows,
             xlabel="Predicted", ylabel="True", title="Confusion Matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_run(result: dict, output_dir: Path, fold, seed) -> Path:
    stem = f"fold_{fold}_seed_{seed}"
    path = output_dir / f"{stem}.json"
    payload = {"fold": fold, "seed": seed, **result}
    answer_references = payload.pop("answer_references")
    answer_hypotheses = payload.pop("answer_hypotheses")
    with path.open("w") as file:
        json.dump(payload, file, indent=2)

    horizontal_bar(output_dir / f"{stem}_incorrect.png",
                   Counter(hypothesis for reference, hypothesis in zip(answer_references, answer_hypotheses)
                           if reference != hypothesis).most_common(),
                   "Top Incorrect Predictions")
    if "confusion_matrix" in result:
        save_confusion_matrix(result["confusion_matrix"], output_dir / f"{stem}_confusion.png")
        horizontal_bar(output_dir / f"{stem}_other.png",
                       Counter(result["other_output_counts"]).most_common(), "Top Other Outputs")
    if "pretrain_breakdown" in result:
        breakdown = result["pretrain_breakdown"]
        horizontal_bar(output_dir / f"{stem}_pretrain.png", [
            ("Matched", breakdown["matched"]), ("Not matched", breakdown["not_matched"]),
            ("Other", breakdown["other"]),
        ], "Pretraining Breakdown")
        horizontal_bar(output_dir / f"{stem}_missed.png", breakdown["top_missed"], "Top Missed Statements", 15)
        horizontal_bar(output_dir / f"{stem}_extra.png", breakdown["top_extra"], "Top Extra Statements", 15)
    return path
