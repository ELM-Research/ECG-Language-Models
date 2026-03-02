import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path

from configs.config import get_args
from utils.dir_file_manager import DirFileManager

DIR_FILE_MANAGER = DirFileManager()
PERTURB_ORDER = ["None", "noise", "only_text", "zeros"]
PASTEL_COLORS = ["#A8D8EA", "#F5B7B1", "#ABEBC6", "#D7BDE2"]
STEP_EPOCH_RE = re.compile(r"step_epoch_(\d+)_step_(\d+)")


def extract_metrics(json_data):
    return {
        "Accuracy": {"mean": json_data["ACC"]["mean"], "std": json_data["ACC"]["std"]},
        "F1": {"mean": json_data["F1"]["mean"], "std": json_data["F1"]["std"]},
    }


def parse_perturbation(filename):
    stem = Path(filename).stem
    for p in ["only_text", "noise", "zeros", "None"]:
        if stem.endswith(f"_{p}"):
            return p
    return None


def load_run_config(json_dir):
    """Load config.yaml from the run directory.
    Expected layout: runs/{llm}_{encoder}/{data}/{run_id}/checkpoints/
    config.yaml lives in {run_id}/ (parent of checkpoints/).
    Returns the config dict or None if not found.
    """
    p = Path(json_dir).resolve()
    run_dir = p.parent if p.name == "checkpoints" else p
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return None


def derive_model_name(json_dir):
    """Derive model label from config.yaml, falling back to directory name."""
    cfg = load_run_config(json_dir)
    if cfg:
        elm = cfg.get("elm", "?")
        llm = cfg.get("llm", "?")
        encoder = cfg.get("encoder", "?")
        return f"ELM: {elm}\nLLM: {llm}\nEncoder: {encoder}"
    p = Path(json_dir).resolve()
    if p.name == "checkpoints":
        return p.parent.parent.parent.name
    return p.name


def derive_dataset_name(json_dir):
    """Derive dataset name from config.yaml, falling back to directory name."""
    cfg = load_run_config(json_dir)
    if cfg:
        data = cfg.get("data", [])
        if isinstance(data, list):
            return "_".join(data) if data else "unknown"
        return str(data)
    p = Path(json_dir).resolve()
    if p.name == "checkpoints":
        return p.parent.parent.name
    return "unknown"


def collect_dir_metrics(json_dir, ckpt_type):
    metrics = {}
    for f in sorted(Path(json_dir).glob("*.json")):
        if ckpt_type in f.name:
            perturb = parse_perturbation(f.name)
            if perturb is not None:
                metrics[perturb] = extract_metrics(DIR_FILE_MANAGER.open_json(f))
    return metrics


def find_aligned_perturbations(all_metrics):
    if not all_metrics:
        return []
    common = set.intersection(*(set(m.keys()) for m in all_metrics.values()))
    return [p for p in PERTURB_ORDER if p in common]


def plot_perturb_comparison(all_metrics, model_names, perturbations, metric, dataset, save_path):
    n_models = len(model_names)
    n_perturbs = len(perturbations)
    x = np.arange(n_models)
    width = 0.8 / n_perturbs

    fig, ax = plt.subplots(figsize=(n_models * 5, 6))
    for i, perturb in enumerate(perturbations):
        means = [all_metrics[name][perturb][metric]["mean"] for name in model_names]
        stds = [all_metrics[name][perturb][metric]["std"] for name in model_names]
        offset = (i - (n_perturbs - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds,
               label=perturb, capsize=4, color=PASTEL_COLORS[i % len(PASTEL_COLORS)],
               edgecolor=PASTEL_COLORS[i % len(PASTEL_COLORS)], linewidth=0.1)

    ax.set_ylabel(f"{metric} (%)", fontsize=17)
    ax.set_title(f"The Effect of Perturbations on {dataset}", fontsize=19)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=14, ha="center")
    ax.tick_params(axis="y", labelsize=15)
    ax.legend(title="Perturbation", fontsize=12, title_fontsize=13,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    if metric == "F1":
        top = 100
    elif "ecg-qa" in dataset:
        top = 100
    elif "ecg-instruct-45k" in dataset:
        top = 30
    ax.set_ylim(bottom=0, top = top)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {metric} plot to {save_path}")


def visualize_perturb_comparison(json_dirs, output_dir, ckpt_type):
    all_metrics = {}
    model_names = []
    datasets = set()
    for d in json_dirs:
        name = derive_model_name(d)
        metrics = collect_dir_metrics(d, ckpt_type)
        if not metrics:
            print(f"Warning: no result JSONs in {d}, skipping.")
            continue
        all_metrics[name] = metrics
        model_names.append(name)
        datasets.add(derive_dataset_name(d))

    if not all_metrics:
        print("No metrics found in any directory.")
        return

    perturbations = find_aligned_perturbations(all_metrics)
    if not perturbations:
        print("No aligned perturbation types found across all directories.")
        return

    dataset_label = "_".join(sorted(datasets))
    print(f"Models: {model_names}")
    print(f"Dataset: {dataset_label}")
    print(f"Aligned perturbations: {perturbations}")
    os.makedirs(output_dir, exist_ok=True)
    for metric in ["Accuracy", "F1"]:
        save_path = os.path.join(output_dir, f"{metric}_{dataset_label}_{ckpt_type}_comparison.png")
        plot_perturb_comparison(all_metrics, model_names, perturbations, metric, dataset_label, save_path)


def parse_step_epoch(filename):
    m = STEP_EPOCH_RE.match(Path(filename).stem)
    return (int(m.group(1)), int(m.group(2))) if m else None


def collect_stepwise_metrics(json_dir):
    seen, entries = set(), []
    for f in sorted(Path(json_dir).glob("step_epoch_*.json")):
        key = parse_step_epoch(f.name)
        if key is None or key in seen:
            continue
        data = DIR_FILE_MANAGER.open_json(f)
        if "per_class_acc" not in data:
            continue
        seen.add(key)
        entries.append({"epoch": key[0], "step": key[1], "per_class_acc": data["per_class_acc"]})
    entries.sort(key=lambda e: (e["epoch"], e["step"]))
    return entries

def create_class_mapping(title):
    labels = title.split("-")[2:]
    if "ecg" in labels:
        order = ["ecg", "noise", "flatline"]
    else:
        order = ["noise", "flatline"]
    keys = "ABC"
    return {keys[i]: l for i, l in enumerate(l for l in order if l in labels)}


def plot_stepwise_per_class_accuracy(entries, title, save_path, dataset):
    class_map = create_class_mapping(dataset)
    classes = sorted(entries[0]["per_class_acc"].keys())
    x = np.arange(len(entries))
    x_labels = [f"{e['step']}" for e in entries]
    fig, ax = plt.subplots(figsize=(max(10, len(entries) * 1.0), 7))
    for i, cls in enumerate(classes):
        means = np.array([e["per_class_acc"][cls]["mean"] for e in entries])
        stds = np.array([e["per_class_acc"][cls]["std"] for e in entries])
        color = plt.cm.tab10(i % 10)
        ax.plot(x, means, marker="o", label=class_map[cls], color=color, linewidth=3, markersize=8)
        ax.fill_between(x, means - stds, means + stds, alpha=0.3, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=14, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=18)
    ax.set_xlabel("Training Step", fontsize=18)
    ax.set_title(title, fontsize=20, fontweight="bold")
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(title="Class", fontsize=12, title_fontsize=13,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved stepwise per-class accuracy plot to {save_path}")

def visualize_stepwise_combined(stepwise_dirs, output_dir):
    """Plot per-class accuracy from multiple runs on a single figure."""
    os.makedirs(output_dir, exist_ok=True)

    all_runs = []
    for d in stepwise_dirs:
        entries = collect_stepwise_metrics(d)
        if not entries:
            print(f"No stepwise JSONs with per_class_acc in {d}, skipping.")
            continue
        cfg = load_run_config(d)
        label = f"ELM: {cfg['elm']} | LLM: {cfg['llm']} | Enc: {cfg['encoder']}" if cfg else Path(d).resolve().name
        dataset = derive_dataset_name(d)
        all_runs.append({"entries": entries, "label": label, "dataset": dataset, "dir": d})

    if not all_runs:
        print("No valid stepwise data found.")
        return

    # use class mapping from the first run
    class_map = create_class_mapping(all_runs[0]["dataset"])
    classes = sorted(all_runs[0]["entries"][0]["per_class_acc"].keys())

    fig, axes = plt.subplots(1, len(classes), figsize=(8 * len(classes), 7), sharey=True)
    if len(classes) == 1:
        axes = [axes]

    linestyles = ["-", "--", "-.", ":"]
    run_colors = [plt.cm.tab10(i) for i in range(len(all_runs))]

    for cls_idx, cls in enumerate(classes):
        ax = axes[cls_idx]
        for run_idx, run in enumerate(all_runs):
            entries = run["entries"]
            steps = [e["step"] for e in entries]
            means = np.array([e["per_class_acc"][cls]["mean"] for e in entries])
            stds = np.array([e["per_class_acc"][cls]["std"] for e in entries])
            color = run_colors[run_idx]
            ls = linestyles[run_idx % len(linestyles)]
            ax.plot(steps, means, marker="o", label=run["label"], color=color,
                    linestyle=ls, linewidth=2.5, markersize=6)
            ax.fill_between(steps, means - stds, means + stds, alpha=0.15, color=color)

        ax.set_title(f"Class: {class_map.get(cls, cls)}", fontsize=16)
        ax.set_xlabel("Training Step", fontsize=14)
        if cls_idx == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=14)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=12)

    # single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=min(len(all_runs), 3), fontsize=11)
    fig.suptitle("Per-Class Accuracy Over Training", fontsize=18, fontweight="bold", y=1.06)
    fig.tight_layout()
    save_path = os.path.join(output_dir, "stepwise_per_class_combined.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined stepwise plot to {save_path}")

def visualize_stepwise(stepwise_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for d in stepwise_dirs:
        entries = collect_stepwise_metrics(d)
        if not entries:
            print(f"No stepwise JSONs with per_class_acc in {d}, skipping.")
            continue
        cfg = load_run_config(d)
        label = f"ELM: {cfg['elm']} LLM: {cfg['llm']} Encoder: {cfg['encoder']}" if cfg else Path(d).resolve().name
        dataset = derive_dataset_name(d)
        # title = f"Per-Class Accuracy Over Training\n{label}\n{dataset}"
        title = "Per-Class Accuracy Evaluated\non Test Set Over Training Steps"
        safe = Path(d).resolve().name
        if safe == "checkpoints":
            safe = Path(d).resolve().parent.name
        save_path = os.path.join(output_dir, f"stepwise_per_class_{safe}_{dataset}.png")
        plot_stepwise_per_class_accuracy(entries, title, save_path, dataset)


def main():
    args = get_args("analyze")
    if args.json_dirs:
        visualize_perturb_comparison(args.json_dirs, args.output_dir, args.ckpt_type)
    if args.stepwise_dirs:
        if len(args.stepwise_dirs) > 1:
            visualize_stepwise_combined(args.stepwise_dirs, args.output_dir)
        else:
            visualize_stepwise(args.stepwise_dirs, args.output_dir)


if __name__ == "__main__":
    main()