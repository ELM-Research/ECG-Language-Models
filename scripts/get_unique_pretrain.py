#!/usr/bin/env python3
"""Plot the most frequent texts in each pretraining dataset."""

import csv
from collections import Counter
from pathlib import Path
from textwrap import shorten

from matplotlib.figure import Figure

from elm.config.load import load_config
from elm.data.build import DataBuilder


TOP_K = 20
MAX_LABEL_LENGTH = 100

CONFIGS = (
    Path("src/elm/config/experiment_9b/pretrain_stage1.yaml"),
    Path("src/elm/config/experiment_9b/pretrain_stage2.yaml"),
)

OUTPUT_DIR = Path("src/logs")


def save_top_texts(name, counts):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    top_texts = counts.most_common(TOP_K)

    csv_path = OUTPUT_DIR / f"{name}_top_texts.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "count", "text"])
        for rank, (text, count) in enumerate(top_texts, start=1):
            writer.writerow([rank, count, text])

    texts = [
        shorten(text.replace("\n", " "), width=MAX_LABEL_LENGTH, placeholder="...")
        for text, _ in reversed(top_texts)
    ]
    frequencies = [count for _, count in reversed(top_texts)]

    figure = Figure(figsize=(12, 8))
    axis = figure.subplots()
    axis.barh(texts, frequencies)
    axis.set(
        title=f"{name}: Top {TOP_K} Most Frequent Texts",
        xlabel="Count",
    )
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()

    plot_path = OUTPUT_DIR / f"{name}_top_texts.png"
    figure.savefig(plot_path, dpi=200)

    return csv_path, plot_path


def main():
    for config_path in CONFIGS:
        builder = DataBuilder(load_config(config_path))
        tokenizer = builder.build_llm_tokenizer()
        dataset = builder.build_torch_dataset(tokenizer)

        counts = Counter(instance["text"].replace("; .;", ";") for instance in dataset.data)
        total = len(dataset)
        unique = len(counts)

        csv_path, plot_path = save_top_texts(config_path.stem, counts)

        print(f"{config_path.stem}:")
        print(f"  total texts: {total:,}")
        print(f"  unique texts: {unique:,}")
        print(f"  unique percentage: {100 * unique / total:.2f}%")
        print(f"  counts: {csv_path}")
        print(f"  plot: {plot_path}")


if __name__ == "__main__":
    main()
