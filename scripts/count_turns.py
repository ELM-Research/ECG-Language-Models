#!/usr/bin/env python3
"""Report the distribution of conversation turns in the SFT stages."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from matplotlib.figure import Figure
from tqdm import tqdm

from elm.config.load import load_config
from elm.data.build import DataBuilder


DEFAULT_CONFIGS = (
    Path("src/elm/config/experiment/sft_stage1.yaml"),
    Path("src/elm/config/experiment/sft_stage2.yaml"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--output-dir", type=Path, default=Path("src/logs"))
    return parser.parse_args()


def count_turns(text):
    """Count messages in one conversation."""
    if isinstance(text, str):
        text = json.loads(text)

    if not isinstance(text, list):
        raise TypeError(
            f"Expected conversation to be a list of messages, got {type(text).__name__}"
        )

    return len(text)


def save_distribution(counts, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    overall = sum(counts.values(), Counter())
    columns = {**counts, "overall": overall}
    turns = sorted(overall)

    csv_path = output_dir / "sft_turn_counts.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["num_turns", *columns])
        writer.writerows(
            [
                num_turns,
                *(column[num_turns] for column in columns.values()),
            ]
            for num_turns in turns
        )

    figure = Figure(figsize=(10, 6))
    axis = figure.subplots()

    width = 0.8 / len(columns)
    offsets = [
        (index - (len(columns) - 1) / 2) * width
        for index in range(len(columns))
    ]

    for offset, (name, column) in zip(offsets, columns.items()):
        axis.bar(
            [num_turns + offset for num_turns in turns],
            [column[num_turns] for num_turns in turns],
            width=width,
            label=name,
        )

    axis.set(
        title="SFT conversation-turn distribution",
        xlabel="Number of turns",
        ylabel="Number of instances",
    )
    axis.set_xticks(turns)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()

    figure.tight_layout()
    plot_path = output_dir / "sft_turn_distribution.png"
    figure.savefig(plot_path, dpi=200)

    return csv_path, plot_path


def main():
    args = parse_args()
    counts = {}

    overall_min = (float("inf"), None, None)
    overall_max = (-1, None, None)

    for config_path in args.configs:
        builder = DataBuilder(load_config(config_path))
        tokenizer = builder.build_llm_tokenizer()
        dataset = builder.build_torch_dataset(tokenizer)

        stage_counts = Counter()
        stage_min = (float("inf"), None)
        stage_max = (-1, None)

        for index, instance in enumerate(
            tqdm(dataset.data, total=len(dataset), desc=config_path.stem)
        ):
            num_turns = count_turns(instance["text"])

            stage_counts[num_turns] += 1
            stage_min = min(stage_min, (num_turns, index))
            stage_max = max(stage_max, (num_turns, index))

        stage_name = config_path.stem
        counts[stage_name] = stage_counts

        print(f"{stage_name}: {len(dataset):,} instances")
        print(f"  min: {stage_min[0]:,} turns at index {stage_min[1]}")
        print(f"  max: {stage_max[0]:,} turns at index {stage_max[1]}")
        print(f"  mean: {sum(stage_counts.elements()) / len(dataset):.2f} turns")

        overall_min = min(overall_min, (*stage_min, stage_name))
        overall_max = max(overall_max, (*stage_max, stage_name))

        del dataset, builder, tokenizer

    print("overall:")
    print(
        f"  min: {overall_min[0]:,} turns "
        f"in {overall_min[2]} at index {overall_min[1]}"
    )
    print(
        f"  max: {overall_max[0]:,} turns "
        f"in {overall_max[2]} at index {overall_max[1]}"
    )

    csv_path, plot_path = save_distribution(counts, args.output_dir)
    print(f"counts: {csv_path}")
    print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()