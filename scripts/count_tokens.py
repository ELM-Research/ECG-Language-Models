#!/usr/bin/env python3
"""Report untruncated token-length extrema for the SFT stages."""

import argparse
import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

from matplotlib.figure import Figure
from tqdm import tqdm

from elm.config.load import load_config
from elm.data.build import DataBuilder


DEFAULT_CONFIGS = (
    # Path("src/elm/config/experiment/sft_stage1.yaml"),
    Path("src/elm/config/experiment/sft_stage2.yaml"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("src/logs"))
    return parser.parse_args()


def threaded_map(function, items, workers):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = iter(items)
        while batch := tuple(islice(iterator, workers * 8)):
            yield from executor.map(function, batch)


def save_distribution(counts, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = sum(counts.values(), Counter())
    columns = {**counts, "overall": overall}
    lengths = sorted(overall)

    csv_path = output_dir / "sft_token_length_counts.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["token_length", *columns])
        writer.writerows([length, *(column[length] for column in columns.values())]
                         for length in lengths)

    figure = Figure(figsize=(12, 6))
    axis = figure.subplots()
    for name, column in columns.items():
        axis.plot(lengths, [column[length] for length in lengths], label=name)
    axis.set(title="SFT token-length distribution", xlabel="Token length", ylabel="Count")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    plot_path = output_dir / "sft_token_length_distribution.png"
    figure.savefig(plot_path, dpi=200)
    return csv_path, plot_path


def main():
    args = parse_args()
    overall_min = (float("inf"), None, None)
    overall_max = (-1, None, None)
    counts = {}

    for config_path in args.configs:
        builder = DataBuilder(load_config(config_path))
        tokenizer = builder.build_llm_tokenizer()
        dataset = builder.build_torch_dataset(tokenizer)
        dataset.text_preparer.truncate = False
        _, placeholders = dataset.ecg_modality_preparer(None)

        stage_min = (float("inf"), None)
        stage_max = (-1, None)
        stage_counts = Counter()

        def token_length(instance):
            data_inst = dataset.prepare_text(instance["text"], placeholders)["input_ids"]
            print(tokenizer.decode(data_inst, skip_special_tokens=False))
            return len(data_inst)

        lengths = threaded_map(token_length, dataset.data, args.workers)
        for index, length in enumerate(tqdm(lengths, total=len(dataset), desc=config_path.stem)):
            stage_counts[length] += 1
            stage_min = min(stage_min, (length, index))
            stage_max = max(stage_max, (length, index))

        counts[config_path.stem] = stage_counts
        print(f"{config_path.stem}: {len(dataset):,} instances")
        print(f"  min: {stage_min[0]:,} tokens at index {stage_min[1]}")
        print(f"  max: {stage_max[0]:,} tokens at index {stage_max[1]}")
        overall_min = min(overall_min, (*stage_min, config_path.stem))
        overall_max = max(overall_max, (*stage_max, config_path.stem))
        del token_length, lengths, dataset, builder, tokenizer

    print("overall:")
    print(f"  min: {overall_min[0]:,} tokens in {overall_min[2]} at index {overall_min[1]}")
    print(f"  max: {overall_max[0]:,} tokens in {overall_max[2]} at index {overall_max[1]}")
    csv_path, plot_path = save_distribution(counts, args.output_dir)
    print(f"counts: {csv_path}")
    print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()