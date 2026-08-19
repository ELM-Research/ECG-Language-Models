#!/usr/bin/env python3
"""Run a short, non-checkpointing training profile on the real data and model."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import distributed

from elm.config.load import load_config
from elm.data.build import build_data
from elm.model import build_model
from elm.optimizer import build_optimizer
from elm.training.checkpoint import Checkpointer
from elm.training.common import build_scheduler
from elm.training.rl.trainer import train_epoch as train_rl_epoch
from elm.training.supervised import train_epoch as train_supervised_epoch
from elm.utils.parallelism import (
    cleanup,
    configure_runtime,
    get_device,
    get_world_size,
    init_dist,
    is_main,
    print_training_setup,
    setup_model,
)
from elm.utils.seed import set_seed

LOG_PATH = Path("src/logs/profiles.jsonl")


class LimitedLoader:
    """Expose only a few batches while retaining the sampler API trainers need."""

    def __init__(self, dataloader, steps: int):
        self.dataloader = dataloader
        self.steps = min(steps, len(dataloader))
        self.sampler = dataloader.sampler
        self.examples = 0
        self.tokens = 0

    def __len__(self) -> int:
        return self.steps

    def __iter__(self):
        for batch in itertools.islice(self.dataloader, self.steps):
            self.examples += batch["input_ids"].shape[0]
            self.tokens += batch["attention_mask"].sum().item()
            yield batch


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--steps", type=positive_int, default=10)
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        help="Override the per-GPU microbatch for this in-memory profile only.",
    )
    return parser.parse_args()


def reduce_profile_stats(loader: LimitedLoader, elapsed: float, device: torch.device):
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    stats = torch.tensor(
        [loader.examples, loader.tokens, elapsed, peak_allocated, peak_reserved],
        dtype=torch.float64,
        device=device,
    )
    if distributed.is_initialized():
        distributed.all_reduce(stats[:2])
        distributed.all_reduce(stats[2:], op=distributed.ReduceOp.MAX)
    return stats.tolist()


def build_record(config, args, steps, result, stats) -> dict:
    """Flatten the knobs worth sweeping plus the measured throughput."""
    examples, tokens, elapsed, peak_allocated, peak_reserved = stats
    training = config["training"]
    model = config["model"]
    world_size = get_world_size()
    return {
        "time": datetime.now().isoformat(timespec="seconds"),
        "config": args.config.stem,
        "model": model["name"],
        "language_model": model["language_model"],
        "vision_model": Path(model["vision_model"]).name,
        "peft": model["peft"],
        "lora_rank": model["lora_rank"],
        "trainable": model["trainable"],
        "training_stage": training["training_stage"],
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "strategy": config["gpu"]["strategy"],
        "world_size": world_size,
        "batch_size": training["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "global_batch": training["batch_size"] * training["gradient_accumulation_steps"] * world_size,
        "num_workers": training["num_workers"],
        "prefetch_factor": training["prefetch_factor"],
        "segment_length": config["segment_length"],
        "num_ecg_tokens": model["num_ecg_tokens"],
        "patch_size": model["patch_size"],
        "truncation_length": model["truncation_length"],
        "steps": steps,
        "wall_time_s": round(elapsed, 2),
        "examples_per_s": round(examples / elapsed, 2),
        "tokens_per_s": round(tokens / elapsed),
        "peak_allocated_gib": round(peak_allocated, 2),
        "peak_reserved_gib": round(peak_reserved, 2),
        "optimizer_steps": result["optimizer_steps"],
    }


def log_record(record: dict) -> None:
    print("\nProfile result")
    for key, value in record.items():
        print(f"  {key}: {value}")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as file:
        file.write(json.dumps(record) + "\n")
    print(f"Appended profile to {LOG_PATH}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    config["training"]["epochs"] = 1
    config["development"] = True
    config["wandb"] = False

    configure_runtime(config)
    init_dist(config["gpu"]["strategy"])
    try:
        print_training_setup(config)
        set_seed(config["seed"])
        tokenizer, dataloader = build_data(config)
        dataloader = LimitedLoader(dataloader, args.steps)
        model = setup_model(build_model(config, tokenizer), config["gpu"])
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, dataloader)
        checkpointer = Checkpointer(model, tokenizer, optimizer, scheduler, None, 1, enabled=False)

        device = get_device()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        if config["training"]["training_stage"] == "rl":
            result = train_rl_epoch(
                model, optimizer, scheduler, checkpointer, dataloader, tokenizer, config, 0
            )
        else:
            result = train_supervised_epoch(
                model, optimizer, scheduler, checkpointer, dataloader, config, 0
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        stats = reduce_profile_stats(dataloader, elapsed, device)

        if is_main():
            record = build_record(config, args, len(dataloader), result, stats)
            if config["training"]["training_stage"] == "rl":
                record["group_size"] = config["rl"]["group_size"]
                record["sampled_responses"] = round(stats[0] * config["rl"]["group_size"])
            log_record(record)
    finally:
        cleanup()


if __name__ == "__main__":
    main()