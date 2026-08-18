#!/usr/bin/env python3
"""Run a short, non-checkpointing training profile on the real data and model."""

from __future__ import annotations

import argparse
import itertools
import time
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    config["training"]["epochs"] = 1
    config["development"] = True
    config["wandb"] = False

    strategy = config["gpu"]["strategy"]
    configure_runtime(config)
    init_dist(strategy)
    try:
        print_training_setup(config)
        set_seed(config["seed"])
        tokenizer, dataloader = build_data(config)
        dataloader = LimitedLoader(dataloader, args.steps)
        model = setup_model(build_model(config, tokenizer), strategy)
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, dataloader)
        checkpointer = Checkpointer(model, tokenizer, None, 1, enabled=False)

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
        examples, tokens, elapsed, peak_allocated, peak_reserved = reduce_profile_stats(
            dataloader, elapsed, device
        )

        if is_main():
            print("\nProfile result")
            print(f"  measured batches/rank: {len(dataloader)}")
            print(f"  wall time: {elapsed:.2f} s")
            print(f"  global examples/s: {examples / elapsed:.2f}")
            print(f"  input tokens/s: {tokens / elapsed:.0f}")
            print(f"  peak allocated/GPU: {peak_allocated:.2f} GiB")
            print(f"  peak reserved/GPU: {peak_reserved:.2f} GiB")
            print(f"  optimizer steps: {result['optimizer_steps']}")
            if config["training"]["training_stage"] == "rl":
                group_size = config["rl"]["group_size"]
                print(f"  sampled responses: {examples * group_size:.0f}")
            print(f"  data-parallel world size: {get_world_size()}")
    finally:
        cleanup()


if __name__ == "__main__":
    main()
