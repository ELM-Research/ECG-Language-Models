#!/usr/bin/env python3
"""Measure Orah forward and backward CUDA memory for a dummy token length."""

import argparse
from pathlib import Path

import torch
from torch import distributed
from torch.nn.functional import pad

from elm.config.load import load_config
from elm.data.build import DataBuilder
from elm.model import build_model
from elm.utils.parallelism import (
    cleanup,
    configure_runtime,
    get_device,
    get_world_size,
    init_dist,
    is_main,
    setup_model,
)


GIB = 1024**3
DEFAULT_CONFIG = Path("src/elm/config/experiment/sft_stage1.yaml")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("length", type=positive_int, help="Total number of tokens in each dummy sequence.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        help="Dummy microbatch size; defaults to training.batch_size in the config.",
    )
    parser.add_argument(
        "--single-device",
        action="store_true",
        help="Ignore the config's distributed strategy and profile one unsharded model.",
    )
    return parser.parse_args()


def gib(value: int) -> str:
    return f"{value / GIB:.2f} GiB"


def print_memory(name: str, current: int, peak: int, peak_reserved: int, baseline: int) -> None:
    print(
        f"{name:<16} allocated={gib(current):>10}  "
        f"peak allocated={gib(peak):>10}  peak reserved={gib(peak_reserved):>10}  "
        f"peak over baseline={gib(peak - baseline):>10}"
    )


def maximum_across_ranks(values: tuple[int, ...], device: torch.device) -> list[int]:
    stats = torch.tensor(values, dtype=torch.int64, device=device)
    if distributed.is_initialized():
        distributed.all_reduce(stats, op=distributed.ReduceOp.MAX)
    return stats.tolist()


def profile(args: argparse.Namespace, config: dict) -> None:
    device = get_device()
    if device.type != "cuda":
        raise RuntimeError("This profiler requires a CUDA device")

    tokenizer = DataBuilder(config).build_llm_tokenizer()
    model = setup_model(build_model(config, tokenizer), config["gpu"])
    model.train()

    num_ecg_tokens = model.config.num_ecg_tokens
    if args.length <= num_ecg_tokens:
        raise ValueError(
            f"length must be greater than num_ecg_tokens ({num_ecg_tokens}) "
            "so the dummy batch has a supervised text token"
        )

    batch_size = config["training"]["batch_size"]
    ecg_token_id = model.config.ecg_token_id
    text_token_id = 0 if ecg_token_id != 0 else 1
    input_ids = torch.full((batch_size, args.length), text_token_id, dtype=torch.long, device=device)
    input_ids[:, :num_ecg_tokens] = ecg_token_id
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :num_ecg_tokens] = -100
    ecg_values = torch.zeros(
        (batch_size, model.config.num_leads, model.config.segment_length),
        device=device,
    )

    # Match the loss construction in elm.training.supervised: only retain logits
    # needed for the supervised text suffix.
    label_start = num_ecg_tokens
    logits_to_keep = args.length - label_start + 1
    shift_labels = pad(labels[:, label_start:], (0, 1), value=-100)

    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)

    torch.cuda.reset_peak_memory_stats(device)
    loss = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        ecg_values=ecg_values,
        logits_to_keep=logits_to_keep,
        shift_labels=shift_labels,
        use_cache=False,
    ).loss
    torch.cuda.synchronize(device)
    forward_current = torch.cuda.memory_allocated(device)
    forward_peak = torch.cuda.max_memory_allocated(device)
    forward_peak_reserved = torch.cuda.max_memory_reserved(device)

    torch.cuda.reset_peak_memory_stats(device)
    loss.backward()
    torch.cuda.synchronize(device)
    backward_current = torch.cuda.memory_allocated(device)
    backward_peak = torch.cuda.max_memory_allocated(device)
    backward_peak_reserved = torch.cuda.max_memory_reserved(device)

    stats = maximum_across_ranks(
        (
            baseline,
            baseline_reserved,
            forward_current,
            forward_peak,
            forward_peak_reserved,
            backward_current,
            backward_peak,
            backward_peak_reserved,
        ),
        device,
    )
    if not is_main():
        return
    (
        baseline,
        baseline_reserved,
        forward_current,
        forward_peak,
        forward_peak_reserved,
        backward_current,
        backward_peak,
        backward_peak_reserved,
    ) = stats
    parameter = next(model.parameters())
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"config: {args.config}")
    print(f"strategy: {config['gpu']['strategy'] or 'single device'} ({get_world_size()} rank(s))")
    print(f"tokens: {args.length:,}")
    print(f"batch size per rank: {batch_size}")
    print(f"parameter dtype: {parameter.dtype}")
    print(f"gradient checkpointing: {config['gpu']['gradient_checkpointing']}")
    print_memory("model + inputs", baseline, baseline, baseline_reserved, baseline)
    print_memory("after forward", forward_current, forward_peak, forward_peak_reserved, baseline)
    print_memory("after backward", backward_current, backward_peak, backward_peak_reserved, baseline)
    print(f"loss: {loss.detach().item():.6f}")
    print(f"overall peak allocated: {gib(max(forward_peak, backward_peak))}")
    print(f"overall peak reserved: {gib(max(forward_peak_reserved, backward_peak_reserved))}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.single_device:
        config["gpu"]["strategy"] = None
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    configure_runtime(config)
    init_dist(config["gpu"]["strategy"])
    try:
        profile(args, config)
    finally:
        cleanup()


if __name__ == "__main__":
    main()