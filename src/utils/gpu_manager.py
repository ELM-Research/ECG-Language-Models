"""GPU placement and parallel-strategy dispatch.

Free helpers (`is_main`, `get_world_size`, `init_dist`, ...) are thin shims over
the global ParallelContext, kept for backward compatibility with call sites
across the codebase.
"""

import argparse
from typing import Iterable

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.parallel_context import (
    ParallelContext,
    get_parallel_context,
    init_parallel_context,
)


def init_dist(strategy: str = "ddp") -> ParallelContext:
    return init_parallel_context(strategy=strategy)


def get_local_rank() -> int:
    return get_parallel_context().local_rank


def get_rank() -> int:
    return get_parallel_context().global_rank


def get_world_size() -> int:
    return get_parallel_context().world_size


def is_main() -> bool:
    return get_parallel_context().is_main


def barrier() -> None:
    get_parallel_context().barrier()


def cleanup() -> None:
    get_parallel_context().cleanup()


def broadcast_value(val, src: int = 0):
    return get_parallel_context().broadcast_value(val, src=src)


def train_dev_break(enabled: bool, batch: dict, loss_value: float) -> bool:
    if not enabled:
        return False
    should_break = False
    if is_main():
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            else:
                print(f"{key}: {type(value)}")
        print("loss", loss_value)
        should_break = True
    return broadcast_value(should_break, src=0)


class GPUSetup:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ctx = get_parallel_context()

    def setup_gpu(self, model: torch.nn.Module, find_unused_parameters: bool) -> torch.nn.Module:
        device = self.get_device()
        model = model.to(device)
        strategy = self.ctx.strategy
        if strategy == "ddp":
            model = DDP(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=find_unused_parameters,
            )
            if self.args.torch_compile:
                model = torch.compile(model)
        elif strategy == "fsdp":
            # FSDP path applies torch.compile per-unit inside _wrap_fsdp.
            # Compiling the outer FSDP-wrapped model traces through the
            # pre-forward all-gather hooks and crashes on DTensor params.
            model = self._wrap_fsdp(model)
        else:
            if self.args.torch_compile:
                model = torch.compile(model)
        if is_main():
            print(f"[GPUSetup] strategy={strategy} | find_unused_parameters={find_unused_parameters} | torch_compile={self.args.torch_compile}")
        return model

    def _wrap_fsdp(self, model: torch.nn.Module) -> torch.nn.Module:
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        # Megatron / torchtitan / nanotron recipe: keep fp32 master weights
        # and fp32 optimizer state, run forward + backward in bf16 via the
        # MixedPrecisionPolicy, and reduce gradients in fp32.
        # Casting up to fp32 also satisfies FSDP2's "uniform original dtype
        # per unit" rule: HF LLMs load in bf16 while the encoder/connector
        # default to fp32, so a single dtype must be chosen — fp32 preserves
        # the encoder's precision and gives the LLM a master copy to update.
        model = model.to(torch.float32)
        dp_mesh = self.ctx.dp_mesh
        compile_units = self.args.torch_compile
        for module in self._collect_wrap_modules(model):
            fully_shard(module, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=True)
            if compile_units:
                # Per-unit compile: each FSDP unit's forward is a fresh Dynamo
                # graph and FSDP's hooks are honoured at the unit boundary.
                module.compile()
        fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy, reshard_after_forward=True)
        return model

    def _collect_wrap_modules(self, model: torch.nn.Module) -> Iterable[torch.nn.Module]:
        if hasattr(model, "fsdp_wrap_modules"):
            return list(model.fsdp_wrap_modules())
        return []

    def get_device(self) -> torch.device:
        if self.ctx.is_distributed and torch.cuda.is_available():
            return torch.device(f"cuda:{self.ctx.local_rank}")
        dev = getattr(self.args, "device", None)
        return torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))

    def print_model_device(self, model: torch.nn.Module, name: str) -> None:
        if is_main():
            print(f"{name} device:", next(model.parameters()).device)
