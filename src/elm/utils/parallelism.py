import os

import torch
from torch import distributed
from torch.distributed.fsdp import FSDPModule, fully_shard, register_fsdp_forward_method
from torch.distributed.tensor import DTensor


def cleanup():
    if distributed.is_initialized():
        try:
            distributed.destroy_process_group()
        except OSError:
            pass


def init_dist(strategy: str | None) -> None:
    if strategy is None:
        return
    if strategy != "fsdp2":
        raise ValueError(f"Unknown distributed strategy: {strategy}")
    device = torch.device("cuda", get_local_rank())
    torch.cuda.set_device(device)
    distributed.init_process_group(device_id=device)


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_rank() -> int:
    return distributed.get_rank() if distributed.is_initialized() else 0


def get_world_size() -> int:
    return distributed.get_world_size() if distributed.is_initialized() else 1


def is_main() -> bool:
    return get_rank() == 0


def distributed_mean(total: float, count: float, device: torch.device) -> float:
    stats = torch.tensor((total, count), dtype=torch.float64, device=device)
    if distributed.is_initialized():
        distributed.all_reduce(stats)
    return (stats[0] / stats[1].clamp_min(1)).item()


def any_process(value: bool, device: torch.device) -> bool:
    flag = torch.tensor(value, dtype=torch.int64, device=device)
    if distributed.is_initialized():
        distributed.all_reduce(flag, op=distributed.ReduceOp.MAX)
    return bool(flag.item())


def set_gradient_sync(model: torch.nn.Module, enabled: bool) -> None:
    if isinstance(model, FSDPModule):
        model.set_requires_gradient_sync(enabled)


def setup_model(model: torch.nn.Module, strategy: str | None) -> torch.nn.Module:
    if strategy is None:
        return model.to(get_device())
    if strategy != "fsdp2":
        raise ValueError(f"Unknown distributed strategy: {strategy}")

    block_names = {name for module in model.modules()
                   for name in (getattr(module, "_no_split_modules", None) or ())}
    for module in reversed(list(model.modules())):
        if type(module).__name__ in block_names:
            fully_shard(module)
    model = fully_shard(model)
    if hasattr(model, "generate"):
        register_fsdp_forward_method(model, "generate")
    print_parallelism(model)
    return model


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_parallelism(model: torch.nn.Module) -> None:
    parameters = list(model.parameters())
    sharded = [parameter for parameter in parameters if isinstance(parameter, DTensor)]
    groups = sum(isinstance(module, FSDPModule) for module in model.modules())
    status = (f"{groups} groups, {len(sharded)}/{len(parameters)} parameter tensors sharded, "
              f"{sum(p.to_local().numel() for p in sharded):,}/"
              f"{sum(p.numel() for p in sharded):,} elements local")
    print(f"[rank {get_rank()}/{get_world_size()}] fsdp2 on {parameters[0].device}: {status}", flush=True)
