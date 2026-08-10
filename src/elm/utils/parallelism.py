import os

import torch
from torch import distributed
from torch.distributed.fsdp import FSDPModule, fully_shard, register_fsdp_forward_method
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel as DDP

def barrier():
    if distributed.is_initialized():
        distributed.barrier()

def cleanup():
    if distributed.is_initialized():
        try:
            distributed.destroy_process_group()
        except OSError:
            pass

def broadcast_value(value, src: int = 0):
    if not distributed.is_initialized():
        return value

    values = [value]
    distributed.broadcast_object_list(values, src=src)
    return values[0]

def init_dist():
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


def setup_model(model: torch.nn.Module, strategy: str | None) -> torch.nn.Module:
    if strategy == "fsdp2":
        block_names = {name for module in model.modules()
                       for name in (getattr(module, "_no_split_modules", None) or ())}
        for module in reversed(list(model.modules())):
            if type(module).__name__ in block_names:
                fully_shard(module)
        model = fully_shard(model)
        if hasattr(model, "generate"):
            register_fsdp_forward_method(model, "generate")
    elif strategy == "ddp":
        device = get_device(distributed=True)
        model = DDP(model.to(device), device_ids=[device.index])
    elif strategy is None:
        return model.to(get_device())
    else:
        raise ValueError(f"Unknown distributed strategy: {strategy}")
    print_parallelism(model, strategy)
    return model

def get_device(distributed: bool = False) -> torch.device:
    if distributed:
        return torch.device("cuda", get_local_rank())
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_parallelism(model: torch.nn.Module, strategy: str | None) -> None:
    parameters = list(model.parameters())
    if strategy == "fsdp2":
        sharded = [parameter for parameter in parameters if isinstance(parameter, DTensor)]
        groups = sum(isinstance(module, FSDPModule) for module in model.modules())
        status = (f"{groups} groups, {len(sharded)}/{len(parameters)} parameter tensors sharded, "
                  f"{sum(p.to_local().numel() for p in sharded):,}/"
                  f"{sum(p.numel() for p in sharded):,} elements local")
    else:
        status = f"{sum(p.numel() for p in parameters):,} elements {'replicated' if strategy == 'ddp' else 'local'}"
    print(f"[rank {get_rank()}/{get_world_size()}] {strategy or 'single'} on {parameters[0].device}: {status}", flush=True)