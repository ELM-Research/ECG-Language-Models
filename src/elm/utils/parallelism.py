import torch
from torch import distributed
import os

import torch, argparse, os, torch.distributed as dist
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
    distributed.init_process_group(backend=distributed.get_default_backend_for_device(device),
                                   device_id = device)

def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))

def get_rank() -> int:
    return distributed.get_rank() if distributed.is_initialized() else 0

def get_world_size() -> int:
    return distributed.get_world_size() if distributed.is_initialized() else 1

def is_main() -> bool:
    return get_rank() == 0


def setup_gpu(model: torch.nn.Module, find_unused_parameters) -> torch.nn.Module:
    device = get_device()
    model = model.to(device)
    if getattr(self.args, "distributed", False):
        model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=find_unused_parameters)
    if is_main():
        print(f"find_unused_parameters: {find_unused_parameters}")
    if self.args.torch_compile:
        model = torch.compile(model)
    return model

def get_device() -> torch.device:
    return get_multi_device() if getattr(self.args, "distributed", False) else self.get_single_device()

def get_single_device(self) -> torch.device:
    dev = getattr(self.args, "device", None)
    return torch.device(dev or ("cuda" if torch.cuda.is_available() else "cpu"))

def get_multi_device(self) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{get_local_rank()}")
    return torch.device("cpu")

def print_model_device(self, model: torch.nn.Module, name: str) -> None:
    if is_main():
        print(f"{name} device:", next(model.parameters()).device)