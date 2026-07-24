import torch
from torch import distributed
import os

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