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

def apply_fsdp2(model, mesh=None, mp_policy=None):
    """Shard transformer blocks bottom-up, then shard the remaining root parameters."""
    from torch.distributed.fsdp import FSDPModule, fully_shard, register_fsdp_forward_method

    if not distributed.is_initialized():
        raise RuntimeError("Initialize torch.distributed before applying FSDP2")
    if isinstance(model, FSDPModule):
        return model
    kwargs = {"mesh": mesh} if mesh is not None else {}
    if mp_policy is not None:
        kwargs["mp_policy"] = mp_policy
    block_names = {name for module in model.modules()
                   for name in getattr(module, "_no_split_modules", ())}
    for module in tuple(model.modules()):
        if module.__class__.__name__ in block_names:
            fully_shard(module, reshard_after_forward=True, **kwargs)
    fully_shard(model, reshard_after_forward=False, **kwargs)
    if hasattr(model, "generate"):
        register_fsdp_forward_method(model, "generate")
    return model

def get_full_state_dict(model):
    """Gather an FSDP2-compatible CPU state dict on rank 0."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    return get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))

def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))

def get_rank() -> int:
    return distributed.get_rank() if distributed.is_initialized() else 0

def get_world_size() -> int:
    return distributed.get_world_size() if distributed.is_initialized() else 1

def is_main() -> bool:
    return get_rank() == 0
