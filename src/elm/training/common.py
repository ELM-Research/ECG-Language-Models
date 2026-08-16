from collections.abc import Mapping
import torch

def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=device.type == "cuda")
    if isinstance(value, Mapping):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value

def begin_epoch(model, dataloader, epoch: int) -> torch.device:
    if not len(dataloader):
        raise ValueError("Training dataloader is empty")
    model.train()
    if hasattr(dataloader.sampler, "set_epoch"):
        dataloader.sampler.set_epoch(epoch)
    return next(model.parameters()).device

def optimizer_step(model, optimizer, max_grad_norm: float) -> None:
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
