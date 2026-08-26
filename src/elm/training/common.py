from collections.abc import Mapping
import math
import torch
from transformers import get_scheduler

def build_scheduler(config, optimizer, dataloader):
    training = config["training"]
    accumulation_steps = training["gradient_accumulation_steps"]
    if not len(dataloader) or accumulation_steps < 1:
        raise ValueError("Dataloader and gradient_accumulation_steps must be positive")
    steps = math.ceil(len(dataloader) / accumulation_steps)
    steps *= training["epochs"]
    if training["training_stage"] == "rl":
        steps *= config["rl"]["updates_per_rollout"]
    if steps < 1:
        raise ValueError("Total optimizer steps must be positive")
    warmup_ratio = config["optimizer"]["warmup_ratio"]
    if not 0 <= warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    return get_scheduler(config["optimizer"]["scheduler"], optimizer,
                         math.ceil(steps * warmup_ratio), steps)

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

def optimizer_step(model, optimizer, scheduler, max_grad_norm: float) -> None:
    if max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
