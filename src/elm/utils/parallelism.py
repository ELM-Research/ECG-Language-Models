import os
import torch
from torch import distributed
from torch.distributed.fsdp import (
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
    register_fsdp_forward_method,
)
from torch.distributed.tensor import DTensor

def cleanup():
    if distributed.is_initialized():
        try:
            distributed.destroy_process_group()
        except OSError:
            pass


def configure_runtime(config: dict) -> None:
    gpu_config = config.get("gpu", {})
    torch.set_float32_matmul_precision(gpu_config.get("matmul_precision", "high"))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = gpu_config.get("cudnn_benchmark", True)


def init_dist(strategy: str | None) -> None:
    if strategy is None:
        return
    if strategy != "fsdp2":
        raise ValueError(f"Unknown distributed strategy: {strategy}")
    device = torch.device("cuda", get_local_rank())
    torch.cuda.set_device(device)
    distributed.init_process_group(device_id=device)


def get_local_rank() -> int: return int(os.environ.get("LOCAL_RANK", 0))

def get_rank() -> int: return distributed.get_rank() if distributed.is_initialized() else 0

def get_world_size() -> int: return distributed.get_world_size() if distributed.is_initialized() else 1

def is_main() -> bool: return get_rank() == 0

def print_training_setup(config: dict) -> None:
    if not is_main():
        return
    training = config["training"]
    world_size = get_world_size()
    micro_batch = training["batch_size"]
    accumulation = training["gradient_accumulation_steps"]
    global_batch = micro_batch * accumulation * world_size
    details = (
        f"stage={training['training_stage']}, world_size={world_size}, "
        f"micro_batch/gpu={micro_batch}, accumulation={accumulation}, "
        f"global_batch/update={global_batch}"
    )
    if training["training_stage"] == "rl":
        group_size = config["rl"]["group_size"]
        details += f", sampled_responses/update={global_batch * group_size}"
    print(f"Training setup: {details}", flush=True)
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(get_local_rank())
        memory_gib = properties.total_memory / 1024**3
        print(
            f"CUDA setup: {properties.name}, {memory_gib:.1f} GiB, "
            f"compute capability {properties.major}.{properties.minor}; "
            f"matmul={torch.get_float32_matmul_precision()}, "
            f"cuDNN benchmark={torch.backends.cudnn.benchmark}",
            flush=True,
        )

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


def setup_model(model: torch.nn.Module, gpu_config: str | None) -> torch.nn.Module:
    if gpu_config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    if gpu_config["strategy"] is None:
        return model.to(get_device())

    # uniform FP32 originals satisfy FSDP2 and remain the optimizer's sharded master weights.
    model.float()
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    block_names = {name for module in model.modules()
                   for name in (getattr(module, "_no_split_modules", None) or ())}
    for module in reversed(list(model.modules())):
        if type(module).__name__ in block_names:
            fully_shard(module, mp_policy=mp_policy,
                        reshard_after_forward=False)
    model = fully_shard(model, mp_policy=mp_policy)
    if hasattr(model, "generate"):
        register_fsdp_forward_method(model, "generate")
    print_parallelism(model)
    return model

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_parallelism(model: torch.nn.Module) -> None:
    parameters = list(model.parameters())
    sharded = [parameter for parameter in parameters if isinstance(parameter, DTensor)]
    if len(sharded) != len(parameters):
        raise RuntimeError(f"FSDP2 left {len(parameters) - len(sharded)} parameter tensors unsharded")
    if {parameter.dtype for parameter in sharded} != {torch.float32}:
        raise RuntimeError("FSDP2 optimizer parameters must all be float32")
    groups = sum(isinstance(module, FSDPModule) for module in model.modules())
    status = (f"{groups} groups, {len(sharded)}/{len(parameters)} parameter tensors sharded, "
              f"{sum(p.to_local().numel() for p in sharded):,}/"
              f"{sum(p.numel() for p in sharded):,} elements local; "
              "fp32 shards, bf16 compute, fp32 reduce")
    print(f"[rank {get_rank()}/{get_world_size()}] fsdp2 on {parameters[0].device}: {status}", flush=True)