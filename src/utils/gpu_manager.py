import torch, argparse, os, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(get_local_rank())


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def batch_to_device(v, device):
    if isinstance(v, torch.Tensor):
        return v.to(device)
    if isinstance(v, dict):
        return {k: batch_to_device(x, device) for k, x in v.items()}
    return v

def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main() -> bool:
    return get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except OSError: pass


def broadcast_value(val, src: int = 0):
    """Broadcast a small Python object (e.g., str/int) without GPU assumptions."""
    if not (dist.is_available() and dist.is_initialized()):
        return val
    obj = [val]
    dist.broadcast_object_list(obj, src=src)
    return obj[0]


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


def assert_max_batch_fits(model, batch: dict, args, optimizer=None, margin: float = 0.10) -> None:
    if batch is None or not torch.cuda.is_available():
        return
    F, ids = torch.nn.functional, batch["elm_input_ids"]
    pad = args.llm_input_len - ids.shape[1]
    if pad <= 0:
        return  # first batch already at the cap; training itself already stresses it
    probe = dict(batch)
    probe["elm_input_ids"] = F.pad(ids, (pad, 0), value=int(ids.reshape(-1)[0]))
    probe["elm_attention_mask"] = F.pad(batch["elm_attention_mask"], (pad, 0), value=0)
    if "elm_labels" in batch:
        probe["elm_labels"] = F.pad(batch["elm_labels"], (pad, 0), value=-100)
    if "signal_id_indices" in batch:
        probe["signal_id_indices"] = torch.where(batch["signal_id_indices"] >= 0, batch["signal_id_indices"] + pad, batch["signal_id_indices"])
    device = next(model.parameters()).device
    probe = {k: batch_to_device(v, device) for k, v in probe.items()}
    torch.cuda.reset_peak_memory_stats(device)
    try:
        model(**probe).loss.backward()
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(f"[fit-check] worst-case batch (batch_size={ids.shape[0]} x llm_input_len="
                           f"{args.llm_input_len}) does not fit; lower --batch_size / --llm_input_len or "
                           f"add --gradient_checkpointing. ({e})") from e
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    # backward peak (params + grads + activations) + optimizer state resident during real steps
    opt_bytes = optimizer.estimated_state_bytes() if optimizer is not None else 0
    peak = torch.cuda.max_memory_allocated(device) + opt_bytes
    free, _ = torch.cuda.mem_get_info(device)
    budget = free + torch.cuda.memory_allocated(device)  # free accounts for other processes; + our resident params
    gb = 1024 ** 3
    if peak * (1 + margin) > budget:
        raise RuntimeError(f"[fit-check] worst-case peak ~{peak / gb:.1f} GB (incl. optimizer state) leaves "
                           f"<{margin:.0%} headroom of {budget / gb:.1f} GB available; lower --batch_size / "
                           f"--llm_input_len or add --gradient_checkpointing.")
    if is_main():
        print(f"[fit-check] OK: worst-case ~{peak / gb:.1f} GB / {budget / gb:.1f} GB available ({margin:.0%} margin kept).")


class GPUSetup:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def setup_gpu(self, model: torch.nn.Module, find_unused_parameters) -> torch.nn.Module:
        device = self.get_device()
        model = model.to(device)
        if getattr(self.args, "distributed", False):
            model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=find_unused_parameters)
        if is_main():
            print(f"find_unused_parameters: {find_unused_parameters}")
        if self.args.torch_compile:
            model = torch.compile(model, dynamic=True)
        return model

    def get_device(self) -> torch.device:
        return self.get_multi_device() if getattr(self.args, "distributed", False) else self.get_single_device()

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