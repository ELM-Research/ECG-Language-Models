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


def pad_batch_to_len(batch: dict, target_len: int, pad_id: int) -> dict:
    """Left-pad a collated batch up to target_len (used to force worst-case-shape warmup steps)."""
    pad = target_len - batch["elm_input_ids"].shape[1]
    if pad <= 0:
        return batch
    F = torch.nn.functional
    batch = dict(batch)
    batch["elm_input_ids"] = F.pad(batch["elm_input_ids"], (pad, 0), value=pad_id)
    batch["elm_attention_mask"] = F.pad(batch["elm_attention_mask"], (pad, 0), value=0)
    if "elm_labels" in batch:
        batch["elm_labels"] = F.pad(batch["elm_labels"], (pad, 0), value=-100)
    if "signal_id_indices" in batch:
        batch["signal_id_indices"] = torch.where(batch["signal_id_indices"] >= 0, batch["signal_id_indices"] + pad, batch["signal_id_indices"])
    return batch

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