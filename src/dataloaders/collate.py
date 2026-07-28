import torch
from torch.utils.data import default_collate


def _mapping_structure(x):
    return {key: _mapping_structure(value) for key, value in x.items()} if isinstance(x, dict) else None


def left_pad_elm(x: dict, target_len: int, pad_id: int) -> dict:
    """Left-pad ELM sequence tensors along their last dimension."""
    pad_len = target_len - x["elm_input_ids"].shape[-1]
    if pad_len < 0:
        raise ValueError(f"ELM sequence exceeds target length {target_len}")
    if pad_len == 0:
        return x

    x = dict(x)
    for key, value in (("elm_input_ids", pad_id), ("elm_attention_mask", 0), ("elm_labels", -100)):
        if key in x:
            x[key] = torch.nn.functional.pad(x[key], (pad_len, 0), value=value)
    if "signal_id_indices" in x:
        indices = x["signal_id_indices"]
        x["signal_id_indices"] = torch.where(indices >= 0, indices + pad_len, indices)
    return x


def collate_elm(batch: list[dict], pad_id: int) -> dict | None:
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    structure = _mapping_structure(batch[0])
    if any(_mapping_structure(item) != structure for item in batch[1:]):
        raise ValueError("Batch items have different mapping structures")
    target_len = max(item["elm_input_ids"].shape[-1] for item in batch)
    return default_collate([left_pad_elm(item, target_len, pad_id) for item in batch])
