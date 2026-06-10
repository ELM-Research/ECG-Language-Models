import argparse
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from collections.abc import Mapping, Sequence

from utils.gpu_manager import get_world_size, get_rank

from dataloaders.dataset_mixer import DatasetMixer


class BuildDataLoader:
    def __init__(
        self,
        args: argparse.Namespace,
    ):
        self.args = args
        self.dataset_mixer = DatasetMixer(self.args)

    def build_dataloader(
        self,
    ):
        torch_dataset = self.dataset_mixer.build_torch_dataset()
        torch_data_loader = self.build_torch_dataloader(torch_dataset)
        return torch_data_loader

    def build_torch_dataloader(self, torch_dataset):
        self.pad_token_id = torch_dataset.llm_tokenizer.pad_token_id
        sampler = self.get_torch_dataloader_sampler(torch_dataset)
        if "train" in self.args.mode:
            torch_data_loader = DataLoader(
                torch_dataset,
                batch_size=self.args.batch_size,
                shuffle=(sampler is None),
                num_workers=self.args.num_workers,
                sampler=sampler,
                pin_memory=torch.cuda.is_available(),
                collate_fn=self.collate_fn,
                persistent_workers=(self.args.num_workers > 0),
                prefetch_factor=4 if self.args.num_workers > 0 else None,
            )
        elif "eval" in self.args.mode:
            torch_data_loader = DataLoader(
                torch_dataset,
                batch_size=1,  # batched inference/eval not implemented
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                collate_fn=self.collate_fn,
            )
        return torch_data_loader

    def get_torch_dataloader_sampler(
        self,
        torch_dataset,
    ):
        if self.args.distributed:
            sampler = DistributedSampler(torch_dataset, num_replicas=get_world_size(),
                                         rank=get_rank(), seed=self.args.seed, shuffle=True)
        else:
            sampler = None
        return sampler

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        batch = self._pad_items_to_batch_max(batch)
        self._assert_same_structure_and_shapes(batch)
        return torch.utils.data.dataloader.default_collate(batch)

    def _pad_items_to_batch_max(self, batch):
        """Left-pad each item's sequence tensors to the longest sample in the batch.

        Pad values follow the conventions used when building the items:
        pad_token_id for input ids, -100 for labels (ignored by the loss), 0
        for the attention mask. signal_id_indices hold positions into the
        sequence, so each sample's indices shift by its own pad amount; the -1
        "no signal" sentinel stays negative. In train mode the target length
        is rounded up to a multiple of 8 so matmul shapes stay tensor-core
        friendly; eval batches keep their exact length.
        """
        lengths = [item["elm_input_ids"].shape[0] for item in batch]
        target = max(lengths)
        if "train" in self.args.mode:
            target = -(-target // 8) * 8
            target = min(target, self.args.llm_input_len)
        if all(length == target for length in lengths):
            return batch
        padded_batch = []
        for item, length in zip(batch, lengths):
            pad = target - length
            if pad == 0:
                padded_batch.append(item)
                continue
            item = dict(item)
            item["elm_input_ids"] = torch.cat(
                [torch.full((pad,), self.pad_token_id, dtype=item["elm_input_ids"].dtype), item["elm_input_ids"]])
            if "elm_labels" in item:
                item["elm_labels"] = torch.cat(
                    [torch.full((pad,), -100, dtype=item["elm_labels"].dtype), item["elm_labels"]])
            item["elm_attention_mask"] = torch.cat(
                [torch.zeros(pad, dtype=item["elm_attention_mask"].dtype), item["elm_attention_mask"]])
            if "signal_id_indices" in item:
                indices = item["signal_id_indices"]
                item["signal_id_indices"] = torch.where(indices >= 0, indices + pad, indices)
            padded_batch.append(item)
        return padded_batch

    def _get_structure_shapes(self, x, path="root"):
        shapes = {}

        if torch.is_tensor(x):
            shapes[path] = ("tensor", tuple(x.shape))
            return shapes

        if isinstance(x, np.ndarray):
            shapes[path] = ("ndarray", tuple(x.shape))
            return shapes

        if isinstance(x, Mapping):
            for k, v in x.items():
                shapes.update(self._get_structure_shapes(v, f"{path}.{k}"))
            return shapes

        if isinstance(x, Sequence) and not isinstance(x, (str, bytes)):
            for i, v in enumerate(x):
                shapes.update(self._get_structure_shapes(v, f"{path}[{i}]"))
            return shapes

        shapes[path] = type(x).__name__
        return shapes

    def _assert_same_structure_and_shapes(self, batch):
        ref = self._get_structure_shapes(batch[0])

        for i, item in enumerate(batch[1:], start=1):
            cur = self._get_structure_shapes(item)

            assert ref.keys() == cur.keys(), (
                f"Structure mismatch between item 0 and item {i}\n"
                f"item 0 keys: {sorted(ref.keys())}\n"
                f"item {i} keys: {sorted(cur.keys())}"
            )

            for k in ref:
                if ref[k] != cur[k]:
                    print(f"\nMismatch at item {i}, key={k}")
                    print(f"item 0: {ref[k]}")
                    print(f"item {i}: {cur[k]}")
                    raise AssertionError(
                        f"Shape/type mismatch at {k} between item 0 and item {i}: "
                        f"{ref[k]} vs {cur[k]}"
                    )
