import argparse
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from collections.abc import Mapping, Sequence

from utils.parallel_context import get_parallel_context

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
        if self.args.parallel_strategy:
            ctx = get_parallel_context()
            sampler = DistributedSampler(torch_dataset, num_replicas=ctx.dp_size,
                                         rank=ctx.dp_rank, seed=self.args.seed, shuffle=True)
        else:
            sampler = None
        return sampler

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        self._assert_same_structure_and_shapes(batch)
        return torch.utils.data.dataloader.default_collate(batch)

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
