import argparse

import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

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
        def _ensure_resizable_storage(item):
            if torch.is_tensor(item):
                return item.clone()
            if isinstance(item, np.ndarray):
                return torch.tensor(item)
            if isinstance(item, dict):
                return {key: _ensure_resizable_storage(value) for key, value in item.items()}
            if isinstance(item, list):
                return [_ensure_resizable_storage(value) for value in item]
            if isinstance(item, tuple):
                return tuple(_ensure_resizable_storage(value) for value in item)
            return item

        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        batch = [_ensure_resizable_storage(item) for item in batch]
        return torch.utils.data.dataloader.default_collate(batch)
