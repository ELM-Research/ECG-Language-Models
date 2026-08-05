import argparse
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from utils.gpu_manager import get_world_size, get_rank

from dataloaders.collate import collate_elm
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
        return self.build_torch_dataloader(torch_dataset)

    def build_torch_dataloader(self, torch_dataset):
        self.pad_id = torch_dataset.llm_tokenizer.pad_token_id
        sampler = self.get_torch_dataloader_sampler(torch_dataset)
        if "train" in self.args.mode:
            return DataLoader(
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
        if "eval" in self.args.mode:
            return DataLoader(
                torch_dataset,
                batch_size=1,  # batched inference/eval not implemented
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                collate_fn=self.collate_fn,
            )
        return ValueError(f"Unsupported mode: {self.args.mode}")

    def get_torch_dataloader_sampler(
        self,
        torch_dataset,
    ):
        if self.args.distributed:
            return DistributedSampler(torch_dataset, num_replicas=get_world_size(),
                                         rank=get_rank(), seed=self.args.seed, shuffle=True)
        return None

    def collate_fn(self, batch):
        return collate_elm(batch, self.pad_id)
