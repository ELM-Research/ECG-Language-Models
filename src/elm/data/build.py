import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Literal
from elm.utils.parallelism import get_rank, get_world_size, is_main
from elm.utils.constants import SIGNAL_TOKEN_PLACEHOLDER, RL_TOKENS

class BuildDataloader:
    def __init__(self, data_names: list,
                 data_subset: float,
                 llm_tokenizer: str,
                 modality: str,
                 batch_size: int,
                 num_workers: int,
                 seed: int,
                 training: bool,
                 augmentation: bool = False,
                 perturbation: Literal["blackout", "gaussian"] | None = None):
        self.llm_tokenizer = llm_tokenizer
        self.data_names = data_names
        self.data_subset = data_subset
        self.modality = modality
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.training = training
        self.augmentation = augmentation
        self.perturbation = perturbation

    ### TORCH DATALOADER
    def build_dataloader(self,):
        torch_dataset = self.build_torch_dataset()
        return self.build_torch_dataloader(torch_dataset)

    def build_torch_dataloader(self, torch_dataset):
        sampler = self.get_torch_dataloader_sampler(torch_dataset)
        return DataLoader(
            torch_dataset,
            batch_size = self.batch_size if self.training else 1,
            shuffle = (sampler is None) if self.training else False,
            num_workers = self.num_workers if self.training else 0,
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            collate_fn = self.collate_fn,
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=4 if self.num_workers > 0 else None,

        )

    def get_torch_dataloader_sampler(self, torch_dataset,):
        if get_world_size() > 1:
            return DistributedSampler(torch_dataset, num_replicas=get_world_size(),
                                      rank=get_rank(), seed=self.seed, shuffle=True)
        return None

    def collate_fn(self, batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    ### TORCH DATASET
    def build_torch_dataset(self, ):
        from elm.data.modality.base import Base
        from elm.data.modality.text import Text
        data = []
        for data_name in self.data_names:
            dataset = self.build_hf_dataset(data_name)
            data.extend(dataset)
        if is_main():
            print(f"Length of Dataset: {len(data)}")
            print(f"Using {self.modality} modality")
        llm_tokenizer_components = self.build_llm_tokenizer()
        text_preparer = Text(llm_tokenizer_components)
        ecg_modality_preparer = self.build_ecg_modality()
        torch_dataset = Base(data, ecg_modality_preparer, text_preparer,
                             augmentation = self.augmentation,
                             perturbation = self.perturbation)
        return torch_dataset

    def build_ecg_modality(self,):
        if self.modality == "signal":
            from elm.data.modality.signal import Signal
            return Signal()

        raise ValueError(f"Unknown data modality: {self.modality}")

    def build_hf_dataset(self, data_name):
        if self.args.mode in ["train", "post_train"]:
            split_name = f"fold{self.args.fold}_train"
        elif self.args.mode in ["eval", "inference"]:
            split_name = f"fold{self.args.fold}_test"
        data = load_dataset(
                f"ELM-Research/{data_name}",
                split=split_name,
            ).with_transform(self.decode_batch)
        if self.data_subset:
            n = int(len(data) * self.data_subset)
            data = data.shuffle(seed=self.seed).select(range(n))
        if is_main():
            print("Length of Dataset Considered:", len(data))
        return data

    def decode_batch(self, batch: dict) -> dict:
        if "text" in batch:
            out = []
            for t in batch["text"]:
                try:
                    out.append(json.loads(t))
                except Exception:
                    out.append(t)
            batch["text"] = out
        return batch

    def build_llm_tokenizer(
        self,
    ):
        llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_tokenizer)
        return self.modify_llm_tokenizer(llm_tokenizer)

    def modify_llm_tokenizer(self, llm_tokenizer):
        if self.args.dev and is_main():
            print("Before Modification\n")
            self.print_llm_tokenizer_info(llm_tokenizer)

        if getattr(llm_tokenizer, "pad_token", None) is None:  # llama 3.2
            llm_tokenizer.pad_token = llm_tokenizer.eos_token

        tokens_to_add = {"additional_special_tokens": [],}
        tokens_to_add["additional_special_tokens"].append(SIGNAL_TOKEN_PLACEHOLDER)
        if self.args.train_phase in ["sft", "rl"]:
            vocab = llm_tokenizer.get_vocab()
            tokens_to_add["additional_special_tokens"].extend(t for t in RL_TOKENS if t not in vocab)
        llm_tokenizer.add_special_tokens(tokens_to_add)

        if self.modality == "ecg_byte":
            new_vocab, ecg_byte_builder = self.build_ecg_byte()
            llm_tokenizer.add_tokens(new_vocab)
            out = {"llm_tokenizer": llm_tokenizer, "ecg_tokenizer": ecg_byte_builder}
        else:
            out = {"llm_tokenizer": llm_tokenizer}

        if self.args.dev and is_main():
            print("After Modification\n")
            self.print_llm_tokenizer_info(llm_tokenizer)
        return out

    ### DEV FUNCTIONS ###
    def print_llm_tokenizer_info(self, llm_tokenizer):
        print("Vocab Size:", len(llm_tokenizer))
        print("special_tokens_map:", llm_tokenizer.special_tokens_map)
        print("all_special_tokens:", llm_tokenizer.all_special_tokens)
        print("all_special_ids:", llm_tokenizer.all_special_ids)
        for k in ("pad", "bos", "eos", "unk"):
            t = getattr(llm_tokenizer, f"{k}_token", None)
            i = getattr(llm_tokenizer, f"{k}_token_id", None)
            print(f"{k.upper()} -> token: {t!r}, id: {i}")
        print("-" * 20)