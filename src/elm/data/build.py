import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from transformers import AutoTokenizer
from datasets import load_dataset
from typing import Literal
from elm.utils.parallelism import get_rank, get_world_size, is_main
from elm.utils.constants import ECG_TOKEN_PLACEHOLDER, RL_TOKENS

class BuildDataloader:
    def __init__(self, data_names: list,
                 split_names: list,
                 llm_tokenizer_name: str,
                 ecg_tokens,
                 modality: str,
                 batch_size: int,
                 num_workers: int,
                 seed: int,
                 training_stage: Literal["pretrain", "sft", "rl"] | None = None,
                 augmentation: bool = False,
                 perturbation: Literal["blackout", "gaussian"] | None = None,
                 development: bool = False,):
        self.data_names = data_names
        self.split_names = split_names
        self.llm_tokenizer_name = llm_tokenizer_name
        self.ecg_tokens = ecg_tokens
        self.modality = modality
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.training_stage = training_stage
        self.augmentation = augmentation
        self.perturbation = perturbation
        self.development = development

    ### TORCH DATALOADER
    def build_dataloader(self,):
        torch_dataset = self.build_torch_dataset()
        return self.build_torch_dataloader(torch_dataset)

    def build_torch_dataloader(self, torch_dataset):
        sampler = self.get_torch_dataloader_sampler(torch_dataset)
        return DataLoader(
            torch_dataset,
            batch_size = self.batch_size if self.training_stage else 1,
            shuffle = (sampler is None) if self.training_stage else False,
            num_workers = self.num_workers if self.training_stage else 0,
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            collate_fn = partial(self.custom_collate_fn,
                                 self.pad_token_id),
            persistent_workers=(self.num_workers > 0),
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def get_torch_dataloader_sampler(self, torch_dataset,):
        if get_world_size() > 1:
            return DistributedSampler(torch_dataset, num_replicas=get_world_size(),
                                      rank=get_rank(), seed=self.seed, shuffle=True)
        return None


    def custom_collate_fn(self, batch, pad_token_id):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    ### TORCH DATASET
    def build_torch_dataset(self, ):
        from elm.data.modality.elm_dataset import ELMDataset
        from elm.data.modality.text import Text
        data = []
        for data_name, split_name in zip(self.data_names, self.split_names):
            dataset = self.build_hf_dataset(data_name, split_name)
            data.extend(dataset)
        if is_main(): print(f"Length of Dataset: {len(data)}", f"Using {self.modality} modality")
        llm_tokenizer = self.build_llm_tokenizer()
        self.pad_token_id = llm_tokenizer.pad_token_id
        text_preparer = Text(llm_tokenizer, self.training_stage)
        ecg_modality_preparer = self.build_ecg_modality()
        torch_dataset = ELMDataset(data, ecg_modality_preparer, text_preparer,
                             augmentation = self.augmentation,
                             perturbation = self.perturbation)
        return torch_dataset

    def build_ecg_modality(self,):
        if self.modality == "signal":
            from elm.data.modality.signal import Signal
            return Signal(self.ecg_tokens)

        raise ValueError(f"Unknown data modality: {self.modality}")

    def build_hf_dataset(self, data_name, split_name):
        data = load_dataset(data_name, split=split_name).with_transform(self.decode_batch)
        if is_main(): print("Length of Dataset Considered:", len(data))
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
        llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_tokenizer_name)
        return self.modify_llm_tokenizer(llm_tokenizer)

    def modify_llm_tokenizer(self, llm_tokenizer):
        if self.development and is_main():
            print("Before Modification\n")
            self.print_llm_tokenizer_info(llm_tokenizer)

        tokens_to_add = [ECG_TOKEN_PLACEHOLDER]
        if self.training_stage in ["sft", "rl"]:
            vocab = llm_tokenizer.get_vocab()
            for key, value in RL_TOKENS.items():
                if value not in vocab: tokens_to_add.append(RL_TOKENS)

        llm_tokenizer.add_tokens(tokens_to_add)
        if self.development and is_main():
            print("After Modification\n")
            self.print_llm_tokenizer_info(llm_tokenizer)
        return llm_tokenizer

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