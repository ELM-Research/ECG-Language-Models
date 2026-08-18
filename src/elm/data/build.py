import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset
from elm.utils.parallelism import get_rank, get_world_size, is_main
from elm.utils.constants import ECG_TOKEN_PLACEHOLDER, RL_TOKENS

class DataBuilder:
    def __init__(self, config: dict, training: bool = True):
        data, model, training_config = config["data"], config["model"], config["training"]
        runtime = config["training" if training else "evaluation"]
        self.is_training = training
        self.data_names = data["data_names"]
        self.split_names = data["split_names"]
        self.llm_tokenizer_name = model["language_model"]
        self.truncation_length = model["truncation_length"]
        self.num_ecg_tokens = model["num_ecg_tokens"]
        self.batch_size = runtime["batch_size"]
        self.num_workers = runtime["num_workers"]
        self.training_stage = training_config["training_stage"]
        self.enable_thinking = config["enable_thinking"]
        self.system_prompt_path = config["system_prompt_path"]
        self.modality = config["modality"]
        self.seed = config["seed"]
        self.augmentation = config["augment_ecg"] and self.is_training
        self.perturbation = config["perturbation"]
        self.development = config["development"]

    ### TORCH DATALOADER
    def build_dataloader(self, llm_tokenizer=None):
        if llm_tokenizer is None:
            llm_tokenizer = self.build_llm_tokenizer()
        torch_dataset = self.build_torch_dataset(llm_tokenizer)
        return self.build_torch_dataloader(torch_dataset, llm_tokenizer)

    def build_torch_dataloader(self, torch_dataset, llm_tokenizer):
        sampler = self.get_torch_dataloader_sampler(torch_dataset)
        return DataLoader(
            torch_dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None and self.is_training,
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            collate_fn = DataCollatorForSeq2Seq(llm_tokenizer,
                                                label_pad_token_id=-100),
            persistent_workers=self.is_training and self.num_workers > 0,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def get_torch_dataloader_sampler(self, torch_dataset,):
        if get_world_size() > 1:
            return DistributedSampler(torch_dataset, num_replicas=get_world_size(),
                                      rank=get_rank(), seed=self.seed, shuffle=True)
        return None

    ### TORCH DATASET
    def build_torch_dataset(self, llm_tokenizer):
        from elm.data.modality.elm_dataset import ELMDataset
        from elm.data.modality.text import Text
        data = []
        for data_name, split_name in zip(self.data_names, self.split_names):
            dataset = self.build_hf_dataset(data_name, split_name)
            data.extend(dataset)
        if is_main(): print(f"Length of Dataset: {len(data)}", f"Using {self.modality} modality")
        text_preparer = Text(llm_tokenizer,
                             self.truncation_length,
                             self.training_stage,
                             self.enable_thinking,
                             self.system_prompt_path)
        ecg_modality_preparer = self.build_ecg_modality()
        torch_dataset = ELMDataset(data, ecg_modality_preparer, text_preparer,
                             augmentation = self.augmentation,
                             perturbation = self.perturbation)
        return torch_dataset

    def build_ecg_modality(self,):
        if self.modality == "signal":
            from elm.data.modality.signal import Signal
            return Signal(self.num_ecg_tokens, self.perturbation == "only_text")

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
            for token in RL_TOKENS:
                if token not in vocab: tokens_to_add.append(token)

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


def build_data(config: dict, training: bool = True):
    builder = DataBuilder(config, training)
    tokenizer = builder.build_llm_tokenizer()
    return tokenizer, builder.build_dataloader(tokenizer)
