import numpy as np
from PIL import Image
import torch
from typing import Optional

from dataloaders.data_representation.base import Base
from utils.gpu_manager import is_main

class StackedSignal(Base):
    def __init__(self, data, llm_tokenizer_components, encoder_tokenizer_components, args):
        super().__init__(data, args)
        self.llm_tokenizer = llm_tokenizer_components["llm_tokenizer"] if llm_tokenizer_components else None
        self.encoder_tokenizer = encoder_tokenizer_components["encoder_tokenizer"]

    def __getitem__(self, index):
        instance = self.data[index]
        if instance["ecg_path"] == "noise" or self.args.perturb == "noise":
            ecg_signal = self.gauss_noise_ecg()
        elif instance["ecg_path"] == "flatline" or self.args.perturb == "zeros":
            ecg_signal = self.blackout_ecg()
        else:
            ecg_np_file = self.fm.open_npy(instance["ecg_path"])
            ecg_signal = ecg_np_file["ecg"][self.args.leads]
            if self.args.augment_ecg:
                ecg_signal = self.augment_ecg(ecg_signal)
        ecg_stacked_signal = self.signal_to_stacked_signal(ecg_signal)

        if self.encoder_tokenizer is not None:
            diagnostic_report = ecg_np_file["report"]
            if self.args.encoder == "clip-vit-base-patch32":
                encoder_tokenizer_out = self.prepare_clip_input(diagnostic_report, ecg_stacked_signal)
            elif self.args.encoder == "siglip2-so400m-patch16-naflex":
                encoder_tokenizer_out = self.prepare_siglip_input(diagnostic_report, ecg_stacked_signal)
            elif self.args.encoder == "vit-base-patch16-224-in21k":
                encoder_tokenizer_out = self.prepare_vit_input(ecg_stacked_signal)
        else:
            encoder_tokenizer_out = {"ecg_signal": ecg_signal}

        text = instance["text"]
        prompt = self.make_prompt(text)
        if self.args.dev and is_main():
            print("prompt\n", prompt)

        if "train" in self.args.mode:
            return self.prepare_training_set(prompt, encoder_tokenizer_out)
        else:
            return self.prepare_eval_inference_set(prompt, encoder_tokenizer_out)

    def prepare_training_set(
        self,
        prompt: Optional[str],
        encoder_tokenizer_out: dict,
    ):
        input_ids = self.prepare_input_ids(prompt)
        signal_id_indices = self.find_signal_token_indices(input_ids)
        attention_mask = self.create_attention_mask(input_ids)
        labels = self.create_labels(input_ids)
        assert len(input_ids) == len(attention_mask) == len(labels) <= self.args.llm_input_len, (
            f"Length mismatch: {len(input_ids)} != {len(attention_mask)} != {len(labels)} or exceeds {self.args.llm_input_len}"
        )
        elm = {
            "elm_input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "elm_labels": torch.tensor(labels, dtype=torch.int64),
            "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "signal_id_indices": torch.tensor(signal_id_indices, dtype=torch.int64),
        }
        return {**elm, "encoder_tokenizer_out": encoder_tokenizer_out}

    def prepare_eval_inference_set(
        self,
        prompt: Optional[str],
        encoder_tokenizer_out: dict,
    ):
        input_ids = self.prepare_input_ids(prompt)
        signal_id_indices = self.find_signal_token_indices(input_ids)
        attention_mask = self.create_attention_mask(input_ids)
        assert len(input_ids) == len(attention_mask), f"Length mismatch: {len(input_ids)} != {len(attention_mask)}"
        elm = {
            "elm_input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "signal_id_indices": torch.tensor(signal_id_indices, dtype=torch.int64),
        }
        return {**elm, "encoder_tokenizer_out": encoder_tokenizer_out}

    ### SIGNAL TO STACKED SIGNAL FUNCTIONS ###
    def signal_to_stacked_signal(self, signal):
        normalized_signal, _ = self.normalize(signal)
        rgb_norm_signal = np.stack([normalized_signal * 255] * 3, axis=-1).astype(np.uint8)
        return Image.fromarray(rgb_norm_signal)
