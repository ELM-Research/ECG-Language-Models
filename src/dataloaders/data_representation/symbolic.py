import numpy as np
import torch

from dataloaders.data_representation.base import Base
from utils.gpu_manager import is_main
from configs.constants import ECG_TOKEN_PREFIX


class Symbolic(Base):
    def __init__(self, data, llm_tokenizer_components, args):
        super().__init__(data, args)
        self.llm_tokenizer = llm_tokenizer_components["llm_tokenizer"]
        self.ecg_byte_builder = llm_tokenizer_components["ecg_tokenizer"]

    def __getitem__(self, index):
        instance = self.data[index]
        if instance["ecg_path"] == "noise" or self.args.perturb == "noise":
            ecg_signal = self.gauss_noise_ecg()
        elif instance["ecg_path"] == "flatline" or self.args.perturb == "zeros":
            ecg_signal = self.blackout_ecg()
        else:
            ecg_np_file = self.fm.open_npy(instance["ecg_path"])
            ecg_signal = ecg_np_file["ecg"][self.args.leads]
            if self.is_train and self.args.augment_ecg:
                ecg_signal = self.augment_ecg(ecg_signal)

        ### PREPARE ECG INPUT ###
        symbols, _ = self.ecg_byte_builder.ecg_to_symbol(ecg_signal)
        ecg_tokens = self.ecg_byte_builder.encode(symbols)
        # print("ecg_tokens", ecg_tokens)
        ecg_tokens = self.llm_tokenizer.convert_tokens_to_ids([f"{ECG_TOKEN_PREFIX}{ids}" for ids in ecg_tokens])

        ### PREPARE TEXT INPUTS ###
        text = instance["text"]
        prompt = self.make_prompt(text)
        if self.args.dev and is_main():
            print("prompt\n", prompt)

        if "train" in self.args.mode:
            return self.prepare_training_set(ecg_tokens, prompt)
        else:
            return self.prepare_eval_inference_set(ecg_tokens, prompt)

    ### PREPARE TRAINING/EVAL/INFERENCE SETS ###
    def prepare_training_set(
        self,
        ecg_tokens: np.array,
        prompt: str,
    ):
        truncated_padded_input = self.trunc_pad_input(ecg_tokens, prompt)
        attention_mask = self.create_attention_mask(truncated_padded_input)
        labels = self.create_labels(truncated_padded_input)
        if self.args.dev and is_main():
            self.decode_and_print_mapping(truncated_padded_input)
            self.check_labels(labels)
            self.check_attention_mask(truncated_padded_input, attention_mask)

        assert len(truncated_padded_input) == len(attention_mask) == len(labels) == self.args.llm_input_len, (
            f"Length mismatch: {len(truncated_padded_input)} != {len(attention_mask)} != {len(labels)} != {self.args.llm_input_len}"
        )
        # print("truncated_padded_ecg_tokens", truncated_padded_ecg_tokens)
        # print("signal_id_indices", signal_id_indices)
        return {
            "elm_input_ids": torch.tensor(truncated_padded_input, dtype=torch.int64),
            "elm_labels": torch.tensor(labels, dtype=torch.int64),
            "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        }

    def prepare_eval_inference_set(
        self,
        ecg_tokens: np.array,
        prompt: str,
    ):
        truncated_padded_input = self.trunc_pad_input(ecg_tokens, prompt)
        attention_mask = self.create_attention_mask(truncated_padded_input)
        assert len(truncated_padded_input) == len(attention_mask), f"Length mismatch: {len(truncated_padded_input)} != {len(attention_mask)}"
        return {
            "elm_input_ids": torch.tensor(truncated_padded_input, dtype=torch.int64),
            "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        }

    ### PADDING/TRUNCATION FUNCTIONS ###
    def trunc_pad_input(self, ecg_tokens: np.ndarray, prompt: str):
        before, after = self.split_prompt(prompt)
        if "train" in self.args.mode:
            min_ecg_token_len = int(self.args.min_ecg_tokens_len)
            before_len, after_len, ecg_token_len = len(before), len(after), len(ecg_tokens)

            if before_len + after_len + ecg_token_len == self.args.llm_input_len:
                # return before + ecg_tokens + after, self.convert_ecg_tokens(ecg_tokens)
                return before + ecg_tokens + after
            elif before_len + after_len + ecg_token_len < self.args.llm_input_len:
                # return self.pad_input(before + ecg_tokens + after), self.convert_ecg_tokens(ecg_tokens)
                return self.pad_input(before + ecg_tokens + after)

            if before_len + min_ecg_token_len > self.args.llm_input_len:
                raise ValueError("before + min_ecg exceeds llm_input_len; lower min_ecg_tokens_len.")

            target_ecg = min(ecg_token_len, max(min_ecg_token_len, self.args.llm_input_len - (before_len + after_len)))
            ecg_tokens = ecg_tokens[:target_ecg]
            remaining_after = self.args.llm_input_len - before_len - len(ecg_tokens)
            after = after[: max(remaining_after, 0)]
        return before + ecg_tokens + after

    def convert_ecg_tokens(self, ecg_tokens):
        ecg_tokens = self.llm_tokenizer.convert_ids_to_tokens(ecg_tokens)
        ecg_tokens = [int(tok.replace(ECG_TOKEN_PREFIX, "")) for tok in ecg_tokens]
        return ecg_tokens
