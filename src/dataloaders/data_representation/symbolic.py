import torch

from dataloaders.data_representation.base import Base
from utils.gpu_manager import is_main
from configs.constants import ECG_TOKEN_PREFIX, SIGNAL_TOKEN_PLACEHOLDER


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
            if self.args.augment_ecg:
                ecg_signal = self.augment_ecg(ecg_signal)

        symbols, _ = self.ecg_byte_builder.ecg_to_symbol(ecg_signal)
        ecg_tokens = self.ecg_byte_builder.encode(symbols)
        ecg_tokens = self.llm_tokenizer.convert_tokens_to_ids([f"{ECG_TOKEN_PREFIX}{ids}" for ids in ecg_tokens])
        return self.prepare_symbolic_inputs(ecg_tokens, instance["text"])

    def prepare_symbolic_inputs(self, ecg_tokens, text):
        """Tokenize the conversation, then splice the ECG token ids in at the first <signal> placeholder."""
        ids, labels = self.make_inputs(text)
        split = ids.index(self.llm_tokenizer.convert_tokens_to_ids(SIGNAL_TOKEN_PLACEHOLDER))
        before, after, after_labels = ids[:split], ids[split + 1:], labels[split + 1:]  # the <signal> is replaced by the ECG tokens
        ecg_tokens = list(ecg_tokens)
        if "train" in self.args.mode:
            before, ecg_tokens, after, after_labels = self.fit_symbolic(before, ecg_tokens, after, after_labels)
        ids = before + ecg_tokens + after
        labels = labels[:split] + [-100] * len(ecg_tokens) + after_labels
        if "train" in self.args.mode:
            ids, labels, attention_mask = self.pad_and_mask(ids, labels)
            assert len(ids) == len(attention_mask) == len(labels) == self.args.llm_input_len, (
                f"Length mismatch: {len(ids)} != {len(attention_mask)} != {len(labels)} != {self.args.llm_input_len}"
            )
            if self.args.dev and is_main():
                self.decode_and_print_mapping(ids)
                self.check_labels(labels)
        else:
            attention_mask = [1] * len(ids)
        return {
            "elm_input_ids": torch.tensor(ids, dtype=torch.int64),
            "elm_labels": torch.tensor(labels, dtype=torch.int64),
            "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
        }

    def fit_symbolic(self, before, ecg_tokens, after, after_labels):
        """Cap ECG tokens (>= min_ecg_tokens_len) and trim the trailing text so the spliced sequence fits llm_input_len."""
        limit = self.args.llm_input_len
        if len(before) + len(ecg_tokens) + len(after) <= limit:
            return before, ecg_tokens, after, after_labels
        min_ecg = int(self.args.min_ecg_tokens_len)
        if len(before) + min_ecg > limit:
            raise ValueError("before + min_ecg exceeds llm_input_len; lower min_ecg_tokens_len.")
        target_ecg = min(len(ecg_tokens), max(min_ecg, limit - (len(before) + len(after))))
        ecg_tokens = ecg_tokens[:target_ecg]
        remaining_after = max(limit - len(before) - len(ecg_tokens), 0)
        return before, ecg_tokens, after[:remaining_after], after_labels[:remaining_after]
