from torch.utils.data import Dataset
import random
import numpy as np
from typing import List, Tuple
from PIL import Image
import torch

from utils.dir_file_manager import DirFileManager
from utils.chat_template_manager import encode_with_labels, assistant_stop_ids
from utils.gpu_manager import is_main
from configs.constants import (
    HF_LLMS,
    SIGNAL_TOKEN_PLACEHOLDER,
    LEADING_PREFIX_RE,
    TAG_RE,
    IMAGE_WORD_RE,
    case_preserving_signal,
    ECG_ENCODERS,
    VISION_ENCODERS,
)

class Base(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args
        self.fm = DirFileManager()
        if self.args.encoder:
            if self.args.encoder in ECG_ENCODERS:
                self.max_len = ECG_ENCODERS[self.args.encoder]["encoder_input_len"]
            elif self.args.encoder in VISION_ENCODERS:
                self.max_len = VISION_ENCODERS[self.args.encoder]["encoder_input_len"]

    def __len__(self):
        return len(self.data)

    ### ENCODER TRAINING FUNCTIONS ###
    def prepare_clip_input(self, diagnostic_report: str, ecg_image: Image.Image):
        clip_out = self.encoder_tokenizer(
            text=[diagnostic_report], images=[ecg_image], return_tensors="pt", padding="max_length", max_length=self.max_len, truncation=True
        )
        return {
            "encoder_input_ids": clip_out["input_ids"][0].contiguous(),
            "encoder_attention_mask": clip_out["attention_mask"][0].contiguous(),
            "encoder_pixels": clip_out["pixel_values"][0].contiguous(),
        }

    def prepare_siglip_input(self, diagnostic_report: str, ecg_image: Image.Image):
        siglip_out = self.encoder_tokenizer(
            text=[diagnostic_report], images=[ecg_image], return_tensors="pt", padding="max_length", max_length=self.max_len, truncation=True
        )
        return {
            "encoder_input_ids": siglip_out["input_ids"][0].contiguous(),
            "encoder_attention_mask": siglip_out["pixel_attention_mask"][0].contiguous(),
            "encoder_pixels": siglip_out["pixel_values"][0].contiguous(),
            "spatial_shapes": siglip_out["spatial_shapes"][0].contiguous(),
        }

    def prepare_vit_input(self, ecg_image: Image.Image):
        vit_out = self.encoder_tokenizer(images=ecg_image, return_tensors="pt")
        mask = torch.rand(size=(1, VISION_ENCODERS[self.args.encoder]["num_patches"])) < 0.75
        return {
            "encoder_pixels": vit_out["pixel_values"][0].contiguous(),
            "encoder_mask": mask[0].contiguous(),
        }

    ### CHAT TEMPLATE / TOKENIZATION (HuggingFace-native) ###
    @property
    def stop_ids(self) -> frozenset:
        """Token ids that terminate an assistant turn (derived from the tokenizer's chat template)."""
        if not hasattr(self, "_stop_ids"):
            self._stop_ids = assistant_stop_ids(self.llm_tokenizer)
        return self._stop_ids

    @property
    def think_prefix_ids(self) -> list[int]:
        if not hasattr(self, "_think_prefix_ids"):
            self._think_prefix_ids = self.llm_tokenizer.encode("<think>\n", add_special_tokens=False)
        return self._think_prefix_ids

    def get_system_prompt(self) -> str:
        with open(self.args.system_prompt, encoding="utf-8") as file:
            return file.read()

    def clean_text(self, message_value: str) -> str:
        message_value = TAG_RE.sub("", message_value)
        message_value = IMAGE_WORD_RE.sub(case_preserving_signal, message_value)
        message_value = LEADING_PREFIX_RE.sub("", message_value)
        return message_value

    def build_messages(self, text) -> list[dict]:
        """Turn dataset conversation turns into HF chat messages, prepending the
        signal placeholders to the first human turn and the system prompt (if any)."""
        messages = []
        if HF_LLMS[self.args.llm]["system_prompt"] and getattr(self.args, "system_prompt", None):
            messages.append({"role": "system", "content": self.get_system_prompt()})
        signals = "" if self.args.perturb == "only_text" else SIGNAL_TOKEN_PLACEHOLDER * self.args.num_encoder_tokens + "\n"
        first_human = True
        for turn in text:
            if self.args.dev and is_main():
                print("turn", turn)
            is_human = turn.get("from", turn.get("role", "")).lower() in ("human", "user")
            content = self.clean_text(turn.get("value", turn.get("content", "")))
            if is_human and first_human:
                content, first_human = signals + content, False
            messages.append({"role": "user" if is_human else "assistant", "content": content})
        return messages

    def make_inputs(self, text) -> Tuple[List[int], List[int]]:
        """Tokenize one sample into (input_ids, labels). Assistant turns are labeled, the rest is -100."""
        if getattr(self.args, "train_phase", "sft") == "pretrain":
            return self.make_pretrain_inputs(text)
        think = tuple(self.think_prefix_ids) if getattr(self.args, "explicit_thinking", False) else ()
        return encode_with_labels(self.llm_tokenizer, self.build_messages(text), think)

    def make_pretrain_inputs(self, text: str) -> Tuple[List[int], List[int]]:
        tok = self.llm_tokenizer
        signals = "" if self.args.perturb == "only_text" else SIGNAL_TOKEN_PLACEHOLDER * self.args.num_encoder_tokens + "\n"
        ids = tok.encode(f"{tok.bos_token or ''}{signals}{text}{tok.eos_token}", add_special_tokens=False)
        sig_id = tok.convert_tokens_to_ids(SIGNAL_TOKEN_PLACEHOLDER)
        last_sig = max((i for i, t in enumerate(ids) if t == sig_id), default=-1)
        if last_sig >= 0:
            prefix_end = last_sig + 2  # mask the signals and the newline that follows them
        elif tok.bos_token_id is not None and tok.bos_token_id in ids:
            prefix_end = ids.index(tok.bos_token_id) + 1
        else:
            prefix_end = 0
        return ids, [-100 if i < prefix_end else t for i, t in enumerate(ids)]

    ### TRUNCATION / PADDING ###
    def find_signal_token_indices(self, input_ids: list[int]) -> list[int]:
        signal_token_id = self.llm_tokenizer.convert_tokens_to_ids(SIGNAL_TOKEN_PLACEHOLDER)
        indices = [i for i, x in enumerate(input_ids) if x == signal_token_id]
        if not indices:
            if self.args.dev and is_main():
                print(f"Signal token ID {signal_token_id} not found in input IDs.")
            return [-1]
        return indices

    def pad_and_mask(self, ids: list[int], labels: list[int]) -> Tuple[List[int], List[int], List[int]]:
        """Left-pad ids/labels to llm_input_len and build the matching attention mask."""
        pad = self.args.llm_input_len - len(ids)
        return ([self.llm_tokenizer.pad_token_id] * pad + ids, [-100] * pad + labels, [0] * pad + [1] * len(ids))

    def fit_to_len(self, ids: list[int], labels: list[int]) -> Tuple[List[int], List[int], List[int]]:
        """Truncate (preserving signal tokens) then left-pad to llm_input_len."""
        if len(ids) > self.args.llm_input_len:
            ids, labels = self.truncate_preserving_signal_tokens(ids, labels)
        return self.pad_and_mask(ids, labels)

    def truncate_preserving_signal_tokens(self, ids: list[int], labels: list[int]) -> Tuple[List[int], List[int]]:
        limit = self.args.llm_input_len
        overflow = len(ids) - limit
        signal_token_id = self.llm_tokenizer.convert_tokens_to_ids(SIGNAL_TOKEN_PLACEHOLDER)
        first_signal_idx = next((i for i, t in enumerate(ids) if t == signal_token_id), len(ids))

        def priority(i):  # drop prompt tokens first, then responses; always keep signals and everything before them
            if i < first_signal_idx:
                return 2
            return 1 if labels[i] != -100 else 0

        droppable = sorted((i for i, t in enumerate(ids) if t != signal_token_id), key=priority)
        drop = set(droppable[:overflow])
        ids = [t for i, t in enumerate(ids) if i not in drop][-limit:]
        labels = [l for i, l in enumerate(labels) if i not in drop][-limit:]
        return ids, labels

    ### ELM TRAINING / EVAL / INFERENCE ###
    def prepare_signal_inputs(self, text, encoder_tokenizer_out: dict) -> dict:
        """Shared item builder for signal/stacked_signal/rgb representations."""
        ids, labels = self.make_inputs(text)
        training = "train" in self.args.mode
        if training:
            ids, labels, attention_mask = self.fit_to_len(ids, labels)
        else:
            attention_mask = [1] * len(ids)
        signal_id_indices = self.find_signal_token_indices(ids)
        if training:
            assert len(signal_id_indices) == self.args.num_encoder_tokens
            assert len(ids) == len(attention_mask) == len(labels) == self.args.llm_input_len, (
                f"Length mismatch: {len(ids)} != {len(attention_mask)} != {len(labels)} != {self.args.llm_input_len}"
            )
        if self.args.dev and is_main():
            self.decode_and_print_mapping(ids)
            self.check_labels(labels)
        return {
            "elm_input_ids": torch.tensor(ids, dtype=torch.int64),
            "elm_labels": torch.tensor(labels, dtype=torch.int64),
            "elm_attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "signal_id_indices": torch.tensor(signal_id_indices, dtype=torch.int64),
            "encoder_tokenizer_out": encoder_tokenizer_out,
        }

    def slice_continuation(self, prompt_ids: list[int], generated_ids: list[int]) -> list[int]:
        K = len(prompt_ids)
        if len(generated_ids) >= K and generated_ids[:K] == prompt_ids:
            return generated_ids[K:]
        return generated_ids

    def get_response_ranges(self, labels: List[int]) -> List[Tuple[int, int]]:
        """Contiguous spans of labeled (assistant) tokens — one per assistant turn."""
        ranges, start = [], None
        for i, lab in enumerate(labels):
            if lab != -100 and start is None:
                start = i
            if lab == -100 and start is not None:
                ranges.append((start, i))
                start = None
        if start is not None:
            ranges.append((start, len(labels)))
        return ranges

    def get_ground_truth_responses(self, input_ids: List[int], ranges: List[Tuple[int, int]]) -> List[str]:
        return [self.llm_tokenizer.decode(input_ids[s:e], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for s, e in ranges]

    def get_generated_response_for_turn(self, prompt_input_ids: list[int], generated_ids: list[int]) -> str:
        cont = self.slice_continuation(prompt_input_ids, generated_ids)
        cut = next((i for i, t in enumerate(cont) if t in self.stop_ids), len(cont))
        return self.llm_tokenizer.decode(cont[:cut], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()

    ### SIGNAL FUNCTIONS ###
    def normalize(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        min_vals = np.min(ecg_signal)
        max_vals = np.max(ecg_signal)
        normalized = (ecg_signal - min_vals) / (max_vals - min_vals + self.args.norm_eps)
        clipped_normalized = np.clip(normalized, 0, 1)
        return clipped_normalized, (min_vals, max_vals)

    def blackout_ecg(self):
        c = np.random.choice(np.arange(10))
        return np.full((len(self.args.leads), self.args.segment_len), c)

    def gauss_noise_ecg(self):
        return np.random.randn(len(self.args.leads), self.args.segment_len)

    def augment_ecg(self, signal):
        if random.random() < 0.5:
            noise_level = 0.05
            noise = np.random.normal(0, noise_level * np.std(signal), signal.shape)
            perturbed_signal = signal + noise

            if random.random() < 0.5:
                wander_amplitude = 0.07 * np.max(np.abs(signal))
                wander = wander_amplitude * np.sin(np.linspace(0, random.randint(1, 5) * np.pi, signal.shape[1]))
                wander = np.tile(wander, (signal.shape[0], 1))
                perturbed_signal += wander

            return perturbed_signal
        return signal

    ### DEBUGGING FUNCTIONS ###
    def decode_and_print_mapping(self, truncated_padded_input: list[int]) -> None:
        tokens = self.llm_tokenizer.convert_ids_to_tokens(truncated_padded_input)
        decoded = self.llm_tokenizer.decode(truncated_padded_input, skip_special_tokens=False)

        print("=== ECG Token Mapping ===")
        for tid, tok in zip(truncated_padded_input, tokens):
            print(f"ID {tid:<6} | Token {tok}")

        print("\n=== Full Decoded String ===")
        print(decoded)

    def check_labels(self, labels):
        labels_np = np.array(labels)
        non_neg_indices = np.where(labels_np != -100)[0]
        if len(non_neg_indices) > 0:
            non_neg_values = labels_np[non_neg_indices].tolist()
            tokens = self.llm_tokenizer.convert_ids_to_tokens(non_neg_values)
            for idx, (token, token_id) in enumerate(zip(tokens, non_neg_values)):
                print(f"{idx}: {token} -> {token_id}")
        else:
            print("No valid labels found (all are -100)")
        print("=" * 100)

    def assert_range_alignment(self, input_ids: List[int], ranges: List[Tuple[int, int]]) -> None:
        for s, e in ranges:
            assert e > s, f"Empty response range at {s}."
            assert any(input_ids[i] in self.stop_ids for i in range(s, e)), f"No turn terminator in response range [{s}, {e})."
