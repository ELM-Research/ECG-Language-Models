import argparse
import pickle
from pathlib import Path
from typing import Tuple, Union
import numpy as np

import bpe

class BuildECGTokenizers:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.normalize_epsilon = 1e-6

    @staticmethod
    def open_tokenizer(path: Union[str, Path]) -> Tuple[dict, dict]:
        """Open a pickled tokenizer file and return the vocabulary and merges."""
        with open(path, "rb") as f:
            vocab, merges = pickle.load(f)
        return vocab, merges

    def normalize(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        min_vals = np.min(ecg_signal)
        max_vals = np.max(ecg_signal)
        normalized = (ecg_signal - min_vals) / (max_vals - min_vals + self.args.norm_eps)
        clipped_normalized = np.clip(normalized, 0, 1)
        return clipped_normalized, (min_vals, max_vals)

    def denormalize(self, ecg_signal: np.ndarray, min_max_vals: Tuple[float, float]) -> np.ndarray:
        min_vals, max_vals = min_max_vals
        return ecg_signal * (max_vals - min_vals) + min_vals


class BuildECGByte(BuildECGTokenizers):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.symbols = list("abcdefghijklmnopqrstuvwxyz")
        self.len_symbols = len(self.symbols)
        self.get_tokenizer()

    def get_tokenizer(self):
        self.vocab, self.merges = self.open_tokenizer(self.args.ecg_tokenizer)

    def encode(self, symbols: str):
        return bpe.encode_symbol(symbols, self.merges)

    def decode(self, ecg_tokens):
        return "".join(self.vocab[token_id] for token_id in ecg_tokens)

    def ecg_to_symbol(self, ecg_signal: np.ndarray) -> str:
        normalized_ecg_signal, min_max_vals = self.normalize(ecg_signal)
        quantized_signal = self.quantize(normalized_ecg_signal)
        symbols = self.quantized_to_symbol(quantized_signal)
        return "".join(symbols.flatten()), min_max_vals

    def quantize(self, clipped_normalized: np.ndarray) -> np.ndarray:
        return np.minimum(
            np.floor(clipped_normalized * self.len_symbols),
            self.len_symbols - 1,
        ).astype(np.uint8)

    def quantized_to_symbol(self, quantized_signal: np.ndarray) -> np.ndarray:
        return np.vectorize(lambda x: self.symbols[x])(quantized_signal)

    def symbol_to_ecg(self, symbols: str, min_max_vals: Tuple[float, float]) -> np.ndarray:
        quantized_signal = self.symbol_to_quantized(symbols)
        dequantized_signal = self.dequantize(quantized_signal)
        ecg_signal = self.denormalize(dequantized_signal, min_max_vals)
        return ecg_signal

    def symbol_to_quantized(self, symbols: str):
        return np.vectorize(lambda x: self.symbols.index(x))(symbols)

    def dequantize(self, quantized_signal: np.ndarray) -> np.ndarray:
        return quantized_signal / (len(self.symbols) - 1)
