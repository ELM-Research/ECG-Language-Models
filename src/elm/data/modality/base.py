import random
import numpy as np
from typing import Tuple

class Base:
    def __init__(self, data, ecg_modality, text_preparer):
        self.data = data
        self.ecg_modality = ecg_modality

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        instance = self.data[index]
        opened_npy = np.load(instance["ecg_path"], allow_pickle=True).item()
        ecg = self.ecg_modality(opened_npy["ecg"])

    def convert(self,):
        pass

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