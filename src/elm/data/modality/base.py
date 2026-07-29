import random
import numpy as np
from typing import Tuple, Literal

class Base:
    def __init__(self, data,
                 ecg_modality_preparer,
                 text_preparer,
                 norm_eps: float = 1e-6,
                 augmentation: bool = False,
                 perturbation: Literal["blackout", "gaussian"] | None = None):
        self.data = data
        self.ecg_modality_preparer = ecg_modality_preparer
        self.text_preparer = text_preparer
        self.norm_eps = norm_eps
        self.augmentation = augmentation
        # no only_text perturbation here, we do that at the LLM level
        self.perturbation = perturbation

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        instance = self.data[index]
        text = instance["text"]

        opened_npy = np.load(instance["ecg_path"], allow_pickle=True).item()
        ecg_signal = opened_npy["ecg"]
        prepared_ecg_input = self.prepare_ecg(ecg_signal)
        prepared_text_input = self.prepare_text(text)


    def prepare_ecg(self, ecg_signal):
        if self.perturbation == "blackout": ecg_input = self.blackout_ecg(ecg_signal.shape)
        elif self.perturbation == "gaussian": ecg_input = self.gaussian_ecg(ecg_signal.shape)

        if self.augmentation: ecg_input = self.augment_ecg(ecg_signal)

        if type(self.ecg_modality_preparer).__name__ not in ["RGB", "StackedSignal"]:
            ecg_input = self.normalize_ecg(ecg_input)

    def prepare_text(self, text):
        pass

    def normalize_ecg(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        min_vals, max_vals = np.min(ecg_signal), np.max(ecg_signal)
        normalized = (ecg_signal - min_vals) / (max_vals - min_vals + self.norm_eps)
        clipped_normalized = np.clip(normalized, 0, 1)
        return clipped_normalized

    ### Perturbations
    def blackout_ecg(self, ecg_shape): return np.full(ecg_shape, np.random.choice(np.arange(10)))

    def gaussian_ecg(self, ecg_shape): return np.random.randn(ecg_shape)

    ### Augmentations
    def augment_ecg(self, ecg_signal):
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.05 * np.std(ecg_signal), ecg_signal.shape)
            perturbed_signal = ecg_signal + noise

            if random.random() < 0.5:
                wander_amplitude = 0.07 * np.max(np.abs(ecg_signal))
                wander = wander_amplitude * np.sin(np.linspace(0, random.randint(1, 5) * np.pi, ecg_signal.shape[1]))
                wander = np.tile(wander, (ecg_signal.shape[0], 1))
                perturbed_signal += wander

            return perturbed_signal
        return ecg_signal