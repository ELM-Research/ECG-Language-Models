import torch
from transformers import DataCollatorForSeq2Seq


class ELMDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features = [dict(feature) for feature in features]
        ecg_values = [feature.pop("ecg_values", None) for feature in features]
        batch = super().__call__(features, return_tensors)
        if any(ecg is None for ecg in ecg_values):
            if not all(ecg is None for ecg in ecg_values):
                raise ValueError("A batch cannot mix examples with and without ECG input")
        else:
            batch["ecg_values"] = torch.stack([torch.as_tensor(ecg) for ecg in ecg_values])
        return batch
