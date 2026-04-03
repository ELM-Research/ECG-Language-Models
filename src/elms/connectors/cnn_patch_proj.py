import torch.nn.functional as F
from torch import nn

from configs.constants import HF_LLMS


class CNNPatchProjection(nn.Module):
    """Patch-based CNN projection for ECG signals.

    Splits the ECG into N non-overlapping time patches, then processes each
    patch with two 1D conv layers (treating leads as channels), global average
    pools the temporal dimension, and linearly projects to the LLM hidden size.

    z_i = W * Pool(G(X_i)) + b   where G is the two conv layers.
    """

    def __init__(self, num_patches, num_leads, llm_id):
        super().__init__()
        self.num_patches = num_patches
        self.input_dtype = HF_LLMS[llm_id]["native_dtype"]

        self.conv1 = nn.Conv1d(num_leads, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, HF_LLMS[llm_id]["model_hidden_size"])

        self.to(dtype=self.input_dtype)

    def forward(self, ecg_signal):
        # ecg_signal: (B, C, L)  C=leads, L=samples
        B, C, L = ecg_signal.shape
        patch_size = L // self.num_patches

        # (B, C, N, P) -> (B, N, C, P) -> (B*N, C, P)
        x = ecg_signal.reshape(B, C, self.num_patches, patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B * self.num_patches, C, patch_size)
        x = x.to(dtype=self.input_dtype)

        # Conv layers with ReLU
        x = F.relu(self.conv1(x))   # (B*N, 64, P)
        x = F.relu(self.conv2(x))   # (B*N, 128, P)

        # Global average pool -> (B*N, 128)
        x = self.pool(x).squeeze(-1)

        # Project to LLM hidden dim -> (B*N, H)
        x = self.fc(x)

        # (B, N, H)
        return x.view(B, self.num_patches, -1)
