from torch import nn

from configs.constants import HF_LLMS

class PatchProjection(nn.Module):
    def __init__(self, num_patches, patch_dim, llm_id):
        super().__init__()
        self.num_patches = num_patches
        self.input_dtype = HF_LLMS[llm_id]["native_dtype"]
        self.projection = nn.Linear(patch_dim,
                                    HF_LLMS[llm_id]["model_hidden_size"]).to(dtype=self.input_dtype)

    def forward(self, ecg_signal):
        # ecg_signal: (B, C, L) -> N non-overlapping patches -> (B, N, C*Lp) -> (B, N, H)
        B, C, L = ecg_signal.shape
        x = ecg_signal.reshape(B, C, self.num_patches, L // self.num_patches)
        x = x.permute(0, 2, 1, 3).reshape(B, self.num_patches, -1)
        return self.projection(x.to(dtype=self.projection.weight.dtype))