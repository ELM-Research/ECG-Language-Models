from torch import nn

from configs.constants import HF_LLMS


class MLPProjection(nn.Module):
    def __init__(self, input_dim, llm_id, hidden_dim=None):
        super().__init__()
        llm_hidden = HF_LLMS[llm_id]["model_hidden_size"]
        self.input_dtype = HF_LLMS[llm_id]["native_dtype"]
        hidden_dim = hidden_dim or llm_hidden
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_hidden),
        ).to(dtype=self.input_dtype)

    def forward(self, ecg_signal):
        return self.projection(ecg_signal.to(dtype=self.projection[0].weight.dtype))

    def project(self, signal_embeds):
        return self.projection(signal_embeds.to(dtype=self.projection[0].weight.dtype))