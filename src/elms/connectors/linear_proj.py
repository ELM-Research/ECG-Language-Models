from torch import nn

from configs.constants import HF_LLMS


class LinearProjection(nn.Module):
    def __init__(self, projection_dim, llm_id):
        super(LinearProjection, self).__init__()
        self.input_dtype = HF_LLMS[llm_id]["native_dtype"]
        self.projection = nn.Linear(projection_dim,
                                    HF_LLMS[llm_id]["model_hidden_size"]).to(dtype=self.input_dtype)
    def forward(self, ecg_signal):
        return self.projection(ecg_signal.to(dtype=self.projection.weight.dtype))

    def project(self, signal_embeds):
        return self.projection(signal_embeds.to(dtype=self.projection.weight.dtype))
