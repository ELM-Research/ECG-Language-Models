from torch import nn

class MLPProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        llm_hidden = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, llm_hidden),
            )

    def forward(self, ecg_signal): return self.projection(ecg_signal)

    def project(self, signal_embeds): return self.projection(signal_embeds)