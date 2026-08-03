from torch import nn

class MLPProjection(nn.Module):
    def __init__(self, input_dim, llm_hidden_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, llm_hidden_dim),
            nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim),
            )

    def forward(self, ecg_signal): return self.projection(ecg_signal)

    def project(self, signal_embeds): return self.projection(signal_embeds)