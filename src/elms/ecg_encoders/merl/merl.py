from torch import nn
from dataclasses import dataclass

from elms.ecg_encoders.merl.blocks import get_resnet

@dataclass
class MerlConfig:
    proj_hidden: int = 256
    proj_out: int = 256
    in_channels: int = 2048
    num_layers: int = 12
    dropout: float = 0.1
    seq_len: int = 2500
    lm: str = "ncbi/MedCPT-Query-Encoder"
    resnet_type: str = "resnet101"
    spacial_dim: int = None
    d_model: int = 2048
    num_encoder_tokens: int = 1

    def __post_init__(self):
        if self.seq_len == 2500:
            self.spacial_dim = 157
        elif self.seq_len == 1250:
            self.spacial_dim = 79
        else:
            self.spacial_dim = 32

class Merl(nn.Module):
    def __init__(self, cfg: MerlConfig):
        super().__init__()
        self.cfg = cfg
        self.resnet = get_resnet(cfg.resnet_type)
        self.avgpool = nn.AdaptiveAvgPool1d(cfg.num_encoder_tokens)

    def forward(self,):
        raise NotImplementedError

    def get_encoder_embeddings(self, ecg_signal):
        out = self.resnet(ecg_signal)
        out = self.avgpool(out)
        return out.transpose(1, 2)