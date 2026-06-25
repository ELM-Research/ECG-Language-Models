import torch
from torch import nn
from einops import rearrange


class Ecg1DEmbeddings(nn.Module):
    def __init__(self, patch_dim: int, num_patches: int, hidden_size: int):
        super(Ecg1DEmbeddings, self).__init__()
        self.patch_embedding = nn.Linear(patch_dim, hidden_size)
        self.position_embedding = nn.Embedding(num_patches, hidden_size)

    def forward(self, pixel_values, spatial_shapes=None):  # spatial_shapes unused (1D)
        pos = torch.arange(pixel_values.shape[1], device=pixel_values.device)
        return self.patch_embedding(pixel_values) + self.position_embedding(pos).unsqueeze(0)


class SiglipEcg(nn.Module):
    def __init__(self, vision_encoder, segment_len, patch_size, num_encoder_tokens, num_leads=12):
        super(SiglipEcg, self).__init__()
        assert segment_len % patch_size == 0, "segment_len must be divisible by patch_size"
        self.vision_encoder = vision_encoder
        # keep only the vision encoder path; drop the text tower, contrastive logits, and pooling head
        del self.vision_encoder.text_model, self.vision_encoder.logit_scale, self.vision_encoder.logit_bias
        self.vision_encoder.vision_model.use_head = False
        del self.vision_encoder.vision_model.head
        self.patch_size = patch_size
        hidden = self.vision_encoder.config.vision_config.hidden_size
        self.vision_encoder.vision_model.embeddings = Ecg1DEmbeddings(
            num_leads * patch_size, segment_len // patch_size, hidden
        )
        self.pool = nn.AdaptiveAvgPool1d(num_encoder_tokens)

    def forward(self,):
        raise NotImplementedError

    def get_encoder_embeddings(self, ecg_signal):
        patches = rearrange(ecg_signal, "b c (n p) -> b n (c p)", p=self.patch_size)
        spatial_shapes = torch.zeros(patches.shape[0], 2, dtype=torch.long, device=patches.device)
        out = self.vision_encoder.vision_model(pixel_values=patches,
                                               attention_mask=None,
                                               spatial_shapes=spatial_shapes)
        x = out.last_hidden_state.transpose(1, 2)
        return self.pool(x).transpose(1, 2)
