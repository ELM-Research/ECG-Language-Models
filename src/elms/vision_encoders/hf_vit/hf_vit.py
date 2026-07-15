import torch
from torch import nn


class HFVit(nn.Module):
    def __init__(
        self,
        vision_encoder,
        output_hidden_states
    ):
        super(HFVit, self).__init__()
        self.vision_encoder = vision_encoder
        self.output_hidden_states = output_hidden_states

    def forward(self,):
        raise NotImplementedError

    def get_encoder_embeddings(self, encoder_pixels, encoder_mask,):
        out = self.vision_encoder(pixel_values = encoder_pixels,
                                  bool_masked_pos = encoder_mask,
                                  output_hidden_states = self.output_hidden_states)
        return torch.mean(torch.mean(torch.stack(out.hidden_states), dim=0), dim=1)
