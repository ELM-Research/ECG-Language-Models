from torch import nn


class HFSiglip(nn.Module):
    def __init__(
        self,
        vision_encoder,
        output_hidden_states
    ):
        super(HFSiglip, self).__init__()
        self.vision_encoder = vision_encoder
        self.output_hidden_states = output_hidden_states

    def forward(self,):
        raise NotImplementedError

    def get_encoder_embeddings(self, encoder_input_ids, encoder_attention_mask,
                               encoder_pixels, spatial_shapes):
        out = self.vision_encoder(input_ids = encoder_input_ids,
                                  pixel_attention_mask = encoder_attention_mask,
                                  pixel_values = encoder_pixels,
                                  output_hidden_states = self.output_hidden_states,
                                  spatial_shapes = spatial_shapes)
        return out.image_embeds
