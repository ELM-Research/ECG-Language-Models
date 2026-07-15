from torch import nn


class HFClip(nn.Module):
    def __init__(
        self,
        vision_encoder,
        output_hidden_states
    ):
        super(HFClip, self).__init__()
        self.vision_encoder = vision_encoder
        self.output_hidden_states = output_hidden_states

    def forward(self,):
        raise NotImplementedError

    def get_encoder_embeddings(self, encoder_input_ids, encoder_attention_mask, encoder_pixels,):
        out = self.vision_encoder(input_ids = encoder_input_ids,
                                  attention_mask = encoder_attention_mask,
                                  pixel_values = encoder_pixels,
                                  output_hidden_states = self.output_hidden_states)
        return out.image_embeds
