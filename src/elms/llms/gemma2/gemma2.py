from torch import nn

class Gemma2(nn.Module):
    def __init__(self, llm, pad_token_id, eos_token_id, output_hidden_states):
        super(Gemma2, self).__init__()
        self.llm = llm
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.output_hidden_states = output_hidden_states

    def forward(self, elm_input_ids, elm_attention_mask,
                elm_labels, elm_inputs_embeds = None):
        return self.llm(input_ids = elm_input_ids,
                        inputs_embeds = elm_inputs_embeds,
                        attention_mask = elm_attention_mask,
                        labels = elm_labels,
                        output_hidden_states = self.output_hidden_states,
                        use_cache = False)

    def get_llm_embeddings(self, elm_input_ids):
        out = self.llm.get_input_embeddings()(elm_input_ids.to(self.llm.device))
        return out

    def fsdp_wrap_modules(self):
        from elms.llms._wrap import get_decoder_layers
        return get_decoder_layers(self.llm)

    def generate(self, elm_input_ids, elm_attention_mask,
                 elm_inputs_embeds= None, max_new_tokens=128, **gen_kwargs):
        return self.llm.generate(
                input_ids=elm_input_ids,
                inputs_embeds = elm_inputs_embeds,
                attention_mask=elm_attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                **gen_kwargs,
            )
