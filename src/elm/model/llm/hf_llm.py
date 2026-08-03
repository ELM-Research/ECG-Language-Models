from torch import nn
from contextlib import contextmanager


@contextmanager
def generation_mode(hf_llm):
    # KV cache and gradient checkpointing are mutually exclusive: checkpointing
    # recomputes activations in the backward pass and discards cached states, so
    # HF forces use_cache=False whenever a checkpointed model is in training mode.
    # Autoregressive generation has no backward pass and needs the cache to avoid
    # recomputing the full prefix at every step, so flip the model into a
    # cache-on / checkpointing-off configuration here and restore it afterwards.
    was_gc = getattr(hf_llm, "is_gradient_checkpointing", False)
    prev_use_cache = hf_llm.config.use_cache
    if was_gc:
        hf_llm.gradient_checkpointing_disable()
    hf_llm.config.use_cache = True
    try:
        yield
    finally:
        hf_llm.config.use_cache = prev_use_cache
        if was_gc:
            hf_llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

class HFLLM(nn.Module):
    """Adapter from the ELM argument names to a Hugging Face causal LM.

    Default behavior shared by all supported LLM families; per-model
    differences (chat template, watch tokens, dtype) live in HF_LLMS in
    configs/constants.py. Subclass and override when a new model needs
    family-specific handling.
    """

    def __init__(self, llm):
        super().__init__()
        self.llm = llm

    def forward(self, elm_input_ids, elm_attention_mask,
                elm_labels, elm_inputs_embeds = None):
        return self.llm(input_ids = elm_input_ids,
                        inputs_embeds = elm_inputs_embeds,
                        attention_mask = elm_attention_mask,
                        labels = elm_labels,
                        output_hidden_states = self.output_hidden_states)

    def get_llm_embeddings(self, elm_input_ids):
        out = self.llm.get_input_embeddings()(elm_input_ids.to(self.llm.device))
        return out

    def generate(self, elm_input_ids, elm_attention_mask,
                 elm_inputs_embeds= None, max_new_tokens=128, **gen_kwargs):
        with generation_mode(self.llm):
            return self.llm.generate(
                    input_ids=elm_input_ids,
                    inputs_embeds = elm_inputs_embeds,
                    attention_mask=elm_attention_mask,
                    max_new_tokens=max_new_tokens,
                    **gen_kwargs,
                )
