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
