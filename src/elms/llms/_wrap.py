"""Shared helpers for FSDP wrapping of LLM wrapper modules."""

from torch import nn


def get_decoder_layers(hf_model: nn.Module) -> list[nn.Module]:
    """Return the transformer decoder block ModuleList inside an HF causal LM.

    Handles PEFT (PeftModel -> base_model -> model -> model.layers) and the
    bare HF (model.model.layers) layouts.
    """
    node = hf_model
    if hasattr(node, "base_model") and hasattr(node.base_model, "model"):
        node = node.base_model.model
    if hasattr(node, "model") and hasattr(node.model, "layers"):
        return list(node.model.layers)
    if hasattr(node, "layers"):
        return list(node.layers)
    raise RuntimeError(f"Could not locate decoder layers inside {type(hf_model).__name__}")
