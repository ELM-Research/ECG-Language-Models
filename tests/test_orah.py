import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Siglip2VisionConfig,
    Siglip2VisionModel,
)

from elm.model import build_model
from elm.model import orah


class Tokenizer:
    def __len__(self):
        return 21

    def convert_tokens_to_ids(self, token):
        return 20


def test_build_forward_generate_and_reload(monkeypatch, tmp_path):
    language_model = GPT2LMHeadModel(GPT2Config(
        vocab_size=20,
        n_embd=8,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    ))
    vision_model = Siglip2VisionModel(Siglip2VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_patches=2,
    ))
    monkeypatch.setattr(orah.AutoModelForCausalLM, "from_pretrained", lambda _: language_model)
    monkeypatch.setattr(orah.Siglip2VisionModel, "from_pretrained", lambda _: vision_model)
    config = {
        "leads": [0, 1],
        "segment_length": 4,
        "model": {
            "name": "orah",
            "checkpoint": None,
            "language_model": "text",
            "vision_model": "vision",
            "num_ecg_tokens": 2,
            "patch_size": 2,
        },
    }

    model = build_model(config, Tokenizer())
    input_ids = torch.tensor([[20, 20, 3]])
    attention_mask = torch.ones_like(input_ids)
    ecg_values = torch.randn(1, 2, 4)

    output = model(input_ids=input_ids, attention_mask=attention_mask, ecg_values=ecg_values)
    assert output.logits.shape == (1, 3, 21)

    generation_config = language_model.generation_config
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ecg_values=ecg_values,
        max_new_tokens=1,
    )
    assert generated.shape == (1, 4)
    assert language_model.generation_config is generation_config

    model.save_pretrained(tmp_path)
    config["model"]["checkpoint"] = tmp_path
    reloaded = build_model(config, Tokenizer())
    output = reloaded(input_ids=input_ids, attention_mask=attention_mask, ecg_values=ecg_values)
    assert output.logits.shape == (1, 3, 21)
