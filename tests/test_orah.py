import torch
from peft import LoraConfig, get_peft_model
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
    assert model.language_model.config._attn_implementation == "sdpa"
    assert model.encoder.model.config._attn_implementation == "sdpa"
    model.train()
    assert model.config.trainable == ["projector", "language_model"]
    assert not model.encoder.training
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.projector.parameters())
    assert all(parameter.requires_grad for parameter in model.language_model.parameters())

    model.set_trainable(["encoder"])
    assert model.encoder.training
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
    assert not model.projector.training and not model.language_model.training
    assert not any(parameter.requires_grad for parameter in model.projector.parameters())
    assert not any(parameter.requires_grad for parameter in model.language_model.parameters())
    model.set_trainable(["projector", "language_model"])

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

    model.language_model = get_peft_model(model.language_model, LoraConfig(target_modules=["c_attn"]))
    model.set_trainable([])
    assert model.config.trainable == ["language_model"]
    assert any("lora_" in name and parameter.requires_grad
               for name, parameter in model.language_model.named_parameters())
    assert not any("lora_" not in name and parameter.requires_grad
                   for name, parameter in model.language_model.named_parameters())
