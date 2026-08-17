import json

import pytest
import torch
from peft import PeftModel
from transformers import Qwen2Config, Siglip2VisionConfig

from elm.model.orah import Orah, OrahConfig, build


LORA = {"r": 2, "lora_alpha": 4, "target_modules": "all-linear"}


class Tokenizer:
    def __len__(self):
        return 32


def build_config(checkpoint, peft, rank=2):
    return {"model": {
        "checkpoint": str(checkpoint),
        "peft": peft,
        "lora_rank": rank,
        "lora_alpha": 4,
        "target_modules": "all-linear",
        "trainable": ["language_model"],
    }}


def make_model(peft):
    text = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    vision = Siglip2VisionConfig(hidden_size=16, intermediate_size=32,
                                 num_hidden_layers=1, num_attention_heads=2,
                                 image_size=4, patch_size=2)
    model = Orah(OrahConfig(text, vision, ecg_token_id=31, num_ecg_tokens=2,
                           segment_length=4, patch_size=2, num_leads=1,
                           lora=LORA if peft else None))
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            torch.nn.init.constant_(parameter, 0.1)
    return model.eval()


@pytest.mark.parametrize(("saved_peft", "loaded_peft"), [
    (False, False), (False, True), (True, True), (True, False),
])
def test_peft_checkpoint_transitions(tmp_path, saved_peft, loaded_peft):
    torch.manual_seed(0)
    saved = make_model(saved_peft)
    input_ids = torch.tensor([[1, 2, 3]])
    expected = saved(input_ids=input_ids).logits
    saved.save_pretrained(tmp_path)
    loaded = build(build_config(tmp_path, loaded_peft), Tokenizer()).eval()
    actual = loaded(input_ids=input_ids).logits
    assert isinstance(loaded.language_model, PeftModel) is loaded_peft
    assert loaded.config.lora == (LORA if loaded_peft else None)
    if loaded_peft:
        trainable = [name for name, parameter in loaded.language_model.named_parameters()
                     if parameter.requires_grad]
        assert trainable and all("lora_" in name for name in trainable)
    torch.testing.assert_close(actual, expected)


def test_peft_respects_trainable_components():
    model = make_model(True).set_trainable(["projector"])
    assert all(not parameter.requires_grad for parameter in model.language_model.parameters())
    assert all(parameter.requires_grad for parameter in model.projector.parameters())


def test_rejects_changed_lora_shape(tmp_path):
    make_model(True).save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="LoRA configuration"):
        build(build_config(tmp_path, True, rank=4), Tokenizer())


def test_rejects_legacy_peft_checkpoint(tmp_path):
    make_model(True).save_pretrained(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    del config["lora"]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="do not match the saved configuration"):
        build(build_config(tmp_path, True), Tokenizer())
