import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Siglip2VisionConfig

from elm.data.collator import ELMDataCollator
from elm.data.modality.signal import Signal
from elm.data.modality.text import Text
from elm.model import Orah, OrahConfig
from elm.utils.parallelism import apply_fsdp2, get_full_state_dict


def tiny_model():
    text = Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, bos_token_id=1, eos_token_id=2,
        pad_token_id=0, use_sliding_window=False,
    )
    vision = Siglip2VisionConfig(
        hidden_size=12, intermediate_size=24, num_hidden_layers=1,
        num_attention_heads=3, num_patches=4, patch_size=2,
    )
    return Orah(OrahConfig(
        text, vision, ecg_token_id=3, num_ecg_tokens=2,
        segment_length=8, patch_size=2, num_leads=2,
    )).eval()


def test_text_ecg_and_multimodal_inputs():
    model, ecg = tiny_model(), torch.randn(1, 2, 8)
    cases = (
        ({"input_ids": torch.tensor([[1, 4, 5]])}, 3),
        ({"ecg_values": ecg}, 2),
        ({"input_ids": torch.tensor([[1, 3, 3, 4]]), "ecg_values": ecg}, 4),
    )
    for inputs, length in cases:
        assert model(**inputs).logits.shape == (1, length, 32)
        assert model.generate(**inputs, max_new_tokens=1).shape == (1, length + 1)


def test_ecg_features_replace_only_ecg_token_embeddings(monkeypatch):
    model = tiny_model()
    input_ids = torch.tensor([[4, 3, 5, 3], [3, 6, 3, 7]])
    features = torch.arange(64).reshape(2, 2, 16).to(model.dtype)
    original = model.get_input_embeddings()(input_ids)
    monkeypatch.setattr(model, "get_ecg_features", lambda _: features)

    _, _, actual = model._prepare_inputs(input_ids, None, torch.zeros(2, 2, 8))
    expected = original.clone()
    expected[input_ids == model.config.ecg_token_id] = features.flatten(0, 1)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="Each prompt must contain 2 ECG tokens"):
        model(input_ids=torch.tensor([[3, 4]]), ecg_values=torch.zeros(1, 2, 8))


def test_signal_tokenization_preserves_exact_count_and_masks_labels():
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "report": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.add_tokens(["<ecg>"])
    placeholders = Signal({"mode": "static", "num_ecg_tokens": 2})(torch.zeros(2, 8))[1]
    tokenized = Text(tokenizer, 8, "pretrain")("report", placeholders)
    ecg_token_id = tokenizer.convert_tokens_to_ids("<ecg>")

    assert tokenized["input_ids"].count(ecg_token_id) == 2
    assert all(label == -100 for token, label in zip(tokenized["input_ids"], tokenized["labels"]) if token == ecg_token_id)
    longer = Text(tokenizer, 8, "pretrain")("report report", placeholders)
    batch = ELMDataCollator(tokenizer, label_pad_token_id=-100)([
        {**tokenized, "ecg_values": torch.zeros(2, 8)},
        {**longer, "ecg_values": torch.ones(2, 8)},
    ])
    assert batch["ecg_values"].shape == (2, 2, 8)
    assert torch.equal(batch["input_ids"].eq(ecg_token_id).sum(-1), torch.tensor([2, 2]))
    assert torch.all(batch["labels"][batch["input_ids"] == ecg_token_id] == -100)
    assert tiny_model()(**batch).logits.shape[:2] == batch["input_ids"].shape
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    messages = [{"role": "user", "content": "report"}, {"role": "assistant", "content": "report"}]
    sft = Text(tokenizer, 8, "sft")(messages, placeholders)
    assert sft["input_ids"].count(ecg_token_id) == 2
    with pytest.raises(ValueError, match="Expected 2 ECG tokens"):
        Text(tokenizer, 1, "pretrain")("report", placeholders)


def test_only_text_omits_ecg_values_and_tokens():
    model_inputs, placeholders = Signal({"mode": "static", "num_ecg_tokens": 2}, only_text=True)(torch.zeros(2, 8))
    assert model_inputs == {}
    assert placeholders == ""


def test_huggingface_round_trip(tmp_path, monkeypatch):
    model, ecg = tiny_model(), torch.randn(1, 2, 8)
    inputs = {"input_ids": torch.tensor([[1, 3, 3, 4]]), "ecg_values": ecg}
    expected = model(**inputs).logits
    model.generation_config.max_new_tokens = 1
    model.save_pretrained(tmp_path)

    import transformers.dynamic_module_utils as dynamic_modules
    monkeypatch.setattr(dynamic_modules, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True).eval()

    torch.testing.assert_close(loaded(**inputs).logits, expected)
    assert loaded.generate(**inputs).shape == (1, 5)
    assert (tmp_path / "orah.py").is_file()


def test_set_trainable_preserves_peft_style_freezing():
    model = tiny_model()
    frozen = next(model.llm.parameters())
    frozen.requires_grad_(False)
    model.set_trainable(["llm"])
    assert not frozen.requires_grad


def test_fsdp2_shards_hf_blocks_bottom_up_and_registers_generate(monkeypatch):
    import torch.distributed.fsdp as fsdp
    import elm.utils.parallelism as parallelism

    sharded, registered = [], []
    monkeypatch.setattr(parallelism.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(fsdp, "fully_shard", lambda module, **kwargs: sharded.append((module, kwargs)))
    monkeypatch.setattr(fsdp, "register_fsdp_forward_method", lambda module, method: registered.append((module, method)))
    model = apply_fsdp2(tiny_model())

    assert {module.__class__.__name__ for module, _ in sharded[:-1]} == {"Qwen3DecoderLayer", "Siglip2EncoderLayer"}
    assert all(kwargs["reshard_after_forward"] for _, kwargs in sharded[:-1])
    assert sharded[-1] == (model, {"reshard_after_forward": False})
    assert registered == [(model, "generate")]


def test_full_state_dict_keeps_huggingface_parameter_names():
    model = tiny_model()
    assert get_full_state_dict(model).keys() == model.state_dict().keys()
