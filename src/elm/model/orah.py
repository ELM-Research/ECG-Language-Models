import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
    Siglip2VisionConfig,
    Siglip2VisionModel,
)


class OrahConfig(PretrainedConfig):
    model_type = "orah"
    is_composition = True
    sub_configs = {"text_config": AutoConfig, "vision_config": Siglip2VisionConfig}

    def __init__(self, text_config=None, vision_config=None, ecg_token_id=None,
                 num_ecg_tokens=100, segment_length=2500, patch_size=25,
                 num_leads=12, **kwargs):
        if text_config is None or vision_config is None:
            raise ValueError("text_config and vision_config are required")
        self.text_config = text_config if isinstance(text_config, PretrainedConfig) else AutoConfig.for_model(**text_config)
        if not isinstance(vision_config, PretrainedConfig):
            vision_config = {key: value for key, value in vision_config.items() if key != "model_type"}
            vision_config = Siglip2VisionConfig(**vision_config)
        self.vision_config = vision_config
        if min(num_ecg_tokens, segment_length, patch_size, num_leads) < 1 or segment_length % patch_size:
            raise ValueError("ECG dimensions must be positive and segment_length divisible by patch_size")
        self.vision_config.vision_use_head = False
        self.ecg_token_id = ecg_token_id
        self.num_ecg_tokens = num_ecg_tokens
        self.segment_length = segment_length
        self.patch_size = patch_size
        self.num_leads = num_leads
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "tie_word_embeddings", "vocab_size"):
            kwargs.setdefault(name, getattr(self.text_config, name, None))
        super().__init__(**kwargs)


class ECGEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = nn.Linear(config.num_leads * config.patch_size, config.vision_config.hidden_size)
        self.position_embedding = nn.Embedding(config.segment_length // config.patch_size, config.vision_config.hidden_size)

    def forward(self, patches, spatial_shapes=None):
        positions = torch.arange(patches.shape[1], device=patches.device)
        return self.patch_embedding(patches) + self.position_embedding(positions)[None]


class SigLEP(nn.Module):
    def __init__(self, vision_model, config):
        super().__init__()
        self.model = vision_model
        self.model.vision_model.embeddings = ECGEmbedding(config)
        self.model.vision_model.use_head = False
        self.model.vision_model.head = None
        self.shape = (config.num_leads, config.segment_length)
        self.patch_size = config.patch_size
        self.pool = nn.AdaptiveAvgPool1d(config.num_ecg_tokens)

    def forward(self, ecg_values):
        if tuple(ecg_values.shape[1:]) != self.shape:
            raise ValueError(f"Expected ECG shape (batch, {self.shape[0]}, {self.shape[1]}), got {tuple(ecg_values.shape)}")
        ecg_values = ecg_values.to(self.model.vision_model.embeddings.patch_embedding.weight)
        low = ecg_values.amin((-2, -1), keepdim=True)
        ecg_values = (ecg_values - low) / (ecg_values.amax((-2, -1), keepdim=True) - low + 1e-6)
        patches = ecg_values.unfold(-1, self.patch_size, self.patch_size).transpose(1, 2).flatten(2)
        batch, length = patches.shape[:2]
        output = self.model(
            pixel_values=patches,
            pixel_attention_mask=torch.ones((batch, length), dtype=torch.long, device=patches.device),
            spatial_shapes=torch.zeros((batch, 2), dtype=torch.long, device=patches.device),
        ).last_hidden_state
        return self.pool(output.transpose(1, 2)).transpose(1, 2)


class MLPProjection(nn.Sequential):
    def __init__(self, input_dim, output_dim):
        super().__init__(nn.Linear(input_dim, output_dim), nn.GELU(), nn.Linear(output_dim, output_dim))


class Orah(PreTrainedModel, GenerationMixin):
    config_class = OrahConfig
    supports_gradient_checkpointing = True

    def __init__(self, config, language_model=None, vision_model=None):
        super().__init__(config)
        self.language_model = language_model or AutoModelForCausalLM.from_config(config.text_config)
        vision_model = vision_model or Siglip2VisionModel(config.vision_config)
        self.encoder = SigLEP(vision_model, config)
        self.projector = MLPProjection(config.vision_config.hidden_size, config.text_config.hidden_size)
        self.post_init()

    @classmethod
    def from_components(cls, language_model, vision_model, ecg_token_id, **kwargs):
        config = OrahConfig(language_model.config, vision_model.config, ecg_token_id, **kwargs)
        return cls(config, language_model, vision_model)

    def get_input_embeddings(self): return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value): self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self): return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value): self.language_model.set_output_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens=None, **kwargs):
        embeddings = self.language_model.resize_token_embeddings(new_num_tokens, **kwargs)
        self.config.text_config.vocab_size = self.config.vocab_size = embeddings.num_embeddings
        return embeddings

    def get_ecg_features(self, ecg_values): return self.projector(self.encoder(ecg_values))

    def _prepare_inputs(self, input_ids, attention_mask, ecg_values, inputs_embeds):
        if input_ids is None and ecg_values is not None:
            if self.config.ecg_token_id is None:
                raise ValueError("ecg_token_id is required for ECG input")
            input_ids = torch.full((ecg_values.shape[0], self.config.num_ecg_tokens), self.config.ecg_token_id,
                                   dtype=torch.long, device=self.get_input_embeddings().weight.device)
            attention_mask = torch.ones_like(input_ids)
        if ecg_values is None:
            return input_ids, attention_mask, inputs_embeds
        if self.config.ecg_token_id is None:
            raise ValueError("ecg_token_id is required for ECG input")
        if input_ids.shape[0] != ecg_values.shape[0]:
            raise ValueError("Text and ECG batch sizes must match")
        features = self.get_ecg_features(ecg_values)
        mask = input_ids.eq(self.config.ecg_token_id)
        if not torch.all(mask.sum(-1) == features.shape[1]):
            raise ValueError(f"Each input must contain {features.shape[1]} ECG tokens")
        inputs_embeds = self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[mask] = features.to(inputs_embeds).flatten(0, 1)
        return input_ids, attention_mask, inputs_embeds

    def forward(self, input_ids=None, attention_mask=None, labels=None, ecg_values=None,
                inputs_embeds=None, **kwargs):
        input_ids, attention_mask, inputs_embeds = self._prepare_inputs(
            input_ids, attention_mask, ecg_values, inputs_embeds)
        return self.language_model(
            input_ids=None if inputs_embeds is not None else input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, ecg_values=None, inputs_embeds=None, **kwargs):
        input_ids, attention_mask, inputs_embeds = self._prepare_inputs(
            input_ids, attention_mask, ecg_values, inputs_embeds)
        self.language_model.generation_config = self.generation_config
        if inputs_embeds is None:
            return self.language_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def set_trainable(self, names):
        self._trainable = set(names)
        for name in ("encoder", "projector", "language_model"):
            if name not in self._trainable:
                getattr(self, name).requires_grad_(False)
            elif name != "language_model":
                getattr(self, name).requires_grad_(True)
        return self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for name in ("encoder", "projector", "language_model"):
            if hasattr(self, "_trainable") and name not in self._trainable:
                getattr(self, name).eval()
        return self


def build_orah(text_model, vision_model, vocab_size, ecg_token_id, **kwargs):
    language_model = AutoModelForCausalLM.from_pretrained(text_model)
    language_model.resize_token_embeddings(vocab_size)
    encoder = Siglip2VisionModel.from_pretrained(vision_model)
    return Orah.from_components(language_model, encoder, ecg_token_id, **kwargs)


OrahConfig.register_for_auto_class()
Orah.register_for_auto_class("AutoModelForCausalLM")