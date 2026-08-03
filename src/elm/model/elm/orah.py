import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin


def _config(value, default):
    if isinstance(value, PretrainedConfig):
        return value
    value = dict(value or {"model_type": default})
    return AutoConfig.for_model(value.pop("model_type"), **value)


class OrahConfig(PretrainedConfig):
    model_type = "orah"
    is_composition = True
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(self, text_config=None, vision_config=None, ecg_token_id=None,
                 num_ecg_tokens=100, segment_length=2500, patch_size=25,
                 num_leads=12, **kwargs):
        self.text_config = _config(text_config, "qwen3")
        self.vision_config = _config(vision_config, "siglip2_vision_model")
        if min(num_ecg_tokens, segment_length, patch_size, num_leads) < 1 or segment_length % patch_size:
            raise ValueError("ECG dimensions must be positive and segment_length divisible by patch_size")
        self.vision_config.vision_use_head = False
        self.ecg_token_id = ecg_token_id
        self.num_ecg_tokens = num_ecg_tokens
        self.segment_length = segment_length
        self.patch_size = patch_size
        self.num_leads = num_leads
        for key in ("bos_token_id", "eos_token_id", "pad_token_id", "tie_word_embeddings", "vocab_size"):
            kwargs.setdefault(key, getattr(self.text_config, key, None))
        super().__init__(**kwargs)


class ECGEmbeddings(nn.Module):
    def __init__(self, patch_dim, num_patches, hidden_size):
        super().__init__()
        self.patch_embedding = nn.Linear(patch_dim, hidden_size)
        self.position_embedding = nn.Embedding(num_patches, hidden_size)

    def forward(self, pixel_values, spatial_shapes=None):
        positions = torch.arange(pixel_values.shape[1], device=pixel_values.device)
        return self.patch_embedding(pixel_values.to(self.patch_embedding.weight)) + self.position_embedding(positions)[None]


class SigLEP(nn.Module):
    def __init__(self, vision_model, config):
        super().__init__()
        self.vision_model = vision_model
        self.patch_size = config.patch_size
        self.shape = (config.num_leads, config.segment_length)
        self.vision_model.embeddings = ECGEmbeddings(
            config.num_leads * config.patch_size,
            config.segment_length // config.patch_size,
            config.vision_config.hidden_size,
        )
        self.vision_model.use_head = False
        if hasattr(self.vision_model, "head"):
            del self.vision_model.head
        self.pool = nn.AdaptiveAvgPool1d(config.num_ecg_tokens)

    def forward(self, ecg_values):
        if tuple(ecg_values.shape[1:]) != self.shape:
            raise ValueError(f"Expected ECG shape (batch, {self.shape[0]}, {self.shape[1]}), got {tuple(ecg_values.shape)}")
        low = ecg_values.amin(dim=(-2, -1), keepdim=True)
        ecg_values = (ecg_values - low) / (ecg_values.amax(dim=(-2, -1), keepdim=True) - low + 1e-6)
        patches = ecg_values.unfold(-1, self.patch_size, self.patch_size).permute(0, 2, 1, 3).flatten(2)
        batch, length = patches.shape[:2]
        output = self.vision_model(
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
    input_modalities = ("text", "time")

    def __init__(self, config, llm=None, vision_model=None):
        super().__init__(config)
        self.llm = llm or AutoModelForCausalLM.from_config(config.text_config)
        vision_model = vision_model or AutoModel.from_config(config.vision_config)
        self.encoder = SigLEP(vision_model, config)
        self.connector = MLPProjection(config.vision_config.hidden_size, config.text_config.hidden_size)
        self.post_init()

    @classmethod
    def from_components(cls, llm, encoder, ecg_token_id, **kwargs):
        vision_model = getattr(encoder, "vision_model", encoder)
        config = OrahConfig(llm.config, vision_model.config, ecg_token_id, **kwargs)
        return cls(config, llm, vision_model)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, value):
        self.llm.set_output_embeddings(value)

    def get_ecg_features(self, ecg_values):
        return self.connector(self.encoder(ecg_values))

    def _prepare_inputs(self, input_ids, attention_mask, ecg_values, inputs_embeds=None):
        if input_ids is None:
            if ecg_values is None or self.config.ecg_token_id is None:
                return input_ids, attention_mask, inputs_embeds
            input_ids = torch.full(
                (ecg_values.shape[0], self.config.num_ecg_tokens), self.config.ecg_token_id,
                dtype=torch.long, device=ecg_values.device,
            )
            attention_mask = torch.ones_like(input_ids)
        if ecg_values is None:
            return input_ids, attention_mask, inputs_embeds
        if inputs_embeds is not None:
            raise ValueError("Pass input_ids, not inputs_embeds, when ecg_values is provided")
        if self.config.ecg_token_id is None:
            raise ValueError("config.ecg_token_id is required for ECG input")
        if input_ids.shape[0] != ecg_values.shape[0]:
            raise ValueError("input_ids and ecg_values must have the same batch size")
        features = self.get_ecg_features(ecg_values)
        mask = input_ids.eq(self.config.ecg_token_id)
        if not torch.all(mask.sum(-1) == features.shape[1]):
            raise ValueError(f"Each prompt must contain {features.shape[1]} ECG tokens")
        inputs_embeds = self.get_input_embeddings()(input_ids)
        mask = mask.unsqueeze(-1).expand_as(inputs_embeds)
        return input_ids, attention_mask, inputs_embeds.masked_scatter(mask, features.to(inputs_embeds).flatten())

    def forward(self, input_ids=None, attention_mask=None, labels=None,
                ecg_values=None, inputs_embeds=None, **kwargs):
        input_ids, attention_mask, inputs_embeds = self._prepare_inputs(
            input_ids, attention_mask, ecg_values, inputs_embeds)
        return self.llm(
            input_ids=None if inputs_embeds is not None else input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, ecg_values=None,
                 inputs_embeds=None, **kwargs):
        input_ids, attention_mask, inputs_embeds = self._prepare_inputs(
            input_ids, attention_mask, ecg_values, inputs_embeds)
        self.llm.generation_config = self.generation_config
        return self.llm.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def set_trainable(self, names):
        self._trainable = set(names)
        for name in ("encoder", "connector", "llm"):
            if name != "llm" or name not in self._trainable:
                getattr(self, name).requires_grad_(name in self._trainable)
        return self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for name in ("encoder", "connector", "llm"):
            if hasattr(self, "_trainable") and name not in self._trainable:
                getattr(self, name).eval()
        return self


OrahConfig.register_for_auto_class()
Orah.register_for_auto_class("AutoModelForCausalLM")
