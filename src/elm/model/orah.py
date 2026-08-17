import torch
from peft import PeftModel, LoraConfig, get_peft_model
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
from elm.model.siglep import SigLEP, SigLEPConfig
from elm.utils.constants import ECG_TOKEN_PLACEHOLDER

class OrahConfig(PretrainedConfig):
    # HuggingFace Pretrained Config Variables
    # https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L163
    model_type = "orah"
    has_no_defaults_at_init = True
    sub_configs = {"text_config": AutoConfig, "vision_config": Siglip2VisionConfig}

    def __init__(self, text_config=None, vision_config=None, ecg_token_id=None,
                 num_ecg_tokens=100, segment_length=2500, patch_size=25,
                 num_leads=12, trainable=("projector", "language_model"), **kwargs):
        if text_config is None or vision_config is None:
            raise ValueError("text_config and vision_config are required")
        self.text_config = text_config if isinstance(text_config, PretrainedConfig) else AutoConfig.for_model(**text_config)
        self.vision_config = (vision_config if isinstance(vision_config, Siglip2VisionConfig)
                              else Siglip2VisionConfig(**vision_config))
        kwargs.setdefault("attn_implementation", {
            name: getattr(self, name)._attn_implementation for name in self.sub_configs
        })
        if min(num_ecg_tokens, segment_length, patch_size, num_leads) < 1 or segment_length % patch_size:
            raise ValueError("ECG dimensions must be positive and segment_length divisible by patch_size")
        self.ecg_token_id = ecg_token_id
        self.num_ecg_tokens = num_ecg_tokens
        self.segment_length = segment_length
        self.patch_size = patch_size
        self.num_leads = num_leads
        self.trainable = list(trainable)
        super().__init__(**kwargs)


class Orah(PreTrainedModel, GenerationMixin):
    # HuggingFace Pretrained Config Variables
    config_class = OrahConfig
    supports_gradient_checkpointing = True
    _components = ("encoder", "projector", "language_model")

    def __init__(self, config, language_model=None, vision_model=None):
        super().__init__(config)
        self.language_model = language_model or AutoModelForCausalLM.from_config(config.text_config)
        if isinstance(vision_model, SigLEP):
            dimensions = (config.num_ecg_tokens, config.segment_length, config.patch_size, config.num_leads)
            encoder_dimensions = tuple(getattr(vision_model.config, name) for name in (
                "num_ecg_tokens", "segment_length", "patch_size", "num_leads"))
            if dimensions != encoder_dimensions:
                raise ValueError(f"SigLEP dimensions {encoder_dimensions} do not match Orah dimensions {dimensions}")
            self.encoder = vision_model
        else:
            encoder_config = SigLEPConfig(**(config.vision_config.to_dict() | {
                "num_ecg_tokens": config.num_ecg_tokens,
                "segment_length": config.segment_length,
                "patch_size": config.patch_size,
                "num_leads": config.num_leads,
            }))
            self.encoder = SigLEP(encoder_config, vision_model)
        self.projector = nn.Sequential(
            nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size),
            nn.GELU(),
            nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size),
        )
        self.post_init()
        self.set_trainable(config.trainable)

    def get_input_embeddings(self): return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value): self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self): return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, value): self.language_model.set_output_embeddings(value)

    def resize_token_embeddings(self, new_num_tokens=None, **kwargs):
        embeddings = self.language_model.resize_token_embeddings(new_num_tokens, **kwargs)
        self.vocab_size = embeddings.num_embeddings
        return embeddings

    def prepare_inputs(self, input_ids, attention_mask, ecg_values, inputs_embeds):
        if ecg_values is None:
            return input_ids, attention_mask, inputs_embeds
        if self.config.ecg_token_id is None:
            raise ValueError("ecg_token_id is required for ECG input")
        if input_ids is None:
            raise ValueError("input_ids are required with ECG input")
        if input_ids.shape[0] != ecg_values.shape[0]:
            raise ValueError("Text and ECG batch sizes must match")
        features = self.projector(self.encoder(ecg_values))
        mask = input_ids.eq(self.config.ecg_token_id)
        if not torch.all(mask.sum(-1) == features.shape[1]):
            raise ValueError(f"Each input must contain {features.shape[1]} ECG tokens")
        inputs_embeds = self.get_input_embeddings()(input_ids) if inputs_embeds is None else inputs_embeds
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[mask] = features.to(inputs_embeds).flatten(0, 1)
        return input_ids, attention_mask, inputs_embeds

    def forward(self, input_ids=None, attention_mask=None, labels=None, ecg_values=None,
                inputs_embeds=None, **kwargs):
        input_ids, attention_mask, inputs_embeds = self.prepare_inputs(
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
        input_ids, attention_mask, inputs_embeds = self.prepare_inputs(
            input_ids, attention_mask, ecg_values, inputs_embeds)
        if inputs_embeds is None:
            return self.language_model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def set_trainable(self, names):
        trainable = set(names)
        unknown = trainable.difference(self._components)
        if unknown:
            raise ValueError(f"Unknown trainable components: {sorted(unknown)}")
        if isinstance(self.language_model, PeftModel):
            trainable.add("language_model")
        self.config.trainable = [name for name in self._components if name in trainable]
        for name in self._components:
            module = getattr(self, name)
            if not isinstance(module, PeftModel):
                module.requires_grad_(name in trainable)
        return self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for name in self._components:
            if name not in self.config.trainable:
                getattr(self, name).eval()
        return self

def build(config, tokenizer):
    if config["model"].get("checkpoint"):
        model = Orah.from_pretrained(config["model"]["checkpoint"])
    else:
        language_model = AutoModelForCausalLM.from_pretrained(config["model"]["language_model"])
        vision_config = AutoConfig.from_pretrained(config["model"]["vision_model"])
        vision_model = (SigLEP.from_pretrained(config["model"]["vision_model"], config=vision_config)
                        if isinstance(vision_config, SigLEPConfig)
                        else Siglip2VisionModel.from_pretrained(config["model"]["vision_model"]))
        orah_config = OrahConfig(
            language_model.config,
            vision_model.config,
            tokenizer.convert_tokens_to_ids(ECG_TOKEN_PLACEHOLDER),
            num_ecg_tokens=config["model"]["num_ecg_tokens"],
            segment_length=config["segment_length"],
            patch_size=config["model"]["patch_size"],
            num_leads=len(config["leads"]),
        )
        model = Orah(orah_config, language_model, vision_model)
    if config["model"]["peft"] and not isinstance(model.language_model, PeftModel):
        lora_config = LoraConfig(
            r=config["model"]["lora_rank"], lora_alpha=config["model"]["lora_alpha"],
            target_modules=config["model"]["target_modules"],
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        model.language_model.print_trainable_parameters()
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))
    return model.set_trainable(config["model"].get("trainable", model.config.trainable))

OrahConfig.register_for_auto_class()
Orah.register_for_auto_class("AutoModelForCausalLM")
