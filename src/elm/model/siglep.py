import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PreTrainedModel, Siglip2VisionConfig, Siglip2VisionModel


class SigLEPConfig(Siglip2VisionConfig):
    model_type = "siglep"

    def __init__(self, num_ecg_tokens=100, segment_length=2500, patch_size=25, num_leads=12, **kwargs):
        if min(num_ecg_tokens, segment_length, patch_size, num_leads) < 1 or segment_length % patch_size:
            raise ValueError("ECG dimensions must be positive and segment_length divisible by patch_size")
        self.num_ecg_tokens = num_ecg_tokens
        self.segment_length = segment_length
        self.patch_size = patch_size
        self.num_leads = num_leads
        super().__init__(patch_size=patch_size, **kwargs)


class ECGEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embedding = nn.Linear(config.num_leads * config.patch_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.segment_length // config.patch_size, config.hidden_size)

    def forward(self, patches, spatial_shapes=None):
        positions = torch.arange(patches.shape[1], device=patches.device)
        return self.patch_embedding(patches) + self.position_embedding(positions)[None]


class SigLEP(PreTrainedModel):
    config_class = SigLEPConfig
    base_model_prefix = "model"
    main_input_name = "ecg_values"
    supports_gradient_checkpointing = True

    def __init__(self, config, vision_model=None):
        super().__init__(config)
        vision_model = vision_model or Siglip2VisionModel(config)
        self.model = getattr(vision_model, "vision_model", vision_model)
        self.model.embeddings = ECGEmbedding(config)
        self.model.use_head = False
        self.model.head = None
        self.shape = (config.num_leads, config.segment_length)
        self.patch_size = config.patch_size
        self.pool = nn.AdaptiveAvgPool1d(config.num_ecg_tokens)
        self.post_init()
        self._init_weights(self.model.embeddings)

    def _init_weights(self, module):
        if isinstance(module, ECGEmbedding):
            nn.init.trunc_normal_(module.patch_embedding.weight, std=module.patch_embedding.in_features**-0.5)
            nn.init.zeros_(module.patch_embedding.bias)
            nn.init.normal_(module.position_embedding.weight, std=self.config.hidden_size**-0.5)
        else:
            super()._init_weights(module)

    def forward(self, ecg_values):
        if tuple(ecg_values.shape[1:]) != self.shape:
            raise ValueError(f"Expected ECG shape (batch, {self.shape[0]}, {self.shape[1]}), got {tuple(ecg_values.shape)}")
        ecg_values = ecg_values.to(self.model.embeddings.patch_embedding.weight)
        low = ecg_values.amin((-2, -1), keepdim=True)
        ecg_values = (ecg_values - low) / (ecg_values.amax((-2, -1), keepdim=True) - low + 1e-6)
        patches = ecg_values.unfold(-1, self.patch_size, self.patch_size).transpose(1, 2).flatten(2)
        batch, length = patches.shape[:2]
        output = self.model(
            patches,
            torch.ones((batch, length), dtype=torch.long, device=patches.device),
            torch.zeros((batch, 2), dtype=torch.long, device=patches.device),
        ).last_hidden_state
        return self.pool(output.transpose(1, 2)).transpose(1, 2)


AutoConfig.register(SigLEPConfig.model_type, SigLEPConfig)
AutoModel.register(SigLEPConfig, SigLEP)
SigLEPConfig.register_for_auto_class()
SigLEP.register_for_auto_class("AutoModel")