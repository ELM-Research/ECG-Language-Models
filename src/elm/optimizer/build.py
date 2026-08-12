import torch
from torch import distributed, nn
from torch.distributed.optim import ZeroRedundancyOptimizer


class OrahOptimizer:
    def __init__(self, muon, adamw):
        self.muon = muon
        self.adamw = adamw

    @property
    def optimizers(self):
        return self.muon, self.adamw

    @property
    def param_groups(self):
        return self.muon.param_groups + self.adamw.param_groups

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def zero_grad(self, set_to_none=True):
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def consolidate_state_dict(self):
        for optimizer in self.optimizers:
            if isinstance(optimizer, ZeroRedundancyOptimizer):
                optimizer.consolidate_state_dict()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adamw.load_state_dict(state_dict["adamw"])


def build(config, model):
    if config.get("gpu", {}).get("strategy") == "fsdp2":
        raise ValueError("ZeRO requires DDP; FSDP2 already shards optimizer state")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Orah has no trainable parameters")
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise ValueError("Orah trainable parameters must be float32; use bfloat16 autocast for compute")

    base_model = model.module if hasattr(model, "module") else model
    adamw_only = {
        parameter
        for module in base_model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    output = base_model.get_output_embeddings()
    if output is not None:
        adamw_only.update(output.parameters())

    muon_parameters = [
        parameter for parameter in parameters
        if parameter.ndim == 2 and parameter not in adamw_only
    ]
    muon_set = set(muon_parameters)
    adamw_parameters = [parameter for parameter in parameters if parameter not in muon_set]
    if not muon_parameters or not adamw_parameters:
        raise ValueError("Orah requires parameters for both Muon and AdamW")

    training = config["training"]
    learning_rate = training["learning_rate"]
    weight_decay = training["weight_decay"]
    adamw_groups = [
        {"params": [parameter for parameter in adamw_parameters if parameter.ndim > 1],
         "weight_decay": weight_decay},
        {"params": [parameter for parameter in adamw_parameters if parameter.ndim <= 1],
         "weight_decay": 0.0},
    ]

    if distributed.is_initialized():
        muon = ZeroRedundancyOptimizer(
            muon_parameters, optimizer_class=torch.optim.Muon, lr=learning_rate,
            weight_decay=weight_decay, adjust_lr_fn="match_rms_adamw")
        adamw = ZeroRedundancyOptimizer(
            adamw_groups, optimizer_class=torch.optim.AdamW, lr=learning_rate)
    else:
        muon = torch.optim.Muon(
            muon_parameters, lr=learning_rate, weight_decay=weight_decay,
            adjust_lr_fn="match_rms_adamw")
        adamw = torch.optim.AdamW(adamw_groups, lr=learning_rate)
    return OrahOptimizer(muon, adamw)
