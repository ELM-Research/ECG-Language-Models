from torch import nn

from elm.optimizer.muon import Muon


def build(config, model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    adamw_only = {
        parameter
        for module in model.modules()
        if isinstance(module, nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    output = model.get_output_embeddings()
    if output is not None:
        adamw_only.update(output.parameters())

    muon_parameters = [parameter for parameter in parameters if parameter.ndim == 2 and parameter not in adamw_only]
    muon_set = set(muon_parameters)
    adamw_parameters = [parameter for parameter in parameters if parameter not in muon_set]
    learning_rate = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]

    return Muon([
        {"params": muon_parameters, "lr": learning_rate, "weight_decay": weight_decay, "use_muon": True},
        {"params": [parameter for parameter in adamw_parameters if parameter.ndim > 1],
         "lr": learning_rate, "weight_decay": weight_decay, "use_muon": False},
        {"params": [parameter for parameter in adamw_parameters if parameter.ndim <= 1],
         "lr": learning_rate, "weight_decay": 0.0, "use_muon": False},
    ])
