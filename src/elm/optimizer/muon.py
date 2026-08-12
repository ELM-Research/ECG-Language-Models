# Original implementatino https://github.com/samsja/muon_fsdp_2
# Modified by willxxy

import math

import torch
from torch import nn
from torch import distributed
from torch.distributed.tensor import DTensor


def orthogonalize(gradient, steps):
    update = gradient.bfloat16()
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.T
    update /= update.norm() + 1e-7
    for _ in range(steps):
        gram = update @ update.T
        update = 3.4445 * update + (-4.775 * gram + 2.0315 * gram @ gram) @ update
    return update.T if transposed else update


def momentum_update(gradient, momentum_buffer, momentum, nesterov):
    momentum_buffer.lerp_(gradient, 1 - momentum)
    return gradient.lerp_(momentum_buffer, momentum) if nesterov else momentum_buffer


def scale_update(update):
    return update.mul_(0.2 * math.sqrt(max(update.shape)))


def muon_update(parameter, state, group, index):
    update = momentum_update(parameter.grad, state["momentum_buffer"], group["momentum"], group["nesterov"])
    if not isinstance(update, DTensor):
        return scale_update(orthogonalize(update, group["ns_steps"]))
    if update.device_mesh.ndim != 1:
        raise ValueError("Muon supports only a one-dimensional FSDP2 mesh")

    mesh = update.device_mesh
    process_group = mesh.get_group()
    destination = distributed.get_global_rank(process_group, index % mesh.size())
    shards = [torch.zeros_like(update.to_local()) for _ in range(mesh.size())] if mesh.get_rank() == destination else None
    distributed.gather(update.to_local(), shards, dst=destination, group=process_group)
    if mesh.get_rank() == destination:
        full_update = torch.cat(shards)
        full_update.copy_(orthogonalize(full_update, group["ns_steps"]))
        distributed.scatter(update.to_local(), list(full_update.chunk(mesh.size())), src=destination, group=process_group)
    else:
        distributed.scatter(update.to_local(), None, src=destination, group=process_group)
    return scale_update(update)


def adamw_update(parameter, state, group):
    gradient = parameter.grad
    state["step"] += 1
    state["exp_avg"].lerp_(gradient, 1 - group["betas"][0])
    state["exp_avg_sq"].lerp_(gradient.square(), 1 - group["betas"][1])
    average = state["exp_avg"] / (1 - group["betas"][0] ** state["step"])
    variance = state["exp_avg_sq"] / (1 - group["betas"][1] ** state["step"])
    return average / (variance.sqrt() + group["eps"])


class Muon(torch.optim.Optimizer):
    def __init__(self, parameter_groups):
        for group in parameter_groups:
            if group["use_muon"]:
                group.setdefault("momentum", 0.95)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
            else:
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
        super().__init__(parameter_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for index, parameter in enumerate(group["params"]):
                if parameter.grad is None:
                    if not group["use_muon"] or not isinstance(parameter, DTensor):
                        continue
                    parameter.grad = torch.zeros_like(parameter)
                state = self.state[parameter]
                if not state and group["use_muon"]:
                    state["momentum_buffer"] = torch.zeros_like(parameter)
                elif not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                if group["use_muon"]:
                    update = muon_update(parameter, state, group, index)
                else:
                    update = adamw_update(parameter, state, group)
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update.reshape(parameter.shape), alpha=-group["lr"])
        return loss

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
    learning_rate = config["optimizer"]["learning_rate"]
    weight_decay = config["optimizer"]["weight_decay"]

    return Muon([
        {"params": muon_parameters, "lr": learning_rate, "weight_decay": weight_decay, "use_muon": True},
        {"params": [parameter for parameter in adamw_parameters if parameter.ndim > 1],
         "lr": learning_rate, "weight_decay": weight_decay, "use_muon": False},
        {"params": [parameter for parameter in adamw_parameters if parameter.ndim <= 1],
         "lr": learning_rate, "weight_decay": 0.0, "use_muon": False},
    ])