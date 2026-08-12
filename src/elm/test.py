import torch
from torch import nn

from elm.optimizer import muon
from elm.optimizer.muon import build


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.hidden = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.output = nn.Linear(4, 8, bias=False)

    def get_output_embeddings(self):
        return self.output


CONFIG = {"optimizer": {"learning_rate": 1e-4, "weight_decay": 1e-2}}


def test_builds_muon_with_adamw_parameter_groups():
    model = Model()
    optimizer = build(CONFIG, model)
    muon, adamw_decay, adamw_no_decay = optimizer.param_groups

    assert muon["use_muon"] and muon["params"] == [model.hidden.weight]
    assert not adamw_decay["use_muon"]
    assert set(adamw_decay["params"]) == {model.embedding.weight, model.output.weight}
    assert not adamw_no_decay["use_muon"]
    assert set(adamw_no_decay["params"]) == {model.hidden.bias, model.norm.weight, model.norm.bias}
    assert [group["lr"] for group in optimizer.param_groups] == [1e-4] * 3
    assert [group["weight_decay"] for group in optimizer.param_groups] == [1e-2, 1e-2, 0.0]
    assert len({parameter for group in optimizer.param_groups for parameter in group["params"]}) == len(list(model.parameters()))


def test_excludes_frozen_parameters():
    model = Model()
    model.hidden.requires_grad_(False)
    optimizer = build(CONFIG, model)
    parameters = {parameter for group in optimizer.param_groups for parameter in group["params"]}

    assert model.hidden.weight not in parameters
    assert model.hidden.bias not in parameters


def test_single_device_step_keeps_state_in_parameter_dtype():
    model = Model()
    optimizer = build(CONFIG, model)
    output = model.output(model.norm(model.hidden(model.embedding(torch.tensor([1, 2])))))
    output.square().mean().backward()
    gradients = {parameter: parameter.grad.clone() for parameter in model.parameters()}
    optimizer.step()

    assert all(torch.equal(parameter.grad, gradient) for parameter, gradient in gradients.items())
    for parameter, state in optimizer.state.items():
        assert all(value.dtype == parameter.dtype for value in state.values() if torch.is_tensor(value))

if __name__ == "__main__":
    test_builds_muon_with_adamw_parameter_groups()
    test_excludes_frozen_parameters()
    test_single_device_step_keeps_state_in_parameter_dtype()