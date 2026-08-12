import torch
from torch import nn

from elm.optimizer.build import build


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.hidden = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.output = nn.Linear(4, 8, bias=False)

    def get_output_embeddings(self):
        return self.output


def test_builds_muon_with_adamw_parameter_groups():
    model = Model()
    optimizer = build({"training": {"learning_rate": 1e-4, "weight_decay": 1e-2}}, model)
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
    optimizer = build({"training": {"learning_rate": 1e-4, "weight_decay": 1e-2}}, model)
    parameters = {parameter for group in optimizer.param_groups for parameter in group["params"]}

    assert model.hidden.weight not in parameters
    assert model.hidden.bias not in parameters


def test_single_device_step_keeps_state_in_parameter_dtype():
    model = Model()
    optimizer = build({"training": {"learning_rate": 1e-4, "weight_decay": 1e-2}}, model)
    output = model.output(model.norm(model.hidden(model.embedding(torch.tensor([1, 2])))))
    output.square().mean().backward()
    optimizer.step()

    for parameter, state in optimizer.state.items():
        assert all(value.dtype == parameter.dtype for value in state.values() if torch.is_tensor(value))

if __name__ == "__main__":
    test_builds_muon_with_adamw_parameter_groups()
    test_excludes_frozen_parameters()
    test_single_device_step_keeps_state_in_parameter_dtype()