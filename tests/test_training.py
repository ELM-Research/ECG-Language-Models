import pytest
import torch

from elm.training.checkpoint import Checkpointer
from elm.training.common import build_scheduler, optimizer_step


class Loader:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


@pytest.mark.parametrize(("stage", "scheduler_name", "updates", "final_lr"), [
    ("sft", "cosine", 10, 0.0),
    ("rl", "constant_with_warmup", 30, 1.0),
])
def test_scheduler_counts_optimizer_updates(stage, scheduler_name, updates, final_lr):
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    config = {
        "training": {"training_stage": stage, "gradient_accumulation_steps": 2, "epochs": 2},
        "optimizer": {"scheduler": scheduler_name, "warmup_ratio": 0.03},
        "rl": {"updates_per_rollout": 3},
    }
    scheduler = build_scheduler(config, optimizer, Loader(10))
    assert optimizer.param_groups[0]["lr"] == 0.0
    for _ in range(updates):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(final_lr)


def test_optimizer_step_advances_scheduler_and_checkpoint():
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=1)
    checkpointer = Checkpointer(model, None, None, save_steps=2, enabled=False)
    model(torch.ones(1, 1)).sum().backward()
    optimizer_step(model, optimizer, scheduler, checkpointer, max_grad_norm=1.0)
    assert optimizer.param_groups[0]["lr"] == 1.0
    assert checkpointer.steps == 1
    assert all(parameter.grad is None for parameter in model.parameters())


def test_checkpoint_cadence_and_best_loss(tmp_path):
    checkpointer = Checkpointer(None, None, tmp_path, save_steps=2, enabled=True)
    saved = []
    checkpointer.save = saved.append
    checkpointer.step()
    checkpointer.step()
    checkpointer.save_best(2.0)
    checkpointer.save_best(3.0)
    checkpointer.save_best(1.0)
    assert saved == ["step_2", "epoch_best", "epoch_best"]
