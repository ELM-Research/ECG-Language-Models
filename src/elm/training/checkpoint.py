from pathlib import Path
import torch
from torch import distributed
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from elm.utils.parallelism import is_main


class Checkpointer:
    def __init__(self, model, tokenizer, optimizer, scheduler, run_dir: Path | None,
                 save_steps: int | None, enabled: bool):
        if enabled and is_main() and run_dir is None:
            raise ValueError("run_dir is required when checkpointing is enabled")
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.run_dir = run_dir
        self.save_steps = save_steps
        self.enabled = enabled
        self.steps = 0
        self.epoch = self.batch = 0
        self.best_loss = float("inf")

    def step(self, epoch: int, batch: int, steps: int = 1) -> None:
        previous = self.steps
        self.steps += steps
        self.epoch, self.batch = epoch, batch
        if self.save_steps is None and batch == 0:
            self.save(f"epoch_{epoch}")
        elif self.save_steps and self.steps // self.save_steps > previous // self.save_steps:
            self.save(f"step_{self.steps}")

    def save_best(self, loss: float) -> None:
        if loss < self.best_loss:
            self.best_loss = loss
            self.save("epoch_best")

    def save_crash(self) -> None:
        self.optimizer.zero_grad(set_to_none = True)
        self.save("last_crashed")
        if self.enabled and distributed.is_initialized(): distributed.barrier()

    def save(self, name: str) -> None:
        if not self.enabled:
            return
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = (get_model_state_dict(self.model, options=options)
                      if distributed.is_initialized() else None)
        optimizer_state = (get_optimizer_state_dict(self.model, self.optimizer, options=options)
                           if self.optimizer.state else None)
        if is_main():
            path = self.run_dir / "checkpoints" / name
            (path / "trainer_state.pt").unlink(missing_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)
            state_path = path / "trainer_state.pt"
            torch.save({"optimizer": optimizer_state, "scheduler": self.scheduler.state_dict(),
                        "steps": self.steps, "epoch": self.epoch, "batch": self.batch,
                        "best_loss": self.best_loss}, state_path)
            print(f"Saved checkpoint: {path}")

    def load(self, path: str | Path) -> tuple[int, int]:
        path = Path(path)
        state_path = path / "trainer_state.pt"
        if not state_path.is_file():
            raise ValueError(f"Checkpoint has no training state: {path}")
        state = (torch.load(state_path, map_location="cpu", weights_only=False)
                 if is_main() else None)
        optimizer_state = state.pop("optimizer") if is_main() else {}
        values = [state, optimizer_state is not None if is_main() else None]
        if distributed.is_initialized():
            distributed.broadcast_object_list(values)
        state, optimizer_initialized = values
        if optimizer_initialized:
            set_optimizer_state_dict(
                self.model, self.optimizer, optimizer_state,
                options=StateDictOptions(full_state_dict=True,
                                         broadcast_from_rank0=distributed.is_initialized()))
        self.scheduler.load_state_dict(state["scheduler"])
        self.steps, self.epoch, self.batch = state["steps"], state["epoch"], state["batch"]
        self.best_loss = state["best_loss"]
        if is_main():
            print(f"Resumed checkpoint: {path}")
        return self.epoch, self.batch
