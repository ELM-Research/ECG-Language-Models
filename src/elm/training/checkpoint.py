from pathlib import Path

from torch import distributed
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)

from elm.utils.parallelism import is_main


class Checkpointer:
    def __init__(self, model, tokenizer, run_dir: Path | None, save_steps: int, enabled: bool):
        if save_steps < 1:
            raise ValueError("save_steps must be positive")
        if enabled and is_main() and run_dir is None:
            raise ValueError("run_dir is required when checkpointing is enabled")
        self.model = model
        self.tokenizer = tokenizer
        self.run_dir = run_dir
        self.save_steps = save_steps
        self.enabled = enabled
        self.steps = 0
        self.best_loss = float("inf")

    def step(self) -> None:
        self.steps += 1
        if self.steps % self.save_steps == 0:
            self.save(f"step_{self.steps}")

    def save_best(self, loss: float) -> None:
        if loss < self.best_loss:
            self.best_loss = loss
            self.save("epoch_best")

    def save(self, name: str) -> None:
        if not self.enabled:
            return
        state_dict = (get_model_state_dict(
            self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            if distributed.is_initialized() else None)
        if is_main():
            path = self.run_dir / "checkpoints" / name
            self.model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)
            print(f"Saved checkpoint: {path}")
