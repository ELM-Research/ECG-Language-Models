import os

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from utils.gpu_manager import is_main


_SAVE_OPTS = StateDictOptions(full_state_dict=True, cpu_offload=True)


def _load_opts() -> StateDictOptions:
    # broadcast_from_rank0/cpu_offload route through the default process group;
    # disable them when running single-process so DCP doesn't require dist init.
    is_dist = dist.is_available() and dist.is_initialized()
    return StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=is_dist,
        cpu_offload=is_dist,
    )


class CheckpointManager:
    def __init__(self, run_dir, args):
        self.run_dir = run_dir
        self.args = args
        self.checkpoint_dir = os.path.join(run_dir, "checkpoints")
        self.best_loss = float("inf")
        self.epoch_losses = []
        if is_main():
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, epoch, step, is_best=False, prefix=""):
        # All ranks participate in the gather; only rank 0 writes.
        model_sd = get_model_state_dict(model, options=_SAVE_OPTS)
        optimizer_sd = self._get_optimizer_state_dict(model, optimizer)

        if not is_main():
            return

        filename = f"{prefix}epoch_{epoch}_step_{step}.pt"
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model_sd,
            "optimizer_state_dict": optimizer_sd,
            "n_current_steps": optimizer.n_current_steps,
            "best_loss": self.best_loss,
        }
        torch.save(checkpoint, filepath)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{prefix}best.pt")
            torch.save(checkpoint, best_path)

    def save_epoch(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.epoch_losses.append(loss)
            return True
        self.epoch_losses.append(loss)
        return False

    def save_step(self, step, total_steps_per_epoch):
        if step == 0:
            return True
        save_interval = max(1, total_steps_per_epoch // 10)
        return step % save_interval == 0

    def stop_early(self):
        if len(self.epoch_losses) < self.args.patience + 1:
            return False
        best_loss = min(self.epoch_losses[: -self.args.patience])
        current_loss = min(self.epoch_losses[-self.args.patience :])
        return current_loss > best_loss - self.args.patience_delta

    def resume_checkpoint(self, path, model, optimizer):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        set_model_state_dict(model, model_state_dict=ckpt["model_state_dict"], options=_load_opts())
        self._load_optimizer_state_dict(model, optimizer, ckpt["optimizer_state_dict"])
        optimizer.n_current_steps = ckpt["n_current_steps"]
        self.best_loss = ckpt.get("best_loss", float("inf"))
        start_epoch = ckpt["epoch"] + 1
        if is_main():
            print(f"Resumed from {path} | epoch {start_epoch} | step {optimizer.n_current_steps}")
        del ckpt
        return start_epoch

    def _get_optimizer_state_dict(self, model, optimizer):
        from optimizers.optimizer_setup import MuonAdamW
        inner = optimizer.optimizer
        if isinstance(inner, MuonAdamW):
            return {
                "muon": get_optimizer_state_dict(model, inner.muon, options=_SAVE_OPTS),
                "adamw": get_optimizer_state_dict(model, inner.adamw, options=_SAVE_OPTS),
            }
        return get_optimizer_state_dict(model, inner, options=_SAVE_OPTS)

    def _load_optimizer_state_dict(self, model, optimizer, optimizer_sd):
        from optimizers.optimizer_setup import MuonAdamW
        inner = optimizer.optimizer
        if isinstance(inner, MuonAdamW):
            set_optimizer_state_dict(model, inner.muon, optim_state_dict=optimizer_sd["muon"], options=_load_opts())
            set_optimizer_state_dict(model, inner.adamw, optim_state_dict=optimizer_sd["adamw"], options=_load_opts())
        else:
            set_optimizer_state_dict(model, inner, optim_state_dict=optimizer_sd, options=_load_opts())
