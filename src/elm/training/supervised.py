from torch.nn.functional import pad
from tqdm import tqdm

from elm.training.common import begin_epoch, move_to_device, optimizer_step
from elm.utils.logging import log_wandb
from elm.utils.parallelism import distributed_mean, is_main, set_gradient_sync


def train_epoch(model, optimizer, scheduler, checkpointer, dataloader, config: dict,
                epoch: int, start_batch: int = 0) -> dict:
    training = config["training"]
    accumulation_steps = training["gradient_accumulation_steps"]
    if accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")

    device = begin_epoch(model, dataloader, epoch)
    num_batches = len(dataloader)
    progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}", disable=not is_main(), leave=False)
    optimizer.zero_grad(set_to_none=True)
    total_loss = window_loss = 0.0
    all_masked_batches = 0

    for step, batch in enumerate(progress):
        if step < start_batch:
            continue
        window_start = step - step % accumulation_steps
        window_size = min(accumulation_steps, num_batches - window_start)
        update = step + 1 == window_start + window_size
        set_gradient_sync(model, update)

        target_columns = batch["labels"].ne(-100).any(0).nonzero()
        all_masked_batches += not len(target_columns)
        label_start = (max(target_columns[0].item(), 1) if len(target_columns)
                       else batch["labels"].shape[1] - 1)
        logits_to_keep = batch["labels"].shape[1] - label_start + 1
        batch = move_to_device(batch, device)
        shift_labels = pad(batch["labels"][:, label_start:], (0, 1), value=-100)
        output = model(**batch, logits_to_keep=logits_to_keep,
                       shift_labels=shift_labels, use_cache=False)
        loss = output.loss if len(target_columns) else output.logits.sum() * 0
        (loss / window_size).backward()
        loss_value = loss.detach().item()
        total_loss += loss_value
        window_loss += loss_value

        if update:
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer_step(model, optimizer, scheduler, training["max_grad_norm"])
            checkpointer.step(epoch + (step + 1 == num_batches), (step + 1) % num_batches)
            step_loss = distributed_mean(window_loss, window_size, device)
            progress.set_postfix(loss=f"{step_loss:.4f}")
            if config["wandb"]:
                metrics = {"loss": step_loss, "lr": learning_rate, "epoch": epoch}
                log_wandb(metrics, "train")
            window_loss = 0.0

    if is_main():
        print(f"All-masked batches: {all_masked_batches}")
    return {
        "average_loss": distributed_mean(total_loss, num_batches - start_batch, device),
        "optimizer_steps": (num_batches - start_batch + accumulation_steps - 1) // accumulation_steps,
    }
