import torch
from tqdm import tqdm
import wandb

from utils.gpu_manager import is_main, train_dev_break, batch_to_device, all_reduce_sum


def run_train(nn, dataloader, epoch, args, optimizer=None, checkpoint_manager=None):
    train = optimizer is not None
    if getattr(args, "distributed", False) and hasattr(getattr(dataloader, "sampler", None), "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    nn.train() if train else nn.eval()
    device = next(nn.parameters()).device
    accum_steps = getattr(args, "grad_accum_steps", 1)
    total_steps_per_epoch = len(dataloader)
    total_loss = 0.0
    total_steps = 0
    accum_loss_for_log = 0.0
    if train:
        optimizer.zero_grad()

    verb = "Training" if train else "Validating"
    progress = tqdm(
        dataloader,
        desc=f"{verb} LLM: {args.llm} ENCODER: {args.encoder};Epoch: {epoch}",
        disable=not is_main(),
        leave=False,
    )

    for step, batch in enumerate(progress):
        batch = {k: batch_to_device(v, device) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            raw_loss = nn(**batch).loss
        total_loss += raw_loss.item()
        total_steps += 1

        if train:
            (raw_loss / accum_steps).backward()
            accum_loss_for_log += raw_loss.item()

            if (step + 1) % accum_steps == 0 or (step + 1) == total_steps_per_epoch:
                grad_clip = getattr(args, "grad_clip", 0.0)
                if grad_clip > 0:
                    params = (p for p in nn.parameters() if p.grad is not None)
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)

                optimizer.step_and_update_lr()
                optimizer.zero_grad()

                if getattr(args, "wandb", False) and is_main():
                    wandb.log(
                        {
                            "train/step_loss": accum_loss_for_log,
                            "train/lr": optimizer.learning_rate,
                            "epoch": epoch,
                        }
                    )

                accum_loss_for_log = 0.0

            if args.save_step and checkpoint_manager and is_main():
                if checkpoint_manager.save_step(step, total_steps_per_epoch):
                    checkpoint_manager.save_checkpoint(nn, optimizer, epoch, step, prefix="step_")

        if train_dev_break(getattr(args, "dev", False), batch, raw_loss.item()):
            break

    if not train:
        total_loss = all_reduce_sum(total_loss)
        total_steps = all_reduce_sum(total_steps)

    average_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    if not train and getattr(args, "wandb", False) and is_main():
        wandb.log({"val/loss": average_loss, "epoch": epoch})
    return {"average_loss": average_loss, "total_steps": total_steps}
