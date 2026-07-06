import torch
from tqdm import tqdm
import wandb

from utils.gpu_manager import is_main, train_dev_break, batch_to_device, all_reduce_sum


@torch.no_grad()
def run_validation(
    nn,
    dataloader,
    epoch,
    args,
):
    if getattr(args, "distributed", False) and hasattr(getattr(dataloader, "sampler", None), "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    show_progress = is_main()
    total_loss = 0.0
    total_steps = 0
    progress = tqdm(
        dataloader,
        desc=f"Validating LLM: {args.llm} ENCODER: {args.encoder};Epoch: {epoch}",
        disable=not show_progress,
        leave=False,
    )

    device = next(nn.parameters()).device

    nn.eval()
    for step, batch in enumerate(progress):
        batch = {k: batch_to_device(v, device) for k, v in batch.items()}

        out = nn(**batch)
        raw_loss = out.loss

        total_loss += raw_loss.item()
        total_steps += 1

        if train_dev_break(getattr(args, "dev", False), batch, raw_loss.item()):
            break
    nn.train()

    total_loss = all_reduce_sum(total_loss)
    total_steps = all_reduce_sum(total_steps)
    average_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    if getattr(args, "wandb", False) and is_main():
        wandb.log({"val/loss": average_loss, "epoch": epoch})
    return {"average_loss": average_loss, "total_steps": total_steps}
