from tqdm import tqdm

from elm.training.common import begin_epoch, move_to_device, optimizer_step
from elm.training.rl.rollout import current_log_prob, rollout_group
from elm.training.rl.sapo.sapo_loss import compute_policy_loss_sapo
from elm.utils.logging import log_wandb
from elm.utils.parallelism import (
    any_process,
    distributed_mean,
    get_world_size,
    is_main,
    set_gradient_sync,
)


def train_epoch(model, optimizer, dataloader, tokenizer, config: dict, epoch: int) -> dict:
    training, rl = config["training"], config["rl"]
    accumulation_steps = training["gradient_accumulation_steps"]
    if (accumulation_steps < 1 or rl["group_size"] < 2 or
            rl["updates_per_rollout"] < 1 or rl["temperature"] <= 0):
        raise ValueError("Invalid gradient accumulation, group size, rollout update count, or temperature")

    device = begin_epoch(model, dataloader, epoch)
    progress = tqdm(dataloader, desc=f"RL epoch {epoch + 1}", disable=not is_main(), leave=False)
    rollouts = []
    reward_sum = total_loss = 0.0
    loss_windows = optimizer_steps = 0

    for step, batch in enumerate(progress):
        batch = move_to_device(batch, device)
        for item in range(batch["input_ids"].shape[0]):
            rollout = rollout_group(model, batch, item, tokenizer, rl, config["enable_thinking"])
            rollouts.append(rollout)
            reward_sum += rollout["mean_reward"]

        update = (step + 1) % accumulation_steps == 0 or step + 1 == len(dataloader)
        if not update:
            continue

        has_signal = any_process(any(not rollout["degenerate"] for rollout in rollouts), device)
        loss_sum = 0.0
        if has_signal:
            world_size = get_world_size()
            global_batch_size = len(rollouts) * rl["group_size"] * world_size
            for _ in range(rl["updates_per_rollout"]):
                optimizer.zero_grad(set_to_none=True)
                for index, rollout in enumerate(rollouts):
                    set_gradient_sync(model, index == len(rollouts) - 1)
                    loss = compute_policy_loss_sapo(
                        old_log_prob=rollout["old_log_prob"],
                        log_prob=current_log_prob(model, rollout),
                        advantages=rollout["advantages"],
                        response_mask=rollout["response_mask"],
                        global_batch_size=global_batch_size,
                        dp_size=world_size,
                        tau_pos=rl["tau_pos"],
                        tau_neg=rl["tau_neg"],
                    )
                    valid = not rollout["degenerate"]
                    (loss * valid).backward()
                    loss_sum += loss.detach().item() * valid
                optimizer_step(model, optimizer, training["max_grad_norm"])

        loss = distributed_mean(loss_sum, rl["updates_per_rollout"], device) if has_signal else 0.0
        reward = distributed_mean(reward_sum, len(rollouts), device)
        total_loss += loss
        loss_windows += has_signal
        optimizer_steps += has_signal * rl["updates_per_rollout"]
        progress.set_postfix(loss=f"{loss:.4f}", reward=f"{reward:.4f}")
        if config["wandb"]:
            metrics = {"loss": loss, "lr": optimizer.param_groups[0]["lr"],
                       "mean_reward": reward, "epoch": epoch}
            log_wandb(metrics, "train")
        rollouts.clear()
        reward_sum = 0.0

    return {
        "average_loss": total_loss / max(loss_windows, 1),
        "optimizer_steps": optimizer_steps,
    }
