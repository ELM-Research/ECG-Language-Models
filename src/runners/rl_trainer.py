"""Agnostic RL training loop (mirrors runners/trainer.py; policy loss is pluggable via args.rl_algo)."""
import torch
from tqdm import tqdm
import wandb

from utils.gpu_manager import is_main, get_world_size, train_dev_break, batch_to_device
from rl.rl_loss import get_rl_loss, get_loss_kwargs
from rl.rollout import rollout_group, current_log_prob


def _global_mean(total, count, device, dp_size):
    stats = torch.tensor([total, count], dtype=torch.float64, device=device)
    if dp_size > 1:
        torch.distributed.all_reduce(stats)
    return (stats[0] / stats[1].clamp_min(1)).item()


def run_rl_train(nn, optimizer, dataloader, epoch, args, checkpoint_manager=None):
    nn.train()
    if getattr(args, "distributed", False) and hasattr(getattr(dataloader, "sampler", None), "set_epoch"):
        dataloader.sampler.set_epoch(epoch)

    show_progress = is_main()
    total_loss, total_steps = 0.0, 0
    progress = tqdm(dataloader, desc=f"RL[{args.rl_algo}] LLM:{args.llm} Epoch:{epoch}",
                    disable=not show_progress, leave=False)

    device = next(nn.parameters()).device
    accum_steps = getattr(args, "grad_accum_steps", 1)
    updates = getattr(args, "rl_updates_per_rollout", 2)
    total_steps_per_epoch = len(dataloader)
    loss_fn = get_rl_loss(args.rl_algo)
    algo_kw = get_loss_kwargs(args.rl_algo, args)
    dp_size = get_world_size()
    tokenizer = dataloader.dataset.llm_tokenizer

    optimizer.zero_grad()
    rollouts, reward_sum, reward_component_sums = [], 0.0, {}
    for step, batch in enumerate(progress):
        batch = {k: batch_to_device(v, device) for k, v in batch.items()}
        B = batch["elm_input_ids"].shape[0]
        for i in range(B):
            ro = rollout_group(nn, batch, i, tokenizer, args)
            rollouts.append(ro)
            reward_sum += ro["mean_reward"]
            for name, value in ro["mean_reward_components"].items():
                reward_component_sums[name] = reward_component_sums.get(name, 0.0) + value

        avg_loss = 0.0

        if (step + 1) % accum_steps == 0 or (step + 1) == total_steps_per_epoch:
            gbs = len(rollouts) * args.rl_group_size * dp_size
            metric_sums, metric_weight = {}, 0.0
            has_signal = torch.tensor(any(not ro["degenerate"] for ro in rollouts), device=device, dtype=torch.int)
            if dp_size > 1:
                torch.distributed.all_reduce(has_signal, op = torch.distributed.ReduceOp.MAX)
            for _ in range(updates * has_signal.item()):
                update_loss = 0.0
                update_metric_sums, update_metric_weight = {}, 0.0
                for ro in rollouts:
                    log_prob = current_log_prob(nn, ro)
                    loss, metrics = loss_fn(old_log_prob=ro["old_log_prob"], log_prob=log_prob,
                                            advantages=ro["advantages"], response_mask = ro["resp_mask"],
                                            global_batch_size=gbs, dp_size=dp_size, **algo_kw)
                    (loss * (not ro["degenerate"])).backward()
                    weight = ro["resp_mask"].sum().item() * (not ro["degenerate"])
                    for name, value in metrics.items():
                        update_metric_sums[name] = update_metric_sums.get(name, 0.0) + value * weight
                    update_metric_weight += weight
                    if not ro["degenerate"]:
                        update_loss += loss.detach().item()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_((p for p in nn.parameters() if p.grad is not None), args.grad_clip)
                optimizer.step_and_update_lr()
                optimizer.zero_grad()
                avg_loss += update_loss / updates
                metric_sums, metric_weight = update_metric_sums, update_metric_weight
            avg_loss = _global_mean(avg_loss, 1, device, dp_size)
            total_loss += avg_loss
            total_steps += 1

            if getattr(args, "wandb", False):
                logged_metrics = {
                    f"train/{name}": _global_mean(value, metric_weight, device, dp_size)
                    for name, value in metric_sums.items()
                }
                logged_rewards = {
                    f"train/reward/{name}": _global_mean(value, len(rollouts), device, dp_size)
                    for name, value in reward_component_sums.items()
                }
                mean_reward = _global_mean(reward_sum, len(rollouts), device, dp_size)
                if is_main():
                    wandb.log({"train/step_loss": avg_loss, "train/lr": optimizer.learning_rate,
                               "train/mean_reward": mean_reward,
                               "epoch": epoch, **logged_rewards, **logged_metrics})
            rollouts, reward_sum, reward_component_sums = [], 0.0, {}

        if args.save_step and checkpoint_manager and is_main():
            if checkpoint_manager.save_step(step, total_steps_per_epoch):
                checkpoint_manager.save_checkpoint(nn, optimizer, epoch, step, prefix="step_")

        if train_dev_break(getattr(args, "dev", False), batch, avg_loss):
            break

    average_loss = total_loss / total_steps if total_steps > 0 else float("inf")
    return {"average_loss": average_loss, "total_steps": total_steps}
