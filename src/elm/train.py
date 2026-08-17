from elm.config.load import get_config
from elm.data.build import build_data
from elm.model import build_model
from elm.optimizer import build_optimizer
from elm.training.rl.trainer import train_epoch as train_rl_epoch
from elm.training.supervised import train_epoch as train_supervised_epoch
from elm.utils.logging import cleanup_wandb, setup_experiment_folder, setup_wandb
from elm.utils.parallelism import cleanup, init_dist, is_main, setup_model
from elm.utils.seed import set_seed
import torch
torch.set_float32_matmul_precision("high")
RUNS_DIR = "./src/runs"


def main():
    config, exp_name = get_config()
    strategy = config["gpu"]["strategy"]
    init_dist(strategy)

    try:
        if is_main():
            if not config["development"]:
                setup_experiment_folder(f"{RUNS_DIR}/{exp_name}", config)
            if config["wandb"]:
                setup_wandb(config, name=exp_name)

        set_seed(config["seed"])
        tokenizer, dataloader = build_data(config)
        model = setup_model(build_model(config, tokenizer), strategy)
        optimizer = build_optimizer(config, model)
        for epoch in range(config["training"]["epochs"]):
            if config["training"]["training_stage"] == "rl":
                result = train_rl_epoch(model, optimizer, dataloader, tokenizer, config, epoch)
            else:
                result = train_supervised_epoch(model, optimizer, dataloader, config, epoch)
            if is_main():
                print(f"Epoch {epoch + 1}: loss={result['average_loss']:.4f}")

    finally:
        if config["wandb"] and is_main():
            cleanup_wandb()
        cleanup()


if __name__ == "__main__":
    main()
