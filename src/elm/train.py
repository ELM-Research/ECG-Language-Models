from elm.config.load import get_config
from elm.data.build import build_data
from elm.model import build_model
from elm.optimizer import build_optimizer
from elm.training.checkpoint import Checkpointer
from elm.training.common import build_scheduler
from elm.training.rl.trainer import train_epoch as train_rl_epoch
from elm.training.supervised import train_epoch as train_supervised_epoch
from elm.utils.logging import cleanup_wandb, setup_experiment_folder, setup_wandb
from elm.utils.parallelism import (
    cleanup,
    configure_runtime,
    init_dist,
    is_main,
    print_training_setup,
    setup_model,
)
from elm.utils.seed import set_seed
RUNS_DIR = "./src/runs"

def main():
    config, exp_name = get_config()
    resume = config["training"].get("resume")
    if resume:
        config["model"]["checkpoint"] = resume
    strategy = config["gpu"]["strategy"]
    configure_runtime(config)
    init_dist(strategy)
    print_training_setup(config)

    try:
        run_dir = None
        if is_main():
            if not config["development"]:
                run_dir = setup_experiment_folder(f"{RUNS_DIR}/{exp_name}", config)
            if config["wandb"]:
                setup_wandb(config, name=exp_name)

        set_seed(config["seed"])
        tokenizer, dataloader = build_data(config)
        model = setup_model(build_model(config, tokenizer), strategy)
        optimizer = build_optimizer(config, model)
        scheduler = build_scheduler(config, optimizer, dataloader)
        checkpointer = Checkpointer(model, tokenizer, optimizer, scheduler, run_dir,
                                    config["training"]["save_steps"],
                                    enabled=not config["development"])
        start_epoch, start_batch = checkpointer.load(resume) if resume else (0, 0)
        for epoch in range(start_epoch, config["training"]["epochs"]):
            skip_batches = start_batch if epoch == start_epoch else 0
            if config["training"]["training_stage"] == "rl":
                result = train_rl_epoch(model, optimizer, scheduler, checkpointer,
                                        dataloader, tokenizer, config, epoch, skip_batches)
            else:
                result = train_supervised_epoch(model, optimizer, scheduler, checkpointer,
                                                dataloader, config, epoch, skip_batches)
            if not skip_batches:
                checkpointer.save_best(result["average_loss"])
            if is_main():
                print(f"Epoch {epoch + 1}: loss={result['average_loss']:.4f}")

    finally:
        if config["wandb"] and is_main():
            cleanup_wandb()
        cleanup()

if __name__ == "__main__":
    main()
