import gc
import torch
from elm.config.load import get_config
from elm.utils.parallelism import init_dist, cleanup, is_main
from elm.utils.seed import set_seed
from elm.utils.logging import setup_experiment_folder, setup_wandb, cleanup_wandb

RUNS_DIR = "./src/runs"

if __name__ == "__main__":
    config, exp_name = get_config()
    if config["gpu"]["distributed"]: init_dist()
    gc.collect()
    torch.cuda.empty_cache()

    try:
        if not config["dev"] and is_main():
            run_folder = setup_experiment_folder(
                f'{RUNS_DIR}/{exp_name}',
                config,)
            if config["wandb"]: setup_wandb(config)
        set_seed(config["seed"])


    finally:
        if config["wandb"] and is_main(): cleanup_wandb()
        if config["gpu"]["distributed"]: cleanup()