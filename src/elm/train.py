import gc
import torch
from elm.config.load import get_config
from elm.utils.parallelism import init_dist, cleanup, is_main
from elm.utils.seed import set_seed
from elm.utils.logging import setup_experiment_folder

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
            print(f"Run folder: {run_folder}")
            # if args.wandb:
            #     setup_wandb(args)
        set_seed(config["seed"])


    finally:
        if config["gpu"]["distributed"]:
            cleanup()