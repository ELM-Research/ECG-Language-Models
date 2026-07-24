import gc
import torch
from elm.config.load import get_config
from elm.utils.parallelism import init_dist, cleanup, is_main
from elm.utils.seed import set_seed
from elm.utils.logging import setup_experiment_folders

RUNS_DIR = "./src/runs"

if __name__ == "__main__":
    config = get_config()
    if config["gpu"]["distributed"]: init_dist()
    gc.collect()
    torch.cuda.empty_cache()

    try:
        if not config["dev"] and is_main():
            run_folder = setup_experiment_folders(
                f'{RUNS_DIR}/{config["model"]["llm"]}_{config["model"]["encoder"]}'
            )
        print("hi")
        # if not args.dev:
        #     data_name = "_".join(args.data)
        #     run_folder = setup_experiment_folders(
        #         f"{RUNS_DIR}/{args.elm}_{args.llm}_{args.encoder}/{data_name}", # add args.elm as a name
        #         args,
        #     )
        # if is_main() and not args.dev:
        #     print(f"Run folder: {run_folder}")
        #     if args.wandb:
        #         setup_wandb(args)
        set_seed(config["seed"])


    finally:
        if config["gpu"]["distributed"]:
            cleanup()