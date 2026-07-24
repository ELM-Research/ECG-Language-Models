import gc
import torch
import bpe
from elm.config.load import get_config
from elm.utils.parallelism import init_dist, cleanup

if __name__ == "__main__":
    config = get_config()
    print(config)
    if config["gpu"]["distributed"]: init_dist()
    gc.collect()
    torch.cuda.empty_cache()

    try:
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
        # set_seed(args.seed)


    finally:
        if config["gpu"]["distributed"]:
            cleanup()