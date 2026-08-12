import gc
import torch
from elm.data.build import BuildDataloader
from elm.config.load import get_config
from elm.model import build_model
from elm.optimizer import build_optimizer
from elm.utils.parallelism import cleanup, init_dist, is_main, setup_model
from elm.utils.seed import set_seed
from elm.utils.logging import setup_experiment_folder, setup_wandb, cleanup_wandb

RUNS_DIR = "./src/runs"

if __name__ == "__main__":
    config, exp_name = get_config()
    strategy = config["gpu"]["strategy"]

    if strategy: init_dist()

    gc.collect()
    torch.cuda.empty_cache()

    try:
        if not config["development"] and is_main():
            run_folder = setup_experiment_folder(
                f'{RUNS_DIR}/{exp_name}',
                config,)

            if config["wandb"]: setup_wandb(config)

        set_seed(config["seed"])
        dataloader_builder = BuildDataloader(config["data"]["data_names"],
                                             config["data"]["split_names"],
                                             config["model"]["language_model"],
                                             config["model"]["truncation_length"],
                                             config["enable_thinking"],
                                             config["system_prompt_path"],
                                             config["model"]["num_ecg_tokens"],
                                             config["modality"], config["training"]["batch_size"],
                                             config["training"]["num_workers"], config["seed"],
                                             training_stage=config["training"]["training_stage"],
                                             augmentation=config["augment_ecg"],
                                             perturbation=config["perturbation"],
                                             development=config["development"],)
        tokenizer = dataloader_builder.build_llm_tokenizer()
        model = setup_model(build_model(config, tokenizer), strategy)
        optimizer = build_optimizer(config, model)
        dataloader = dataloader_builder.build_dataloader(tokenizer)

    finally:
        if config["wandb"] and is_main(): cleanup_wandb()
        if strategy: cleanup()
