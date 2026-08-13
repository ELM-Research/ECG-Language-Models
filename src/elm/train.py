import gc
import torch
from elm.data.build import BuildDataloader
from elm.config.load import get_config
from elm.model import build_model
from elm.optimizer import build_optimizer
from elm.utils.parallelism import cleanup, init_dist, is_main, setup_model
from elm.utils.seed import set_seed
from elm.utils.logging import setup_experiment_folder, setup_wandb, cleanup_wandb
from elm.training.trainers import run_pretrain_sft_test, run_rl_train_test

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
            checkpoint_manager = None # Implement checkpoint manager here
        else:
            checkpoint_manager = None

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
        runner = run_rl_train_test if config["training"]["training_stage"] == "rl" else run_pretrain_sft_test
        for epoch in range(0, config["training"]["epochs"]): # for resuming checkpoint 0 shouldb e the last epoch
            epoch_result = runner(model, optimizer, dataloader, epoch,
                                  checkpoint_manager = checkpoint_manager)

    finally:
        if config["wandb"] and is_main(): cleanup_wandb()
        if strategy: cleanup()
