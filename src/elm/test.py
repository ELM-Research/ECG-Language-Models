import gc
import torch
from elm.data.build import BuildDataloader
from elm.config.load import get_config


if __name__ == "__main__":
    config, exp_name = get_config()

    gc.collect()
    torch.cuda.empty_cache()

    dataloader_builder = BuildDataloader(config["data"]["data_names"],
                                            config["data"]["split_names"],
                                            config["model"]["llm_tokenizer_name"],
                                            config["modality"], config["training"]["batch_size"],
                                            config["training"]["num_workers"], config["seed"],
                                            training_stage=config["training"]["training_stage"],
                                            development=config["development"])
    dataloader = dataloader_builder.build_dataloader()
    for batch in dataloader:
        print(batch)
        break