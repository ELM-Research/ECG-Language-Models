import gc
import torch
from elm.data.build import BuildDataloader
from elm.config.load import get_config
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model

if __name__ == "__main__":
    # config, _ = get_config()
    # print(config)

    # gc.collect()
    # torch.cuda.empty_cache()
    lora_config = LoraConfig(
                    r = 16,
                    lora_alpha = 32,
                    target_modules = ["q_proj", "k_proj", "v_proj",
                                      "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
    language_model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3.5-0.8B-Base",)
    language_model = get_peft_model(language_model, lora_config)
    language_model.print_trainable_parameters()

    # dataloader_builder = BuildDataloader(config["data"]["data_names"],
    #                                         config["data"]["split_names"],
    #                                         config["model"]["llm"]["llm_tokenizer_name"],
    #                                         config["model"]["llm"]["truncation_length"],
    #                                         config["enable_thinking"],
    #                                         config["system_prompt_path"],
    #                                         config["model"]["ecg_tokens"],
    #                                         config["modality"], config["training"]["batch_size"],
    #                                         config["training"]["num_workers"], config["seed"],
    #                                         training_stage=config["training"]["training_stage"],
    #                                         augmentation=config["augment_ecg"],
    #                                         perturbation=config["perturbation"],
    #                                         development=config["development"],)
    # dataloader = dataloader_builder.build_dataloader()
    # for batch in dataloader:
    #     print(batch["input_ids"].shape)
    #     print(batch["labels"].shape)
    #     print(batch["labels"])
    #     print(batch["ecg"].shape)
    #     print(dataloader.dataset.text_preparer.llm_tokenizer.decode(batch["input_ids"]))
    #     break