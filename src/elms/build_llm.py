import argparse
from transformers import AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, TaskType, get_peft_model

from configs.constants import HF_LLMS
from utils.chat_template_manager import assistant_stop_ids
from utils.gpu_manager import is_main


class BuildLLM:
    def __init__(self, args: argparse.Namespace, llm_tokenizer):
        self.args = args
        self.llm_tokenizer = llm_tokenizer
        self.pad_token_id = llm_tokenizer.pad_token_id
        # Stop generation at any assistant turn terminator (e.g. Gemma's <end_of_turn>), derived from the chat template.
        self.eos_token_id = list(assistant_stop_ids(llm_tokenizer))

    def build_llm(
        self,
    ):
        if self.args.llm in HF_LLMS:
            llm = self.build_hf_llm()
            if "llama-3" in self.args.llm:
                from elms.llms.llama3.llama3 import Llama3
                model = Llama3(llm, self.pad_token_id, self.eos_token_id,
                               HF_LLMS[self.args.llm]["output_hidden_states"])
            elif "qwen2.5" in self.args.llm:
                from elms.llms.qwen25.qwen25 import Qwen25
                model = Qwen25(llm, self.pad_token_id, self.eos_token_id,
                               HF_LLMS[self.args.llm]["output_hidden_states"])
            elif "gemma-2" in self.args.llm:
                from elms.llms.gemma2.gemma2 import Gemma2
                model = Gemma2(llm, self.pad_token_id, self.eos_token_id,
                               HF_LLMS[self.args.llm]["output_hidden_states"])
        else:
            raise ValueError(f"{self.args.llm} not supported.")

        if self.args.dev and is_main():
            self.print_llm_dtype(model)
        components = {
            "llm": model,
            "find_unused_parameters": HF_LLMS[self.args.llm]["find_unused_parameters"],
        }
        return components

    ### HF FUNCTIONS ###
    def build_hf_llm(
        self,
    ):
        if self.args.scratch:
            config = AutoConfig.from_pretrained(HF_LLMS[self.args.llm]["model"])
            config._attn_implementation = self.args.attention_type
            hf_llm = AutoModelForCausalLM.from_config(config).to(HF_LLMS[self.args.llm]["native_dtype"])
        else:
            hf_llm = AutoModelForCausalLM.from_pretrained(
                HF_LLMS[self.args.llm]["model"],
                dtype=HF_LLMS[self.args.llm]["native_dtype"],
                attn_implementation=self.args.attention_type,
            )
        HF_LLMS[self.args.llm]["model_hidden_size"] = hf_llm.config.hidden_size
        assert HF_LLMS[self.args.llm]["model_hidden_size"] is not None, print("model_hidden_size")
        hf_llm = self.resize_and_report_embeddings(hf_llm)
        if self.args.gradient_checkpointing:
            hf_llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            hf_llm.config.use_cache = False
        if self.args.peft:
            hf_llm = self.build_peft(
                hf_llm,
            )
        return hf_llm

    ### HF LLM FUNCTIONS ###
    def print_llm_dtype(self, llm):
        print(
            f"{self.args.llm} native dtype:", HF_LLMS[self.args.llm]["native_dtype"], f"\n{self.args.llm} actual dtype:", next(llm.parameters()).dtype
        )
        assert HF_LLMS[self.args.llm]["native_dtype"] == next(llm.parameters()).dtype, print(f"{self.args.llm} native and actual dtype do not match")

    def resize_and_report_embeddings(self, hf_llm):
        old_size = hf_llm.get_input_embeddings().weight.shape[0]
        if is_main():
            print(f"[{self.args.llm}] embedding size before: {old_size}")
        hf_llm.resize_token_embeddings(len(self.llm_tokenizer))
        new_size = hf_llm.get_input_embeddings().weight.shape[0]
        if is_main():
            print(f"[{self.args.llm}] embedding size after: {new_size}")
        assert new_size == len(self.llm_tokenizer), f"Embedding size {new_size} does not match tokenizer vocab size {len(self.llm_tokenizer)}"
        return hf_llm

    ### PEFT FUNCTIONS ###
    def build_peft(
        self,
        llm,
    ):
        lora_config = self.get_lora_configs()
        llm = get_peft_model(llm, lora_config)
        if is_main():
            llm.print_trainable_parameters()
        return llm

    def get_lora_configs(
        self,
    ):
        lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_alpha,
            task_type=TaskType.CAUSAL_LM,
        )
        return lora_config
