import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from elm.model.elm.orah import Orah


class BuildELM:
    def __init__(self, elm_name, llm_name, vocab_size, encoder_name,
                 elm_checkpoint=None, encoder_checkpoint=None,
                 attn_implementation="sdpa", llm_pretrained=True, peft=None,
                 gradient_checkpointing=True, lora_rank=16, lora_alpha=32,
                 ecg_token_id=None, num_ecg_tokens=100, segment_length=2500,
                 patch_size=25, num_leads=12, update=("connector",)):
        if elm_name != "orah":
            raise ValueError(f"Unknown ELM: {elm_name}")
        self.llm_name = llm_name
        self.vocab_size = vocab_size
        self.encoder_name = encoder_checkpoint or encoder_name
        self.elm_checkpoint = elm_checkpoint
        self.attn_implementation = attn_implementation
        self.llm_pretrained = llm_pretrained
        self.peft = peft
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.model_kwargs = {
            "ecg_token_id": ecg_token_id,
            "num_ecg_tokens": num_ecg_tokens,
            "segment_length": segment_length,
            "patch_size": patch_size,
            "num_leads": num_leads,
        }
        self.update = update

    def build_elm(self):
        llm = self._build_llm()
        encoder = AutoModel.from_pretrained(self.encoder_name)
        model = Orah.from_components(llm, encoder, **self.model_kwargs)
        if self.elm_checkpoint:
            checkpoint = torch.load(self.elm_checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        model.set_trainable(self.update)
        return {"elm": model, "llm": model.llm, "encoder": model.encoder,
                "projection": model.connector}

    def _build_llm(self):
        if self.llm_pretrained:
            llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name, attn_implementation=self.attn_implementation)
        else:
            config = AutoConfig.from_pretrained(self.llm_name)
            config._attn_implementation = self.attn_implementation
            llm = AutoModelForCausalLM.from_config(config)
        llm.resize_token_embeddings(self.vocab_size)
        if self.gradient_checkpointing:
            llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            llm.config.use_cache = False
        if self.peft:
            llm = get_peft_model(llm, LoraConfig(
                r=self.lora_rank, lora_alpha=self.lora_alpha,
                task_type=TaskType.CAUSAL_LM))
        return llm
