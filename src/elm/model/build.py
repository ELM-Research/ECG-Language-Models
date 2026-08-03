import torch
from elm.utils.parallelism import is_main

class BuildELM:
    def __init__(self, elm_name: str,
                 llm_name: str,
                 vocab_size: str,
                 encoder_name: str,
                 elm_checkpoint: str = None,
                 encoder_checkpoint: str = None,
                 attn_implementation: str = "sdpa",
                 llm_pretrained: bool = True,
                 peft: str = None,
                 gradient_checkpointing: bool = True,
                 lora_rank: int =  16,
                 lora_alpha: int = 32,
                 ):
        self.elm_checkpoint = elm_checkpoint
        self.llm_builder = BuildLLM(llm_name, vocab_size,
                                    llm_pretrained, peft,
                                    attn_implementation,
                                    gradient_checkpointing,
                                    lora_rank, lora_alpha)
        self.encoder_builder = BuildEncoder(encoder_name,
                                            encoder_checkpoint)

    def build_elm(self, ):
        llm_components = self.llm_builder.build_llm()
        encoder_components = self.encoder_builder.build_encoder()
        elm_components = ConnectNN(llm_components, encoder_components, self.args).connect_nn()
        assert elm_components is not None, print("ELM Components is None")
        if self.elm_checkpoint: self.load_elm_checkpoint(elm_components)
        return elm_components

    def load_elm_checkpoint(self, elm_components):
        elm_checkpoint = torch.load(self.elm_checkpoint, map_location="cpu", weights_only=False)
        model = elm_components["elm"]
        model.load_state_dict(elm_checkpoint, strict=False)
        if is_main(): print(f"Loaded ELM checkpoint from {self.elm_checkpoint}")

from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from elm.model.llm.hf_llm import HFLLM
class BuildLLM:
    def __init__(self, llm_name: str,
                 vocab_size: str,
                 llm_pretrained: bool = True,
                 peft: str = None,
                 attn_implementation: str = "sdpa",
                 gradient_checkpointing: bool = True,
                 lora_rank: int =  16,
                 lora_alpha: int = 32,):
        self.llm_name = llm_name
        self.vocab_size = vocab_size
        self.llm_pretrained = llm_pretrained
        self.peft = peft
        self.attn_implementation = attn_implementation
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

    def build_llm(
        self,
    ):
        llm = self.build_hf_llm()
        model = HFLLM(llm)
        if self.args.dev and is_main():
            self.print_llm_dtype(model)
        return {"llm": model,}

    ### HF FUNCTIONS ###
    def build_hf_llm(
        self,
    ):
        if self.llm_pretrained:
            hf_llm = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                attn_implementation=self.attn_implementation,
                )
        else:
            config = AutoConfig.from_pretrained(self.llm_name)
            hf_llm = AutoModelForCausalLM.from_config(config)
        hf_llm = self.resize_and_report_embeddings(hf_llm)
        if self.gradient_checkpointing:
            hf_llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            hf_llm.config.use_cache = False
        if self.peft: hf_llm = self.build_peft(hf_llm,)
        return hf_llm

    ### HF LLM FUNCTIONS ###
    def print_llm_dtype(self, llm): print(f"\n{self.llm_name} actual dtype:", next(llm.parameters()).dtype)

    def resize_and_report_embeddings(self, hf_llm):
        old_size = hf_llm.get_input_embeddings().weight.shape[0]
        if is_main(): print(f"Embedding size before: {old_size}")
        hf_llm.resize_token_embeddings(self.vocab_size)
        new_size = hf_llm.get_input_embeddings().weight.shape[0]
        if is_main(): print(f"Embedding size after: {new_size}")
        assert new_size == self.vocab_size, f"Embedding size {new_size} does not match tokenizer vocab size {self.vocab_size}"
        return hf_llm

    ### PEFT FUNCTIONS ###
    def build_peft(
        self,
        llm,
    ):
        lora_config = self.get_lora_configs()
        lora_config = LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    task_type=TaskType.CAUSAL_LM,
                )
        llm = get_peft_model(llm, lora_config)
        if is_main(): llm.print_trainable_parameters()
        return llm

class BuildEncoder:
    def __init__(self, encoder_name: str,
                 encoder_checkpoint : str = None,
                 ):
        self.encoder_name = encoder_name
        self.encoder_checkpoint = encoder_checkpoint

    def build_encoder(self):
        if self.encoder_name == "siglep":
            encoder_components = self.prepare_siglep()

        assert encoder_components is not None, print("NN Components is None")

        if self.encoder_checkpoint:
            self.load_nn_checkpoint(encoder_components)

        return encoder_components

    def prepare_siglep(self,):
        from transformers import AutoModel, AutoConfig
        from elm.model.encoder.siglep.siglep import SigLEP
        config = AutoConfig.from_pretrained(self.encoder_checkpoint)
        hf_encoder = AutoModel.from_pretrained(self.encoder_checkpoint, config = config)
        hidden = hf_encoder.config.vision_config.hidden_size
        model = SigLEP(hf_encoder, segment_len=self.args.segment_len,
                          patch_size=25,
                          num_encoder_tokens=self.args.num_encoder_tokens,
                          num_leads=len(self.args.leads))
        return {"encoder": model,}

    def load_nn_checkpoint(self, encoder_components):
        ckpt = torch.load(self.args.encoder_ckpt, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]

        model_keys = set(encoder_components["encoder"].state_dict().keys())
        ckpt_keys = set(state.keys())

        loaded_keys = model_keys & ckpt_keys
        missing_from_ckpt = model_keys - ckpt_keys
        unused_in_ckpt = ckpt_keys - model_keys

        encoder_components["encoder"].load_state_dict(state, strict=False)

        if is_main():
            print(f"\nLoaded {self.args.encoder} checkpoint from {self.args.encoder_ckpt}")
            if not missing_from_ckpt and not unused_in_ckpt:
                print("  All layers loaded from checkpoint.")
            else:
                if missing_from_ckpt:
                    print(f"  Layers NOT loaded (not in checkpoint) [{len(missing_from_ckpt)}]:")
                    for k in sorted(missing_from_ckpt):
                        print(f"    - {k}")
                if unused_in_ckpt:
                    print(f"  Checkpoint layers ignored (not in model) [{len(unused_in_ckpt)}]:")
                    for k in sorted(unused_in_ckpt):
                        print(f"    - {k}")
                print(f"  Loaded: {len(loaded_keys)}/{len(model_keys)} model layers\n")


def merge_dicts(*parts: Mapping[str, Any], allow_override: Iterable[str] = ()) -> Dict[str, Any]:
    """Merge dict-like parts with duplicate-key protection.
    Keys in `allow_override` are allowed to be overwritten by later parts.
    Later parts win for allowed keys; duplicates for other keys raise."""
    out: Dict[str, Any] = {}
    allowed = set(allow_override)
    for p in parts:
        for k, v in p.items():
            if k in out and k not in allowed:
                raise ValueError(f"Duplicate component keys when merging: {k}")
            out[k] = v
    return out


class ConnectNN:
    def __init__(self, llm_components: dict, encoder_components: dict, args: argparse.Namespace):
        self.args = args
        self.llm_components = llm_components
        self.encoder_components = encoder_components

    def connect_nn(
        self,
    ):
        if self.args.elm  == "mlp_llava":
            encoder_llm_components = self.build_mlp_llava()
        elif self.args.elm == "linear_llava":
            encoder_llm_components = self.build_linear_llava()
        elif self.args.elm == "base_elf":
            encoder_llm_components = self.build_base_elf()
        elif self.args.elm == "patch_elf":
            encoder_llm_components = self.build_patch_elf()
        elif self.args.elm == "conv_elf":
            encoder_llm_components = self.build_conv_elf()
        elif self.args.elm == "ecg_byte":
            encoder_llm_components = {"elm": self.llm_components["llm"]}
        return merge_dicts(
            self.encoder_components,
            self.llm_components,
            encoder_llm_components,
            allow_override=("find_unused_parameters",),
        )

    def build_mlp_llava(
        self,
    ):
        from elms.llm_encoders.llava import LLaVA
        if self.args.encoder in VISION_ENCODERS:
            projection_dim = VISION_ENCODERS[self.args.encoder]["projection_dim"]
        else:
            projection_dim = ECG_ENCODERS[self.args.encoder]["projection_dim"]
        projection_layer = MLPProjection(projection_dim, self.args.llm)
        encoder_llm = LLaVA(
            self.llm_components["llm"], self.encoder_components["encoder"],
            projection_layer, set(self.args.update),
            True if self.args.perturb == "only_text" else False)
        return {"elm": encoder_llm}

    def build_linear_llava(
        self,
    ):
        from elms.llm_encoders.llava import LLaVA
        if self.args.encoder in VISION_ENCODERS:
            projection_dim = VISION_ENCODERS[self.args.encoder]["projection_dim"]
        else:
            projection_dim = ECG_ENCODERS[self.args.encoder]["projection_dim"]
        projection_layer = LinearProjection(projection_dim, self.args.llm)
        encoder_llm = LLaVA(
            self.llm_components["llm"], self.encoder_components["encoder"],
            projection_layer, set(self.args.update),
            True if self.args.perturb == "only_text" else False)
        return {"elm": encoder_llm}

    def build_base_elf(
        self,
    ):
        from elms.llm_encoders.base_elf import BaseElf
        projection_dim = len(self.args.leads) * self.args.segment_len
        projection_layer = LinearProjection(projection_dim, self.args.llm)
        encoder_llm = BaseElf(self.llm_components["llm"], projection_layer,
                           set(self.args.update),
                           True if self.args.perturb == "only_text" else False)
        return {"elm": encoder_llm}

    def build_patch_elf(self):
        from elms.llm_encoders.base_elf import BaseElf
        num_leads = len(self.args.leads)
        num_patches = self.args.num_encoder_tokens
        assert self.args.segment_len % num_patches == 0, \
            f"segment_len ({self.args.segment_len}) must be divisible by num_encoder_tokens ({num_patches})"
        patch_dim = num_leads * (self.args.segment_len // num_patches)
        projection_layer = PatchProjection(num_patches, patch_dim, self.args.llm)
        encoder_llm = BaseElf(self.llm_components["llm"], projection_layer,
                           set(self.args.update),
                           True if self.args.perturb == "only_text" else False)
        return {"elm": encoder_llm}

    def build_conv_elf(self):
        from elms.llm_encoders.base_elf import BaseElf
        num_leads = len(self.args.leads)
        num_patches = self.args.num_encoder_tokens
        assert self.args.segment_len % num_patches == 0, \
            f"segment_len ({self.args.segment_len}) must be divisible by num_encoder_tokens ({num_patches})"
        projection_layer = CNNPatchProjection(num_patches, num_leads, self.args.llm)
        encoder_llm = BaseElf(self.llm_components["llm"], projection_layer,
                           set(self.args.update),
                           True if self.args.perturb == "only_text" else False)
        return {"elm": encoder_llm}