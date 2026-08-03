import torch
from elm.utils.parallelism import is_main

class BuildELM:
    def __init__(self, elm_name: str,
                 llm_name: str,
                 llm_tokenizer,
                 elm_checkpoint: str = None,
                 encoder_checkpoint: str = None,
                 attn_implementation: str = "sdpa",
                 llm_pretrained: bool = True,
                 peft: str = None
                 ):
        self.elm_checkpoint = elm_checkpoint
        self.llm_builder = BuildLLM(llm_name, llm_tokenizer,
                                    attn_implementation)
        self.encoder_builder = BuildEncoder()

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
                 llm_tokenizer,
                 llm_pretrained: bool = True,
                 peft: str = None):
        self.llm_name = llm_name
        self.llm_tokenizer = llm_tokenizer
        self.llm_pretrained = llm_pretrained
        self.peft = peft

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
                attn_implementation=self.args.attention_type,
            )
        else:
            config = AutoConfig.from_pretrained(self.llm_name)
            hf_llm = AutoModelForCausalLM.from_config(config)
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


class BuildEncoder:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def build_encoder(self):
        if self.args.encoder == "merl":
            encoder_components = self.prepare_merl()
        elif self.args.encoder == "mlae":
            encoder_components = self.prepare_mlae()
        elif self.args.encoder == "mtae":
            encoder_components = self.prepare_mtae()
        elif self.args.encoder == "st_mem":
            encoder_components = self.prepare_st_mem()
        elif self.args.encoder == "clip-vit-base-patch32":
            encoder_components = self.prepare_hf_clip()
        elif self.args.encoder == "siglip2-so400m-patch16-naflex":
            encoder_components = self.prepare_hf_siglip()
        elif self.args.encoder == "siglip-ecg":
            encoder_components = self.prepare_siglip_ecg()
        elif self.args.encoder == "vit-base-patch16-224-in21k":
            encoder_components = self.prepare_hf_vit()
        else:
            encoder_components = {}
        assert encoder_components is not None, print("NN Components is None")

        if self.args.encoder_ckpt:
            self.load_nn_checkpoint(encoder_components)

        return encoder_components

    def prepare_hf_vit(self, ):
        from transformers import ViTForMaskedImageModeling
        from elms.vision_encoders.hf_vit.hf_vit import HFVit
        hf_encoder = ViTForMaskedImageModeling.from_pretrained(VISION_ENCODERS[self.args.encoder]["model"])
        VISION_ENCODERS[self.args.encoder]["projection_dim"] = hf_encoder.config.hidden_size
        VISION_ENCODERS[self.args.encoder]["model_hidden_size"] = hf_encoder.config.hidden_size
        VISION_ENCODERS[self.args.encoder]["num_patches"] = (hf_encoder.config.image_size // hf_encoder.config.patch_size) ** 2
        assert VISION_ENCODERS[self.args.encoder]["num_patches"] is not None, print("num_patches is None")
        model = HFVit(hf_encoder, VISION_ENCODERS[self.args.encoder]["output_hidden_states"])
        return {"encoder": model}

    def prepare_hf_clip(self,):
        from transformers import AutoModel
        from elms.vision_encoders.hf_clip.hf_clip import HFClip
        hf_encoder = AutoModel.from_pretrained(VISION_ENCODERS[self.args.encoder]["model"])
        VISION_ENCODERS[self.args.encoder]["model_hidden_size"] = hf_encoder.config.projection_dim
        VISION_ENCODERS[self.args.encoder]["projection_dim"] = hf_encoder.config.projection_dim
        model = HFClip(hf_encoder, VISION_ENCODERS[self.args.encoder]["output_hidden_states"])
        return {"encoder": model}

    def prepare_hf_siglip(self,):
        from transformers import AutoModel
        from elms.vision_encoders.hf_siglip.hf_siglip import HFSiglip
        hf_encoder = AutoModel.from_pretrained(VISION_ENCODERS[self.args.encoder]["model"])
        VISION_ENCODERS[self.args.encoder]["model_hidden_size"] = hf_encoder.config.text_config.hidden_size
        VISION_ENCODERS[self.args.encoder]["projection_dim"] = hf_encoder.config.text_config.hidden_size
        model = HFSiglip(hf_encoder, VISION_ENCODERS[self.args.encoder]["output_hidden_states"])
        return {"encoder": model}

    def prepare_siglip_ecg(self,):
        from transformers import AutoModel, AutoConfig
        from elms.vision_encoders.siglip_ecg.siglip_ecg import SiglipEcg
        config = AutoConfig.from_pretrained(VISION_ENCODERS[self.args.encoder]["model"])
        config.text_config.bos_token_id = None
        config.text_config.eos_token_id = None
        hf_encoder = AutoModel.from_pretrained(VISION_ENCODERS[self.args.encoder]["model"], config = config)
        hidden = hf_encoder.config.vision_config.hidden_size
        VISION_ENCODERS[self.args.encoder]["model_hidden_size"] = hidden
        VISION_ENCODERS[self.args.encoder]["projection_dim"] = hidden
        model = SiglipEcg(hf_encoder, segment_len=self.args.segment_len,
                          patch_size=25,
                          num_encoder_tokens=self.args.num_encoder_tokens,
                          num_leads=len(self.args.leads))
        return {"encoder": model}

    def prepare_merl(self,):
        from elms.ecg_encoders.merl.merl import MerlConfig, Merl
        cfg = MerlConfig(distributed=self.args.distributed,
                         num_encoder_tokens=self.args.num_encoder_tokens)
        model = Merl(cfg)
        return {"encoder": model}

    def prepare_mlae(self):
        from elms.ecg_encoders.mlae.mlae import MLAEConfig, MLAE
        cfg = MLAEConfig(seq_len=self.args.segment_len,
                         num_encoder_tokens=self.args.num_encoder_tokens) # Each lead is patch, so default
        model = MLAE(cfg)
        return {"encoder": model}

    def prepare_mtae(self):
        from elms.ecg_encoders.mtae.mtae import MTAEConfig, MTAE
        cfg = MTAEConfig(seq_len=self.args.segment_len, patch_size=self.calculate_patch_size(),
                         num_encoder_tokens=self.args.num_encoder_tokens)
        model = MTAE(cfg)
        return {"encoder": model}

    def prepare_st_mem(self):
        from elms.ecg_encoders.st_mem.st_mem import ST_MEMConfig, ST_MEM
        cfg = ST_MEMConfig(seq_len=self.args.segment_len, patch_size=self.calculate_patch_size(),
                           num_encoder_tokens=self.args.num_encoder_tokens)
        model = ST_MEM(cfg)
        return {"encoder": model}

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

    def calculate_patch_size(self):
        min_patches = 16
        max_patches = 64
        factors = [i for i in range(1, self.args.segment_len + 1) if self.args.segment_len % i == 0]
        patch_candidates = []
        for patch_size in factors:
            num_patches = self.args.segment_len // patch_size
            if min_patches <= num_patches <= max_patches:
                patch_candidates.append(patch_size)
        if not patch_candidates:
            target = int(np.sqrt(self.args.segment_len / 32))
            patch_size = min(factors, key=lambda x: abs(x - target))
        else:
            patch_size = min(patch_candidates, key=lambda x: abs(self.args.segment_len // x - 32))
        return patch_size


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