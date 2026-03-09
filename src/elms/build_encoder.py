import argparse
import torch
import numpy as np

from utils.gpu_manager import is_main

from configs.constants import VISION_ENCODERS

class BuildEncoder:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def build_encoder(self):
        encoder_components = None

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
        # Strip wrapper prefixes (e.g. 'st_mem.' from ECG-Bench checkpoints)
        model_keys = set(encoder_components["encoder"].state_dict().keys())
        if not (model_keys & set(state.keys())):
            prefixes = {k.split('.', 1)[0] for k in state.keys() if '.' in k}
            for prefix in prefixes:
                stripped = {k[len(prefix)+1:]: v for k, v in state.items() if k.startswith(prefix + '.')}
                if model_keys & set(stripped.keys()):
                    state = stripped
                    break
        encoder_components["encoder"].load_state_dict(state, strict=False)
        if is_main():
            print(f"Loaded {self.args.encoder} checkpoint from {self.args.encoder_ckpt}")

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

    def check_ckpt(self, model):
        import hashlib
        h = hashlib.md5()
        for p in model.parameters():
            h.update(p.data.cpu().numpy().tobytes())
        return h.hexdigest()