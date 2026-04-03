import argparse
from typing import Any, Dict, Iterable, Mapping

from elms.connectors.linear_proj import LinearProjection
from elms.connectors.patch_proj import PatchProjection
from elms.connectors.cnn_patch_proj import CNNPatchProjection

from configs.constants import ECG_ENCODERS, VISION_ENCODERS


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
        if self.args.elm == "llava":
            encoder_llm_components = self.build_llava()
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

    def build_llava(
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
            projection_layer, self.args.update_encoder,
            True if self.args.perturb == "only_text" else False)
        return {"elm": encoder_llm}

    def build_base_elf(
        self,
    ):
        from elms.llm_encoders.base_elf import BaseElf
        projection_dim = len(self.args.leads) * self.args.segment_len
        projection_layer = LinearProjection(projection_dim, self.args.llm)
        encoder_llm = BaseElf(self.llm_components["llm"], projection_layer,
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
                           True if self.args.perturb == "only_text" else False)
        return {"elm": encoder_llm}
