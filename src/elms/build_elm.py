import argparse
import torch

from elms.build_llm import BuildLLM
from elms.build_encoder import BuildEncoder
from elms.connect_nns import ConnectNN

from utils.gpu_manager import is_main

class BuildELM:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def build_elm(self, llm_tokenizer):
        llm_components = BuildLLM(self.args, llm_tokenizer,).build_llm()
        encoder_components = BuildEncoder(self.args).build_encoder()
        elm_components = ConnectNN(llm_components, encoder_components, self.args).connect_nn()
        assert elm_components is not None, print("ELM Components is None")
        if self.args.elm_ckpt:
            self.load_elm_checkpoint(elm_components)
        return elm_components

    def load_elm_checkpoint(self, elm_components):
        elm_checkpoint = torch.load(self.args.elm_ckpt, map_location="cpu", weights_only=False)
        model = elm_components["elm"]
        state = self._align_vocab_expanded_tensors(model, elm_checkpoint["model_state_dict"])
        model.load_state_dict(state, strict=False)
        if is_main():
            print(f"Loaded ELM checkpoint from {self.args.elm_ckpt}")

    def _align_vocab_expanded_tensors(self, model, ckpt_state: dict) -> dict:
        """Allow loading checkpoints trained with smaller tokenizer vocab.

        Copies all matching tensors directly. For 2D tensors where only dim-0 grew
        (e.g., input embeddings and lm_head), copy old rows into current tensor and
        keep newly-added rows initialized from the current model init.
        """
        model_state = model.state_dict()
        aligned = {}

        for key, tensor in ckpt_state.items():
            if key not in model_state:
                continue
            target = model_state[key]
            if tensor.shape == target.shape:
                aligned[key] = tensor
                continue

            can_row_expand = (
                tensor.ndim == target.ndim == 2
                and tensor.shape[1] == target.shape[1]
                and tensor.shape[0] <= target.shape[0]
            )
            if can_row_expand:
                merged = target.clone()
                merged[:tensor.shape[0]] = tensor
                aligned[key] = merged
                if is_main():
                    print(f"[ckpt-load] Expanded '{key}' from {tuple(tensor.shape)} -> {tuple(target.shape)}")
                continue

            if is_main():
                print(f"[ckpt-load] Skipping shape-mismatch '{key}': ckpt={tuple(tensor.shape)} model={tuple(target.shape)}")

        return aligned