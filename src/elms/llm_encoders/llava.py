from torch import nn
import torch


class LLaVA(nn.Module):
    def __init__(self, llm: nn.Module, encoder: nn.Module,
                 projection: nn.Module, update: set, only_text: bool = False):
        super(LLaVA, self).__init__()
        self.llm = llm
        self.encoder = encoder
        self.projection = projection
        self.update = update
        self.only_text = only_text
        for name, module in [("encoder", self.encoder), ("connector", self.projection), ("llm", self.llm)]:
            requires_grad = name in self.update
            for p in module.parameters():
                p.requires_grad = requires_grad

    def forward(self, elm_input_ids, elm_attention_mask, elm_labels, signal_id_indices,
                encoder_tokenizer_out):
        projected_embeds = self.get_projections(encoder_tokenizer_out)
        llm_embeddings = self.llm.get_llm_embeddings(elm_input_ids)
        elm_inputs_embeds = self.inject_projected_embeds(llm_embeddings, projected_embeds,
                                                         signal_id_indices)
        out = self.llm(elm_input_ids = None,
                       elm_inputs_embeds = elm_inputs_embeds,
                       elm_attention_mask = elm_attention_mask,
                       elm_labels = elm_labels)
        return out

    def train(self, mode: bool = True):
        super().train(mode)
        for name, module in [("encoder", self.encoder), ("connector", self.projection), ("llm", self.llm)]:
            module.train(mode if name in self.update else False)
        return self

    def get_projections(self, encoder_tokenizer_out):
        if "encoder" in self.update:
            signal_embeds = self.encoder.get_encoder_embeddings(**encoder_tokenizer_out)
        else:
            with torch.no_grad():
                signal_embeds = self.encoder.get_encoder_embeddings(**encoder_tokenizer_out)
        projected_embeds = self.projection.project(signal_embeds)
        return projected_embeds

    def inject_projected_embeds(self, llm_embeddings: torch.Tensor, projected_embeds: torch.Tensor, signal_id_indices: torch.Tensor) -> torch.Tensor:
        if self.only_text:
            return llm_embeddings
        else:
            assert llm_embeddings.ndim == 3
            B, T, H = llm_embeddings.shape

            if projected_embeds.ndim == 2:
                projected_embeds = projected_embeds.unsqueeze(1)
            if signal_id_indices.ndim == 1:
                signal_id_indices = signal_id_indices.unsqueeze(0)

            assert projected_embeds.shape[:2] == signal_id_indices.shape
            assert projected_embeds.shape[0] == B and projected_embeds.shape[2] == H
            embedding_mask = (projected_embeds != 0).any(dim=-1)
            indices_mask = signal_id_indices >= 0
            valid_mask = embedding_mask & indices_mask

            if valid_mask.any():
                valid_indices = signal_id_indices[valid_mask]
                assert (valid_indices >= 0).all() and (valid_indices < T).all(), (
                    f"Valid indices must be in [0, {T}), got min={valid_indices.min()}, max={valid_indices.max()}"
                )

            N = signal_id_indices.shape[1]
            dev = llm_embeddings.device
            batch_idx = torch.arange(B, device=dev).unsqueeze(1).expand(B, N)

            out = llm_embeddings.clone()
            if valid_mask.any():
                valid_batch_idx = batch_idx[valid_mask]
                valid_signal_idx = signal_id_indices[valid_mask]
                valid_projected = projected_embeds[valid_mask]

                out[valid_batch_idx, valid_signal_idx] = valid_projected

                injected = out[valid_batch_idx, valid_signal_idx]
                assert torch.allclose(injected, valid_projected, atol=1e-6), "Injection failed: projected embeddings not correctly written."

            return out

    def generate(self, elm_input_ids, encoder_tokenizer_out, elm_attention_mask, signal_id_indices, **gen_kwargs):
        projected_embeds = self.get_projections(encoder_tokenizer_out)
        llm_embeddings = self.llm.get_llm_embeddings(elm_input_ids)
        elm_inputs_embeds = self.inject_projected_embeds(llm_embeddings, projected_embeds, signal_id_indices)
        out = self.llm.generate(elm_input_ids = None,
                                elm_inputs_embeds = elm_inputs_embeds,
                                elm_attention_mask = elm_attention_mask,
                                **gen_kwargs)
        return out
