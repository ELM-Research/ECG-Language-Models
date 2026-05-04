# Original work Copyright (c) Meta Platforms, Inc. and affiliates. <https://github.com/facebookresearch/mae>
# Modified work Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>
# Modified work Copyright 2025 ECG-Language-Models authors. <https://github.com/ELM-Research/ECG-Language-Models>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this repository https://github.com/bakqui/ST-MEM.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
### VIT ###
from typing import Optional
from torch import nn
from dataclasses import dataclass
import torch
from einops import pack, rearrange, repeat, unpack
from einops.layers.torch import Rearrange


__all__ = ["ViT"]

@dataclass
class ST_MEMConfig:
    seq_len: int = 2250
    patch_size: int = 75
    num_leads: int = 12
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    decoder_embed_dim: int = 256
    decoder_depth: int = 4
    decoder_num_heads: int = 4
    mlp_ratio: int = 4
    qkv_bias: bool = True
    norm_pix_loss: bool = False
    d_model: int = None
    num_encoder_tokens: int = 1

    def __post_init__(self):
        if self.d_model is None:
            self.d_model = self.embed_dim


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """MLP Module with GELU activation fn + dropout."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, drop_out_rate=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(drop_out_rate), nn.Linear(hidden_dim, output_dim), nn.Dropout(drop_out_rate)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        qkv_bias: bool = True,
        drop_out_rate: float = 0.0,
        attn_drop_out_rate: float = 0.0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == input_dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop_out_rate)
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=qkv_bias)

        if project_out:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, output_dim), nn.Dropout(drop_out_rate))
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        heads: int = 8,
        dim_head: int = 32,
        qkv_bias: bool = True,
        drop_out_rate: float = 0.0,
        attn_drop_out_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        attn = Attention(
            input_dim=input_dim,
            output_dim=output_dim,
            heads=heads,
            dim_head=dim_head,
            qkv_bias=qkv_bias,
            drop_out_rate=drop_out_rate,
            attn_drop_out_rate=attn_drop_out_rate,
        )
        self.attn = PreNorm(dim=input_dim, fn=attn)
        self.droppath1 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        ff = FeedForward(input_dim=output_dim, output_dim=output_dim, hidden_dim=hidden_dim, drop_out_rate=drop_out_rate)
        self.ff = PreNorm(dim=output_dim, fn=ff)
        self.droppath2 = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.droppath1(self.attn(x)) + x
        x = self.droppath2(self.ff(x)) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_size: int,
        num_leads: int,
        num_classes: Optional[int] = None,
        width: int = 768,
        depth: int = 12,
        mlp_dim: int = 3072,
        heads: int = 12,
        dim_head: int = 64,
        qkv_bias: bool = True,
        drop_out_rate: float = 0.0,
        attn_drop_out_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        assert seq_len % patch_size == 0, "The sequence length must be divisible by the patch size."
        self._repr_dict = {
            "seq_len": seq_len,
            "patch_size": patch_size,
            "num_leads": num_leads,
            "num_classes": num_classes if num_classes is not None else "None",
            "width": width,
            "depth": depth,
            "mlp_dim": mlp_dim,
            "heads": heads,
            "dim_head": dim_head,
            "qkv_bias": qkv_bias,
            "drop_out_rate": drop_out_rate,
            "attn_drop_out_rate": attn_drop_out_rate,
            "drop_path_rate": drop_path_rate,
        }
        self.width = width
        self.depth = depth

        # embedding layers
        num_patches = seq_len // patch_size
        patch_dim = num_leads * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (n p) -> b n (p c)", p=patch_size), nn.LayerNorm(patch_dim), nn.Linear(patch_dim, width), nn.LayerNorm(width)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, width))
        self.cls_embedding = nn.Parameter(torch.randn(width))

        # transformer layers
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(
                input_dim=width,
                output_dim=width,
                hidden_dim=mlp_dim,
                heads=heads,
                dim_head=dim_head,
                qkv_bias=qkv_bias,
                drop_out_rate=drop_out_rate,
                attn_drop_out_rate=attn_drop_out_rate,
                drop_path_rate=drop_path_rate_list[i],
            )
            self.add_module(f"block{i}", block)
        self.dropout = nn.Dropout(drop_out_rate)
        self.norm = nn.LayerNorm(width)

        # classifier head
        self.head = nn.Identity() if num_classes is None else nn.Linear(width, num_classes)

    def reset_head(self, num_classes: Optional[int] = None):
        del self.head
        self.head = nn.Identity() if num_classes is None else nn.Linear(self.width, num_classes)

    def forward_encoding(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape
        cls_embeddings = repeat(self.cls_embedding, "d -> b d", b=b)
        x, ps = pack([cls_embeddings, x], "b * d")
        x = x + self.pos_embedding[:, : n + 1]

        x = self.dropout(x)
        for i in range(self.depth):
            x = getattr(self, f"block{i}")(x)

        cls_embeddings, _ = unpack(x, ps, "b * d")

        return self.norm(cls_embeddings)

    def forward(self, series):
        x = self.forward_encoding(series)
        return self.head(x)

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f"    {k}={v},\n"
        print_str += ")"
        return print_str


### ENCODER / STE_MEM_VIT ###
from torch import nn


class ST_MEM_ViT(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_size: int,
        num_leads: int,
        num_classes: Optional[int] = None,
        width: int = 768,
        depth: int = 12,
        mlp_dim: int = 3072,
        heads: int = 12,
        dim_head: int = 64,
        qkv_bias: bool = True,
        drop_out_rate: float = 0.0,
        attn_drop_out_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        assert seq_len % patch_size == 0, "The sequence length must be divisible by the patch size."
        self._repr_dict = {
            "seq_len": seq_len,
            "patch_size": patch_size,
            "num_leads": num_leads,
            "num_classes": num_classes if num_classes is not None else "None",
            "width": width,
            "depth": depth,
            "mlp_dim": mlp_dim,
            "heads": heads,
            "dim_head": dim_head,
            "qkv_bias": qkv_bias,
            "drop_out_rate": drop_out_rate,
            "attn_drop_out_rate": attn_drop_out_rate,
            "drop_path_rate": drop_path_rate,
        }
        self.width = width
        self.depth = depth

        # embedding layers
        num_patches = seq_len // patch_size
        patch_dim = patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (n p) -> b c n p", p=patch_size), nn.LayerNorm(patch_dim), nn.Linear(patch_dim, width), nn.LayerNorm(width)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, width))
        self.sep_embedding = nn.Parameter(torch.randn(width))
        self.lead_embeddings = nn.ParameterList(nn.Parameter(torch.randn(width)) for _ in range(num_leads))

        # transformer layers
        drop_path_rate_list = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        for i in range(depth):
            block = TransformerBlock(
                input_dim=width,
                output_dim=width,
                hidden_dim=mlp_dim,
                heads=heads,
                dim_head=dim_head,
                qkv_bias=qkv_bias,
                drop_out_rate=drop_out_rate,
                attn_drop_out_rate=attn_drop_out_rate,
                drop_path_rate=drop_path_rate_list[i],
            )
            self.add_module(f"block{i}", block)
        self.dropout = nn.Dropout(drop_out_rate)
        self.norm = nn.LayerNorm(width)

    def forward_encoding(self, series):
        num_leads = series.shape[1]
        if num_leads > len(self.lead_embeddings):
            raise ValueError(f"Number of leads ({num_leads}) exceeds the number of lead embeddings")

        x = self.to_patch_embedding(series)
        b, _, n, _ = x.shape
        x = x + self.pos_embedding[:, 1 : n + 1, :].unsqueeze(1)

        # lead indicating modules
        sep_embedding = self.sep_embedding[None, None, None, :]
        left_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, :1, :].unsqueeze(1)
        right_sep = sep_embedding.expand(b, num_leads, -1, -1) + self.pos_embedding[:, -1:, :].unsqueeze(1)
        x = torch.cat([left_sep, x, right_sep], dim=2)
        lead_embeddings = torch.stack(list(self.lead_embeddings)).unsqueeze(0)
        lead_embeddings = lead_embeddings.unsqueeze(2).expand(b, -1, n + 2, -1)
        x = x + lead_embeddings
        x = rearrange(x, "b c n p -> b (c n) p")

        x = self.dropout(x)
        for i in range(self.depth):
            x = getattr(self, f"block{i}")(x)

        # remove SEP embeddings
        x = rearrange(x, "b (c n) p -> b c n p", c=num_leads)
        x = x[:, :, 1:-1, :]

        # x = torch.mean(x, dim=(1, 2))
        # print("inside2", x.shape)
        return self.norm(x)

    def forward(self, series):
        x = self.forward_encoding(series)
        return x

    def __repr__(self):
        print_str = f"{self.__class__.__name__}(\n"
        for k, v in self._repr_dict.items():
            print_str += f"    {k}={v},\n"
        print_str += ")"
        return print_str


#####

class ST_MEM(nn.Module):
    def __init__(self, cfg: ST_MEMConfig):
        super().__init__()
        self.cfg = cfg
        seq_len = cfg.seq_len
        patch_size = cfg.patch_size
        num_leads = cfg.num_leads
        embed_dim = cfg.embed_dim
        depth = cfg.depth
        num_heads = cfg.num_heads
        mlp_ratio = cfg.mlp_ratio
        qkv_bias = cfg.qkv_bias

        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.num_leads = num_leads
        self.encoder = ST_MEM_ViT(
            seq_len=seq_len,
            patch_size=patch_size,
            num_leads=num_leads,
            width=embed_dim,
            depth=depth,
            mlp_dim=mlp_ratio * embed_dim,
            heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.avgpool = nn.AdaptiveAvgPool1d(cfg.num_encoder_tokens)

    def forward_encoder(self, x,):
        x = self.encoder(x)
        return x

    def get_encoder_embeddings(self, ecg_signal):
        x_latents = self.forward_encoder(ecg_signal.to(next(self.parameters()).dtype))
        out = rearrange(x_latents, 'b c n d -> b (c n) d')
        out = out.transpose(1, 2)
        out = self.avgpool(out)
        return out.transpose(1, 2)

    def forward(self, ):
        raise NotImplementedError


