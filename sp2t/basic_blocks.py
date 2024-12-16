from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import ops

import numpy as np

from einops import rearrange, reduce, repeat

from mmcv.cnn import build_activation_layer, build_norm_layer

import math

class SampleNorm(nn.Module):
    """Samplewise normalization with 1D LN parameters."""

    ALPHABET_TABLE = list("abcdefghijklmnopqrstuvwxyz")
    
    def __init__(self, normalize_shape = 0) -> None:
        """Initialize SampleNorm.

        Args:
            normalize_shape (int, optional): Channel count. Value < 0 will disable the learnable parameters. Defaults to 0.
        """
        super().__init__()
        self.learnable = normalize_shape > 0
        self.normalize_shape = normalize_shape
        if self.learnable:
            self.scale = nn.Parameter(torch.ones(normalize_shape), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(normalize_shape), requires_grad=True)
        
    def forward(self, x):
        """Do samplewise normalization.

        Args:
            x (torch.Tensor): tensor of shape [B x ... x C]

        Returns:
            x: normalized tensor
        """
        # record original shape
        shape = x.shape
        
        # do normalization
        x = x.reshape(x.shape[0], -1)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = (x - mean) / std
        
        # reshape back
        shape_info = {self.ALPHABET_TABLE[i]: shape[i] for i in range(len(shape))}
        dim_str = " ".join(self.ALPHABET_TABLE[1:len(shape)])
        fmt_str = f"a ({dim_str}) -> a {dim_str}"
        x = rearrange(x, fmt_str, **shape_info)
        
        # use LN learnable weights
        if self.learnable:
            assert shape[-1] == self.normalize_shape
            x = x * self.scale + self.bias
            
        return x
    

class FourierEmbed(nn.Module):
    """Fourier Embedding Layer from https://arxiv.org/pdf/2106.02795.pdf"""
    def __init__(self, d_embed, n_dim, temperature=16):
        super().__init__()
        assert d_embed % 2 == 0
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32), requires_grad=False)
        self.scaler = math.sqrt(2)
        self.proj = nn.Linear(n_dim, d_embed // 2, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.GELU(),
            nn.Linear(d_embed, d_embed),
        )
        self.norm = SampleNorm(d_embed)
        
    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): [... x D]
        Returns:
            torch.Tensor: [... x E]
        """
        x = self.proj(x * self.temperature)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1) * self.scaler
        x = self.mlp(x)
        x = self.norm(x)
        return x
    
    
class RelativeEmbed(nn.Module):
    def __init__(self, d_embed, n_dim, d_hidden):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(n_dim, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_embed),
        )
        
    def forward(self, x: torch.Tensor):
        return self.head(x)
    
    
class SIREN(nn.Module):
    def __init__(self, d_in, d_out, d_embed=None, num_layers=3, norm=True, mult=30):
        super().__init__()
        self.mult = mult
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        d_embed = d_out if d_embed is None else d_embed
        ch_in = d_in
        for l in range(num_layers):
            ch_out = d_embed if l != num_layers - 1 else d_out
            self.linears.append(nn.Linear(ch_in, ch_out))
            self.norms.append(SampleNorm(ch_out) if norm else nn.Identity())
            ch_in = ch_out
    
    def forward(self, x):
        x = x * self.mult
        for linear, norm in zip(self.linears, self.norms):
            x = norm(torch.sin(linear(x)))
        return x


class SIRENEmbed(nn.Module):
    def __init__(self, d_embed, n_dim, d_hidden, num_layers=3, temperature=16):
        super().__init__()
        self.temperature = temperature
        self.num_layers = num_layers
        assert num_layers >= 2
        self.layers = nn.ModuleList([nn.Linear(n_dim, d_hidden)])
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(d_hidden, d_hidden))
        self.layers.append(nn.Sequential(
            nn.Linear(d_hidden, d_embed),
            SampleNorm(d_embed)
        ))
        
    def forward(self, x: torch.Tensor):
        x = x * self.temperature
        for l in range(self.num_layers - 1):
            x = torch.sin(self.layers[l](x))
        x = self.layers[-1](x)
        return x
    

def identity_mapping(x):
    return x

class TableEmbed(nn.Module):
    """
    learn a [*table_size, d_embed] mapping
    defaults to scaled -1 to 1 gaussian
    """
    IDENTITY_MAPPING = identity_mapping
    
    def __init__(
        self, 
        d_embed, 
        table_size: tuple,
        strength = 1.0,                 # might be per embed
        temperature = 1.0,              # might be per embed
        input_mapping = identity_mapping,
    ) -> None:
        super().__init__()        
        s = torch.tensor(strength)
        t = torch.tensor(temperature)
        p = torch.meshgrid(*[torch.linspace(-1, 1, size) for size in table_size])           # ^TABLE_SIZE * D * E
        p = torch.stack(p, dim=-1).unsqueeze_(-1).repeat(*[1 for _ in table_size], d_embed)  # ^TABLE_SIZE * D * E
        c = torch.randn(len(table_size), d_embed) * 0.5
        for _ in table_size:
            c.unsqueeze_(0)                                                              # ^1 * D x E 
        r2 = ((p - c) * (p - c)).sum(dim=-2)                                                # ^TABLE_SIZE * D * E
        table = s * torch.exp(-r2) / math.sqrt(2 * math.pi)
        tmean = table
        for d in range(len(table_size)):
            tmean = tmean.mean(dim=d, keepdim=True)
        self.table = nn.Parameter(table - tmean, requires_grad=True)                            # [*table_size x D]
        self.table_size = nn.Parameter(torch.tensor(table_size, dtype=float), requires_grad=False)
        self.input_mapping = input_mapping
        self.dim = len(table_size)
        
    def forward(self, x: torch.Tensor):
        x = self.input_mapping(x)
        loc = (x * 0.5 + 0.5) * self.table_size
        loc = loc.clamp(max=self.table_size-1).clamp(min=0).floor().long()
        loc = loc.unbind(dim=-1)
        return self.table[loc]


class TableEmbed3D(nn.Module):
    def __init__(
        self,
        d_embed,
        strength=1.0,
        temperature=None,
        table_size=8,
        input_scale=1.0,
        norm=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        if isinstance(table_size, int):
            table_size = [table_size] * 3
        if isinstance(temperature, (list, tuple)):
            t = torch.rand(d_embed)
            t *= temperature[1] - temperature[0]
            t += temperature[0]
            temperature = t
        elif isinstance(temperature, (float, int)):
            temperature = [temperature] * d_embed
        if isinstance(strength, (list, tuple)):
            s = torch.rand(d_embed)
            s *= strength[1] - strength[0]
            s += strength[0]
            strength = s
        if isinstance(strength, (float, int)):
            strength = [strength] * d_embed
        self.input_scale = input_scale
        # uniformly split [-1, 1] to table_size voxels and calc center
        inds = [
            (torch.arange(size) * 2 - size + 1) / size
            for size in table_size
        ]
        grid_pos = torch.stack(
            torch.meshgrid(*inds, indexing='ij'),
            dim=-1
        )
        # add some randomness for each embed
        layers = []
        for d in range(d_embed):
            t = temperature[d]
            r2 = (grid_pos ** 2).sum(dim=-1) * t
            p = torch.exp(-r2) / math.sqrt(2 * math.pi)
            p = p - p.mean() + torch.randn_like(p) * 0.02 / (t + 1)
            if norm:
                p = p / p.std()
            layers.append(p * strength[d])
        table = torch.stack(layers, dim=0).unsqueeze(0)
        self.table = nn.Parameter(table)    # 1 x EMBED x *VOXEL_SIZE

    def forward(self, x: torch.Tensor):
        x = (x * self.input_scale).clamp(-1, 1)
        x = rearrange(x, "n d -> 1 1 1 n d")
        res = F.grid_sample(self.table, x)
        res = rearrange(res, "1 e 1 1 n -> n e")
        return res


class MultiHeadAttentionNoParam(nn.Module):
    def __init__(self, d_embed, num_heads, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads
        
    def forward(self, q, k, v, bias=None, kv_mask=None):
        H = self.num_heads
        q = rearrange(q, "b n (h d) -> (b h) n d", h=H)
        k = rearrange(k, "b n (h d) -> (b h) n d", h=H)
        v = rearrange(v, "b n (h d) -> (b h) n d", h=H)
        weights = torch.einsum("xqd,xkd->xqk", q, k)
        if bias is not None:
            weights = weights + bias
        if kv_mask is not None:
            kv_mask = repeat(1 - kv_mask.float(), "b k -> (b h) q k", h=H, q=q.shape[1])
            weights = weights * kv_mask
        weights = weights.softmax(dim=-1)
        weights = self.dropout(weights)
        out = torch.einsum("xqk,xkd->xqd", weights, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=H)
        return out, weights