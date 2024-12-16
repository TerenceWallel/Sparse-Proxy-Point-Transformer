import math

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.profiler import record_function

from einops import repeat, rearrange, reduce

from mmcv.ops.points_sampler import get_sampler_cls
from mmcv.ops import knn
from mmdet3d.models.layers import make_sparse_convmodule

from spconv.pytorch import SparseConvTensor

from .basic_blocks import SIRENEmbed, TableEmbed3D, MultiHeadAttentionNoParam

class ProxyFuser(nn.Module):
    def __init__(
        self, 
        d_embed,
        dropout,
        with_ffn,
        norm_loc,
    ):
        super().__init__()
        self.norm_loc = norm_loc
        self.with_ffn = with_ffn
        if with_ffn:
            self.ffn = nn.Sequential(
                nn.Linear(d_embed, d_embed * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_embed * 2, d_embed)
            )
            self.ffn_norm = nn.BatchNorm1d(d_embed)
    def process_ffn(self, features):
        if self.with_ffn and self.norm_loc == "pre":
            return features + self.ffn(self.ffn_norm(features))
        if self.with_ffn and self.norm_loc == "post":
            return self.ffn_norm(features + self.ffn(features))
        return features
    def forward(self, features: torch.Tensor, info: dict):
        pass


class ProxyFuserAttn(ProxyFuser):    
    def __init__(
        self, 
        d_embed,
        num_heads,
        dropout,
        pe_cfg = None,
        rpe_cfg=None,
        residual = True,
        norm_loc = "pre",
        with_ffn = False,
        **kwargs,
    ):
        super().__init__(d_embed, dropout, with_ffn, norm_loc)
        self.residual = residual
        self.norm_loc = norm_loc
        self.num_heads = num_heads
        self.d_embed = d_embed
        self.with_pe = pe_cfg is not None
        self.with_rpe = rpe_cfg is not None
        self.attn = nn.MultiheadAttention(d_embed, num_heads, dropout, batch_first=True)
        if pe_cfg is not None:
            self.pe = SIRENEmbed(**pe_cfg)
        if rpe_cfg is not None:
            self.rpe = TableEmbed3D(**rpe_cfg)
        if residual:
            self.norm = nn.BatchNorm1d(d_embed)
        else:
            self.norm_loc = "none"
        
    def forward(self, features: torch.Tensor, info: dict):
        # extract info
        assert all(k in info for k in ["proxy_pos", "batch_num_px"])
        positions : torch.Tensor = info["proxy_pos"]   # N x D
        batch_num_px: list = info["batch_num_px"]
        batch_px_mask = info["batch_px_mask"]
        
        # constants
        N, C = features.shape
        B = len(batch_num_px)
        H = self.num_heads
        maxlen = max(batch_num_px)
        device = features.device
        has_mask = len(batch_px_mask) > 0
        
        residual = features
        if self.norm_loc == "pre":
            features = self.norm(features)
            
        if self.with_pe:
            features_with_pe = features + self.pe(positions)

        with record_function("Fuser RPE"):
            if self.with_rpe:
                if B > 1:
                    rpes = []
                    start = 0
                    for b, n in enumerate(batch_num_px):
                        end = start + n
                        pad = maxlen - n
                        batch_pos = positions[start:end]
                        batch_rel_pos = batch_pos.unsqueeze(0) - batch_pos.unsqueeze(1)  # N x N x D
                        batch_rel_pos = rearrange(batch_rel_pos, "n1 n2 d -> (n1 n2) d")
                        batch_rpe = self.rpe(batch_rel_pos)
                        batch_rpe = rearrange(batch_rpe, "(n1 n2) h -> 1 h n1 n2", n1=n, n2=n)
                        batch_rpe = F.pad(batch_rpe, (0, pad, 0, pad), "constant", 0)
                        rpes.append(batch_rpe.squeeze_(0))
                        start = end
                    rpe = torch.cat(rpes, dim=0) * math.sqrt(self.d_embed)
                else:
                    rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # N x N x D
                    rel_pos = rearrange(rel_pos, "n1 n2 d -> (n1 n2) d")
                    rpe = self.rpe(rel_pos)
                    rpe = rearrange(rpe, "(n1 n2) h -> h n1 n2", n1=batch_num_px[0], n2=batch_num_px[0])
            elif "proxy_rpe" in info:
                rpe = info["proxy_rpe"]
            else:
                rpe = None
        
        # batch and build pad mask
        if B > 1:
            token = torch.zeros(B, maxlen, C, device=device)
            token_with_pe = torch.zeros(B, maxlen, C, device=device) if self.with_pe else None
            pad_mask = torch.zeros(B, maxlen, device=device)
            start = 0
            for b, n in enumerate(batch_num_px):
                end = start + n
                token[b, :n] = features[start:end]
                if self.with_pe:
                    token_with_pe[b, :n] = features_with_pe[start:end]
                pad_mask[b, n:] = float("-inf")
                if has_mask:
                    pad_mask[b, :n] = batch_px_mask[b]
                start = end
            if self.with_pe:
                q = k = token_with_pe
                v = token
            else:
                q = k = v = token
        else:
            pad_mask = None
            if self.with_pe:
                q = k = features_with_pe
                v = features
            else:
                q = k = v = features
        
        # there is a bug related to MHA's fast path before 2.0.1
        # batching is disabled to skip fast path when B == 1
        with record_function("Fuser Attn"):
            token, attn = self.attn(q, k, v, key_padding_mask=pad_mask, attn_mask=rpe)
            
        # unbatch
        if B > 1:
            out = []
            start = 0
            for b, n in enumerate(batch_num_px):
                end = start + n
                out.append(token[b, :n])
                start = end
            out = torch.cat(out, dim=0)
        else:
            out = token
            
        if self.residual:
            out = out + residual
        if self.norm_loc == "post":
            out = self.norm(out)

        with record_function("Fuser FFN"):
            out = self.process_ffn(out)
        
        return out, { "attn": attn }
            

class SELayer(nn.Module):
    def __init__(
        self,
        channel,
        pool_mode="mean",
        expansion=2,
        reduction=4,
        dropout=0.1
    ):
        super().__init__()
        assert pool_mode in ["mean", "min", "max"]
        self.pool_mode = pool_mode
        self.local = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel * expansion, channel),
        )
        self.dense = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()
        
    def do_pool(self, x):
        if self.pool_mode == "mean":
            return x.mean(dim=0)
        elif self.pool_mode == "min":
            return x.min(dim=0).values
        elif self.pool_mode == "max":
            return x.max(dim=0).values
        else:
            raise NotImplementedError(f"Unknown pool mode {self.pool_mode}")

    def forward(self, x: torch.Tensor, batch_cnts):
        """
            x: (N, C)
            batch_cnts: list of batch sizes
        """
        B = len(batch_cnts)
        C = x.shape[1]
        # residual and local block
        residual = x
        x = self.local(x)
        # squeeze and excitation
        if B == 1:
            s = self.do_pool(x)
            s = self.dense(x)
            e = self.activation(x * s)
        else:
            start = 0
            s = x.new_zeros(B, C)
            for b, n in enumerate(batch_cnts):
                end = start + n
                s[b] = self.do_pool(x[start:end])
                start = end
            s = self.dense(s)
            e = []
            start = 0
            for b, n in enumerate(batch_cnts):
                end = start + n
                e.append(x[start:end] * s[[b]])
                start = end
            e = self.activation(torch.cat(e, dim=0))
        # add residual
        out = e + residual
        return out


class ProxyFuserSE(ProxyFuser):
    def __init__(
        self,
        d_embed,
        pool_mode="mean",
        expansion=2,
        reduction=4,
        num_layers=2,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(d_embed, dropout, False, None)
        self.se_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.se_layers.append(
                SELayer(d_embed, pool_mode, expansion, reduction, dropout))

    def forward(self, features: torch.Tensor, info: dict):
        # extract info
        assert all(k in info for k in ["batch_num_px"])
        batch_num_px: list = info["batch_num_px"]

        for layer in self.se_layers:
            features = layer(features, batch_num_px)

        return features, {}


def get_proxy_fuser(cfg) -> ProxyFuser:
    mode = cfg["mode"]
    if mode == "attn":
        return ProxyFuserAttn(**cfg)
    elif mode == "se":
        return ProxyFuserSE(**cfg)
    else:
        raise NotImplementedError(f"Unknown proxy initializer mode {mode}")