import math

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.profiler import record_function

from einops import repeat, rearrange, reduce

class SparseAttention(nn.Module):
    """compute sparsely connected attention"""
    def __init__(
        self,
        num_ch,
        num_heads = 1.0,
        drop_sim = 0.1,
        drop_path = 0.3,
        reduce_query = "none",
        with_proj = True,
        same_kv = True,
        residual = True,
        norm_loc = "pre",
        sim_scale = 1.0,
        enable_checkpoint = False,
        norm_after_proj = False,
        include_self = False,
        use_taichi=False,
        use_warp=False,
    ):
        super().__init__()
        native_available = not include_self and reduce_query == "none"
        self.use_warp = use_warp and native_available
        self.use_taichi = use_taichi and native_available and not use_warp
        self.enable_checkpoint = enable_checkpoint
        self.num_heads = num_heads
        self.drop_sim = drop_sim
        self.drop_path = drop_path
        self.reduce_query = reduce_query
        self.with_proj = with_proj
        self.same_kv = same_kv
        self.residual = residual
        self.norm_loc = norm_loc
        self.sim_scale = sim_scale / math.sqrt(num_ch)
        self.include_self = include_self
        if self.with_proj:
            def build_proj(out_ch = num_ch):
                if norm_after_proj:
                    return nn.Sequential(nn.Linear(num_ch, out_ch), nn.BatchNorm1d(out_ch, affine=False))
                else:
                    return nn.Linear(num_ch, out_ch)
            self.proj_q = build_proj(num_ch * 3 if include_self else num_ch)
            self.proj_k = build_proj()
            self.proj_v = build_proj() if not self.same_kv else None
            self.proj_o = build_proj()
        else:
            self.proj_q = self.proj_k = self.proj_v = self.proj_o = nn.Identity()
        if residual:
            self.norm = nn.LayerNorm(num_ch)
        else:
            self.norm_loc = "none"
        self.debug = False
        self.debug_states = {}

    def set_debug(self, debug):
        self.debug = debug
        self.debug_states = {}

    def get_debug_states(self):
        return self.debug_states

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, associations, bias=None):
        """compute sparsely connected attention
        
        Args:
            q (torch.Tensor): query of dim Q x C
            k (torch.Tensor): key of dim K x C 
            v (torch.Tensor): value of dim K x C (same as key)
            associations (torch.Tensor): describe KV -> Q connections of dim N x 2
            bias (torch.Tensor): bias of dim N x H
            num_heads (int, optional): number of heads. Defaults to 1
            drop_sim (float, optional): dropout rate. Defaults to 0
            init_query (str, optional): reduce mode for query initialization.
    
        Returns:
            out (torch.Tensor): output of dim Q x C
            weights (torch.Tensor): normalized weights of dim N x num_heads
        """
        dtype = q.dtype
        q = q.float()
        k = k.float()
        v = k if self.same_kv else v.float()
        use_native = self.use_warp or self.use_taichi
        if self.training or not use_native:
            if self.enable_checkpoint:
                out, weights = checkpoint(self.forward_torch, q, k, v, associations, bias, use_reentrant=False)
            else:
                out, weights = self.forward_torch(q, k, v, associations, bias)
        else:
            if self.use_warp:
                out, weights = self.forward_warp(q, k, v, associations, bias)
            # elif self.use_taichi:
            #     out, weights = self.forward_taichi(q, k, v, associations, bias)
            else:
                raise ValueError("Invalid use case")
        return out.to(dtype), weights

    def forward_warp(self, q, k, v, associations, bias):
        from .warp_sparse_attention import warp_sparse_attention
        Q, C = q.shape
        H = self.num_heads
        D = C // H
        N = associations.shape[0]
        device = q.device
        assert not self.include_self

        # prepare residual and norm
        residual = q
        if self.norm_loc == "pre":
            q = self.norm(q)

        # project and rearrange
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v) if not self.same_kv else k

        # apply sparse attention
        with record_function("Warp"):
            out, weights = warp_sparse_attention(q, k, v, associations, bias, self.sim_scale, D, H)

        # rest of things
        out = self.proj_o(out)
        if self.residual:
            out = out + residual
        if self.norm_loc == "post":
            out = self.norm(out)

        return out, weights

    # def forward_taichi(self, q, k, v, associations, bias):
    #     from .taichi_sparse_attention import taichi_sparse_attn, taichi_sparse_attn_nobias, ti_sync, ensure_taichi_initialized
    #     Q, C = q.shape
    #     H = self.num_heads
    #     D = C // H
    #     N = associations.shape[0]
    #     device = q.device
    #     assert not self.include_self

    #     ensure_taichi_initialized()

    #     # prepare residual and norm
    #     residual = q
    #     if self.norm_loc == "pre":
    #         q = self.norm(q)

    #     # project and rearrange
    #     q = rearrange(self.proj_q(q), "x (h d) -> x h d", h=H)
    #     k = rearrange(self.proj_k(k), "x (h d) -> x h d", h=H)
    #     v = rearrange(self.proj_v(v), "x (h d) -> x h d", h=H) if not self.same_kv else k

    #     # to contiguous
    #     q = q.contiguous()
    #     k = k.contiguous()
    #     v = v.contiguous()
    #     associations = associations.contiguous()
    #     if bias is not None:
    #         bias = bias.contiguous()

    #     # apply sparse attention
    #     out = torch.zeros(Q, H, D, device=device, dtype=torch.float32)
    #     tmp_sim = torch.zeros(N, H, device=device, dtype=torch.float32)
    #     tmp_sum = torch.zeros(Q, H, device=device, dtype=torch.float32)
    #     tmp_max = torch.zeros(Q, H, device=device, dtype=torch.float32)
    #     with record_function("Taichi"):
    #         if bias is not None:
    #             taichi_sparse_attn(q, k, v, associations, bias, out, tmp_sim, tmp_sum, tmp_max,
    #                                N, 8, H, D, self.sim_scale)
    #         else:
    #             taichi_sparse_attn_nobias(q, k, v, associations, out, tmp_sim, tmp_sum, tmp_max,
    #                                       N, 8, H, D, self.sim_scale)
    #         ti_sync()

    #     # rest of things
    #     out = self.proj_o(rearrange(out, "x h d -> x (h d)"))
    #     if self.residual:
    #         out = out + residual
    #     if self.norm_loc == "post":
    #         out = self.norm(out)

    #     return out, tmp_sim

    def forward_torch(self, q, k, v, associations, bias):
        """forward function for training"""
        Q, C = q.shape
        KV = k.shape[0]
        H = self.num_heads
        D = C // H
        N = associations.shape[0]
        device = q.device

        if self.debug:
            self.debug_states["in_norm"] = [
                q.norm(dim=-1).mean().item(),
                k.norm(dim=-1).mean().item(),
                v.norm(dim=-1).mean().item(),
            ]

        # prepare residual and norm
        residual = q
        if self.norm_loc == "pre":
            q = self.norm(q)
        
        # project
        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v) if not self.same_kv else k
        if self.include_self:
            self_k = q[:, :C]
            self_v = q[:, C:C*2]
            q = q[:, C*2:C*3]
        
        if self.debug:
            self.debug_states["in_proj_norm"] = [
                q.norm(dim=-1).mean().item(),
                k.norm(dim=-1).mean().item(),
                v.norm(dim=-1).mean().item(),
            ]
            if self.include_self:
                self.debug_states["in_proj_norm"].extend([
                    self_k.norm(dim=-1).mean().item(),
                    self_v.norm(dim=-1).mean().item()
                ])

        with record_function("Sparse Expand"):
            # generate association ids
            src_ids = associations[..., 0].long()
            dst_ids = associations[..., 1].long()

            # expand kv
            k_expand = k[src_ids]
            v_expand = v[src_ids] if not self.same_kv else k_expand

            # get q by pooling
            if self.reduce_query != "none":
                q.index_reduce_(0, dst_ids, k_expand, self.reduce_query, include_self=True)
            q_expand = q[dst_ids]
            
            if self.include_self:
                self_k = rearrange(self_k, "n (h d) -> n h d", h=H)
                self_v = rearrange(self_v, "n (h d) -> n h d", h=H)
                q = rearrange(q, "n (h d) -> n h d", h=H)

            # convert to head based
            q_expand = rearrange(q_expand, "n (h d) -> n h d", h=H)
            k_expand = rearrange(k_expand, "n (h d) -> n h d", h=H)
            v_expand = rearrange(v_expand, "n (h d) -> n h d", h=H)

        with record_function("Sparse Sim"):
            if self.include_self:
                self_sim = (q * self_k).sum(dim=-1) * self.sim_scale
            
            # compute sim
            sim = (q_expand * k_expand).sum(dim=-1)
            if bias is not None:
                sim = sim + bias
            sim = sim * self.sim_scale

            # # normalize for each query
            # max_sim = torch.zeros(Q, H, device=device, dtype=torch.float32)
            # min_sim = torch.zeros(Q, H, device=device, dtype=torch.float32)
            # max_sim = max_sim.scatter_reduce(0, dst_ids[..., :H], sim, "amax", include_self=False)
            # min_sim = - min_sim.scatter_reduce(0, dst_ids[..., :H], -sim, "amax", include_self=False)
            # scale = (max_sim - min_sim).clamp(min=MAX_SIM_RANGE) / MAX_SIM_RANGE
            # scale = scale.gather(0, dst_ids[..., :H])
            # center = (max_sim + min_sim) / 2
            # center = center.gather(0, dst_ids[..., :H])
            # sim = (sim - center) / scale

            # subtract max
            if self.include_self:
                max_sim = self_sim
            else:
                max_sim = torch.full([Q, H], -torch.inf, device=device, dtype=torch.float32)
            max_sim.index_reduce_(0, dst_ids, sim, "amax", include_self=True)
            sim = sim - max_sim[dst_ids]

            # if self.debug:
            #     s = sim.float().exp()
            #     if self.include_self:
            #         ss = (self_sim - max_sim).float().exp()
            #         w = ss.clone()
            #     else:
            #         w = torch.full([Q, H], 1e-8, device=device, dtype=torch.float32)
            #     w = w.index_add_(0, dst_ids, s)
            #     w = s / w[dst_ids]
            #     self.debug_states["raw_weights"] = w.detach()
            #     self.debug_states["raw_sim"] = sim.detach()
            #     self.debug_states["max_sim"] = max_sim.detach()
            #     if self.include_self:
            #         self.debug_states["self_sim"] = ss.detach()

            # apply dropout
            if self.drop_sim > 0 and self.training:
                sim = sim - (torch.rand_like(sim) < self.drop_sim) * 1e12

        with record_function("Sparse Softmax"):
            # softmax
            sim = sim.float().exp()
            if self.include_self:
                self_sim = (self_sim - max_sim).float().exp()
                weights_sum = self_sim.clone()
            else:
                weights_sum = torch.full([Q, H], 1e-8, device=device, dtype=torch.float32)
            weights_sum = weights_sum.index_add_(0, dst_ids, sim)
            weights = sim / weights_sum[dst_ids]

        # if self.training and (weights.isnan().any() or weights.isinf().any()):
        #     raise ValueError("Invalid value in weights")

        with record_function("Sparse Matmul"):
            # matmul
            v_expand = v_expand * weights.unsqueeze_(-1)
            v_expand = rearrange(v_expand, "n h d -> n (h d)")
            if self.include_self:
                out = self_v * (self_sim / weights_sum).unsqueeze_(-1)
                out = rearrange(out, "n h d -> n (h d)")       
            else:
                out = torch.zeros(Q, C, device=device, dtype=torch.float32)
            out = out.index_add_(0, dst_ids, v_expand)
        
        if self.debug:
            self.debug_states["out_norm"] = out.norm(dim=-1).mean().item()

        # out proj
        out = self.proj_o(out)

        if self.debug:
            self.debug_states["out_proj_norm"] = out.norm(dim=-1).mean().item()

        # drop path
        if self.drop_path > 0 and self.training:
            mask = torch.rand_like(out[..., [0]]) > self.drop_path
            out = out * mask
        
        # residual and norm
        if self.residual:
            out = out + residual
        if self.norm_loc == "post":
            out = self.norm(out)
        
        return out, weights