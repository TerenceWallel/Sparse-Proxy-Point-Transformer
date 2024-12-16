import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange, reduce

from mmcv.ops.points_sampler import get_sampler_cls
from mmcv.ops import knn, furthest_point_sample

from .basic_blocks import SIRENEmbed, TableEmbed3D

from pointcept.utils import comm


class FPSInit:
    def __init__(
        self,
        num_proxies=512,
    ):
        self.num_proxies = num_proxies

    @torch.no_grad()
    @torch.compile()
    def __call__(self, batch_pt_pos):
        pos = batch_pt_pos.unsqueeze(0)
        inds = furthest_point_sample(pos, self.num_proxies)[0]
        px_pos = batch_pt_pos[inds]
        return px_pos, dict()


class FixGridInit:
    def __init__(
        self,
        grid_shape = 8,
    ):
        self.grid_shape = grid_shape
    
    @torch.no_grad()
    @torch.compile()
    def __call__(self, batch_pt_pos):
        n_dim = batch_pt_pos.shape[-1]
        pc_start = batch_pt_pos.min(dim=0, keepdim=True).values
        pc_range = batch_pt_pos.max(dim=0, keepdim=True).values - pc_start
        grid_shape = self.grid_shape
        if isinstance(grid_shape, int):
            grid_shape = [grid_shape] * n_dim
        grid_shape = batch_pt_pos.new_tensor(grid_shape)
        grid_extent = pc_range / grid_shape * 0.5
        px_pos = torch.stack(torch.meshgrid(
            *[torch.arange(s, device=grid_shape.device) for s in grid_shape]
        ), dim=-1).view(-1, n_dim) + 0.5
        px_pos = px_pos * grid_extent * 2 + pc_start
        init_info = dict(
            grid_start=pc_start, grid_range=pc_range,
            grid_extent=grid_extent, grid_shape=grid_shape,
        )
        return px_pos, init_info


class FixSquareInit:
    def __init__(
        self,
        pc_start: list,
        pc_range: list,
        grid_shape: list
    ):
        self.grid_start = pc_start
        self.grid_range = pc_range
        self.grid_shape = grid_shape
        self.dim = len(grid_shape)
        grid = torch.meshgrid(*[torch.arange(n) for n in grid_shape], indexing='ij')
        grid = torch.stack(grid, dim=-1).reshape(-1, len(grid_shape))
        grid = (grid + 0.5) / torch.tensor(grid_shape)
        grid = torch.tensor(pc_start) + grid * torch.tensor(pc_range)
        self.grid = grid
        self.grid_extent = [0.5 * pc_range[i] / grid_shape[i] for i in range(self.dim)]

    @torch.no_grad()
    @torch.compile()
    def __call__(self, batch_pt_pos: torch.Tensor):
        n_dim = batch_pt_pos.shape[-1]
        assert n_dim == self.dim
        px_pos = self.grid.to(batch_pt_pos.device)
        init_info = dict(
            grid_start=batch_pt_pos.new_tensor([self.grid_start]), 
            grid_range=batch_pt_pos.new_tensor([self.grid_range]),
            grid_extent=batch_pt_pos.new_tensor([self.grid_extent]), 
            grid_shape=batch_pt_pos.new_tensor(self.grid_shape),
        )
        return px_pos, init_info
    
class FixSizeSquareInit:
    def __init__(self, grid_size: float, valid_dims=3):
        self.grid_size = grid_size
        self.valid_dims = valid_dims

    @torch.no_grad()
    @torch.compile()
    def __call__(self, batch_pt_pos: torch.Tensor):
        pc_start = batch_pt_pos.min(dim=0, keepdim=True).values
        pc_range = batch_pt_pos.max(dim=0, keepdim=True).values - pc_start
        n_dim = pc_range.shape[-1]
        if n_dim > self.valid_dims:
            remaining_vals = pc_start[:, self.valid_dims:] + pc_range[:, self.valid_dims:] * 0.5
            pc_start = pc_start[:, :self.valid_dims]
            pc_range = pc_range[:, :self.valid_dims]
            n_dim = self.valid_dims
        else:
            remaining_vals = None
        if isinstance(self.grid_size, (float, int)):
            grid_extent = self.grid_size / 2
        else:
            grid_extent = pc_start.new_tensor(self.grid_size) / 2
        grid_shape = torch.ceil(pc_range / grid_extent * 0.5).clamp(min=1).long()[0]
        # initialize dense proxy pos
        px_pos = torch.stack(torch.meshgrid(
            *[torch.arange(s, device=grid_shape.device) for s in grid_shape]
        ), dim=-1).view(-1, n_dim) + 0.5 - grid_shape.unsqueeze(0) / 2
        px_pos = px_pos * grid_extent * 2 + (pc_start + pc_range * 0.5)
        if remaining_vals is not None:
            remaining_vals = repeat(remaining_vals, "1 d -> n d", n=px_pos.shape[0])
            px_pos = torch.cat([px_pos, remaining_vals], dim=-1)
        init_info = dict(
            grid_start=px_pos[0] - grid_extent, grid_range=grid_extent*grid_shape*2,
            grid_extent=grid_extent, grid_shape=grid_shape,
        )
        return px_pos, init_info


class SquareInit:
    def __init__(
        self,
        target_proxy_range=[128, 512],
        proxy_search_range=[0.0, 1.0],
        max_search_iter=10,
        valid_dims=3,
    ):
        if isinstance(target_proxy_range, int):
            target_proxy_range = [target_proxy_range, target_proxy_range]
        self.target_proxy_range = target_proxy_range
        self.proxy_search_range = proxy_search_range
        self.max_search_iter = max_search_iter
        self.valid_dims = valid_dims

    @torch.no_grad()
    @torch.compile()
    def __call__(self, batch_pt_pos):
        pc_start = batch_pt_pos.min(dim=0, keepdim=True).values
        pc_range = batch_pt_pos.max(dim=0, keepdim=True).values - pc_start
        n_dim = pc_range.shape[-1]
        if n_dim > self.valid_dims:
            remaining_vals = pc_start[:, self.valid_dims:] + pc_range[:, self.valid_dims:] * 0.5
            pc_start = pc_start[:, :self.valid_dims]
            pc_range = pc_range[:, :self.valid_dims]
            n_dim = self.valid_dims
        else:
            remaining_vals = None
        # binary search for grid extent (half size)
        l, r = self.proxy_search_range
        grid_extent = (l + r) / 2
        grid_shape = None
        for _ in range(self.max_search_iter):
            grid_shape = torch.ceil(pc_range / grid_extent * 0.5).clamp(min=1).long()[0]
            grid_cnt = grid_shape.prod().item()
            if self.target_proxy_range[0] <= grid_cnt <= self.target_proxy_range[1]:
                break
            elif grid_cnt < self.target_proxy_range[0]:
                r = grid_extent
            elif grid_cnt > self.target_proxy_range[1]:
                l = grid_extent
            grid_extent = (l + r) / 2
        # initialize dense proxy pos
        px_pos = torch.stack(torch.meshgrid(
            *[torch.arange(s, device=grid_shape.device) for s in grid_shape]
        ), dim=-1).view(-1, n_dim) + 0.5 - grid_shape.unsqueeze(0) / 2
        px_pos = px_pos * grid_extent * 2 + (pc_start + pc_range * 0.5)
        if remaining_vals is not None:
            remaining_vals = repeat(remaining_vals, "1 d -> n d", n=px_pos.shape[0])
            px_pos = torch.cat([px_pos, remaining_vals], dim=-1)
        init_info = dict(
            grid_start=px_pos[0] - grid_extent, grid_range=grid_extent*grid_shape*2,
            grid_extent=grid_extent, grid_shape=grid_shape,
        )
        return px_pos, init_info


class SparseSquareInit:
    def __init__(
        self,
        target_proxy_range=[128, 512],
        proxy_search_range=[0.0, 1.0],
        max_search_iter=10
    ):
        self.target_proxy_range = target_proxy_range
        self.proxy_search_range = proxy_search_range
        self.max_search_iter = max_search_iter
        
    def try_generate(self, batch_pt_pos, grid_start, grid_range, grid_shape):
        voxel_coord = torch.floor((batch_pt_pos - grid_start) / grid_range * grid_shape).long()
        voxel_ind = 0
        for s, c in zip(grid_shape, voxel_coord.unbind(-1)):
            voxel_ind = voxel_ind * s + c
        grid_ind = torch.unique(voxel_ind)
        return grid_ind

    @torch.no_grad()
    @torch.compile()
    def __call__(self, batch_pt_pos):
        device = batch_pt_pos.device
        pc_start = batch_pt_pos.min(dim=0, keepdim=True).values
        pc_range = batch_pt_pos.max(dim=0, keepdim=True).values - pc_start
        # binary search for grid extent (half size)
        l, r = self.proxy_search_range
        grid_extent = (l + r) / 2
        grid_shape = None
        for _ in range(self.max_search_iter):
            grid_shape = torch.ceil(pc_range / grid_extent * 0.5)[0].long()
            grid_range = grid_shape * grid_extent * 2
            grid_start = pc_start + pc_range * 0.5 - grid_range * 0.5
            grid_ind = self.try_generate(batch_pt_pos, grid_start, grid_range, grid_shape)
            grid_cnt = grid_ind.shape[0]
            if self.target_proxy_range[0] <= grid_cnt <= self.target_proxy_range[1]:
                break
            elif grid_cnt < self.target_proxy_range[0]:
                r = grid_extent
            elif grid_cnt > self.target_proxy_range[1]:
                l = grid_extent
            grid_extent = (l + r) / 2
        # generate grid coords
        curr_ind = grid_ind
        grid_coords = []
        for s in grid_shape.tolist()[::-1]:
            grid_coords.append(curr_ind % s)
            curr_ind = curr_ind // s
        grid_coords = torch.stack(grid_coords[::-1], dim=-1)
        # generate dense 2 sparse id
        dense2sparse = torch.full([int(grid_shape.prod().item())], -1, dtype=torch.long, device=device)
        dense2sparse[grid_ind] = torch.arange(grid_ind.shape[0], dtype=torch.long, device=device)
        # initialize dense proxy pos
        px_pos = (grid_coords + 0.5) * grid_extent * 2 + grid_start
        init_info = dict(
            grid_start=grid_start, grid_range=grid_range,
            grid_extent=grid_extent, grid_shape=grid_shape,
            dense2sparse=dense2sparse,
        )
        return px_pos, init_info


def get_obb2world(points):
    cov = torch.cov(points.T)
    _, _, eigen_vectors = torch.svd(cov.float())
    return eigen_vectors / torch.norm(eigen_vectors, dim=1, keepdim=True, p=2)


class OBBSparseSquareInit:
    def __init__(
        self,
        target_proxy_range=[128, 512],
        proxy_search_range=[0.0, 1.0],
        max_search_iter=10
    ):
        self.target_proxy_range = target_proxy_range
        self.proxy_search_range = proxy_search_range
        self.max_search_iter = max_search_iter

    def try_generate(self, batch_pt_pos, grid_start, grid_range, grid_shape):
        voxel_coord = torch.floor((batch_pt_pos - grid_start) / grid_range * grid_shape).long()
        voxel_ind = 0
        for s, c in zip(grid_shape, voxel_coord.unbind(-1)):
            voxel_ind = voxel_ind * s + c
        grid_ind = torch.unique(voxel_ind)
        return grid_ind

    @torch.no_grad()
    @torch.compile()
    def __call__(self, batch_pt_pos):
        obb2world = get_obb2world(batch_pt_pos[:, :2])
        device = batch_pt_pos.device

        # OBB
        rotation_pt_pos = batch_pt_pos.clone()
        rotation_pt_pos[:, :2] = (obb2world.T @ rotation_pt_pos[:, :2].T).T
        pc_start = rotation_pt_pos.min(dim=0, keepdim=True).values
        pc_range = rotation_pt_pos.max(dim=0, keepdim=True).values - pc_start

        # binary search for grid extent (half size)
        l, r = self.proxy_search_range
        grid_extent = (l + r) / 2
        grid_shape = None
        for _ in range(self.max_search_iter):
            grid_shape = torch.ceil(pc_range / grid_extent * 0.5)[0].long()
            grid_range = grid_shape * grid_extent * 2
            grid_start = pc_start + pc_range * 0.5 - grid_range * 0.5
            grid_ind = self.try_generate(rotation_pt_pos, grid_start, grid_range, grid_shape)
            grid_cnt = grid_ind.shape[0]
            if self.target_proxy_range[0] <= grid_cnt <= self.target_proxy_range[1]:
                break
            elif grid_cnt < self.target_proxy_range[0]:
                r = grid_extent
            elif grid_cnt > self.target_proxy_range[1]:
                l = grid_extent
            grid_extent = (l + r) / 2
        # generate grid coords
        curr_ind = grid_ind
        grid_coords = []
        for s in grid_shape.tolist()[::-1]:
            grid_coords.append(curr_ind % s)
            curr_ind = curr_ind // s
        grid_coords = torch.stack(grid_coords[::-1], dim=-1)
        # generate dense 2 sparse id
        dense2sparse = torch.full([int(grid_shape.prod().item())], -1, dtype=torch.long, device=device)
        dense2sparse[grid_ind] = torch.arange(grid_ind.shape[0], dtype=torch.long, device=device)
        # initialize dense proxy pos
        px_pos = (grid_coords + 0.5) * grid_extent * 2 + grid_start

        rotation_px_pos = px_pos.clone()
        rotation_px_pos[:, :2] = (obb2world @ px_pos[:, :2].T).T

        init_info = dict(
            grid_start=grid_start, grid_range=grid_range,
            grid_extent=grid_extent, grid_shape=grid_shape,
            dense2sparse=dense2sparse, obb2world=obb2world,
            origin_px_pos=px_pos,
        )
        return rotation_px_pos, init_info


class OBBSquareInit:
    def __init__(
        self,
        target_proxy_range=[128, 512],
        proxy_search_range=[0.0, 1.0],
        max_search_iter=10,
        valid_dims=3,
    ):
        if isinstance(target_proxy_range, int):
            target_proxy_range = [target_proxy_range, target_proxy_range]
        self.target_proxy_range = target_proxy_range
        self.proxy_search_range = proxy_search_range
        self.max_search_iter = max_search_iter
        self.valid_dims = valid_dims

    @torch.no_grad()
    # @torch.compile()
    def __call__(self, batch_pt_pos):
        obb2world = get_obb2world(batch_pt_pos[:, :2])
        device = batch_pt_pos.device

        # OBB
        rotation_pt_pos = batch_pt_pos.clone()
        rotation_pt_pos[:, :2] = (obb2world.T @ rotation_pt_pos[:, :2].T).T
        pc_start = rotation_pt_pos.min(dim=0, keepdim=True).values
        pc_range = rotation_pt_pos.max(dim=0, keepdim=True).values - pc_start

        n_dim = pc_range.shape[-1]
        if n_dim > self.valid_dims:
            remaining_vals = pc_start[:, self.valid_dims:] + pc_range[:, self.valid_dims:] * 0.5
            pc_start = pc_start[:, :self.valid_dims]
            pc_range = pc_range[:, :self.valid_dims]
            n_dim = self.valid_dims
        else:
            remaining_vals = None
        # binary search for grid extent (half size)
        l, r = self.proxy_search_range
        grid_extent = (l + r) / 2
        grid_shape = None
        for _ in range(self.max_search_iter):
            grid_shape = torch.ceil(pc_range / grid_extent * 0.5).clamp(min=1).long()[0]
            grid_cnt = grid_shape.prod().item()
            if self.target_proxy_range[0] <= grid_cnt <= self.target_proxy_range[1]:
                break
            elif grid_cnt < self.target_proxy_range[0]:
                r = grid_extent
            elif grid_cnt > self.target_proxy_range[1]:
                l = grid_extent
            grid_extent = (l + r) / 2
        # initialize dense proxy pos
        px_pos_os = torch.stack(torch.meshgrid(
            *[torch.arange(s, device=grid_shape.device) for s in grid_shape]
        ), dim=-1).view(-1, n_dim) + 0.5 - grid_shape.unsqueeze(0) / 2

        px_pos_os = px_pos_os * grid_extent * 2 + (pc_start + pc_range * 0.5)

        px_pos_ws = px_pos_os.clone()
        px_pos_ws[:, :2] = (obb2world @ px_pos_os[:, :2].T).T
        if remaining_vals is not None:
            remaining_vals = repeat(remaining_vals, "1 d -> n d", n=px_pos_ws.shape[0])
            px_pos_ws = torch.cat([px_pos_ws, remaining_vals], dim=-1)

        init_info = dict(
            grid_start=px_pos_os[0] - grid_extent, grid_range=grid_extent * grid_shape * 2,
            grid_extent=grid_extent, grid_shape=grid_shape, origin_px_pos=px_pos_os,
            obb2world=obb2world
        )
        return px_pos_ws, init_info

def get_init(cfg):
    cfg = cfg.copy()
    name = cfg.pop("type")
    if name == "fps":
        return FPSInit(**cfg)
    elif name == "fix grid":
        return FixGridInit(**cfg)
    elif name == "fix square":
        return FixSquareInit(**cfg)
    elif name == "fix size":
        return FixSizeSquareInit(**cfg)
    elif name == "square":
        return SquareInit(**cfg)
    elif name == "sparse square":
        return SparseSquareInit(**cfg)
    elif name == "OBB square":
        return OBBSquareInit(**cfg)
    elif name == "OBB sparse square":
        return OBBSparseSquareInit(**cfg)
    else:
        raise NotImplementedError(f"Unknown initializer type {name}")


class KNNAssociate:
    def __init__(self, num_associate, *args, **kwargs):
        self.num_associate = num_associate

    @torch.no_grad()
    def __call__(self, info):
        px_pos = info["proxy_pos"]  # S x D
        pt_pos = info["point_pos"]  # P x D
        px_offset = info["proxy_offset"]
        pt_offset = info["point_offset"]
        device = pt_pos.device
        S = px_pos.shape[0]
        P = pt_pos.shape[0]
        A = self.num_associate
        # perform knn
        pt_ids = torch.arange(P, device=device).long() + pt_offset
        pt_ids = repeat(pt_ids, "n -> (n a)", a=A)
        px_ids = knn(A, px_pos.unsqueeze(0), pt_pos.unsqueeze(0))
        px_ids = px_ids[0].long() + px_offset
        px_ids = rearrange(px_ids, "a n -> (n a)")
        # merge
        return torch.stack([pt_ids, px_ids], dim=-1)


class FastGridAssociate:
    """associate nearest grid corners"""

    def __init__(self, dim=3, *args, **kwargs):
        self.dim = dim
        self.num_variants = 2 ** dim
        # create weights
        base_weights = []
        for i in range(2 ** dim):
            w = []
            for _ in range(dim):
                w.append(i % 2)
                i = i >> 1
            base_weights.append(w)
        # lower and upper bound weights
        self.weights = [base_weights + base_weights[::-1]]

    @torch.no_grad()
    def __call__(self, info):
        pt_pos = info["point_pos"]
        px_offset = info["proxy_offset"]
        pt_offset = info["point_offset"]
        grid_shape = info["grid_shape"].long()
        grid_range = info["grid_range"]
        grid_start = info["grid_start"]
        grid_extent = info["grid_extent"]
        P, D = pt_pos.shape
        device = pt_pos.device
        # to voxel coord [P x D]
        pt_coord = (pt_pos - grid_start + grid_extent)[..., :self.dim] / grid_range[..., :self.dim]
        pt_coord = pt_coord.clamp(min=1e-6, max=1-1e-6) * (grid_shape[:self.dim] - 1)
        lc = torch.floor(pt_coord).unsqueeze_(1).to(torch.int16)        # P x 1 x D
        uc = torch.ceil(pt_coord).unsqueeze_(1).to(torch.int16)         # P x 1 x D
        # expand to generate 2^D coords
        weights = pt_pos.new_tensor(self.weights, dtype=torch.int16)    # 1 x 2^D x D
        coords = lc * weights[:, :self.num_variants] + uc * weights[:, self.num_variants:]
        # to dense grid index [P x 2^D]
        ind = 0
        for i in range(self.dim):
            ind *= grid_shape[i]
            ind = ind + coords[..., i]
        # generate asso
        pt_ids = torch.arange(P, device=device, dtype=torch.long)
        pt_ids = pt_ids if pt_offset == 0 else pt_ids + pt_offset
        pt_ids = repeat(pt_ids, "n -> (n a)", a=self.num_variants)
        px_ids = rearrange(ind.long(), "n a -> (n a)")
        # handle sparse case
        if "dense2sparse" in info:
            dense2sparse = info["dense2sparse"]
            px_ids = dense2sparse[px_ids]
            mask = px_ids >= 0
            px_ids = px_ids[mask]
            pt_ids = pt_ids[mask]
        # merge
        return torch.stack([pt_ids, px_ids + px_offset], dim=-1)

class OBBFastGridAssociate:
    """associate nearest grid corners"""

    def __init__(self, dim=3, *args, **kwargs):
        self.dim = dim
        self.num_variants = 2 ** dim
        # create weights
        base_weights = []
        for i in range(2 ** dim):
            w = []
            for _ in range(dim):
                w.append(i % 2)
                i = i >> 1
            base_weights.append(w)
        # lower and upper bound weights
        self.weights = [base_weights + base_weights[::-1]]

    @torch.no_grad()
    def __call__(self, info):
        pt_pos = info["point_pos"]
        px_offset = info["proxy_offset"]
        pt_offset = info["point_offset"]
        grid_shape = info["grid_shape"].long()
        grid_range = info["grid_range"]
        grid_start = info["grid_start"]
        grid_extent = info["grid_extent"]
        rotation = info["obb2world"]
        
        pt_pos_os = pt_pos.clone()
        pt_pos_os[:, :2] = (rotation.T @ pt_pos_os[:, :2].T).T
        
        P, D = pt_pos_os.shape
        device = pt_pos_os.device
        # to voxel coord [P x D]
        pt_coord = (pt_pos_os - grid_start + grid_extent)[..., :self.dim] / grid_range[..., :self.dim]
        pt_coord = pt_coord.clamp(min=1e-6, max=1-1e-6) * (grid_shape[:self.dim] - 1)
        lc = torch.floor(pt_coord).unsqueeze_(1).to(torch.int16)        # P x 1 x D
        uc = torch.ceil(pt_coord).unsqueeze_(1).to(torch.int16)         # P x 1 x D
        # expand to generate 2^D coords
        weights = pt_pos_os.new_tensor(self.weights, dtype=torch.int16)    # 1 x 2^D x D
        coords = lc * weights[:, :self.num_variants] + uc * weights[:, self.num_variants:]
        # to dense grid index [P x 2^D]
        ind = 0
        for i in range(self.dim):
            ind *= grid_shape[i]
            ind = ind + coords[..., i]
        # generate asso
        pt_ids = torch.arange(P, device=device, dtype=torch.long)
        pt_ids = pt_ids if pt_offset == 0 else pt_ids + pt_offset
        pt_ids = repeat(pt_ids, "n -> (n a)", a=self.num_variants)
        px_ids = rearrange(ind.long(), "n a -> (n a)")
        # handle sparse case
        if "dense2sparse" in info:
            dense2sparse = info["dense2sparse"]
            px_ids = dense2sparse[px_ids]
            mask = px_ids >= 0
            px_ids = px_ids[mask]
            pt_ids = pt_ids[mask]
        # merge
        return torch.stack([pt_ids, px_ids + px_offset], dim=-1)

def get_asso(cfg):
    name = cfg["type"]
    if name == "knn":
        return KNNAssociate(**cfg)
    elif name == "fgrid":
        return FastGridAssociate(**cfg)
    elif name == "obbfgrid":
        return OBBFastGridAssociate(**cfg)


class ProxyInitializer(nn.Module):
    GLB_STATES = {}

    def __init__(
        self,
        d_embed,
        n_dim,
        init_cfg=dict(type="square", target_st_range=[128, 512]),
        asso_cfg=dict(type="knn", num_associate=8),
        pe_cfg=dict(num_layers=2, temperature=1),
        rel_bias_cfg=None,
        rpe_cfg=None,
        reuse_proxy_feat=False,
        reuse_rel_bias=False,
        reuse_rpe=False,
        lvl_wise_reuse=False,
        mask_empty=False,
        d_last=None,
        block_idx=-1,
        layer_idx=-1,
        level_key=None,
        **kwargs
    ):
        super().__init__()
        assert not lvl_wise_reuse or level_key is not None
        self.d_embed = d_embed
        self.is_first_layer = layer_idx == 0
        self.is_first_block = block_idx == 0
        self.is_first = self.is_first_block and self.is_first_layer
        self.reuse_proxy_feat = reuse_proxy_feat
        self.mask_empty = mask_empty
        self.lvl_wise_reuse = lvl_wise_reuse and level_key[0] == "d"
        self.level_key = level_key
        # main modules
        self.init = get_init(init_cfg)
        self.asso = get_asso(asso_cfg)
        # pos embed
        self.need_pe = not reuse_proxy_feat or self.is_first_block
        if self.need_pe:
            self.pe = SIRENEmbed(d_embed, n_dim, d_embed, **pe_cfg)
        # rel bias (if needed)
        self.need_rel_bias = rel_bias_cfg is not None
        self.need_rel_bias = self.need_rel_bias and (not reuse_rel_bias or self.is_first_block)
        if self.need_rel_bias:
            self.rel_bias = TableEmbed3D(**rel_bias_cfg)
        # rpe (if needed)
        self.need_rpe = rpe_cfg is not None
        self.need_rpe = self.need_rpe and (not reuse_rpe or self.is_first_block)
        if self.need_rpe:
            self.rpe = TableEmbed3D(**rpe_cfg)
        # layer-wise caching
        self.need_upsample = reuse_proxy_feat and not self.is_first_layer and self.is_first_block and d_last is not None
        if self.need_upsample:
            self.upsample_feat = nn.Sequential(
                nn.Linear(d_last, d_embed),
                nn.GELU(),
                nn.BatchNorm1d(d_embed),
            )

    def clr_glb_state(self):
        ProxyInitializer.GLB_STATES = {}

    def store_glb_state(self, key, val):
        key = f"rank{comm.get_rank()}/{key}"
        ProxyInitializer.GLB_STATES[key] = val

    def get_glb_state(self, key, default=None):
        key = f"rank{comm.get_rank()}/{key}"
        if default is None and key not in ProxyInitializer.GLB_STATES:
            existing_keys = list(ProxyInitializer.GLB_STATES.keys())
            raise ValueError(f"given key [{key}] is not in GLB_STATES with keys [{existing_keys}]")
        return ProxyInitializer.GLB_STATES.get(key, default)

    def gen_px_rpe(self, proxy_positions, batch_num_px):
        maxlen = max(batch_num_px)
        rpes = []
        start = 0
        for n in batch_num_px:
            end = start + n
            pad = maxlen - n
            batch_pos = proxy_positions[start:end]
            batch_rel_pos = batch_pos.unsqueeze(0) - batch_pos.unsqueeze(1)  # N x N x D
            batch_rel_pos = rearrange(batch_rel_pos, "n1 n2 d -> (n1 n2) d")
            batch_rpe = self.rpe(batch_rel_pos)
            batch_rpe = rearrange(batch_rpe, "(n1 n2) h -> 1 h n1 n2", n1=n, n2=n)
            batch_rpe = F.pad(batch_rpe, (0, pad, 0, pad), "constant", 0)
            rpes.append(batch_rpe.squeeze_(0))
            start = end
        return torch.cat(rpes, dim=0) * math.sqrt(self.d_embed)

    def forward(self, point_positions, point_offsets):
        # use cached data by default
        if not self.is_first_block:
            info = self.get_glb_state("proxy_info")
            if self.need_pe:
                proxy_feats = self.pe(info["proxy_pos"])
                info["proxy_feat"] = proxy_feats
            elif self.reuse_proxy_feat:
                proxy_feats = self.get_glb_state("proxy_feat")
                info["proxy_feat"] = proxy_feats
            if self.need_rel_bias:
                rel_bias = self.rel_bias(info["rel_pos"])
                info["rel_bias"] = rel_bias * math.sqrt(self.d_embed)
        # need to recompute something
        else:
            # init proxy positions (skip if possible)
            batch_num_px = []
            batch_init_info = []
            proxy_positions = None
            if not self.reuse_proxy_feat or self.is_first:
                batch_px_pos = []
                pt_start = 0
                for b, pt_end in enumerate(point_offsets):
                    pt_pos = point_positions[pt_start:pt_end]
                    px_pos, init_info = self.init(pt_pos)
                    num_px = px_pos.shape[0]
                    batch_num_px.append(num_px)
                    batch_px_pos.append(px_pos)
                    batch_init_info.append(init_info)
                proxy_positions = torch.cat(batch_px_pos, dim=0)
            else:
                info = self.get_glb_state("proxy_info")
                batch_num_px = info["batch_num_px"]
                batch_init_info = info["batch_init_info"]
                proxy_positions = info["proxy_pos"]
            # associate (cannot skip because the points has changed)
            if self.lvl_wise_reuse and self.level_key[0] == "d":
                asso = self.get_glb_state(f"asso_{self.level_key[1:]}")
                rel_pos = self.get_glb_state(f"rel_pos_{self.level_key[1:]}", 0)
                batch_px_mask = self.get_glb_state(f"batch_px_mask_{self.level_key[1:]}", [])
            else:
                batch_asso = []
                batch_rel_pos = []
                batch_px_mask = []
                pt_start = 0
                px_start = 0
                for b, pt_end in enumerate(point_offsets):
                    pt_pos = point_positions[pt_start:pt_end]
                    init_info = batch_init_info[b]
                    px_pos = proxy_positions[px_start:px_start+batch_num_px[b]]
                    # build associations
                    asso_info = dict(
                        point_pos=pt_pos, point_offset=pt_start,
                        proxy_pos=px_pos, proxy_offset=px_start,
                    )
                    asso_info.update(batch_init_info[b])
                    asso = self.asso(asso_info)
                    batch_asso.append(asso)
                    # calc relative pos
                    if self.need_rel_bias:
                        pt_pos_expand = pt_pos[asso[:, 0] - pt_start]
                        px_pos_expand = px_pos[asso[:, 1] - px_start]
                        batch_rel_pos.append(px_pos_expand - pt_pos_expand)
                    if self.mask_empty:
                        asso_px_ids = asso[..., 1] - px_start
                        asso_px_ids = asso_px_ids.unique()
                        px_mask = torch.full((batch_num_px[b], ), float("-inf"), device=asso.device)
                        px_mask[asso_px_ids] = 0
                        batch_px_mask.append(px_mask)
                    # record new start offset and write results
                    pt_start = pt_end
                    px_start += batch_num_px[b]
                asso = torch.cat(batch_asso, dim=0)
                self.store_glb_state(f"asso_{self.level_key[1:]}", asso)
                if self.need_rel_bias:
                    rel_pos = torch.cat(batch_rel_pos, dim=0)
                    self.store_glb_state(
                        f"rel_pos_{self.level_key[1:]}", rel_pos)
                if self.mask_empty:
                    self.store_glb_state(
                        f"batch_px_mask_{self.level_key[1:]}", batch_px_mask)
            # compute new proxy features
            proxy_feat = self.pe(proxy_positions)
            if self.need_upsample:
                prev_feat = self.get_glb_state("proxy_feat")
                proxy_feat += self.upsample_feat(prev_feat)
            info = dict(
                asso=asso,
                proxy_feat=proxy_feat,
                proxy_pos=proxy_positions,
                batch_num_px=batch_num_px,
                batch_px_mask=batch_px_mask,
                batch_init_info=batch_init_info,
                rel_pos=rel_pos if self.need_rel_bias else None,
            )
            # add proxy rpe to info if needed
            if self.need_rpe:
                info["proxy_rpe"] = self.gen_px_rpe(
                    proxy_positions, batch_num_px)
            # add relative pos and bias to info if needed
            if self.need_rel_bias:
                if self.lvl_wise_reuse:
                    info["rel_bias"] = self.get_glb_state(
                        f"rel_bias_{self.level_key[1:]}")
                else:
                    info["rel_pos"] = rel_pos
                    info["rel_bias"] = self.rel_bias(
                        rel_pos) * math.sqrt(self.d_embed)
                    self.store_glb_state(
                        f"rel_bias_{self.level_key[1:]}", info["rel_bias"])
        self.store_glb_state("proxy_info", info)
        return info
