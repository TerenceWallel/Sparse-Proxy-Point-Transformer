from torch.profiler import record_function

from functools import partial
from addict import Dict
import math
import torch
import torch.nn as nn
import spconv.pytorch as spconv
import torch_scatter
from timm.models.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pointcept.models.point_prompt_training import PDNorm
from pointcept.models.builder import MODELS
from pointcept.models.utils.misc import offset2bincount
from pointcept.models.utils.structure import Point
from pointcept.models.modules import PointModule, PointSequential
from pointcept.utils import comm


from .sparse_attention import SparseAttention
from .proxy_fuser import get_proxy_fuser
from .proxy_initializer import ProxyInitializer
from .basic_blocks import SIRENEmbed

import os
from torch.utils.checkpoint import checkpoint


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        enable_checkpoint=False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        self.enable_checkpoint = enable_checkpoint
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                        _offset_pad[i + 1]
                        - self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size) : _offset_pad[i + 1]
                        - self.patch_size
                    ]
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def attn_main(self, q, k, v, rel_pos):
        attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
        if self.enable_rpe:
            attn = attn + self.rpe(rel_pos)
        if self.upcast_softmax:
            attn = attn.float()
        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)
        feat = (attn @ v).transpose(1, 2).reshape(-1, self.channels)
        return feat
    
    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            rel_pos = self.get_rel_pos(point, order)
            feat = (
                self.attn_main(q, k, v, rel_pos)
                if not self.enable_checkpoint
                else checkpoint(self.attn_main, q, k, v, rel_pos, use_reentrant=False)
            )
        else:
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
        enable_checkpoint=False
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)
        self.enable_checkpoint = enable_checkpoint

    def forward(self, x):
        def fwd(x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x
        return fwd(x) if not self.enable_checkpoint else checkpoint(fwd, x, use_reentrant=False)


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        enable_checkpoint=False,
        spa_cfg=None,
        block_idx=-1,
        layer_idx=-1,
        level_key="",
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.block_idx = block_idx
        self.layer_idx = layer_idx
        self.with_spa = spa_cfg is not None
        if self.with_spa:
            pe_cfg = spa_cfg["pe_cfg"]
            num_heads = spa_cfg.get("num_heads", 8)
            norm_loc = spa_cfg.get("norm_loc", "pre")
            self.num_heads = num_heads
            # initializer
            init_cfg = spa_cfg["initializer_cfg"]
            init_cfg["d_embed"] = channels
            init_cfg["n_dim"] = 3
            init_cfg["pe_cfg"] = pe_cfg
            init_cfg["block_idx"] = block_idx
            init_cfg["layer_idx"] = layer_idx
            init_cfg["level_key"] = level_key
            self.split_ca_bias = False
            if "rel_bias_cfg" in init_cfg:
                self.split_ca_bias = init_cfg["rel_bias_cfg"].get("split", False)
                init_cfg["rel_bias_cfg"]["d_embed"] = num_heads
                init_cfg["rel_bias_cfg"]["d_embed"] *= 2 if self.split_ca_bias else 1
            if "rpe_cfg" in init_cfg:
                init_cfg["rpe_cfg"]["d_embed"] = num_heads
            self.proxy_initializer = ProxyInitializer(**init_cfg)
            # fuser
            fuser_cfg = spa_cfg["fuser_cfg"]
            fuser_cfg["d_embed"] = channels
            fuser_cfg["num_heads"] = num_heads
            fuser_cfg["dropout"] = attn_drop
            fuser_cfg["norm_loc"] = norm_loc
            if "pe_cfg" in fuser_cfg:
                fuser_cfg["pe_cfg"]["n_dim"] = 3
                fuser_cfg["pe_cfg"]["d_embed"] = channels
                fuser_cfg["pe_cfg"]["d_hidden"] = channels
            if "rpe_cfg" in fuser_cfg:
                fuser_cfg["rpe_cfg"]["d_embed"] = num_heads
            self.proxy_fuser = get_proxy_fuser(fuser_cfg)
            # sparse attn
            ca_cfg = spa_cfg["ca_cfg"]
            ca_cfg["norm_loc"] = norm_loc
            ca_cfg["enable_checkpoint"] = ca_cfg.get("enable_checkpoint", enable_checkpoint)
            self.ca_down = SparseAttention(
                channels, num_heads, attn_drop, drop_path,
                residual=True, **ca_cfg
            )
            self.ca_up = SparseAttention(
                channels, num_heads, attn_drop, drop_path,
                residual=True, **ca_cfg
            )

        self.with_ptv3 = not self.with_spa or spa_cfg.get("with_ptv3", True)
        if self.with_ptv3:
            self.cpe = PointSequential(
                spconv.SubMConv3d(
                    channels,
                    channels,
                    kernel_size=3,
                    bias=True,
                    indice_key=cpe_indice_key,
                ),
                nn.Linear(channels, channels),
                norm_layer(channels),
            )

            self.norm1 = PointSequential(norm_layer(channels))
            self.attn = SerializedAttention(
                channels=channels,
                patch_size=patch_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                order_index=order_index,
                enable_rpe=enable_rpe,
                enable_flash=enable_flash,
                upcast_attention=upcast_attention,
                upcast_softmax=upcast_softmax,
                enable_checkpoint=enable_checkpoint,
            )
            self.norm2 = PointSequential(norm_layer(channels))
            self.mlp = PointSequential(
                MLP(
                    in_channels=channels,
                    hidden_channels=int(channels * mlp_ratio),
                    out_channels=channels,
                    act_layer=act_layer,
                    drop=proj_drop,
                    enable_checkpoint=True,
                )
            )
            self.drop_path = PointSequential(
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

        self.set_debug(False)
        
    def set_debug(self, debug):
        self.debug = debug
        self.debug_states = {}
        if self.with_spa:
            self.ca_down.set_debug(debug)
            self.ca_up.set_debug(debug)
        
    def get_debug_states(self):
        if self.with_spa:
            self.debug_states["ca_down"] = self.ca_down.get_debug_states()
            self.debug_states["ca_up"] = self.ca_up.get_debug_states()
        return self.debug_states

    def forward(self, point: Point, info=None):
        if self.with_spa:
            feat, coord, offset = point.feat, point.coord, point.offset
            with record_function(f"proxy init L{self.layer_idx}B{self.block_idx}"):
                proxy_info = self.proxy_initializer(coord, offset) if info is None else info
            asso = proxy_info["asso"].contiguous()
            bias = proxy_info.get("rel_bias", None)
            if bias is not None and self.split_ca_bias:
                bias = bias[:, :self.num_heads].contiguous()
            q = proxy_info["proxy_feat"]
            k = v = feat
            with record_function(f"pt2px CA L{self.layer_idx}B{self.block_idx}"):
                tokens, weights0 = self.ca_down(q, k, v, asso, bias)
            if self.debug:
                self.debug_states["proxy_info"] = {k: v for k, v in proxy_info.items()}
                self.debug_states["tokens0"] = tokens.detach()
                self.debug_states["weight0"] = weights0.detach()
                self.debug_states["feat0"] = feat.detach()
                self.debug_states["coord"] = coord.detach()
                self.debug_states["offset"] = offset.detach()
                
        if self.with_ptv3:
            with record_function(f"ptv3"):
                shortcut = point.feat
                point = self.cpe(point)
                point.feat = shortcut + point.feat
                shortcut = point.feat
                if self.pre_norm:
                    point = self.norm1(point)
                point = self.drop_path(self.attn(point))
                point.feat = shortcut + point.feat
                if not self.pre_norm:
                    point = self.norm1(point)
                shortcut = point.feat
                if self.pre_norm:
                    point = self.norm2(point)
                point = self.drop_path(self.mlp(point))
                point.feat = shortcut + point.feat
                point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
                if not self.pre_norm:
                    point = self.norm2(point)
            if self.debug:
                self.debug_states["feat1"] = point.feat.detach()
        
        if self.with_spa:
            with record_function(f"proxy fuse L{self.layer_idx}B{self.block_idx}"):
                tokens, fuser_info = self.proxy_fuser(tokens, proxy_info)
            self.proxy_initializer.store_glb_state("proxy_feat", tokens)
            q = point.feat
            k = v = tokens
            asso = asso[..., [1, 0]].contiguous()
            bias = proxy_info.get("rel_bias", None)
            if bias is not None and self.split_ca_bias:
                bias = bias[:, self.num_heads:].contiguous()
            with record_function(f"px2pt CA L{self.layer_idx}B{self.block_idx}"):
                feat, weights1 = self.ca_up(q, k, v, asso, bias)
            if self.debug:
                self.debug_states["feat2"] = feat.detach()
                self.debug_states["tokens1"] = tokens.detach()
                self.debug_states["weight1"] = weights1.detach()
                self.debug_states["fuser_info"] = fuser_info
            point.feat = feat
            point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert stride == 2 ** (math.ceil(stride) - 1).bit_length()  # 2, 4, 8
        # TODO: add support to grid pool (any stride)
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(
            point.keys()
        ), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        code_, cluster, counts = torch.unique(
            code[0],
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


@MODELS.register_module("SPPT")
class SparseProxyPTv3(PointModule):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z_trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(48, 48, 48, 48, 48),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(48, 48, 48, 48),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        enable_checkpoint=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        spa_cfg=None,
        spa_skip_layer=0,
        debug_interval=-1,
        debug_save=True,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.cls_mode = cls_mode
        self.shuffle_orders = shuffle_orders
        
        self.debug = debug_interval > 0
        self.debug_interval = debug_interval
        self.debug_save = debug_save
        self.acc_iter = 0
        self.reports = []
        self.num_reports_keep = 2

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.cls_mode or self.num_stages == len(dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(dec_patch_size) + 1

        # norm layers
        if pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(
                    nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=pdnorm_affine
                ),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=pdnorm_affine),
                conditions=pdnorm_conditions,
                decouple=pdnorm_decouple,
                adaptive=pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        spa_layer_idx = 0
        spa_d_last = None
        if isinstance(spa_skip_layer, int):
            spa_skip_layer = [spa_skip_layer, 1e10]

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            if s >= spa_skip_layer[0] and s <= spa_skip_layer[1]:
                spa_cfg["initializer_cfg"]["d_last"] = spa_d_last
                spa_cfg["num_heads"] = enc_num_head[s]
                spa_d_last = enc_channels[s]
                layer_spa_cfg = spa_cfg
                spa_layer_idx += 1
            else:
                layer_spa_cfg = None
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        enable_checkpoint=enable_checkpoint,
                        spa_cfg=layer_spa_cfg,
                        block_idx=i,
                        layer_idx=spa_layer_idx - 1,
                        level_key=f"e{s}",
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                if s >= spa_skip_layer[0] and s <= spa_skip_layer[1]:
                    spa_cfg["initializer_cfg"]["d_last"] = spa_d_last
                    spa_cfg["num_heads"] = dec_num_head[s]
                    spa_d_last = dec_channels[s]
                    spa_layer_idx += 1
                    layer_spa_cfg = spa_cfg
                else:
                    layer_spa_cfg = None
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            enable_checkpoint=enable_checkpoint,
                            spa_cfg=layer_spa_cfg,
                            block_idx=i,
                            layer_idx=spa_layer_idx - 1,
                            level_key=f"d{s}",
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")
        self.debug_states = {}

    def forward(self, data_dict):
        self.acc_iter += 1
        self.debug_states = {}
        is_vis = int(os.environ.get("SPA_VIS_CNT", "0")) > 0
        debug = self.debug and self.training and comm.is_main_process()
        debug = debug and self.acc_iter % self.debug_interval == 0
        debug = debug or is_vis or os.environ.get("SPA_DEBUG", "0") == "1"
        for layer in self.enc._modules.values():
            for block in layer._modules.values():
                if isinstance(block, Block):
                    block.set_debug(debug)
        for layer in self.dec._modules.values():
            for block in layer._modules.values():
                if isinstance(block, Block):
                    block.set_debug(debug)
        if debug:
            self.debug_states = dict(module=self, input=data_dict.copy())
        
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)
        if not self.cls_mode:
            point = self.dec(point)
        # else:
        #     point.feat = torch_scatter.segment_csr(
        #         src=point.feat,
        #         indptr=nn.functional.pad(point.offset, (1, 0)),
        #         reduce="mean",
        #     )
        
        if debug:
            for lname, layer in self.enc._modules.items():
                l = int(lname[3:])
                for bname, block in layer._modules.items():
                    if isinstance(block, Block):
                        b = int(bname[5:])
                        block_states = block.get_debug_states()
                        if block.with_spa:
                            self.debug_states[f"enc_l{l}_b{b}"] = block_states
                            point[f"stat_enc_l{l}_b{b}_pt0_norm"] = block_states["feat0"].norm(dim=-1).mean().detach()
                            point[f"stat_enc_l{l}_b{b}_px0_norm"] = block_states["tokens0"].norm(dim=-1).mean().detach()
                            point[f"stat_enc_l{l}_b{b}_pt1_norm"] = block_states["feat1"].norm(dim=-1).mean().detach()
                            point[f"stat_enc_l{l}_b{b}_px1_norm"] = block_states["tokens1"].norm(dim=-1).mean().detach()
                            point[f"stat_enc_l{l}_b{b}_pt2_norm"] = block_states["feat2"].norm(dim=-1).mean().detach()
            for lname, layer in self.dec._modules.items():
                l = int(lname[3:])
                for bname, block in layer._modules.items():
                    if isinstance(block, Block):
                        b = int(bname[5:])
                        block_states = block.get_debug_states()
                        if block.with_spa:
                            self.debug_states[f"dec_l{l}_b{b}"] = block_states
                            point[f"stat_dec_l{l}_b{b}_pt0_norm"] = block_states["feat0"].norm(dim=-1).mean().detach()
                            point[f"stat_dec_l{l}_b{b}_px0_norm"] = block_states["tokens0"].norm(dim=-1).mean().detach()
                            point[f"stat_dec_l{l}_b{b}_pt1_norm"] = block_states["feat1"].norm(dim=-1).mean().detach()
                            point[f"stat_dec_l{l}_b{b}_px1_norm"] = block_states["tokens1"].norm(dim=-1).mean().detach()
                            point[f"stat_dec_l{l}_b{b}_pt2_norm"] = block_states["feat2"].norm(dim=-1).mean().detach()
            if not is_vis and self.debug_save:
                path = f"debug_states_{self.acc_iter}iter.pth"
                torch.save(self.debug_states, path)
                self.reports.append(path)
                if len(self.reports) > self.num_reports_keep:
                    rm_path = self.reports.pop(0)
                    os.remove(rm_path)
        return point
