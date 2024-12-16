import warp as wp
import torch


wp.init()
wp.set_module_options(dict(fast_math=True))


wp_f32_arr2d = wp.array2d(dtype=wp.float32)
wp_f32_arr3d = wp.array3d(dtype=wp.float32)
wp_i32_arr1d = wp.array(dtype=wp.int32)
wp_i32_arr2d = wp.array2d(dtype=wp.int32)
A = wp.constant(8)


# @wp.kernel
# def warp_indexed_dot_fixasso(
#     in_pt_feat: wp_f32_arr2d,           # [PT x H * C]
#     in_px_feat: wp_f32_arr2d,           # [PX x H * C]
#     in_asso: wp_i32_arr1d,              # [PT x A]
#     in_bias: wp_f32_arr3d,              # [PT x A x H]
#     scale: wp.float32,
#     num_channels: wp.int32,
#     num_heads: wp.int32,
#     is_pt2px: wp.bool,
#     out_sim: wp_f32_arr3d,              # [PT x A x H]
#     out_max: wp_f32_arr2d,              # [PX x H] if IS_PT2PX else [PT x H]
# ):
#     """
#     Launch with shape [PT * A x H], each thread handel one association
#     """
#     tid, h = wp.tid()
#     pt_id = tid // A
#     asso_id = tid % A
#     px_id = in_asso[tid]
#     if is_pt2px:
#         out_id = px_id
#     else:
#         out_id = pt_id
#     sim = in_bias[pt_id, asso_id, h]
#     for c in range(h * num_channels, (h + 1) * num_channels):
#         sim += in_pt_feat[pt_id, c] * in_px_feat[px_id, c]
#     sim = sim * scale
#     out_sim[pt_id, asso_id, h] = sim
#     wp.atomic_max(out_max, out_id, h, sim)


# @wp.kernel
# def warp_indexed_sum_fixasso(
#     in_asso: wp_i32_arr1d,        # [PT x A]
#     in_max: wp_f32_arr2d,         # [PX x H] if IS_PT2PX else [PT x H]
#     inout_sim: wp_f32_arr3d,      # [PT x A x H]
#     num_heads: wp.int32,
#     is_pt2px: wp.bool,
#     out_sum: wp_f32_arr2d,        # [PX x H] if IS_PT2PX else [PT x H]
# ):
#     """
#     Launch with shape [PT * A x H], each thread handel one association
#     """
#     tid, h = wp.tid()
#     pt_id = tid // A
#     asso_id = tid % A
#     px_id = in_asso[tid]
#     if is_pt2px:
#         out_id = px_id
#     else:
#         out_id = pt_id
#     sim = inout_sim[pt_id, asso_id, h] - in_max[out_id, h]
#     sim = wp.exp(sim)
#     inout_sim[pt_id, asso_id, h] = sim
#     wp.atomic_add(out_sum, out_id, h, sim)


# @wp.kernel
# def warp_indexed_weight_fixasso(
#     inout_sim: wp_f32_arr3d,       # [PT x A x H]
#     in_sum: wp_f32_arr2d,          # [PX x H] if IS_PT2PX else [PT x H]
#     in_asso: wp_i32_arr1d,         # [PT x A]
#     in_value: wp_f32_arr2d,        # [PT x H * C] if IS_PT2PX else [PX x H * C]
#     num_channels: wp.int32,
#     num_heads: wp.int32,
#     is_pt2px: wp.bool,
#     out_feat: wp_f32_arr2d,        # [PT x H * C] if IS_PT2PX else [PX x H * C]
# ):
#     """
#     Launch with shape [PT * A x H], each thread handel one association
#     """
#     tid, h = wp.tid()
#     pt_id = tid // A
#     asso_id = tid % A
#     px_id = in_asso[tid]
#     if is_pt2px:
#         src_id, dst_id = pt_id, px_id
#     else:
#         src_id, dst_id = px_id, pt_id
#     weight = inout_sim[pt_id, asso_id, h] / in_sum[dst_id, h]
#     inout_sim[pt_id, asso_id, h] = weight
#     for c in range(h * num_channels, (h + 1) * num_channels):
#         val = in_value[src_id, c] * weight
#         wp.atomic_add(out_feat, dst_id, c, val)


# def warp_sparse_attention_fixasso(
#     query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
#     asso: torch.Tensor, bias: torch.Tensor = None, scale: float = 1,
#     num_channels: int = -1, num_heads: int = -1, num_asso: int = -1,
# ):
#     # do checking
#     assert num_asso == A
#     assert query.device.type == 'cuda'

#     # basic settings values / feats
#     has_bias = bias is not None
#     is_pt2px = query.shape[0] != (asso.shape[0] // num_asso)
#     device = f"cuda:{query.device.index}"
#     pt_feat = key if is_pt2px else query
#     px_feat = query if is_pt2px else key
#     num_pt = pt_feat.shape[0]
#     num_px = px_feat.shape[0]
#     num_out = num_px if is_pt2px else num_pt
#     in_asso = asso[:, 1] if is_pt2px else asso[:, 0]
#     in_asso = in_asso.to(torch.int32).contiguous()

#     # allocate temp and output tensors
#     out_weights = torch.zeros([num_pt, num_asso, num_heads], device=device)
#     out_feats = torch.zeros_like(query)
#     tmp_max = torch.zeros([num_out, num_heads], device=device)
#     tmp_sum = torch.zeros([num_out, num_heads], device=device)

#     # convert to warp in/out
#     warp_pt_feat = wp.from_torch(pt_feat)
#     warp_px_feat = wp.from_torch(px_feat)
#     warp_value_feats = wp.from_torch(value)
#     warp_out_feats = wp.from_torch(out_feats)
#     warp_out_weights = wp.from_torch(out_weights)
#     warp_tmp_max = wp.from_torch(tmp_max)
#     warp_tmp_sum = wp.from_torch(tmp_sum)
#     warp_asso = wp.from_torch(in_asso)
#     warp_bias = warp_out_weights if not has_bias else wp.from_torch(
#         bias.reshape(num_pt, num_asso, num_heads))

#     # launch kernels
#     wp.launch(
#         kernel=warp_indexed_dot_fixasso,
#         dim=[num_pt * num_asso, num_heads],
#         inputs=[warp_pt_feat, warp_px_feat, warp_asso, warp_bias, scale, num_channels, num_heads, is_pt2px],
#         outputs=[warp_out_weights, warp_tmp_max],
#         device=device,
#     )
#     wp.launch(
#         kernel=warp_indexed_sum_fixasso,
#         dim=[num_pt * num_asso, num_heads],
#         inputs=[warp_asso, warp_tmp_max, warp_out_weights, num_heads, is_pt2px],
#         outputs=[warp_tmp_sum],
#         device=device,
#     )
#     wp.launch(
#         kernel=warp_indexed_weight_fixasso,
#         dim=[num_pt * num_asso, num_heads],
#         inputs=[warp_out_weights, warp_tmp_sum, warp_asso, warp_value_feats, num_channels, num_heads, is_pt2px],
#         outputs=[warp_out_feats],
#         device=device,
#     )

#     # sync and return
#     wp.synchronize_device(device)
#     return out_feats, out_weights

warp_lds_dot_native = """
    __shared__ float groupSum[256];
    
    if (is_major) groupSum[group] = 0;
    __syncthreads(); // sync 256 threads
    atomicAdd(&groupSum[group], ((float *)v1)[p1] * ((float *)v2)[p2]);
    __syncthreads(); // sync 256 threads
    
    return groupSum[group];
"""
@wp.func_native(snippet=warp_lds_dot_native)
def warp_lds_dot(
    v1: wp_f32_arr2d,
    v2: wp_f32_arr2d,
    p1: wp.int32,
    p2: wp.int32,
    group: wp.int32,
    is_major: wp.bool,
) -> wp.float32:
    ...


@wp.kernel
def warp_indexed_dot_serialized(
    in_q_feat: wp_f32_arr2d,            # [Q x H * C]
    in_k_feat: wp_f32_arr2d,            # [K x H * C]
    in_asso: wp_i32_arr2d,              # [N x 2]
    in_bias: wp_f32_arr2d,              # [N x H]
    scale: wp.float32,
    num_heads: wp.int32,
    num_channels: wp.int32,
    out_sim: wp_f32_arr2d,              # [N x H]
    tmp_max_sum: wp_f32_arr3d,          # [2 x Q x H]
):
    """
    Launch with shape [N * H * C], each thread handel one association
    """
    tid = wp.tid()
    feat_size = num_heads * num_channels
    feat_offset = tid % feat_size
    group_id = tid // num_channels
    h = group_id % num_heads
    a = tid // feat_size
    c = tid % num_channels
    maxgroup = 256 // num_channels
    group = group_id % maxgroup
    is_major = c == 0
    src_id, dst_id = in_asso[a, 0], in_asso[a, 1]
    src_p = src_id * feat_size + feat_offset
    dst_p = dst_id * feat_size + feat_offset
    sim = warp_lds_dot(in_q_feat, in_k_feat, dst_p, src_p, group, is_major)
    if is_major:
        sim += in_bias[a, h]
        sim = sim * scale
        out_sim[a, h] = sim
        wp.atomic_max(tmp_max_sum, 0, dst_id, h, sim)


@wp.kernel
def warp_indexed_dot(
    in_q_feat: wp_f32_arr2d,            # [Q x H * C]
    in_k_feat: wp_f32_arr2d,            # [K x H * C]
    in_asso: wp_i32_arr2d,              # [N x 2]
    in_bias: wp_f32_arr2d,              # [N x H]
    scale: wp.float32,
    num_channels: wp.int32,
    out_sim: wp_f32_arr2d,              # [N x H]
    tmp_max_sum: wp_f32_arr3d,          # [2 x Q x H]
):
    """
    Launch with shape [N x H], each thread handel one association
    """
    tid, h = wp.tid()
    src_id, dst_id = in_asso[tid, 0], in_asso[tid, 1]
    sim = in_bias[tid, h]
    for c in range(h * num_channels, (h + 1) * num_channels):
        sim += in_q_feat[dst_id, c] * in_k_feat[src_id, c]
    sim = sim * scale
    out_sim[tid, h] = sim
    wp.atomic_max(tmp_max_sum, 0, dst_id, h, sim)


@wp.kernel
def warp_indexed_sum(
    in_asso: wp_i32_arr2d,        # [N x 2]
    tmp_max_sum: wp_f32_arr3d,    # [2 x Q x H]
    inout_sim: wp_f32_arr2d,      # [N x H]
):
    """
    Launch with shape [N x H], each thread handel one association
    """
    tid, h = wp.tid()
    dst_id = in_asso[tid, 1]
    sim = inout_sim[tid, h] - tmp_max_sum[0, dst_id, h]
    sim = wp.exp(sim)
    inout_sim[tid, h] = sim
    wp.atomic_add(tmp_max_sum, 1, dst_id, h, sim)


@wp.kernel
def warp_indexed_div(
    inout_sim: wp_f32_arr2d,       # [N x H]
    tmp_max_sum: wp_f32_arr3d,     # [2 x Q x H]
    in_asso: wp_i32_arr2d,         # [N x 2]
):
    """
    Launch with shape [N x H], each thread handel one association
    """
    tid, h = wp.tid()
    dst_id = in_asso[tid, 1]
    weight = inout_sim[tid, h] / tmp_max_sum[1, dst_id, h]
    inout_sim[tid, h] = weight


@wp.kernel
def warp_indexed_weight(
    inout_sim: wp_f32_arr2d,       # [N x H]
    in_asso: wp_i32_arr2d,         # [N x 2]
    in_value: wp_f32_arr2d,        # [V x H * C]
    num_channels: wp.int32,
    out_feat: wp_f32_arr2d,        # [Q x H * C]
):
    """
    Launch with shape [N x C], each thread handel one association
    """
    tid, c = wp.tid()
    h = c // num_channels
    src_id, dst_id = in_asso[tid, 0], in_asso[tid, 1]
    val = in_value[src_id, c] * inout_sim[tid, h]
    wp.atomic_add(out_feat, dst_id, c, val)


def warp_sparse_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    asso: torch.Tensor, bias: torch.Tensor = None, scale: float = 1,
    num_channels: int = -1, num_heads: int = -1, *args, **kwargs,
):
    # do checking
    assert query.device.type == 'cuda'

    # basic settings values / feats
    Q, KV = query.shape[0], key.shape[0]
    N = asso.shape[0]
    has_bias = bias is not None
    device = query.device
    stream = wp.stream_from_torch(torch.cuda.current_stream(device))
    in_asso = asso.to(torch.int32).contiguous()

    # allocate temp and output tensors
    out_weights = torch.zeros([N, num_heads], device=device)
    out_feats = torch.zeros_like(query)
    tmp_max_sum = torch.zeros([2, Q, num_heads], device=device)

    # convert to warp in/out
    warp_q_feats = wp.from_torch(query)
    warp_k_feats = wp.from_torch(key)
    warp_v_feats = wp.from_torch(value)
    warp_out_feats = wp.from_torch(out_feats)
    warp_out_weights = wp.from_torch(out_weights)
    warp_tmp_max_sum = wp.from_torch(tmp_max_sum)
    warp_asso = wp.from_torch(in_asso)
    warp_bias = warp_out_weights if not has_bias else wp.from_torch(bias)

    # launch kernels
    wp.launch(
        kernel=warp_indexed_dot,
        dim=[N, num_heads],
        inputs=[warp_q_feats, warp_k_feats, warp_asso,
                warp_bias, scale, num_channels],
        outputs=[warp_out_weights, warp_tmp_max_sum],
        stream=stream,
    )
    # wp.launch(
    #     kernel=warp_indexed_dot_serialized,
    #     dim=[N * num_heads * num_channels],
    #     inputs=[warp_q_feats, warp_k_feats, warp_asso,
    #             warp_bias, scale, num_heads, num_channels],
    #     outputs=[warp_out_weights, warp_tmp_max_sum],
    #     stream=stream,
    # )
    wp.launch(
        kernel=warp_indexed_sum,
        dim=[N, num_heads],
        inputs=[warp_asso, warp_tmp_max_sum, warp_out_weights],
        outputs=[],
        stream=stream,
    )
    wp.launch(
        kernel=warp_indexed_div,
        dim=[N, num_heads],
        inputs=[warp_out_weights, warp_tmp_max_sum, warp_asso],
        outputs=[],
        stream=stream,
    )
    wp.launch(
        kernel=warp_indexed_weight,
        dim=[N, num_heads * num_channels],
        inputs=[warp_out_weights, warp_asso, warp_v_feats, num_channels],
        outputs=[warp_out_feats],
        stream=stream,
    )

    # sync and return
    # wp.synchronize_device(device)
    return out_feats, out_weights
