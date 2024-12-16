
import torch
import torch.utils
import torch.utils.data

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # False at runtime
    import taichi as ti
    tensor = ti.types.ndarray()
else:
    # only initialize in main process, else use a temporary placeholder
    if torch.utils.data.get_worker_info() is None:
        import taichi as ti
        tensor = ti.types.ndarray()
    else:
        from argparse import Namespace
        ti = Namespace()
        tensor = torch.Tensor


def ti_sync():
    ti.sync()


@ti.kernel
def taichi_sparse_attn(
    query: tensor,      # Q x H x C
    key: tensor,        # K x H x C
    value: tensor,      # K x H x C
    asso: tensor,       # N x 2
    bias: tensor,       # N x H
    out: tensor,        # Q x H x C
    tmp_sim: tensor,    # N x H
    tmp_sum: tensor,    # Q x H
    tmp_max: tensor,    # Q x H
    N: ti.i32,
    A: ti.i32,
    H: ti.i32,
    C: ti.i32,
    scale: ti.f32,
):
    # calculate similarity and headwise max
    for n, h in ti.ndrange(N, H):
        k, q = asso[n, 0], asso[n, 1]
        sim = bias[n, h]
        for c in range(C):
            sim += query[q, h, c] * key[k, h, c]
        ti.atomic_max(tmp_max[q, h], sim)
        tmp_sim[n, h] = sim
    # compute softmax sum
    for n, h in ti.ndrange(N, H):
        q = asso[n, 1]
        sim = (tmp_sim[n, h] - tmp_max[q, h]) * scale
        sim = ti.exp(sim)
        tmp_sim[n, h] = sim
        ti.atomic_add(tmp_sum[q, h], sim)
    # weight output
    for n, h in ti.ndrange(N, H):
        k, q = asso[n, 0], asso[n, 1]
        w = tmp_sim[n, h] / tmp_sum[q, h]
        for c in range(C):
            ti.atomic_add(out[q, h, c], w * value[k, h, c])


@ti.kernel
def taichi_sparse_attn_nobias(
    query: tensor,      # Q x H x C
    key: tensor,        # K x H x C
    value: tensor,      # K x H x C
    asso: tensor,       # N x 2
    out: tensor,        # Q x H x C
    tmp_sim: tensor,    # N x H
    tmp_sum: tensor,    # Q x H
    tmp_max: tensor,    # Q x H
    N: ti.i32,
    A: ti.i32,
    H: ti.i32,
    C: ti.i32,
    scale: ti.f32,
):
    # calculate similarity and headwise max
    for n, h in ti.ndrange(N, H):
        k, q = asso[n, 0], asso[n, 1]
        sim = 0
        for c in range(C):
            sim += query[q, h, c] * key[k, h, c]
        ti.atomic_max(tmp_max[q, h], sim)
        tmp_sim[n, h] = sim
    # compute softmax sum
    for n, h in ti.ndrange(N, H):
        q = asso[n, 1]
        sim = (tmp_sim[n, h] - tmp_max[q, h]) * scale
        sim = ti.exp(sim)
        tmp_sim[n, h] = sim
        ti.atomic_add(tmp_sum[q, h], sim)
    # weight output
    for n, h in ti.ndrange(N, H):
        k, q = asso[n, 0], asso[n, 1]
        w = tmp_sim[n, h] / tmp_sum[q, h]
        for c in range(C):
            ti.atomic_add(out[q, h, c], w * value[k, h, c])


TI_INITIALIZED = False

def ensure_taichi_initialized():
    if not TI_INITIALIZED:
        ti.init(arch=ti.gpu)