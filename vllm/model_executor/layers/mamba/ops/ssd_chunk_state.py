# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py

# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from .mamba_ssm import softplus


# 已阅
# 说明：计算 dt_out 和 dA_cumsum 的 kernel，
# dt_out 是经过 bias 和 softplus 处理后的 dt，dA_cumsum 是 dt * A 的前缀和（dt 是处理后的）
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_H": 2}),
        triton.Config({"BLOCK_SIZE_H": 4}),
        triton.Config({"BLOCK_SIZE_H": 8}),
        triton.Config({"BLOCK_SIZE_H": 16}),
        triton.Config({"BLOCK_SIZE_H": 32}),
        triton.Config({"BLOCK_SIZE_H": 64}),
    ],
    # 说明：缓存调优结果的 cache key
    key=["chunk_size", "nheads"],
)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    # 说明：dt 的 shape 是 (seqlen, nheads)
    dt_ptr,
    A_ptr,
    # 说明：dt_bias 的 shape 是 (nheads,)
    dt_bias_ptr,
    # 说明：dt_out 的 shape 是 (nheads, nchunks, chunk_size)
    dt_out_ptr,
    # 说明：dA_cumsum 的 shape 是 (nheads, nchunks, chunk_size)
    dA_cumsum_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimension
    seqlen,
    nheads: tl.constexpr,
    chunk_size: tl.constexpr,
    dt_min: tl.constexpr,
    dt_max: tl.constexpr,
    # Strides
    stride_dt_seqlen: tl.int64,
    stride_dt_head: tl.constexpr,
    stride_A_head: tl.constexpr,
    stride_dt_bias_head: tl.constexpr,
    stride_dt_out_head: tl.int64,
    stride_dt_out_chunk: tl.int64,
    stride_dt_out_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
):
    # 说明：chunk 索引
    # if dt is long, may cause problems, so use 64 bit
    # https://github.com/triton-lang/triton/issues/1058
    pid_c = tl.program_id(axis=0).to(tl.int64)
    # 说明：head 索引，每个 block 处理一个 chunk 内 BLOCK_SIZE_H 个 head 的数据
    pid_h = tl.program_id(axis=1)

    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)

    dt_ptr += chunk_seqlen_start * stride_dt_seqlen
    dt_out_ptr += pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (
        offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen
    )
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (
        offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize
    )
    dA_cs_ptrs = dA_cumsum_ptr + (
        offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize
    )
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    dt = tl.load(
        dt_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(
            dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0
        ).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)

    # 说明：保证 dt 非负
    dt = tl.clamp(dt, dt_min, dt_max)
    # 说明：经过了 bias 和 softplus 处理后的 dt，再重新根据 mask 获取有效值
    dt = tl.where(
        (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0
    )
    tl.store(
        dt_out_ptrs,
        dt,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(
        dA_cs_ptrs,
        dA_cs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )


# 已阅
# 说明：计算 Right Factors 的 kernel，输入是 x、b、dt 和 dA_cumsum，输出是 states（每个 chunk 的最终状态）
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            # 说明：num_stages 越大，流水线越充分，内存延迟隐藏得越好，理论性能越高；
            # 但过大的 num_stages 会占用更多的共享内存（SMEM）、寄存器，
            # 导致每个 block 能启动的线程数减少（occupancy 降低），反而可能性能下降。
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
    ],
    key=["hdim", "dstate", "chunk_size"],
)
@triton.jit
def _chunk_state_fwd_kernel(
    # Pointers to matrices
    # 说明：x 的 shape 是 (seqlen, nheads, headdim)
    x_ptr,
    # 说明：b 的 shape 是 (seqlen, ngroups, dstate)
    b_ptr,
    # 说明：states 的 shape 是 (nchunks, nheads, headdim, dstate)
    states_ptr,
    # 说明：dt 的 shape 是 (nheads, nchunks, chunk_size)
    dt_ptr,
    # 说明：dA_cumsum 的 shape 是 (nheads, nchunks, chunk_size)
    dA_cumsum_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimensions
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    seqlen,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_b_dstate: tl.constexpr,
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    stride_dt_head: tl.int64,
    stride_dt_chunk: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # 说明：处理一个 chunk、一个 head 内的 BLOCK_SIZE_M 个 dim 和 BLOCK_SIZE_N 个 state
    # 说明：pid_c 的 chunk 索引
    pid_c = tl.program_id(axis=1).to(tl.int64)
    # 说明：pid_h 是 head 索引
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    # 说明：pid_m 是 dim 索引，对应 BLOCK_SIZE_M 个 dim
    pid_m = tl.program_id(axis=0) // num_pid_n
    # 说明：pid_n 是 state 索引，对应 BLOCK_SIZE_N 个 state
    pid_n = tl.program_id(axis=0) % num_pid_n
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    b_ptr += (
        chunk_seqlen_start * stride_b_seqlen
        # 说明：这里 stride_b_head 应该是 stride_b_group，即 ngroups 维度的 stride；
        # nheads 和 ngroups 都是经过 TP 切分的，所以 nheads_ngroups_ratio 是不变的，表示一个 group 内有多少 heads；
        # pid_h 是 head 索引，pid_h // nheads_ngroups_ratio 就是 group 索引，乘以 stride_b_head 就是跳到对应 group 的位置
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # 说明：chunk_size 维度
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # 说明：x_ptrs 的 shape 是 (BLOCK_SIZE_M, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    # 说明：b_ptrs 的 shape 是 (BLOCK_SIZE_K, BLOCK_SIZE_N)
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    # 说明：chunk 内最后一个位置的 dA_cumsum
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
        tl.float32
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        # 说明：k 是 chunk 内的偏移，chunk_size_limit 是 chunk 内实际的序列长度，BLOCK_SIZE_K 是每次处理的 chunk 内的 token 数量；
        # chunk_size_limit - k 是当前偏移下剩余的 token 数量，如果小于 BLOCK_SIZE_K，就只处理剩余的 token；
        # 以下指针在循环尾部都会增加 BLOCK_SIZE_K * stride，所以计算 mask 时只需考虑剩余 token 数
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        # 说明：以 Mamba2 中矩阵分块为例，right factors or B-block-factors 形如
        # [B_0^\top A_{2:0}, B_1^\top A_{2:1}, B_2^\top A_{2:2}]，这里 scale 计算的就是
        # [A_{2:0}, A_{2:1}, A_{2:2}] 部分
        # 理解：乘上的 dt_k 是属于对 B 做 euler's discretization (一阶近似，delta * B) 中的 delta
        scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)

        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # 说明：final state per chunk supposing that the initial state (to the chunk) is 0 (摘自 Mamba-2 论文 Right Factors. 部分)
    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
    ],
    key=["hdim", "dstate", "chunk_size"],
)
@triton.jit
def _chunk_state_varlen_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    chunk_states_ptr,
    cu_seqlens_ptr,
    states_ptr,
    initstates_ptr,
    # Matrix dimensions
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_b_dstate: tl.constexpr,
    stride_dt_head: tl.int64,
    stride_dt_chunk: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    stride_chunk_states_chunk: tl.int64,
    stride_chunk_states_head: tl.int64,
    stride_chunk_states_hdim: tl.int64,
    stride_chunk_states_dstate: tl.constexpr,
    stride_states_batch: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    stride_init_states_batch: tl.int64,
    stride_init_states_head: tl.int64,
    stride_init_states_hdim: tl.int64,
    stride_init_states_dstate: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    end_idx = tl.load(cu_seqlens_ptr + pid_b + 1)
    pid_c = (end_idx - 1) // chunk_size
    b_ptr += (
        pid_c * chunk_size * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    x_ptr += pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    chunk_states_ptr += (
        pid_c * stride_chunk_states_chunk + pid_h * stride_chunk_states_head
    )

    if HAS_INITSTATES:
        # if there are init states provided, we differentiate between states (which
        # are boundary conditions at a chunk boundary) and initstates (which are boundary
        # conditions when a new example in a cont batch starts)
        initstates_ptr += pid_h * stride_init_states_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(
        dA_cumsum_ptr + (end_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize
    ).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = end_idx - pid_c * chunk_size
    start_idx = tl.load(cu_seqlens_ptr + pid_b)
    start_idx_cur = tl.maximum(start_idx - pid_c * chunk_size, 0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim)
            & (offs_k[None, :] < chunk_size_limit - k)
            & (offs_k[None, :] >= start_idx_cur - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k)
            & (offs_n[None, :] < dstate)
            & (offs_k[:, None] >= start_idx_cur - k),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        scale = tl.where(
            (offs_k >= start_idx_cur - k) & (offs_k < chunk_size_limit - k),
            tl.exp(dA_cs_last - dA_cs_k) * dt_k,
            0.0,
        )
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # If the sequence starts after the last chunk idx, we don't need to add the contribution from the last chunk
    # If HAS_INITSTATES==True need to consider two possibilities
    # - if start_idx < pid_c * chunk_size, then we need to take the past_states_ptrs
    # - if state_idx >= pid * chunk_size, then we need to insert initstates
    if (
        (start_idx < pid_c * chunk_size)  # first chunk
        or (HAS_INITSTATES)
    ):
        dA_cs_boundary = 0.0  # default

        if not HAS_INITSTATES:
            past_states_ptrs = chunk_states_ptr + (
                offs_m[:, None] * stride_chunk_states_hdim
                + offs_n[None, :] * stride_chunk_states_dstate
            )
        else:
            # - this seems repetitive, buts its to help the compiler
            if start_idx < pid_c * chunk_size:
                past_states_ptrs = chunk_states_ptr + (
                    offs_m[:, None] * stride_chunk_states_hdim
                    + offs_n[None, :] * stride_chunk_states_dstate
                )
            else:
                past_states_ptrs = initstates_ptr + (
                    pid_b * stride_init_states_batch
                    + offs_m[:, None] * stride_init_states_hdim
                    + offs_n[None, :] * stride_init_states_dstate
                )

                # need to adjust the boundary
                if start_idx > pid_c * chunk_size:
                    dA_cs_boundary = tl.load(
                        dA_cumsum_ptr
                        + (start_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize
                    ).to(tl.float32)

        past_states = tl.load(
            past_states_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)

        scale = tl.exp(dA_cs_last - dA_cs_boundary)
        acc += past_states * scale

    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


# 已阅
# 说明：计算 dt_out 和 dA_cumsum 的 kernel，
# dt_out 是经过 bias 和 softplus 处理后的 dt，dA_cumsum 是 dt * A 的前缀和（dt 是处理后的）
def _chunk_cumsum_fwd(
    dt,
    A,
    chunk_size,
    cu_chunk_seqlens,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = cu_chunk_seqlens.shape[0] - 1
    dt_out = torch.empty(
        nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    dA_cumsum = torch.empty(
        nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    # 说明：grid 设计为 (nchunks, nheads // BLOCK_SIZE_H)，每个 block 处理一个 chunk 内 BLOCK_SIZE_H 个 head 的数据
    grid_chunk_cs = lambda META: (nchunks, triton.cdiv(nheads, META["BLOCK_SIZE_H"]))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](
            dt_ptr=dt,
            A_ptr=A,
            dt_bias_ptr=dt_bias,
            dt_out_ptr=dt_out,
            dA_cumsum_ptr=dA_cumsum,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            seqlen=seqlen,
            nheads=nheads,
            chunk_size=chunk_size,
            dt_min=dt_limit[0],
            dt_max=dt_limit[1],
            stride_dt_seqlen=dt.stride(0),
            stride_dt_head=dt.stride(1),
            stride_A_head=A.stride(0),
            stride_dt_bias_head=dt_bias.stride(0) if dt_bias is not None else 0,
            stride_dt_out_head=dt_out.stride(0),
            stride_dt_out_chunk=dt_out.stride(1),
            stride_dt_out_csize=dt_out.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            DT_SOFTPLUS=dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_cumsum, dt_out


# 已阅
# 说明：计算 Right Factors 的 kernel，输入是 x、b、dt 和 dA_cumsum，输出是 states（每个 chunk 的最终状态）
def _chunk_state_fwd(
    B, x, dt, dA_cumsum, cu_chunk_seqlens, states=None, states_in_fp32=True
):
    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape

    if states is not None:
        assert states.shape == (nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty(
            (nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype
        )

    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        nchunks,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_state_fwd_kernel[grid](
            x_ptr=x,
            b_ptr=B,
            states_ptr=states,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            hdim=headdim,
            dstate=dstate,
            chunk_size=chunk_size,
            seqlen=seqlen,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_x_seqlen=x.stride(0),
            stride_x_head=x.stride(1),
            stride_x_hdim=x.stride(2),
            stride_b_seqlen=B.stride(0),
            stride_b_head=B.stride(1),
            stride_b_dstate=B.stride(2),
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_hdim=states.stride(2),
            stride_states_dstate=states.stride(3),
            stride_dt_head=dt.stride(0),
            stride_dt_chunk=dt.stride(1),
            stride_dt_csize=dt.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
        )
    return states


def chunk_state_varlen(
    B, x, dt, dA_cumsum, cu_seqlens, chunk_states, initial_states=None
):
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    assert nheads % ngroups == 0
    assert B.shape == (total_seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert chunk_states.shape == (nchunks, nheads, headdim, dstate)

    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)

    states = torch.empty(
        batch,
        nheads,
        headdim,
        dstate,
        dtype=chunk_states.dtype,
        device=chunk_states.device,
    )

    initial_states_strides = (
        (
            initial_states.stride(0),
            initial_states.stride(1),
            initial_states.stride(2),
            initial_states.stride(3),
        )
        if initial_states is not None
        else (0, 0, 0, 0)
    )

    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_state_varlen_kernel[grid](
            x_ptr=x,
            b_ptr=B,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            chunk_states_ptr=chunk_states,
            cu_seqlens_ptr=cu_seqlens,
            states_ptr=states,
            initstates_ptr=initial_states,
            hdim=headdim,
            dstate=dstate,
            chunk_size=chunk_size,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_x_seqlen=x.stride(0),
            stride_x_head=x.stride(1),
            stride_x_hdim=x.stride(2),
            stride_b_seqlen=B.stride(0),
            stride_b_head=B.stride(1),
            stride_b_dstate=B.stride(2),
            stride_dt_head=dt.stride(0),
            stride_dt_chunk=dt.stride(1),
            stride_dt_csize=dt.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_chunk_states_chunk=chunk_states.stride(0),
            stride_chunk_states_head=chunk_states.stride(1),
            stride_chunk_states_hdim=chunk_states.stride(2),
            stride_chunk_states_dstate=chunk_states.stride(3),
            stride_states_batch=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_hdim=states.stride(2),
            stride_states_dstate=states.stride(3),
            stride_init_states_batch=initial_states_strides[0],
            stride_init_states_head=initial_states_strides[1],
            stride_init_states_hdim=initial_states_strides[2],
            stride_init_states_dstate=initial_states_strides[3],
            HAS_INITSTATES=initial_states is not None,
        )
    return states
