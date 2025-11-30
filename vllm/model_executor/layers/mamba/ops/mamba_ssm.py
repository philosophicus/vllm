# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/selective_state_update.py

import torch
from packaging import version

from vllm import _custom_ops as ops
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

TRITON3 = HAS_TRITON and (version.parse(triton.__version__) >= version.parse("3.0.0"))

if TRITON3:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt
else:

    @triton.jit
    def softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt


# 已阅
# 说明：根据不同值进行特化 specialization
@triton.heuristics({"HAS_DT_BIAS": lambda args: args["dt_bias_ptr"] is not None})
@triton.heuristics({"HAS_D": lambda args: args["D_ptr"] is not None})
@triton.heuristics({"HAS_Z": lambda args: args["z_ptr"] is not None})
@triton.heuristics(
    {
        "HAS_STATE_BATCH_INDICES": lambda args: args["state_batch_indices_ptr"]
        is not None
    }
)
@triton.heuristics(
    {"IS_SPEC_DECODING": lambda args: args["num_accepted_tokens_ptr"] is not None}
)
@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens_ptr"] is not None})
@triton.heuristics(
    {"BLOCK_SIZE_DSTATE": lambda args: triton.next_power_of_2(args["dstate"])}
)
# 说明：不要对 N 做特化
@triton.jit(do_not_specialize=["N"])
def _selective_scan_update_kernel(
    # Pointers to matrices
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    z_ptr,
    out_ptr,
    state_batch_indices_ptr,
    dst_state_batch_indices_ptr,
    pad_slot_id,
    num_accepted_tokens_ptr,
    cu_seqlens_ptr,
    # Matrix dimensions
    N,
    nheads,
    dim,
    dstate,
    # 说明：每组 head 数
    nheads_ngroups_ratio,
    # Strides
    stride_state_batch,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_dim,
    stride_dt_bias_head,
    stride_dt_bias_dim,
    stride_A_head,
    stride_A_dim,
    stride_A_dstate,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_C_batch,
    stride_C_group,
    stride_C_dstate,
    stride_D_head,
    stride_D_dim,
    stride_z_batch,
    stride_z_head,
    stride_z_dim,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    stride_state_indices_batch,
    stride_state_indices_T,
    stride_dst_state_indices_batch,
    stride_dst_state_indices_T,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    TIE_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_STATE_BATCH_INDICES: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    # 说明：head dimension，即 P
    pid_m = tl.program_id(axis=0)
    # 说明：batch dimension，即 B
    pid_b = tl.program_id(axis=1)
    # 说明：num_heads dimension，即 H
    pid_h = tl.program_id(axis=2)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + pid_b).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + pid_b + 1).to(tl.int64)
        seq_len = eos - bos

        if seq_len == 0:
            return
    else:
        bos = pid_b
        seq_len = 1

    state_ptr_base = state_ptr

    # If HAS_STATE_BATCH_INDICES is true, then the ssm state's batch coordinate
    # is taken from the state_batch_indices_ptr Otherwise, the state coordinate
    # is the same as the batch id.
    if HAS_STATE_BATCH_INDICES:
        if IS_SPEC_DECODING:
            num_accepted = tl.load(num_accepted_tokens_ptr + pid_b).to(tl.int64)
            # 说明：从 accepted tokens 末尾开始加载 state
            init_token_idx = tl.maximum(num_accepted - 1, 0)
        else:
            init_token_idx = 0

        dst_state_batch_indices_ptr += pid_b * stride_dst_state_indices_batch
        # 说明：这里只确定了 not IS_SPEC_DECODING 时的目标状态地址 dst_state_ptr
        if not IS_SPEC_DECODING:
            dst_state_batch_idx = tl.load(
                dst_state_batch_indices_ptr
                + init_token_idx * stride_dst_state_indices_T
            ).to(tl.int64)
            # 说明：dst_state_ptr 仅应用于 not IS_SPEC_DECODING 的情况
            # state 的 shape 为 (batch, nheads, dim, dstate)，这里有了 batch 和 head 维度，
            # 还缺少 dim 和 dstate 维度
            dst_state_ptr = state_ptr + (
                dst_state_batch_idx * stride_state_batch + pid_h * stride_state_head
            )

        state_batch_indices_ptr += (
            pid_b * stride_state_indices_batch + init_token_idx * stride_state_indices_T
        )
        state_batch_idx = tl.load(state_batch_indices_ptr).to(tl.int64)
        # 说明：state_ptr 后续代表输入状态的地址，缺少 dim 和 dstate 维度
        state_ptr += state_batch_idx * stride_state_batch + pid_h * stride_state_head
    else:
        # 说明：没有 state_batch_indices，输入状态的地址和输出状态的地址相同，
        # 且 state 的 batch 维度和 head 维度分别由 pid_b 和 pid_h 定义，缺少 dim 和 dstate 维度
        dst_state_ptr = (
            state_ptr + pid_b * stride_state_batch + pid_h * stride_state_head
        )
        state_ptr += pid_b * stride_state_batch + pid_h * stride_state_head

    # 说明：x[bos, pid_h] 的地址，缺最后的 dim 维度
    x_ptr += bos * stride_x_batch + pid_h * stride_x_head
    # 说明：dt[bos, pid_h] 的地址，缺最后的 dim 维度
    dt_ptr += bos * stride_dt_batch + pid_h * stride_dt_head
    if HAS_DT_BIAS:
        # 说明：dt_bias[pid_h] 的地址，缺最后的 dim 维度
        dt_bias_ptr += pid_h * stride_dt_bias_head
    # 说明：A[pid_h] 的地址，缺最后的 dim 和 dstate 维度
    A_ptr += pid_h * stride_A_head
    # 说明：B[bos, pid_h // nheads_ngroups_ratio] 的地址，缺最后的 dstate 维度
    B_ptr += bos * stride_B_batch + (pid_h // nheads_ngroups_ratio) * stride_B_group
    # 说明：C[bos, pid_h // nheads_ngroups_ratio] 的地址，缺最后的 dstate 维度
    C_ptr += bos * stride_C_batch + (pid_h // nheads_ngroups_ratio) * stride_C_group
    if HAS_Z:
        # 说明：z[bos, pid_h] 的地址，缺最后的 dim 维度
        z_ptr += bos * stride_z_batch + pid_h * stride_z_head
    # 说明：out[bos, pid_h] 的地址，缺最后的 dim 维度
    out_ptr += bos * stride_out_batch + pid_h * stride_out_head

    # 说明：dim 维度，即 head dimension
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # 说明：dstate 维度
    offs_n = tl.arange(0, BLOCK_SIZE_DSTATE)
    state_ptrs = state_ptr + (
        offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
    )
    if not IS_SPEC_DECODING:
        dst_state_ptrs = dst_state_ptr + (
            offs_m[:, None] * stride_state_dim + offs_n[None, :] * stride_state_dstate
        )

    mask = (offs_m[:, None] < dim) & (offs_n[None, :] < dstate)
    if HAS_STATE_BATCH_INDICES:
        mask &= state_batch_idx != pad_slot_id
    state = tl.load(state_ptrs, mask=mask, other=0.0).to(tl.float32)

    if HAS_DT_BIAS:
        dt_bias_ptrs = dt_bias_ptr + offs_m * stride_dt_bias_dim
    if HAS_D:
        # 说明：D[pid_h] 的地址，缺最后的 dim 维度
        D_ptr += pid_h * stride_D_head
        D_ptrs = D_ptr + offs_m * stride_D_dim
    A_ptrs = A_ptr + offs_m[:, None] * stride_A_dim + offs_n[None, :] * stride_A_dstate

    # 说明：batch 维度做遍历
    for i_t in range(seq_len):
        x_ptrs = x_ptr + offs_m * stride_x_dim
        dt_ptrs = dt_ptr + offs_m * stride_dt_dim
        B_ptrs = B_ptr + offs_n * stride_B_dstate
        C_ptrs = C_ptr + offs_n * stride_C_dstate
        if HAS_Z:
            z_ptrs = z_ptr + offs_m * stride_z_dim
        out_ptrs = out_ptr + offs_m * stride_out_dim

        x = tl.load(x_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if not TIE_HDIM:
            dt = tl.load(dt_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(
                A_ptrs,
                mask=(offs_m[:, None] < dim) & (offs_n[None, :] < dstate),
                other=0.0,
            ).to(tl.float32)
            # 说明：A 的 shape 是 (dim, dstate), dt 的 shape 是 (dim, )；
            # 结果的 shape 是 (dim, dstate)
            dA = tl.exp(A * dt[:, None])
        else:
            dt = tl.load(dt_ptr).to(tl.float32)
            if HAS_DT_BIAS:
                dt += tl.load(dt_bias_ptr).to(tl.float32)
            if DT_SOFTPLUS:
                dt = softplus(dt)
            A = tl.load(A_ptr).to(tl.float32)
            dA = tl.exp(A * dt)  # scalar, not a matrix

        B = tl.load(B_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        C = tl.load(C_ptrs, mask=offs_n < dstate, other=0.0).to(tl.float32)
        if HAS_D:
            D = tl.load(D_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)
        if HAS_Z:
            z = tl.load(z_ptrs, mask=offs_m < dim, other=0.0).to(tl.float32)

        # 说明：not TIE_HDIM 时，(1, dstate) * (dim, 1) -> (dim, dstate)；
        # TIE_HDIM 时，B 的 shape 是 (dstate,)，dt 是 scalar，结果也是 (dstate,)
        dB = B[None, :] * dt[:, None] if not TIE_HDIM else B * dt
        # 说明：state 的 shape 是 (dim, dstate)，x 的 shape 是 (dim,)，
        # - not TIE_HDIM 时，dA 和 dB 的 shape 都是 (dim, dstate)，结果的 shape 是 (dim, dstate)；
        # - TIE_HDIM 时，dA 的 shape 是 scalar，dB 的 shape 是 (dstate,)，结果的 shape 也是 (dim, dstate)；
        # 结果 state 的 shape 是 (dim, dstate)
        state = state * dA + dB * x[:, None]

        if IS_SPEC_DECODING:
            # 说明：每个 token 的 state 都要保存到目标位置
            # 说明：token 对应的存储 dst_index 的地址
            dst_idx_ptr = dst_state_batch_indices_ptr + i_t * stride_dst_state_indices_T
            token_dst_idx = tl.load(dst_idx_ptr).to(tl.int64)
            if token_dst_idx != pad_slot_id:
                token_dst_ptrs = (
                    state_ptr_base
                    + token_dst_idx * stride_state_batch
                    + pid_h * stride_state_head
                    + offs_m[:, None] * stride_state_dim
                    + offs_n[None, :] * stride_state_dstate
                )
                tl.store(
                    # 说明：token_dst_ptrs.dtype.element_ty 表示指针指向的元素类型
                    token_dst_ptrs, state.to(token_dst_ptrs.dtype.element_ty), mask=mask
                )

        out = tl.sum(state * C[None, :], axis=1)
        if HAS_D:
            out += x * D
        if HAS_Z:
            out *= z * tl.sigmoid(z)
        tl.store(out_ptrs, out, mask=offs_m < dim)

        x_ptr += stride_x_batch
        dt_ptr += stride_dt_batch
        B_ptr += stride_B_batch
        C_ptr += stride_C_batch
        out_ptr += stride_out_batch
        if HAS_Z:
            z_ptr += stride_z_batch

    if not IS_SPEC_DECODING:
        # 说明：只保存最后一个 token 的 state
        tl.store(dst_state_ptrs, state.to(dst_state_ptrs.dtype.element_ty), mask=mask)


# 已阅
def selective_state_update(
    # 说明：selective_scan_fn 中使用的 state 的 shape 为 (batch, dim, dstate)
    state,
    x,
    dt,
    A,
    B,
    C,
    D=None,
    z=None,
    dt_bias=None,
    dt_softplus=False,
    state_batch_indices=None,
    dst_state_batch_indices=None,
    pad_slot_id=PAD_SLOT_ID,
    out=None,
    num_accepted_tokens=None,
    cu_seqlens=None,
    is_blackwell=False,
):
    """
    Argument:
        state: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        x: (batch, dim) or (batch, nheads, dim)
        dt: (batch, dim) or (batch, nheads, dim)
        A: (dim, dstate) or (nheads, dim, dstate)
        B: (batch, dstate) or (batch, ngroups, dstate)
        C: (batch, dstate) or (batch, ngroups, dstate)
        D: (dim,) or (nheads, dim)
        z: (batch, dim) or (batch, nheads, dim)
        dt_bias: (dim,) or (nheads, dim)
        pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
        out: Preallocated ssm output tensor. Assume same shape as x.
             In-place updated.
        num_accepted_tokens: (batch,)
            number of accepted tokens from previous verification step,
            tells the kernel which initial state to use
        cu_seqlens: (batch,)
            length per sequence, for variable length in speculative decoding cases
    """
    if state.dim() == 3:
        state = state.unsqueeze(1)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if dt.dim() == 2:
        dt = dt.unsqueeze(1)
    if A.dim() == 2:
        A = A.unsqueeze(0)
    if B.dim() == 2:
        B = B.unsqueeze(1)
    if C.dim() == 2:
        C = C.unsqueeze(1)
    if D is not None and D.dim() == 1:
        D = D.unsqueeze(0)
    if z is not None and z.dim() == 2:
        z = z.unsqueeze(1)
    if dt_bias is not None and dt_bias.dim() == 1:
        dt_bias = dt_bias.unsqueeze(0)
    if out.dim() == 2:
        out = out.unsqueeze(1)
    if num_accepted_tokens is not None:
        assert state_batch_indices is not None and state_batch_indices.dim() == 2
        assert dst_state_batch_indices is None or dst_state_batch_indices.dim() == 2
    if state_batch_indices is not None and state_batch_indices.dim() == 1:
        state_batch_indices = state_batch_indices.unsqueeze(1)
    if dst_state_batch_indices is not None and dst_state_batch_indices.dim() == 1:
        dst_state_batch_indices = dst_state_batch_indices.unsqueeze(1)

    _, nheads, dim, dstate = state.shape
    batch = x.shape[0]
    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
        # Only used to verify the shape of
        # state_batch_indices and dst_state_batch_indices
        max_seqlen = (
            state_batch_indices.size(-1) if state_batch_indices is not None else 1
        )
    else:
        N = batch
        max_seqlen = 1

    assert x.shape == (batch, nheads, dim)
    assert dt.shape == x.shape
    assert A.shape == (nheads, dim, dstate)
    ngroups = B.shape[1]
    assert nheads % ngroups == 0, "nheads must be divisible by ngroups"
    assert B.shape == (batch, ngroups, dstate)
    assert C.shape == B.shape
    if D is not None:
        assert D.shape == (nheads, dim)
    if z is not None:
        assert z.shape == x.shape
    if dt_bias is not None:
        assert dt_bias.shape == (nheads, dim)
    if state_batch_indices is not None:
        assert state_batch_indices.shape[0] >= N
        assert state_batch_indices.shape[1] >= max_seqlen
    if dst_state_batch_indices is not None:
        assert dst_state_batch_indices.shape[0] >= N
        assert dst_state_batch_indices.shape[1] >= max_seqlen
    else:
        # revert to the default behavior of in-place state updates
        dst_state_batch_indices = state_batch_indices
    assert out.shape == x.shape
    if num_accepted_tokens is not None:
        assert num_accepted_tokens.shape == (N,)

    grid = lambda META: (triton.cdiv(dim, META["BLOCK_SIZE_M"]), N, nheads)
    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    state_batch_indices_strides = (
        (state_batch_indices.stride(0), state_batch_indices.stride(1))
        if state_batch_indices is not None
        else (0, 0)
    )
    dst_state_batch_indices_strides = (
        (dst_state_batch_indices.stride(0), dst_state_batch_indices.stride(1))
        if dst_state_batch_indices is not None
        else (0, 0)
    )
    # We don't want autotune since it will overwrite the state.
    # We instead tune by hand based on dstate.

    # Default
    BLOCK_SIZE_M, num_warps = 4, 8

    # 说明：根据 state dimension 的值调整 BLOCK_SIZE_M 和 num_warps，优化性能；
    # state dimension 值越大，head dimension 越小，P * N <= 512
    if dstate <= 16:
        BLOCK_SIZE_M, num_warps = 32, 4
    elif dstate <= 32:
        BLOCK_SIZE_M, num_warps = 16, 4
    elif dstate <= 64:
        BLOCK_SIZE_M, num_warps = 8, 4
    else:
        # dstate > 64
        if is_blackwell:
            # Optimized for B200 with dstate>64
            BLOCK_SIZE_M, num_warps = 32, 8
        elif dstate <= 128:
            BLOCK_SIZE_M, num_warps = 4, 4

    # 说明：判断是否绑定/共享 H 维度，即 num_heads 维度
    tie_hdim = (
        # 说明：A 的 dstate 维度共享
        A.stride(-1) == 0
        # 说明：A 的 dim 维度共享
        and A.stride(-2) == 0
        # 说明：dt 的 dim 维度共享
        and dt.stride(-1) == 0
        # 说明：dt_bias 的 dim 维度共享
        and dt_bias.stride(-1) == 0
    )
    with torch.cuda.device(x.device.index):
        _selective_scan_update_kernel[grid](
            state,
            x,
            dt,
            dt_bias,
            A,
            B,
            C,
            D,
            z,
            out,
            state_batch_indices,
            dst_state_batch_indices,
            pad_slot_id,
            num_accepted_tokens,
            cu_seqlens,
            N,
            nheads,
            dim,
            dstate,
            nheads // ngroups,
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dt.stride(0),
            dt.stride(1),
            dt.stride(2),
            *(dt_bias.stride(0), dt_bias.stride(1)) if dt_bias is not None else 0,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            C.stride(0),
            C.stride(1),
            C.stride(2),
            *(D.stride(0), D.stride(1)) if D is not None else 0,
            z_strides[0],
            z_strides[1],
            z_strides[2],
            out.stride(0),
            out.stride(1),
            out.stride(2),
            state_batch_indices_strides[0],
            state_batch_indices_strides[1],
            dst_state_batch_indices_strides[0],
            dst_state_batch_indices_strides[1],
            dt_softplus,
            tie_hdim,
            BLOCK_SIZE_M,
            num_warps=num_warps,
        )


# 已阅
def selective_scan_fn(
    u,
    # 说明：kernel 中使用的 shape 为 (batch, dim, dstate)
    ssm_states,
    delta,
    # 说明：shape 为 [intermediate_size // tp_size, ssm_state_size]
    A,
    # 说明：B, C shape 为 [dstate, total_length]
    B,
    C,
    D=None,
    # 说明：实参为 gate_p
    z=None,
    delta_bias=None,
    delta_softplus=False,
    query_start_loc=None,
    cache_indices=None,
    has_initial_state=None,
    pad_slot_id=PAD_SLOT_ID,
    block_size=1024,
    block_idx_first_scheduled_token=None,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
) -> torch.Tensor:
    """
    u: (dim, total_length) for varlen or (batch, dim, seqlen)
        applies changes in place.
    ssm_states: (batch, dim, dstate) or (batch, nheads, dim, dstate)
        applies changes in place.
    delta: (dim, total_length) for varlen or (batch, dim, seqlen)
    A: (dim, dstate)
    B: (ngroups, dstate, total_length) for varlen or
                                        (batch,ngroups,dstate,seqlen)
    C: (ngroups, dstate, total_length) for varlen or
                                        (batch,ngroups,dstate,seqlen)
    D: (dim,)
    z: (dim, total_length) for varlen or (batch, dim, seqlen)
    dt_bias: (dim,) or (dim)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended with 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch) int32
        A tensor with each cell is a correspondent
        input and output ssm_state indices
      - Without APC: (batch,) - single state index per batch item
      - With APC: (batch, max_positions) - cache block indices for read/write
        Each non-zero value indicates a cache block to load from and/or write to.
    has_initial_state: (batch) bool
        A tensor populated with ones and zeros,
        indicate if the ssm_state at the corresponding index should be
        used as initial state. Not providing argument assumes
        there's no initial state
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padding entries
        that will not be processed,
        for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
        in this case, the kernel will not process entries at indices 0 and 3
    block_size: int
        The block size to align the cached states to
    block_idx_first_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the first
        cache block to be filled is located.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the last cache block
        to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into cache_indices, where the cache block
        containing the initial state is located.
    returns
        output: (dim, total_length) for varlen or (batch, dim, seqlen)
                supports inplace replacement
    """
    if u.stride(-1) != 1:
        u = u.contiguous()
    if delta.stride(-1) != 1:
        delta = delta.contiguous()
    if D is not None:
        D = D.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if z is not None and z.stride(-1) != 1:
        z = z.contiguous()
    if B.dim() == 3 and query_start_loc is None:
        B = B.unsqueeze(1)
    if B.dim() == 2 and query_start_loc is not None:
        # 说明：B, C shape 变为 [1, dstate, total_length]
        B = B.unsqueeze(0)
    if C.dim() == 3 and query_start_loc is None:
        C = C.unsqueeze(1)
    if C.dim() == 2 and query_start_loc is not None:
        C = C.unsqueeze(0)

    ops.selective_scan_fwd(
        u,
        delta,
        A,
        # 说明：B, C shape 变为 [1, dstate, total_length]
        B,
        C,
        D,
        z,
        delta_bias,
        delta_softplus,
        query_start_loc,
        cache_indices,
        has_initial_state,
        ssm_states,
        pad_slot_id,
        block_size,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
    )

    # 说明：delta 一定会被 inplace 写入；存在 gate 机制的情况下，z 也会被写入；
    # 所以 z 不为 None 时，输出以 z 为准；z 为 None 时，输出以 delta 为准
    if z is None:
        return delta  # output written inplace to delta
    else:
        return z  # output written inplace to z
