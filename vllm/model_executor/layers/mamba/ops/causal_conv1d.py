# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py


import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


# 已阅
# 说明：实现了带状态缓存的因果卷积前向传播计算
# 1. 根据 state_len, seqlen, pre 等参数，更新 initial conv_state；或者将 x 写入 output cache，
# 即根据 block_idx_first_scheduled_token 和 block_idx_last_scheduled_token 获取的 cache line；
# 2. 计算 x 与 w 的卷积结果，写入 o；
@triton.jit()
def _causal_conv1d_fwd_kernel(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # (batch, n_blocks + padding) The second dimension contains
    # the block indices relevant for each sequence
    # plus potential 0-padding at the beginning and at the end
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    block_idx_first_scheduled_token,  # (batch,)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    num_computed_tokens,  # (batch,)
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    dim: tl.constexpr,
    seqlen: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # 说明：block_size_to_align // BLOCK_M
    stride_block_m: tl.constexpr,  # Stride block to align divided by BLOCK_M
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = (
        KERNEL_WIDTH - 1
    )  # can be passed via argument if it's not the same as this value

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    # 说明：[0, 0, 1, 1, 1, 1, 2, 2]
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    # 说明：[0, 1, 0, 1, 2, 3, 0, 1]
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    # 说明：即 block_size_to_align，缓存分块的 token 数量
    B_size: tl.constexpr = stride_block_m * BLOCK_M

    if IS_APC_ENABLED:
        # Handle the case if prefix caching is enabled.
        # In particular, if prefix caching is enabled, the program write additional cache states to "cache_indices_ptr"

        # 说明：需要填充的第一个和最后一个 cache block 的 index
        # Get the length of the completed sequence so far and compute the offset.
        current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        sequence_completed_index = tl.load(num_computed_tokens + idx_seq)

        # Compute the offset where the first stride_block_m-aligned first full block is
        # Value in "token-space"
        sequence_completed_offset_token = sequence_completed_index % B_size
        # 说明：block 内在 computed_tokens 之后的剩余的 token 空间
        seq_completed_offset = B_size - sequence_completed_offset_token
        # 说明：填充 seq_completed_offset 后，剩余的 seqlen - seq_completed_offset 个 token 刨除完整 block 后还剩多少 token
        # 说明：B_size = 8, sequence_completed_offset_token = 2，seq_completed_offset = 6
        # 如果 seqlen = 10, seq_end_offset = 4，表示在最后一个 block 里有多少个 token；
        # 如果 seqlen = 2，seq_end_offset = -4 % 8 = 4，这种情况是有问题的，因为 seq_end_offset 已经大于了 seqlen，
        # 此时 sequence_end_index - seq_end_offset 会跳过整个序列
        seq_end_offset = (seqlen - seq_completed_offset) % B_size
        # 说明：last_full_block_token_index = sequence_end_index - 4，
        # 表示最后一个 block 的起始 token 在 sequence 中的索引位置
        last_full_block_token_index = sequence_end_index - seq_end_offset
        # If the sequence without the sequence_offset_index is stride_cache_chunk-aligned, then the last full chunk is the second-to-last one
        if seq_end_offset == 0:
            # 说明：此时 last_full_block_token_index == sequence_end_index；
            # 问题：last_full_block_token_index - B_size 能保证在序列范围内吗？
            # 是负数没有关系，后面会通过 mask 来控制
            last_full_block_token_index = last_full_block_token_index - B_size

        # Get the number of blocks to be filled for the current sequence
        # If n_block_to_fill = 0, then only the state at the sequence end is stored
        n_block_to_fill = current_last_index - current_first_index

        # Get the index of the init block
        conv_state_init_index = tl.load(initial_state_idx + idx_seq)
    else:
        n_block_to_fill = 0
        current_last_index = 0
        conv_state_init_index = 0
        current_first_index = 0
        last_full_block_token_index = 0

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    # 说明：再加上序列内 token_offset * stride_x_token 即可定位到具体 token
    # base of the sequence
    x_base = (
        x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim
    )  # [BLOCK_N,]

    # 说明：(batch, n_blocks + padding) The second dimension contains
    # the block indices relevant for each sequence
    # plus potential 0-padding at the beginning and at the end
    # cache_idx
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_cache_indices + conv_state_init_index
    ).to(tl.int64)

    if USE_PAD_SLOT:  # noqa
        if conv_states_input_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (
        conv_states_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init_state:
            # 说明：长度为 state_len，这里是读取 conv state 的最后一个 token
            # load from conv_states
            prior_tokens = conv_states_base + (state_len - 1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                # 说明：conv state 的最后一个 token 
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                # 说明：conv state 的最后一个 token 
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                # 说明：conv state 的倒数第二个 token 
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # prior-tokens are zeros
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if (
            state_len <= seqlen
        ):  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            # 说明：seqlen >= state_len，作为 chunk 0，将 x 尾部的 state_len 个 token 的数据写入 conv_state（需要填充的最后一个 block），
            # 但 initial conv_state 没有被更新；如果有其他 chunk，那么其他分块会负责填充从 current_first_index 开始的 block；
            # current_first_index 有可能和 initial conv_state 对应的 block 相同，也有可能是 initial conv_state 之后的 block；
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN
            )  # [BLOCK_M]
            x_ptrs = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last) * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]
            mask_x = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            # 说明：用于计算写入 conv_state 的位置的 index，说明最后一个 block 是从头写入的，
            # block 的长度至少为 state_len；如果是 state_len 则表示正好写满一个 block
            # 调研：block 的长度是否正好是 state_len？
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

            # 说明：conv_state_indices[idx_seq, current_last_index]
            # Compute the offset where the last block should be written in the conv_states
            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                # 说明：需要填充的最后一个 block 的 index
                + current_last_index
            ).to(tl.int64)

            # 说明：conv_states[conv_states_output_coord, idx_feats, idx_tokens_conv]
            conv_states_ptrs_target = (
                conv_states_ptr
                + (conv_states_output_coord * stride_conv_state_seq)  # Offset from seq
                + (idx_feats * stride_conv_state_dim)
            )[None, :] + (  # [BLOCK_N,]
                idx_tokens_conv * stride_conv_state_tok
            )[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            # 说明：写到需要填充的最后一个 block 的 conv_state 中
            tl.store(conv_states_ptrs_target, loaded_x, mask)

        else:
            # 说明：state_len > seqlen，需要左移 conv_state，移动长度为 seqlen，然后将 x 的数据写入 initial conv_state 的尾部；
            # 理解：默认情况 state_len = 3，所以此时应该只有一个 chunk（因为 chunk_size = 8）；
            # 理论上这种情况也有可能有多个 block 需要更新，但这里选择了直接更新 initial conv_state；
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                # 说明：读取 conv_state 结尾处的 state_len - seqlen 个 token
                conv_states_ptrs_source = (
                    conv_states_ptr
                    + (conv_states_input_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[None, :]
                    # 说明：根据下面 mask 确定的 idx_tokens_conv 范围，得到 conv_states 取值的范围是 [seqlen, state_len) 
                    + ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (
                    (conv_states_input_coord < num_cache_lines)
                    # 说明：idx_tokens_conv 的范围是 [0, state_len - seqlen)
                    & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    # 说明：根据下面 mask_x 确定的 idx_tokens_conv 范围，得到 x 取值的范围是 [0, seqlen)
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    # 说明：idx_tokens_conv 的取值范围是 [state_len - seqlen, state_len)
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier()  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                # 说明：将 conv_state 最后的 state_len - seqlen 个 token 数据 + x 的前 seqlen 个 token 数据组成新的 conv_state，
                # 长度为 state_len
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                # 说明：conv_states[conv_states_input_coord, idx_feats, idx_tokens_conv]
                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                # 说明：写回到 initial conv_state 位置
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = (
                    x_base[None, :]
                    # 说明：根据下面 mask_x 确定的 idx_tokens_conv 范围，得到 x 取值的范围是 [0, seqlen)
                    + ((idx_tokens_conv - VAL) * stride_x_token)[:, None]
                )  # [BLOCK_M, BLOCK_N]

                mask_x = (
                    # 说明：idx_tokens_conv 的取值范围是 [state_len - seqlen, state_len)
                    (idx_tokens_conv - VAL >= 0)[:, None]
                    & (idx_tokens_conv - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = (
                    conv_states_base
                    + (idx_tokens_conv * stride_conv_state_tok)[:, None]
                )  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[
                    None, :
                ]
                # 说明：写回到 initial conv_state 位置，前面 state_len - seqlen 个 token 数据为 0，
                # 后面 seqlen 个 token 数据为 x 的数据
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        # 说明：chunk_offset > 0，一定有 prior token，从 x 中读取
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            # 说明：x 里前面 1 个位置的 token
            # 说明：https://triton-lang.org/main/python-api/generated/triton.language.load.html
            # cache_modifier (str, optional, should be one of {“”, “.ca”, “.cg”, “.cv”}, 
            # where “.ca” stands for cache at all levels, 
            # “.cg” stands for cache at global level (cache in L2 and below, not L1), 
            # and “.cv” means don’t cache and fetch again.
            # 同时参见 https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 3:
            # 说明：x 里前面 1 个位置的 token
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            # 说明：x 里前面 2 个位置的 token
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
        if KERNEL_WIDTH == 5:
            # ruff: noqa: F841
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier=".ca")

        # Store intermediate states aligned with stride_block_m
        # The additional states are cached starting from the last stride_block_m.
        # For example:
        # If n_block_to_fill = 0, then only the state at the sequence end is cached and the process below is not involved.
        # If n_block_to_fill > 0, then the states at the sequence end and at the n_block_to_fill-last
        # stride_block_m are cached.
        # For example chunk_offset = n_block_to_fill stores the state at last_full_block
        # 说明：对于 Mamba，n_block_to_fill 对应的 block 大小就是 B_size，即 mamba_block_size（见 _compute_prefix_caching_block_indices）
        # 翻译：n_block_to_fill > 0 时，最后 n_block_to_fill 个 B_size = stride_block_m * BLOCK_M 的 token 会被缓存
        # 说明：chunk_offset 的范围是 [1, n_block_to_fill]（从前面开始缓存），最后用于缓存状态的目标 block 的索引范围是
        # [current_first_index, current_first_index + n_block_to_fill - 1]
        if (chunk_offset - 1) < n_block_to_fill:
            # 说明：chunk 的大小为 BLOCK_M = 8；
            # block 的大小为 B_size = block_size_to_align = stride_block_m * BLOCK_M，表示缓存分块的 token 数量；
            # 说明：idx_tokens_last 表示不包含最后一个 block，从 current_first_index 开始共 n_block_to_fill 个 block，
            # 每个 block 最后 state_len 个 token 的起始位置的索引；
            # Store the states at the chunk boundaries from the start of the sequence
            idx_tokens_last = (
                # 说明：最后一个 block 的起始 token 在 x 中的索引位置
                last_full_block_token_index
                # 说明：因为 last_full_block_token_index 是根据 B_size 对齐计算的，所以每个 block 的起始位置应相差 B_size 个 token；
                # 说明：因为 chunk_size <= block_size_to_align，相同长度的序列 chunk 数量一定 >= block 数量，
                # 所以 n_block_to_fill - chunk_offset 的最大范围是 [0, n_block_to_fill - 1]；
                - (n_block_to_fill - chunk_offset) * B_size
                # 说明：state_len 是 conv_state 里需要存储的 token 数量，值为 KERNEL_WIDTH - 1，一般 KERNEL_WIDTH 默认为 4，
                # 此时 state_len = 3
                - state_len
            ) + tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            x_ptrs = (
                x_ptr
                + (idx_tokens_last * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )  # [BLOCK_M,BLOCK_N,]

            # 理解：n_block_to_fill 隐含了序列的长度情况，在这种情况下计算出的 idx_tokens_last 的范围一定不会越界读取其他序列的 x；
            # 理解：idx_tokens_last 可能为负，此时加载的 x 都为 0
            # 问题：即使 idx_tokens_last 为正，怎么保证不会越界读取其他序列的 x？
            # 有 bug ？idx_tokens_last 应该 >= sequence_start_index 吧？
            # 理解：n_block_to_fill 可能是根据序列长度计算得到的，也就是说一定不会越界读取其他序列的 x ？
            mask_x = (idx_tokens_last >= 0)[:, None] & (idx_feats < dim)[
                None, :
            ]  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

            # 说明：conv_state_indices[idx_seq, current_first_index + (chunk_offset - 1)]
            # chunk_offset <= n_block_to_fill，所以最多只能填充到 current_last_index - 1 对应的 block；
            # cache_idx
            conv_states_output_coord = tl.load(
                conv_state_indices_ptr
                + idx_seq * stride_cache_indices
                + current_first_index
                + (chunk_offset - 1)
            ).to(tl.int64)

            conv_states_ptrs_target = (
                conv_states_ptr
                + (conv_states_output_coord * stride_conv_state_seq)  # Offset from seq
                + (idx_feats * stride_conv_state_dim)
            )[None, :] + (  # [BLOCK_N,]
                idx_tokens_conv * stride_conv_state_tok
            )[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats < dim)[None, :]
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, loaded_x, mask)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr
            + (sequence_start_index + token_offset + idx_token) * stride_o_token
            + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


# 已阅
# 说明：用于 prefill
def causal_conv1d_fn(
    x: torch.Tensor,
    # 说明：shape 为 [intermediate_size, conv_kernel_size]
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    # 说明：每个请求对应一组 block indices，用于指定每个请求的 conv state 在 cache 中的位置
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    # 说明：传入的参数可能是的 mamba_block_size
    block_size_to_align=0,
    metadata=None,
    validate_data=False,
):
    """support varlen + continuous batching when x is 2D tensor

    x: (dim,cu_seq_len)
        cu_seq_len = total tokens of all seqs in that batch
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    conv_states: (...,dim,width - 1) itype
        updated inplace if cache_indices are not provided
        [it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_state[i] = True
             and after that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    bias: (dim,)
    activation: either None or "silu" or "swish" or True
    pad_slot_id: int
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3
    block_idx_first_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the first cache block to be filled is located.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into cache_indices, where the last cache block to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into cache_indices, where the cache block containing the initial state is located.
    num_computed_tokens: (batch,), dtype int32
        The number of tokens already completed for each sequence
    block_size_to_align: int
        The block size to align the cached states to
    out: same shape as `x`
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"

    args = None
    # Store original dtype to cast back at the end
    original_x_dtype = x.dtype
    x = x.to(conv_states.dtype)
    out = torch.empty_like(x)
    if metadata is not None:
        # 说明：nums_dict 的计算逻辑见 compute_causal_conv1d_metadata
        nums_dict = metadata.nums_dict
        args = nums_dict
        # 说明：例子为 [0, 0, 1, 1, 1, 1, 2, 2]
        batch_ptr = metadata.batch_ptr
        # 说明：例子为 [0, 1, 0, 1, 2, 3, 0, 1]
        token_chunk_offset_ptr = metadata.token_chunk_offset_ptr
    else:
        seqlens = query_start_loc.diff().to("cpu")
        args = seqlens
        MAX_NUM_PROGRAMS = 1024

        batch_ptr = torch.full(
            (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=x.device
        )  # tracking which seq-idx the Triton program is handling
        token_chunk_offset_ptr = torch.full(
            (MAX_NUM_PROGRAMS,), PAD_SLOT_ID, dtype=torch.int32, device=x.device
        )  # tracking BLOCK_M-based index in the sequence the Triton program is handling

    # 说明：(dim, cu_seq_len)，需要确保 cu_seq_len 在最后
    is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
    dim, cu_seqlen = x.shape
    # 说明：(dim, width)
    _, width = weight.shape
    state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    # 说明：batch_size
    padded_batch = query_start_loc.size(0) - 1
    stride_x_dim = x.stride(0)
    stride_x_token = x.stride(1)
    stride_w_dim = weight.stride(0)
    stride_w_width = weight.stride(1)
    stride_istate_seq = 0
    stride_istate_dim = 0
    stride_istate_token = 0
    num_cache_lines = 0
    # 说明：固定的 BLOCK_M，在 compute_causal_conv1d_metadata 中的 BLOCK_M 目前也只有 8；
    # 表示一个 block/kernel 处理一个序列中的 8 个 token
    BLOCK_M = 8
    if conv_states is not None:
        # extensions to support vLLM:
        # 1. conv_states is used to replaced initial_states
        # 2. conv_states serve as a cache with num cache lines can be larger than batch size
        # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
        # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
        num_cache_lines = conv_states.size(0)
        assert (
            num_cache_lines == conv_states.shape[0]
            and dim == conv_states.shape[1]
            and width - 1 <= conv_states.shape[2]
        )
        stride_istate_seq = conv_states.stride(0)
        stride_istate_dim = conv_states.stride(1)
        stride_istate_token = conv_states.stride(2)
        # 说明：dim 维度连续
        assert stride_istate_dim == 1
    # 说明：dim() 是方法，ndim 是属性
    if out.dim() == 2:
        stride_o_dim = out.stride(0)
        stride_o_token = out.stride(1)
    else:
        # 理解：此时的 shape 应该是 (batch, dim, seqlen)，参考 causal_conv1d_update 中的 x 和 out 形状
        # 在 validate_data=True 时，assert x.dim() == 2 已经限定了 x 和 out 是 2D 的了，
        # 所以这里是无效的分支
        stride_o_dim = out.stride(1)
        stride_o_token = out.stride(2)
    stride_cache_indices = cache_indices.stride(0) if cache_indices is not None else 0

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_state is not None:
            assert has_initial_state.size() == (padded_batch,)
            assert conv_states is not None, (
                "ERROR: `has_initial_state` is used, which needs also `conv_states`"
            )
        # 说明：width 维度连续
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert is_channel_last, "Need to run in channel-last layout"
        if block_size_to_align is not None and block_size_to_align > 0:
            assert (block_size_to_align % BLOCK_M) == 0, (
                "The mamba block size needs to be divisible by the BLOCK_M"
            )
        else:
            block_size_to_align = BLOCK_M

    if metadata is None:

        # 说明：逻辑可以对照 compute_causal_conv1d_metadata 来看
        def num_program(META, seqlens):
            tot = 0

            mlist = []
            offsetlist = []  # type: ignore

            nums = -(-seqlens // META["BLOCK_M"])

            tot = nums.sum().item()
            mlist = np.repeat(np.arange(len(nums)), nums)
            for idx, num in enumerate(nums):
                offsetlist.extend(
                    range(num)
                )  # chunk-idx if a sequence is split into multiple chunks

            if META["batch_ptr"].nelement() < len(mlist):
                newlen = len(mlist) + 1
                META["batch_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)
                META["token_chunk_offset_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)

            if META["batch_ptr"].nelement() >= len(mlist):
                META["batch_ptr"][0 : len(mlist)].copy_(
                    torch.from_numpy(np.array(mlist))
                )
                META["token_chunk_offset_ptr"][0 : len(mlist)].copy_(
                    torch.from_numpy(np.array(offsetlist))
                )

            META["batch_ptr"] = META["batch_ptr"].to(META["x_ptr"].device)
            META["token_chunk_offset_ptr"] = META["token_chunk_offset_ptr"].to(
                META["x_ptr"].device
            )
            return tot
    else:

        def num_program(META, nums_dict):
            tot = nums_dict[META["BLOCK_M"]]["tot"]

            mlist = nums_dict[META["BLOCK_M"]]["mlist"]
            mlist_len = nums_dict[META["BLOCK_M"]]["mlist_len"]

            offsetlist = nums_dict[META["BLOCK_M"]]["offsetlist"]

            if nums_dict[META["BLOCK_M"]]["batch_ptr"] is not None:
                # 说明：使用 metadata 里预先计算好的结果，避免重复计算
                META["batch_ptr"] = nums_dict[META["BLOCK_M"]]["batch_ptr"]
                META["token_chunk_offset_ptr"] = nums_dict[META["BLOCK_M"]][
                    "token_chunk_offset_ptr"
                ]
            else:
                if META["batch_ptr"].nelement() < mlist_len:
                    newlen = mlist_len + 1
                    META["batch_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)
                    META["token_chunk_offset_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)

                if META["batch_ptr"].nelement() >= mlist_len:
                    META["batch_ptr"][0:mlist_len].copy_(mlist)
                    META["token_chunk_offset_ptr"][0:mlist_len].copy_(offsetlist)
            return tot

    def grid(META):
        return (
            num_program(META, args),
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    if batch_ptr.device != x.device:
        batch_ptr = batch_ptr.to(x.device)
        token_chunk_offset_ptr = token_chunk_offset_ptr.to(x.device)

    _causal_conv1d_fwd_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
        out,
        # Matrix dimensions
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_cache_indices,
        stride_o_dim,
        stride_o_token,
        block_size_to_align // BLOCK_M,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        # launch_cooperative_grid=True
        BLOCK_M=BLOCK_M,
        BLOCK_N=256,
        num_stages=2,
    )
    return out.to(original_x_dtype)


# 已阅
# 说明：因果一维卷积状态增量更新算子
# 1. 根据 num_accepted_tokens，对 conv state 进行滑动窗口处理，并添加 x 到 conv state 末尾；
# 2. 计算 x 与 w 的卷积结果，写入 o；
# 说明：处理一个序列的 BLOCK_N 个维度
@triton.jit()
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    # 说明： (num_cache_lines, dim, state_len)
    conv_state_ptr,
    # 说明： w/o APC (batch, ) 或  w/ APC (batch, max_positions)，参见 selective_scan_fn
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,  # (batch + 1)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # ruff: noqa: E501
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # 说明：dim 维度
    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_APC_ENABLED:
        # Get the state from the initial_state_idx
        conv_state_init = tl.load(initial_state_idx + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
    else:
        conv_state_init = 0
        current_last_index = 0

    # 说明：conv_state_indices_ptr 的 shape 为 (batch, ) 或 (batch, max_positions)，
    # 读取的是 conv_state 的 coord，即第几个 cache_line
    # cache_idx
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
    ).to(tl.int64)

    if USE_PAD_SLOT:  # noqa
        if conv_states_input_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    if IS_VARLEN:
        query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
        query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
        # 说明：state_len 等于 width - 1 或者 width - 1 + (seqlen - 1)；
        # 原来的 seqlen 是 max_query_len，现在改为实际的序列长度
        # 说明：state_len 减去 (seqlen - actual_seqlen)，即减去本来不存在的 padding 部分
        # revise state_len and seqlen
        state_len = state_len - (seqlen - (query_end_index - query_start_index))
        seqlen = query_end_index - query_start_index
        # 说明：IS_VARLEN, x 的 shape 是 [num_tokens, dim]，所以 offset 要乘以 stride_x_token
        x_offset = query_start_index * stride_x_token
        o_offset = query_start_index * stride_o_token
    else:
        query_start_index = idx_seq * seqlen
        query_end_index = query_start_index + seqlen
        x_offset = idx_seq * stride_x_seq
        o_offset = idx_seq * stride_o_seq

    if query_start_index == query_end_index:
        return

    if IS_SPEC_DECODING:
        # 说明：注意，这里的 offset 是 num_accepted_tokens - 1，
        # 参考本段后续的注释，可以这样理解：
        # 1. 在 [history1, history2, ..., historyM] 后面拼接 num_accepted_tokens 个 token，
        #    此时读取位置也应该向后偏移，即从 0 变为 num_accepted_tokens；
        # 2. conv_state 会发生 shift，更新后的 conv_state 是 [history2, ..., historyM] + [num_accepted_tokens 个 token]，
        #    即 history1 被丢弃，所以读取位置应该向前移动 1，即 num_accepted_tokens - 1 个 token；
        # 调研/问题：第 2 步的 shift 1 具体是在哪里发生的？
        # The rolling of conv state:
        #
        # Before forward, the conv_state is:
        # [history1, history2, ..., historyM].
        #
        # After forward, the conv_state becomes:
        # [history2, ..., historyM, draft1, draft2, ..., draftN].
        #
        # After acceptance, it becomes:
        #
        # - accept 1 tokens: [history2, ..., historyM, draft1]
        # - accept 2 tokens: [history3, ..., historyM, draft1, draft2]
        # - and so on.
        conv_state_token_offset = (
            tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1
        )
    else:
        # 说明：此时的 state_len = width - 1
        conv_state_token_offset = 0

    # STEP 1: READ init_state data
    conv_states_base = (
        conv_state_ptr
        # 说明：last computed token 对应的 block index
        + (conv_states_input_coord * stride_conv_state_seq)
        + (idx_feats * stride_conv_state_dim)
    )
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok
    # 说明：KERNEL_WIDTH 的值是权重的 width，表示卷积核的宽度；
    # 理解：对于第一个 token，卷积核前 KERNEL_WIDTH - 1 个位置对应的输入存储在 conv_state 窗口的前 KERNEL_WIDTH - 1 个位置，
    # 第 KERNEL_WIDTH 个位置对应当前输入 token；之后，随着每个新 token 的输入，conv_state 里的对应位置会滑动更新；
    if KERNEL_WIDTH >= 2:
        # 说明：卷积核宽度至少为 2 时，加载 offset 为 0 或 num_accepted_tokens - 1 位置的 token
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 6:
        conv_states_ptrs = prior_tokens + 4 * stride_conv_state_tok  # [BLOCK_N]
        col4 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # With speculative decoding, the conv_state updates works in a sliding
    # window manner, at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        # 说明：关注这里
        # speculative decoding 时的偏移为 1，conv_state_token_offset + idx_tokens + 1 的范围是 [num_accepted_tokens, ...]
        # 非 speculative decoding 时偏移为 seqlen，conv_state_token_offset + idx_tokens + seqlen 的范围是 [seqlen, ...]
        + ((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) * stride_conv_state_tok)[
            :, None
        ]
    )  # [BLOCK_M, BLOCK_N]
    mask = (
        (conv_states_input_coord < num_cache_lines)
        # 说明：idx_tokens + seqlen < state_len，是为了给 x 留出 seqlen 长度的位置；
        # 从上面的起始位置开始，读取 state_len - seqlen 个 token 的 conv_state 数据，
        # 问题：剩余的 seqlen 个位置由 x 来填充？
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)  # [BLOCK_N]

    # 说明：NP2_STATELEN >= state_len >= seqlen，所以 idx_tokens - VAL 的最大值 >= seqlen
    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    # 说明：从 x 中从头开始最长加载 seq_len 长度的数据
    # 这里的要求是 idx_tokens 在 [state_len - seqlen, state_len) 范围内，也就是只有当 idx_tokens 在这个范围内时，
    # 才能从 x 中加载数据；而加载数据的范围是 [0, seqlen)；
    # 这一范围的 idx_tokens 对应 conv_state 中的 [state_len - seqlen + 1, state_len + 1) if IS_SPEC_DECODING
    # 或者 [state_len, state_len + seqlen)，而其 mask 又始终要求 idx_tokens 在 [0, state_len - seqlen) 范围内，
    # 所以此时 conv_state 是加载不到数据的；
    # 反之，conv_state 在 idx_tokens 在 [0, state_len - seqlen) 范围内时，
    # 能加载到 [num_accepted_tokens, state_len - seqlen + num_accepted_tokens) if IS_SPEC_DECODING
    # 或者 [seqlen, state_len) 范围内的数据，长度为 state_len - seqlen，而 x 则加载不到数据；
    # state_len 的值受到 IS_SPEC_DECODING 的影响，所以实际范围是
    # [num_accepted_tokens, width - 1 + num_accepted_tokens - 1) if IS_SPEC_DECODING
    # 或者 [seqlen, width - 1)；
    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    # 说明：避免指令重排
    # 参考本文件其他算子里的注释 need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
    tl.debug_barrier()

    # 说明：将读到的 conv_state 和 x 进行合并，token 维度上保持 state_len 长度
    # conv_state 能加载到数据的时候，x 加载不到数据，反之亦然，所以拼接起来得到 new_conv_state，长度为 state_len；
    # 注意，state_len = width - 1 + (seqlen - 1) if IS_SPEC_DECODING else width - 1
    # conv_state 读取的长度是 state_len - seqlen，
    # 只是起点根据 IS_SPEC_DECODING 不同而不同，开启时起点偏移 num_accepted_tokens，
    # 不开启时起点偏移 seqlen；也就是说开启 IS_SPEC_DECODING 时，conv_state 可能会丢弃尾部的若干 token 数据，
    # 这是 spec decoding 的正常效果
    new_conv_state = tl.where(mask, conv_state, loaded_x)

    # 说明：这里注释写错了，应该是 last_idx
    # 凑齐了 state_len 长度的数据后，写入 last_idx 位置
    # Get the state from the initial_state_idx
    # cache_idx
    conv_states_offset = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
    ).to(tl.int64)
    conv_state_ptrs_target = (
        conv_state_ptr
        + (conv_states_offset * stride_conv_state_seq)  # Offset from seq
        + (idx_feats * stride_conv_state_dim)
    )[None, :] + (  # [BLOCK_N,]
        idx_tokens * stride_conv_state_tok
    )[:, None]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    # 说明：weight 的 shape 是 (dim, width)，KERNEL_WIDTH = width
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 5:
        w_ptrs = w_base + (4 * stride_w_width)  # [BLOCK_N] tensor
        w_col4 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 6:
        w_ptrs = w_base + (5 * stride_w_width)  # [BLOCK_N] tensor
        w_col5 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # 说明：weight 的 shape 是 (dim, width)，x 是从移动 + 更新前的 conv_state 中读取的，shape 是 (dim, seqlen)
    # STEP 5: compute each token
    for idx_token in tl.range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    # 说明：从 x 中加载当前 token 的数据
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 5:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 6:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)

            # 说明：j = 0 时，等于 w_col0 * col0
            acc += matrix_x * matrix_w  # [BLOCK_N]

        # 说明：每个 token 计算完后，更新 col0, col1, ...
        # col0 <- col1, col1 <- col2, ... col(KERNEL_WIDTH-2) <- x[current_token]
        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (
            idx_feats < dim
        )  # token-index  # feature-index
        o_ptrs = (
            o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)


# 已阅
# 说明：用于 decode
def causal_conv1d_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    # 说明：shape 为 [intermediate_size, conv_kernel_size]
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    # 说明：conv_state 的索引
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    # 说明：conv_state_indices 的索引，对应的元素是 conv_state 的 cache_line 索引，
    # 对应的 cache_line 用于保存 shift + 更新后的结果
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    # 说明：conv_state_indices 的索引，对应的元素是 conv_state 的 cache_line 索引，
    # 对应的 cache_line 用于获取初始状态
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
):
    """
    x: Input tensor which can take the following shapes:

    - `[batch, dim]` - single token prediction
    - `[batch, dim, seqlen]` - single or multiple tokens prediction
    - `[num_tokens, dim]` - continuous batching, where num_tokens is
        the total tokens of all sequences in that batch

    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into conv_state_indices, where the last cache block to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into conv_state_indices, where the cache block containing the initial state is located.
    num_accepted_tokens: (batch,), dtype int32
        If not None, it indicates the number of accepted tokens for each
        sequence in the batch.
        This is used in speculative decoding, where the conv_state is updated
        in a sliding window manner.
    query_start_loc: (batch + 1,) int32
        If not None, the inputs is given in a varlen fashion and this indicates
        the starting index of each sequence in the batch.
    max_query_len: int
        If query_start_loc is not None, this indicates the maximum query
        length in the batch.
    pad_slot_id: int
            if conv_state_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: conv_state_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen) or (num_tokens, dim), same shape as `x`
    """
    if validate_data:
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    # 说明：确定 batch_size，dim，seqlen
    if query_start_loc is None:
        batch, dim, seqlen = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len
    # 说明：第一个维度是 dim
    _, width = weight.shape
    # 说明：关注 state_len 和权重 width 的关系
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        # 说明：K 维度
        assert dim == weight.size(0)
        # 说明：dim 维度要是连续的
        assert conv_state.stride(-2) == 1, (
            f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        )
        # 说明：关注 state_len 和权重 width 的关系
        assert state_len >= width - 1
        # 说明：dim == x.size(1) == weight.size(0) == conv_state.size(1)
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            # 说明：即 num_cache_lines >= batch
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        # 说明：要求 width 维度是连续的
        assert weight.stride(1) == 1  # Need this

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    if query_start_loc is None:
        # X (batch, dim, seqlen)
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        # 说明：对应 docstring 中说的 [num_tokens, dim] - continuous batching
        # X (dim, cu_seqlen)
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    # 说明：第一个维度为 num_cache_lines
    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        # 说明：0 维度对应 batch
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if num_accepted_tokens is not None:
        # 说明：seqlen 是 query 的最大长度，这里因为 num_accepted_tokens 的存在，需要对 state_len 进行调整
        # 理解/问题：seqlen = 1 + num_draft_tokens，state_len 需要能够容纳 num_draft_tokens 个 token 的状态数据 + 之前的历史状态数据
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_update_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        num_accepted_tokens,
        query_start_loc,
        block_idx_last_scheduled_token,
        initial_state_idx,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_VARLEN=query_start_loc is not None,
        # 说明：APC = Automatic Prefix Caching
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        IS_SPEC_DECODING=num_accepted_tokens is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)
