# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import torch

from vllm.config.cache import MambaDType
from vllm.config.model import ModelDType
from vllm.distributed import divide
from vllm.utils.torch_utils import (
    STR_DTYPE_TO_TORCH_DTYPE,
    get_kv_cache_torch_dtype,
)


class MambaStateDtypeCalculator:
    @classmethod
    def linear_attention_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        # TODO (tdoublep) requires testing
        if mamba_cache_dtype == "float32":
            raise ValueError("fp32 state for minimax is not yet supported")
        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        return (state_dtype,)

    # 已阅
    @classmethod
    def mamba1_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        return cls._mamba_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )

    # 已阅
    @classmethod
    def mamba2_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        return cls._mamba_state_dtype(
            model_dtype, mamba_cache_dtype, mamba_ssm_cache_dtype
        )

    # 已阅
    @classmethod
    def _mamba_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
        mamba_ssm_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        conv_state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        if mamba_ssm_cache_dtype == "auto":
            temporal_state_dtype = conv_state_dtype
        else:
            temporal_state_dtype = STR_DTYPE_TO_TORCH_DTYPE[mamba_ssm_cache_dtype]

        return (conv_state_dtype, temporal_state_dtype)

    @classmethod
    def short_conv_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, ...]:
        conv_state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        return (conv_state_dtype,)

    @classmethod
    def gated_delta_net_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ) -> tuple[torch.dtype, torch.dtype]:
        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        return (state_dtype, state_dtype)

    @classmethod
    def kda_state_dtype(
        cls,
        model_dtype: ModelDType | torch.dtype,
        mamba_cache_dtype: MambaDType,
    ):
        state_dtype = get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)
        return (state_dtype, state_dtype, state_dtype, torch.float32)


class MambaStateShapeCalculator:
    @classmethod
    def linear_attention_state_shape(
        cls,
        num_heads: int,
        tp_size: int,
        head_dim: int,
    ) -> tuple[tuple[int, int, int], ...]:
        state_shape = (num_heads // tp_size, head_dim, head_dim)
        return (state_shape,)

    # 已阅
    @classmethod
    def mamba1_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        state_size: int,
        conv_kernel: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        conv_state_shape = (divide(intermediate_size, tp_world_size), conv_kernel - 1)

        # 说明：(P, N)
        temporal_state_shape = (divide(intermediate_size, tp_world_size), state_size)

        conv_state_shape = conv_state_shape[1], conv_state_shape[0]

        return conv_state_shape, temporal_state_shape

    # 已阅
    @classmethod
    def mamba2_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        n_groups: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel: int,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = n_groups + cls.extra_groups_for_head_shards(n_groups, tp_world_size)
        # heads and n_groups are TP-ed
        conv_dim = intermediate_size + 2 * n_groups * state_size

        # contiguous along 'dim' axis
        conv_state_shape = (conv_kernel - 1, divide(conv_dim, tp_world_size))

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (h_heads, head_dim, state_size) = (128, 64, 128)
        # 说明：与上面注释不一致，num_heads 也做了 shard 
        temporal_state_shape = (divide(num_heads, tp_world_size), head_dim, state_size)
        return conv_state_shape, temporal_state_shape

    @classmethod
    def short_conv_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        conv_kernel: int,
    ) -> tuple[tuple[int, int]]:
        conv_dim = divide(intermediate_size, tp_world_size)
        conv_state_shape = (conv_kernel - 1, conv_dim)
        return (conv_state_shape,)

    # 已阅
    @classmethod
    def extra_groups_for_head_shards(cls, ngroups: int, tp_size: int):
        """Compute the increase in group numbers to account for
        replication in order to accompany the head shards."""

        # in the case ngoups % tp_size == 0, this will be zero
        if ngroups % tp_size == 0:
            return 0

        # for n_groups == 1, this is exactly tp_size - n_groups
        return tp_size - ngroups

    @classmethod
    def gated_delta_net_state_shape(
        cls,
        tp_world_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        num_spec: int = 0,
    ):
        conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
        conv_state_shape = (
            divide(conv_dim, tp_world_size),
            conv_kernel_size - 1 + num_spec,
        )

        conv_state_shape = conv_state_shape[1], conv_state_shape[0]

        temporal_state_shape = (
            divide(num_v_heads, tp_world_size),
            head_v_dim,
            head_k_dim,
        )
        return conv_state_shape, temporal_state_shape

    @classmethod
    def kda_state_shape(
        cls,
        tp_world_size: int,
        num_heads: int,
        head_dim: int,
        num_k_heads: int | None = None,
        head_k_dim: int | None = None,
        conv_kernel_size: int = 4,
        num_spec: int = 0,
    ) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int, int]]:
        if num_k_heads is None:
            num_k_heads = num_heads
        if head_k_dim is None:
            head_k_dim = head_dim

        proj_size = num_heads * head_dim
        proj_k_size = num_k_heads * head_k_dim

        conv_state_shape = (divide(proj_size, tp_world_size), conv_kernel_size - 1)
        conv_state_k_shape = (divide(proj_k_size, tp_world_size), conv_kernel_size - 1)
        recurrent_state_shape = (divide(num_heads, tp_world_size), head_dim, head_dim)

        conv_state_shape = conv_state_shape[1], conv_state_shape[0]
        conv_state_k_shape = conv_state_k_shape[1], conv_state_k_shape[0]
        return (
            conv_state_shape,
            conv_state_k_shape,
            conv_state_k_shape,
            recurrent_state_shape,
        )


# 已阅
@dataclass
class MambaCopySpec:
    """
    Data class specifying the memory-copy parameters for Mamba states used for
    prefix caching in align mode.

    Attributes:
        start_addr (int): Starting address for the memory copy operation.
        num_elements (int): Number of elements to copy from the starting address.
    """

    start_addr: int
    num_elements: int


MambaStateCopyFunc: TypeAlias = Callable[
    [torch.Tensor, list[int], int, int], MambaCopySpec
]
"""
Type alias for a function that computes a MambaCopySpec for copying state slices.
Parameters:
  state: torch.Tensor - the Mamba state tensor (e.g., conv or temporal states).
  block_ids: list[int] - the list of block indices for the state to copy.
  cur_block_idx: int - current block index within `block_ids` to copy from.
  num_accepted_tokens: int - number of accepted tokens used to compute the copy offset.
      Range: 1 .. 1 + num_speculative_tokens (inclusive).
"""


# 已阅
def get_conv_copy_spec(
    state: torch.Tensor,
    # 说明：cache group 全部的 block ids
    block_ids: list[int],
    # 说明：last_computed_token 所在 block 的 index
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec:
    """Return a MambaCopySpec for copying a convolutional state slice."""
    src_block_id = block_ids[cur_block_idx]
    # 说明：conv_state 完整的 shape 是 (block_idx, conv_kernel - 1, dim)，conv_kernel 的值默认为 4
    # 理解：从 src_block 起始位置开始保存了 num_accepted_tokens 个 token 的状态，
    # 因此从结尾 token 的位置，即 num_accepted_tokens - 1 开始拷贝；
    # 同时也说明，last computed token 就是 num_accepted_tokens 中的第一个 token
    # 问题：conv_state token 维度是值默认为 3，那么也就是说明 num_accepted_tokens 最大为 3？
    src_state = state[src_block_id, num_accepted_tokens - 1 :]
    return MambaCopySpec(
        start_addr=src_state.data_ptr(), num_elements=src_state.numel()
    )


# 已阅
def get_temporal_copy_spec(
    state: torch.Tensor,
    # 说明：cache group 全部的 block ids
    block_ids: list[int],
    # 说明：last_computed_token 所在 block 的 index
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec:
    """Return a MambaCopySpec for copying a temporal state slice."""
    # 说明：对于 ssm state/temporal state，每个 block 只保存了一个 token 的状态，
    # 因此从末尾 block 的位置开始拷贝；last computed token 就是 num_accepted_tokens 中的第一个 token
    src_block_id = block_ids[cur_block_idx + num_accepted_tokens - 1]
    # 说明：ssm_state 完整的 shape 是 (block_idx, head_dim, state_size) 或者 (block_idx, h_heads, head_dim, state_size)
    src_state = state[src_block_id]
    return MambaCopySpec(
        start_addr=src_state.data_ptr(), num_elements=src_state.numel()
    )


get_full_copy_spec = get_temporal_copy_spec


class MambaStateCopyFuncCalculator:
    @classmethod
    def linear_attention_state_copy_func(cls):
        return (get_temporal_copy_spec,)

    @classmethod
    def mamba1_state_copy_func(cls):
        return (get_conv_copy_spec, get_temporal_copy_spec)

    @classmethod
    def mamba2_state_copy_func(cls):
        return get_conv_copy_spec, get_temporal_copy_spec

    @classmethod
    def short_conv_state_copy_func(cls):
        return (get_conv_copy_spec,)

    @classmethod
    def gated_delta_net_state_copy_func(cls):
        return (get_conv_copy_spec, get_temporal_copy_spec)

    @classmethod
    def kda_state_copy_func(cls):
        return (
            get_conv_copy_spec,
            get_conv_copy_spec,
            get_conv_copy_spec,
            get_temporal_copy_spec,
        )
