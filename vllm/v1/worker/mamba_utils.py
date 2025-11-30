# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from typing import Any

import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
)
from vllm.triton_utils import tl, triton
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import GPUInputBatch


# 已阅
@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        curr_src_ptr = (src_ptr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst_ptr = (dst_ptr + i + offsets).to(tl.pointer_type(tl.uint8))

        data = tl.load(curr_src_ptr, mask=mask)
        tl.store(curr_dst_ptr, data, mask=mask)


# 已阅
def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    BLOCK_SIZE = 1024
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


# 已阅
# 说明：要求存在 Mamba 组且所有 Mamba 组的 MambaSpec 相同
def get_mamba_groups(kv_cache_config: KVCacheConfig) -> tuple[list[int], MambaSpec]:
    mamba_group_ids: list[int] = []
    mamba_specs: list[MambaSpec] = []
    for i in range(len(kv_cache_config.kv_cache_groups)):
        kv_cache_spec = kv_cache_config.kv_cache_groups[i].kv_cache_spec
        if isinstance(kv_cache_spec, MambaSpec):
            mamba_group_ids.append(i)
            mamba_specs.append(kv_cache_spec)
    assert len(mamba_group_ids) > 0, "no mamba layers in the model"
    assert all(mamba_specs[0] == spec for spec in mamba_specs)
    return mamba_group_ids, mamba_specs[0]


# 已阅
def collect_mamba_copy_meta(
    # 说明：输出
    src_state_list: list[int],
    # 说明：输出
    dest_state_list: list[int],
    # 说明：输出
    num_elements_list: list[int],
    kv_cache_config: KVCacheConfig,
    # 说明：MambaStateCopyFuncCalculator 类中有获取每个模型 copy func 列表的方法，
    # 包含拷贝模型的 conv_state 和 ssm_state 的不同方法
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
):
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    for mamba_group_id in mamba_group_ids:
        # 说明：当前 KV Cache 组的 Block IDs
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        layer_names = kv_cache_config.kv_cache_groups[mamba_group_id].layer_names
        for layer_name in layer_names:
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache[0]
            for state, state_copy_func in zip(kv_caches, mamba_state_copy_funcs):
                copy_spec = state_copy_func(
                    # 说明：accept_token_bias + 1 是上一轮的 num_accepted_tokens
                    state, block_ids, src_block_idx, accept_token_bias + 1
                )

                # 说明：需要执行拷贝的起始地址，
                # 对于 conv state 可能是 src block 的中间位置；对应 ssm state 可能是后面 block 的起始位置
                src_state_list.append(copy_spec.start_addr)
                # 说明：目标 block 的起始地址
                dest_state_list.append(state[dest_block_id].data_ptr())
                # 说明：需要拷贝的总的字节数
                num_elements_list.append(copy_spec.num_elements * state.element_size())


# 已阅
def do_mamba_copy_block(
    src_state_list: list[int],
    dest_state_list: list[int],
    num_elements_list: list[int],
):
    if len(src_state_list) == 0:
        return
    assert len(src_state_list) == len(dest_state_list)
    assert len(src_state_list) == len(num_elements_list)
    src_state_ptrs = torch.tensor(src_state_list, device="cuda", dtype=torch.int64)
    dst_state_ptrs = torch.tensor(dest_state_list, device="cuda", dtype=torch.int64)
    num_elements = torch.tensor(num_elements_list, device="cuda", dtype=torch.int32)

    batch_memcpy(src_state_ptrs, dst_state_ptrs, num_elements)


# 已阅
# 说明：只有在 align 模式下才会调用这个函数
def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    # 说明：MambaStateCopyFuncCalculator 类中有获取每个模型 copy func 列表的方法，
    # 包含拷贝模型的 conv_state 和 ssm_state 的不同方法
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
):
    """
    Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids):
        mamba_state_idx.pop(req_id, None)

    src_state_list: list[int] = []
    dest_state_list: list[int] = []
    num_elements_list: list[int] = []
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # new / resumed request, no previous state
            # if num_computed_tokens is 0, prev_state_idx will be -1
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_blocks = len(req_state.block_ids[mamba_group_ids[0]])

        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        # 说明：我们总是把当前的 running 状态保存在倒数 (1 + num_speculative_blocks) 个 block 中，上面的例子中，
        # 就是保存到倒数第 3 (= 1 + 2) 个 block 中，这个 block 的 0-based index 是 1 (= 4 - 1 - 2)
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            # 说明：搜集拷贝数据
            collect_mamba_copy_meta(
                src_state_list,
                dest_state_list,
                num_elements_list,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                # 说明：last computed token 对应的 block idx
                prev_state_idx,
                # 说明：running 状态所在的 block，倒数 (1 + num_speculative_blocks) 个 block
                curr_state_idx,
                # 说明：上一轮 num_accepted_tokens - 1
                input_batch.num_accepted_tokens_cpu[i] - 1,
                req_state,
                forward_context,
            )
            # 理解：初始化本轮的 num_accepted_tokens，尤其是对于在 postprocess_mamba 中没有初始化的请求 
            input_batch.num_accepted_tokens_cpu[i] = 1
    # 说明：执行数据拷贝
    do_mamba_copy_block(src_state_list, dest_state_list, num_elements_list)


# 说明：只有在 align 模式下才会调用这个函数
def postprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    input_batch: GPUInputBatch,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: tuple[MambaStateCopyFunc, ...],
):
    """
    If a blocks is converted from partial block to full block in this step, copy the
    state from the block for running state to the new full block.
    """
    num_scheduled_tokens_dict = scheduler_output.num_scheduled_tokens
    scheduled_spec_decode_tokens_dict = scheduler_output.scheduled_spec_decode_tokens
    num_accepted_tokens_cpu = input_batch.num_accepted_tokens_cpu
    # NOTE: can be optimized as this function always returns the same result
    mamba_group_ids, mamba_spec = get_mamba_groups(kv_cache_config)
    src_state_list: list[int] = []
    dest_state_list: list[int] = []
    num_elements_list: list[int] = []
    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        # 说明：本轮开始时已经 computed 的 token 数量
        num_computed_tokens = req_state.num_computed_tokens
        # 说明：本轮调度的 draft 的 token 数量
        num_draft_tokens = len(scheduled_spec_decode_tokens_dict.get(req_id, []))
        # 说明：本轮 scheduled 的总 token 数量
        num_scheduled_tokens = num_scheduled_tokens_dict[req_id]
        # 说明：本轮 accepted 的 token 数量，包括 bonus token 和 draft token
        num_accepted_tokens = num_accepted_tokens_cpu[i]
        # 理解/问题：num_scheduled_tokens = 1 + num_draft_tokens
        # 本轮最初 running state 所经过的 token 数量（已缓存的 + 参与计算但未缓存的）?
        num_tokens_running_state = (
            num_computed_tokens + num_scheduled_tokens - num_draft_tokens
        )
        # 说明：num_tokens_running_state + num_accepted_tokens 是最新 running state 所经过的 token 数量，
        # 减 1 是为了下一步正确计算 block 数量
        new_num_computed_tokens = num_tokens_running_state + num_accepted_tokens - 1
        # 说明：cache block 必须是满的
        aligned_new_computed_tokens = (
            new_num_computed_tokens // mamba_spec.block_size * mamba_spec.block_size
        )
        # TODO: how to ensure all blocks that cache_blocks called are cached here?
        if aligned_new_computed_tokens >= num_tokens_running_state:
            # 说明：有多少个 accepted token 需要补充（至 block 结尾）
            accept_token_bias = aligned_new_computed_tokens - num_tokens_running_state
            src_block_idx = mamba_state_idx[req_id]
            dest_block_idx = aligned_new_computed_tokens // mamba_spec.block_size - 1
            collect_mamba_copy_meta(
                src_state_list,
                dest_state_list,
                num_elements_list,
                kv_cache_config,
                mamba_state_copy_funcs,
                mamba_group_ids,
                # 说明：running 状态所在的 block
                src_block_idx,
                # 说明：最新的 full block
                dest_block_idx,
                accept_token_bias,
                req_state,
                forward_context,
            )
            # 理解：当 src_block_ids != dest_block_idx 时，num_accepted_tokens 不会被重新初始化，
            # 问题：在 preprocess_mamba 中会用到？
            if src_block_idx == dest_block_idx:
                # 理解：当 src_block_idx == dest_block_idx，
                # 如果 accepted_token == 0，此时不需要拷贝，collect_mamba_copy_meta 内部会直接返回；
                # 如果 accepted_token > 0，此时需要把运行状态从 
                num_accepted_tokens_cpu[i] = 1
    do_mamba_copy_block(src_state_list, dest_state_list, num_elements_list)
