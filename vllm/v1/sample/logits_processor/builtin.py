# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, TypeVar

import torch

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

T = TypeVar("T")


# 已阅
# 说明：Min-P 的作用是确保生成的 token 概率不低于某个阈值，阈值是基于当前最大概率乘以 min_p 得到的
# 实现思路：对于那些启用了 min_p 的请求，计算出对应的阈值，然后将 logits 中小于该阈值的 token 的 logits 设置为 -inf
class MinPLogitsProcessor(LogitsProcessor):
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        # 说明：用来记录有多少请求启用了 min_p
        self.min_p_count: int = 0

        self.min_p_cpu_tensor = torch.zeros(
            (max_num_reqs,), dtype=torch.float32, device="cpu", pin_memory=is_pin_memory
        )
        self.min_p_cpu = self.min_p_cpu_tensor.numpy()

        self.use_double_tensor = torch.device(device).type != "cpu"

        if self.use_double_tensor:
            # Pre-allocated device tensor
            self.min_p_device: torch.Tensor = torch.empty(
                (max_num_reqs,), dtype=torch.float32, device=device
            )
        else:
            self.min_p_device = self.min_p_cpu_tensor
        # Current slice of the device tensor
        self.min_p: torch.Tensor = self.min_p_device[:0]

    def is_argmax_invariant(self) -> bool:
        """Min-p never impacts greedy sampling"""
        return True

    def get_min_p_by_index(self, index: int) -> float:
        return float(self.min_p_cpu[index])

    # 说明：注意处理顺序：removed, added, moved
    # 但下面看起来是先处理了 added，再处理 removed，最后处理 moved
    # 需要保证删除的请求索引不会出现在 added 列表中
    def update_state(self, batch_update: BatchUpdate | None):
        if not batch_update:
            return

        needs_update = False
        # Process added requests.
        for index, params, _, _ in batch_update.added:
            min_p = params.min_p
            min_p_before = self.min_p_cpu[index]
            if min_p_before != min_p:
                needs_update = True
                # 说明：只要新旧 min_p 不相等，就设置为新的 min_p
                self.min_p_cpu[index] = min_p
                if min_p and not min_p_before:
                    self.min_p_count += 1
                elif not min_p and min_p_before:
                    self.min_p_count -= 1

        if self.min_p_count:
            # Process removed requests.
            # 说明：如果被移除的请求启用了 min_p，则将其计数减一
            if batch_update.removed:
                needs_update = True
                for index in batch_update.removed:
                    if self.min_p_cpu[index]:
                        self.min_p_cpu[index] = 0
                        self.min_p_count -= 1

            # Process moved requests, unidirectional (a->b) and swap (a<->b).
            for adx, bdx, direct in batch_update.moved:
                min_p_a, min_p_b = self.min_p_cpu[adx], self.min_p_cpu[bdx]
                if min_p_a != min_p_b:
                    needs_update = True
                    # b 的值一定要被替换成 a 的值
                    self.min_p_cpu[bdx] = min_p_a
                    if direct == MoveDirectionality.SWAP:
                        # swap 的话，a 的值也要被替换成 b 的值
                        self.min_p_cpu[adx] = min_p_b
                if direct == MoveDirectionality.UNIDIRECTIONAL:
                    # 单向时要清理值
                    if min_p_a:
                        self.min_p_cpu[adx] = 0
                    # 如果原来的 b 有值，说明被覆盖掉了，计数要减一
                    if min_p_b:
                        self.min_p_count -= 1

        # Update tensors if needed.
        size = batch_update.batch_size
        if self.min_p_count and (needs_update or self.min_p.shape[0] != size):
            # 说明：将 min_p_cpu_tensor 的前 size 个元素复制到 min_p_device 中
            self.min_p = self.min_p_device[:size]
            if self.use_double_tensor:
                self.min_p.copy_(self.min_p_cpu_tensor[:size], non_blocking=True)
            # 说明：维度调整为 (size, 1)
            self.min_p.unsqueeze_(1)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.min_p_count:
            return logits

        # Convert logits to probability distribution
        probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # Calculate maximum probabilities per sequence
        max_probabilities = torch.amax(probability_values, dim=-1, keepdim=True)
        # Adjust min_p
        adjusted_min_p = max_probabilities.mul_(self.min_p)
        # Identify valid tokens using threshold comparison
        invalid_token_mask = probability_values < adjusted_min_p
        # Apply mask using boolean indexing
        logits.masked_fill_(invalid_token_mask, -float("inf"))
        return logits


# 已阅
# 说明：给请求的某些 token id 添加 bias 值
# 问题：看实现的话 {token id: bias 值} 是请求级别的，因为来源是 sampling params，
# 所以需要确认 sampling params 是否也是请求级别的，看会不会出现重复存储
class LogitBiasLogitsProcessor(LogitsProcessor):
    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.pin_memory = is_pin_memory
        # 问题：request index -> {？: ？}
        self.biases: dict[int, dict[int, float]] = {}

        self.bias_tensor: torch.Tensor = torch.tensor(())
        # 说明：(req_idx_tensor, tok_id_tensor)
        self.logits_slice = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

    def is_argmax_invariant(self) -> bool:
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    def update_state(self, batch_update: BatchUpdate | None):
        # needs_update 表示是否对 self.biases 字典进行了更新
        needs_update = process_dict_updates(
            self.biases, batch_update, lambda params, _, __: params.logit_bias or None
        )

        # Update tensors if needed.
        if needs_update:
            # 三个长度相同的大列表，分别存储请求索引、token id 和对应的 bias 值，
            # 相同位置的元素构成一个三元组（request index, token id, bias 值）
            reqs: list[int] = []
            tok_ids: list[int] = []
            biases: list[float] = []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = self._device_tensor(biases, torch.float32)
            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            data, device="cpu", dtype=dtype, pin_memory=self.pin_memory
        ).to(device=self.device, non_blocking=True)

    # 说明：logits 的 shape 是 (num_requests, vocab_size)
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


# 已阅
# 说明：作用是确保生成的最少 token 数
# 实现思路：对于那些还没有生成足够 token 的请求，将它们的停止词 token id 的 logits 设置为 -inf
class MinTokensLogitsProcessor(LogitsProcessor):
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ):
        # index -> (min_toks, output_token_ids, stop_token_ids)
        self.device = device
        self.pin_memory = is_pin_memory
        self.min_toks: dict[int, tuple[int, Sequence[int], set[int]]] = {}

        # (req_idx_tensor,eos_tok_id_tensor)
        self.logits_slice: tuple[torch.Tensor, torch.Tensor] = (
            self._device_tensor([], torch.int32),
            self._device_tensor([], torch.int32),
        )

        self.neg_inf_tensor = torch.tensor(
            -float("inf"), dtype=torch.float32, device=self.device
        )

    def is_argmax_invariant(self) -> bool:
        """By censoring stop tokens, min-tokens can change the outcome
        of the argmax operation in greedy sampling."""
        return False

    @staticmethod
    def add_request(
        params: SamplingParams, _: list[int] | None, output_tok_ids: list[int]
    ) -> tuple[int, Sequence[int], set[int]] | None:
        min_tokens = params.min_tokens
        if not min_tokens or len(output_tok_ids) >= min_tokens:
            return None
        # 如果限制了生成的最少 token 数且数量未达到，则同时返回停止词 token id 集合
        return min_tokens, output_tok_ids, params.all_stop_token_ids

    def update_state(self, batch_update: BatchUpdate | None):
        needs_update = process_dict_updates(
            self.min_toks, batch_update, self.add_request
        )
        if self.min_toks:
            # Check for any requests that have attained their min tokens.
            # 说明：删除那些已经生成足够 token 的请求
            to_remove = tuple(
                index
                for index, (min_toks, out_tok_ids, _) in self.min_toks.items()
                if len(out_tok_ids) >= min_toks
            )
            if to_remove:
                needs_update = True
                for index in to_remove:
                    del self.min_toks[index]

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            for req, (_, _, stop_tok_ids) in self.min_toks.items():
                reqs.extend([req] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)

            self.logits_slice = (
                self._device_tensor(reqs, torch.int32),
                self._device_tensor(tok_ids, torch.int32),
            )

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(
            data, device="cpu", dtype=dtype, pin_memory=self.pin_memory
        ).to(device=self.device, non_blocking=True)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.min_toks:
            # Inhibit EOS token for requests which have not reached min length
            logits.index_put_(self.logits_slice, self.neg_inf_tensor)
        return logits


# 已阅
# 作用：根据 new_state 函数对 batch_update 内容的处理结果，更新 req_entries 字典
# 返回结果非 None 则保存到 req_entries 中，为 None 则从 req_entries 中删除请求 index
# 返回值表示 req_entries 字典是否有更新
# 相关文档中注明的处理顺序是：removed, added, moved，实际代码中是先处理 added，
# 再处理 removed，最后处理 moved，这与 
# https://docs.vllm.ai/en/latest/design/logits_processors/#how-the-vllm-engine-builds-the-batchupdate-data-structure
# 中描述的维护 BatchUpdate 的顺序一致，实际上可以保证删除的请求索引不会出现在 added 列表中，
# 也就是可以先处理 added 再处理 removed，不会影响最终结果
def process_dict_updates(
    req_entries: dict[int, T],
    batch_update: BatchUpdate | None,
    new_state: Callable[[SamplingParams, list[int] | None, list[int]], T | None],
) -> bool:
    """Utility function to update dict state for sparse LogitsProcessors."""

    if not batch_update:
        # Nothing to do.
        return False

    updated = False
    for index, params, prompt_tok_ids, output_tok_ids in batch_update.added:
        if (state := new_state(params, prompt_tok_ids, output_tok_ids)) is not None:
            req_entries[index] = state
            updated = True
        elif req_entries.pop(index, None) is not None:
            updated = True

    if req_entries:
        # Process removed requests.
        for index in batch_update.removed:
            if req_entries.pop(index, None):
                updated = True

        # Process moved requests, unidirectional (a->b) and
        # swapped (a<->b)
        for a_index, b_index, direct in batch_update.moved:
            a_entry = req_entries.pop(a_index, None)
            b_entry = req_entries.pop(b_index, None)
            if a_entry is not None:
                req_entries[b_index] = a_entry
                updated = True
            if b_entry is not None:
                updated = True
                if direct == MoveDirectionality.SWAP:
                    req_entries[a_index] = b_entry

    return updated
