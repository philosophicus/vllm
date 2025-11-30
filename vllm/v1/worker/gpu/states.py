# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_uva_available
from vllm.utils.torch_utils import get_cuda_view_from_cpu_tensor
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu.sample.penalties import bincount

_NP_INT64_MIN = np.iinfo(np.int64).min
_NP_INT64_MAX = np.iinfo(np.int64).max
NO_LORA_ID = 0


# 说明：维护每个请求的状态，包括请求的各种参数和中间结果
class RequestState:
    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        num_speculative_steps: int,
        vocab_size: int,
        device: torch.device,
        pin_memory: bool,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size
        self.device = device
        self.pin_memory = pin_memory

        # 问题：请求 ID 和索引的映射，索引在 BatchUpdate 中会用到？
        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        # 说明：空闲位置索引列表
        self.free_indices = list(range(max_num_reqs))
        self.extra_data: dict[str, ExtraData] = {}

        # 说明：请求的 prompt_len 是固定不变的，只在首次添加 request 时设置
        self.prompt_len = np.zeros(self.max_num_reqs, dtype=np.int32)
        # NOTE(woosuk): This tensor can be extremely large (e.g., several GBs)
        # depending on the configured max_num_reqs and max_model_len.
        # 说明：prefill_token_ids = prompt_token_ids + output_token_ids
        self.prefill_token_ids = UvaBuffer(
            self.max_num_reqs, self.max_model_len, dtype=torch.int32
        )
        # NOTE(woosuk): We don't use UVA for prefill_len because its GPU view
        # can be used outside of update_states and prepare_inputs.
        # Without async barrier, using UVA can cause race conditions.
        # 说明：每个请求的 prefill_len，是 prompt_token_ids + output_token_ids 的长度
        # 问题：没有使用 UVA，为什么不会产生竞态条件？
        self.prefill_len = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        # Number of computed tokens.
        self.num_computed_prefill_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

        # Last sampled tokens.
        self.last_sampled_tokens = torch.zeros(
            self.max_num_reqs,
            1,
            dtype=torch.int64,
            device=device,
        )

        # Draft tokens.
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        # 说明：记录下一个需要 prefill 的 token id，原因是本轮没有完成 prefill，需要在下一轮继续
        self.next_prefill_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )

        # LoRA.
        self.lora_ids = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.lora_ids.fill(NO_LORA_ID)

        # Sampling parameters.
        self.temperature = self._make_param(self.max_num_reqs, torch.float32)
        self.top_p = self._make_param(self.max_num_reqs, torch.float32)
        self.top_k = self._make_param(self.max_num_reqs, torch.int32)
        self.min_p = self._make_param(self.max_num_reqs, torch.float32)
        self.repetition_penalty = self._make_param(self.max_num_reqs, torch.float32)
        self.frequency_penalty = self._make_param(self.max_num_reqs, torch.float32)
        self.presence_penalty = self._make_param(self.max_num_reqs, torch.float32)
        self.seeds = self._make_param(self.max_num_reqs, torch.int64)

        # 说明：每个 output token 需要返回的对数概率数量
        self.num_logprobs = np.empty(self.max_num_reqs, dtype=np.int32)
        # -1 means no logprobs are requested.
        self.num_logprobs.fill(-1)
        # 说明：是否需要返回 prompt token 的对数概率
        self.needs_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)

        # Statistics for penalties.
        # 说明：prompt 的 binary mask
        self.prompt_bin_mask = torch.zeros(
            self.max_num_reqs,
            cdiv(self.vocab_size, 32),
            dtype=torch.int32,
            device=self.device,
        )
        # TODO(woosuk): This tensor is rarely used but can be extremely large.
        # Optimize the memory usage.
        # 说明：output 的词表 token 计数
        self.output_bin_counts = torch.zeros(
            self.max_num_reqs, self.vocab_size, dtype=torch.int32, device=self.device
        )

    def _make_param(self, size: int, dtype: torch.dtype) -> "Param":
        return Param(size, dtype=dtype, device=self.device, pin_memory=self.pin_memory)

    def _make_buffer(self, size: int, dtype: torch.dtype) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    # 已阅
    def add_request(
        self,
        req_id: str,
        prompt_len: int,
        prefill_token_ids: list[int],
        num_computed_tokens: int,
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None,
    ) -> None:
        assert len(self.free_indices) > 0, "No free indices"
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id
        self.extra_data[req_id] = ExtraData(lora_request)

        self.prompt_len[req_idx] = prompt_len
        prefill_len = len(prefill_token_ids)
        assert prefill_len >= prompt_len, (
            f"prefill_len {prefill_len} < prompt_len {prompt_len}"
        )
        self.prefill_len.np[req_idx] = prefill_len
        self.prefill_token_ids.np[req_idx, :prefill_len] = prefill_token_ids

        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        # FIXME(woosuk): This triggers a GPU operation whenever adding a new request.
        # Optimize this.
        # 说明：使用 torch.zeros 初始化的 num_computed_tokens 在 GPU 上
        self.num_computed_tokens[req_idx] = num_computed_tokens

        if lora_request is not None:
            self.lora_ids[req_idx] = lora_request.lora_int_id
        else:
            self.lora_ids[req_idx] = NO_LORA_ID

        self.temperature.np[req_idx] = sampling_params.temperature
        self.top_p.np[req_idx] = sampling_params.top_p
        if 0 < sampling_params.top_k < self.vocab_size:
            top_k = sampling_params.top_k
        else:
            top_k = self.vocab_size
        self.top_k.np[req_idx] = top_k
        self.min_p.np[req_idx] = sampling_params.min_p
        self.repetition_penalty.np[req_idx] = sampling_params.repetition_penalty
        self.frequency_penalty.np[req_idx] = sampling_params.frequency_penalty
        self.presence_penalty.np[req_idx] = sampling_params.presence_penalty

        if use_penalty(sampling_params):
            bincount(
                self.prefill_token_ids.gpu[req_idx],
                prefill_len,
                prompt_len,
                self.prompt_bin_mask[req_idx],
                self.output_bin_counts[req_idx],
            )

        if sampling_params.seed is not None:
            seed = sampling_params.seed
        else:
            seed = np.random.randint(_NP_INT64_MIN, _NP_INT64_MAX)
        self.seeds.np[req_idx] = seed

        if sampling_params.logprobs is not None:
            # 说明：Note that the implementation
            # follows the OpenAI API: The API will always return the log probability of
            # the sampled token, so there may be up to `logprobs+1` elements in the
            # response
            num_logprobs = sampling_params.logprobs
        else:
            # 说明：When set to -1, return all `vocab_size` log probabilities
            num_logprobs = -1
        self.num_logprobs[req_idx] = num_logprobs

        # For now, only support prompt logprobs for the prompt tokens.
        # 说明：整型变量转换为布尔值
        needs_prompt_logprobs = sampling_params.prompt_logprobs is not None
        self.needs_prompt_logprobs[req_idx] = needs_prompt_logprobs

    # 已阅
    def remove_request(self, req_id: str) -> None:
        self.extra_data.pop(req_id, None)
        req_idx = self.req_id_to_index.pop(req_id, None)
        if req_idx is None:
            # Request not found.
            return
        self.index_to_req_id.pop(req_idx, None)
        self.free_indices.append(req_idx)

    # 已阅
    def make_sampling_metadata(
        self,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
    ) -> SamplingMetadata:
        temperature = self.temperature.np[idx_mapping_np]
        temperature = self.temperature.copy_np_to_gpu(temperature)

        top_p = self.top_p.np[idx_mapping_np]
        no_top_p = np.all(top_p == 1.0)
        # 说明：top_p 不全为 1.0 时，才需要传递 top_p 参数
        top_p = self.top_p.copy_np_to_gpu(top_p) if not no_top_p else None

        top_k = self.top_k.np[idx_mapping_np]
        no_top_k = np.all(top_k == self.vocab_size)
        # 说明：top_k 不全为 vocab_size 时，才需要传递 top_k 参数
        top_k = self.top_k.copy_np_to_gpu(top_k) if not no_top_k else None

        min_p = self.min_p.np[idx_mapping_np]
        no_min_p = np.all(min_p == 0.0)
        # 说明：min_p 不全为 0.0 时，才需要传递 min_p 参数
        min_p = self.min_p.copy_np_to_gpu(min_p) if not no_min_p else None

        rep_penalty = self.repetition_penalty.np[idx_mapping_np]
        rep_penalty = self.repetition_penalty.copy_np_to_gpu(rep_penalty)
        freq_penalty = self.frequency_penalty.np[idx_mapping_np]
        freq_penalty = self.frequency_penalty.copy_np_to_gpu(freq_penalty)
        pres_penalty = self.presence_penalty.np[idx_mapping_np]
        pres_penalty = self.presence_penalty.copy_np_to_gpu(pres_penalty)

        seeds = self.seeds.np[idx_mapping_np]
        seeds = self.seeds.copy_np_to_gpu(seeds)

        num_logprobs = self.num_logprobs[idx_mapping_np]
        max_num_logprobs: int | None = int(np.max(num_logprobs))
        if max_num_logprobs == -1:
            max_num_logprobs = None

        return SamplingMetadata(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=rep_penalty,
            frequency_penalty=freq_penalty,
            presence_penalty=pres_penalty,
            seeds=seeds,
            pos=pos,
            max_num_logprobs=max_num_logprobs,
            idx_mapping=idx_mapping,
            prompt_bin_mask=self.prompt_bin_mask,
            output_bin_counts=self.output_bin_counts,
        )

    # 已阅
    def make_lora_inputs(
        self,
        req_ids: list[str],
        idx_mapping: np.ndarray,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[tuple[int, ...], tuple[int, ...], set[LoRARequest]]:
        lora_ids = self.lora_ids[idx_mapping]
        prompt_lora_mapping = tuple(lora_ids)
        # 说明：根据 num_scheduled_tokens 扩展 lora_ids，使得每个 token 都有对应的 lora_id
        token_lora_mapping = tuple(lora_ids.repeat(num_scheduled_tokens))

        active_lora_requests: set[LoRARequest] = set()
        # 说明：req_ids 为排序后的请求 ID 列表，与 idx_mapping 一一对应
        for req_id in req_ids:
            lora_request = self.extra_data[req_id].lora_request
            if lora_request is not None:
                active_lora_requests.add(lora_request)
        return prompt_lora_mapping, token_lora_mapping, active_lora_requests


# 已阅
class Param:
    def __init__(
        self,
        size: int,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
    ):
        # 说明：保存紧凑的参数值
        self.buffer = CpuGpuBuffer(
            size,
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )
        # 说明：保存稀疏的参数值
        self.np = np.zeros_like(self.buffer.np)

    def copy_np_to_gpu(self, x: np.ndarray) -> torch.Tensor:
        n = x.shape[0]
        self.buffer.np[:n] = x
        return self.buffer.copy_to_gpu(n)


# 已阅
@dataclass
class ExtraData:
    lora_request: LoRARequest | None
    in_progress_prompt_logprobs: list[LogprobsTensors] = field(default_factory=list)


# 已阅
# 说明：UVA (Unified Virtual Addressing) 缓冲区，用于在 CPU 和 GPU 之间共享数据
# UVA 的核心是让 CPU 和 GPU 共享同一个虚拟地址空间，即 CPU 和 GPU 可以访问同一个地址
class UvaBuffer:
    def __init__(self, *size: int | torch.SymInt, dtype: torch.dtype):
        assert is_uva_available()
        self.cpu = torch.zeros(*size, dtype=dtype, device="cpu", pin_memory=True)
        self.np = self.cpu.numpy()
        self.gpu = get_cuda_view_from_cpu_tensor(self.cpu)


# 已阅
def use_penalty(sampling_params: SamplingParams) -> bool:
    return (
        sampling_params.repetition_penalty != 1.0
        or sampling_params.frequency_penalty != 0.0
        or sampling_params.presence_penalty != 0.0
    )
