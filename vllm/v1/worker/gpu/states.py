# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor


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
    ):
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_speculative_steps = num_speculative_steps
        self.vocab_size = vocab_size
        self.device = device

        # 问题：请求 ID 和索引的映射，索引在 BatchUpdate 中会用到？
        self.req_id_to_index: dict[str, int] = {}
        self.index_to_req_id: dict[int, str] = {}
        # 说明：空闲位置索引列表
        self.free_indices = list(range(max_num_reqs))

        # 说明：请求的 prompt_len 是固定不变的，只在首次添加 request 时设置
        self.prompt_len = np.zeros(self.max_num_reqs, dtype=np.int32)
        # NOTE(woosuk): This tensor can be extremely large (e.g., several GBs)
        # depending on the configured max_num_reqs and max_model_len.
        # To save GPU memory, we use UVA instead of GPU for this tensor.
        # 说明：prefill_token_ids = prompt_token_ids + output_token_ids
        self.prefill_token_ids = StagedWriteTensor(
            (self.max_num_reqs, self.max_model_len),
            dtype=torch.int32,
            device=device,
            uva_instead_of_gpu=True,
        )
        self.prefill_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)

        # Number of computed tokens.
        self.num_computed_prefill_tokens = np.zeros(self.max_num_reqs, dtype=np.int32)
        self.num_computed_tokens = StagedWriteTensor(
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
    ) -> None:
        assert len(self.free_indices) > 0, "No free indices"
        req_idx = self.free_indices.pop()
        self.req_id_to_index[req_id] = req_idx
        self.index_to_req_id[req_idx] = req_id

        self.prompt_len[req_idx] = prompt_len
        prefill_len = len(prefill_token_ids)
        assert prefill_len >= prompt_len, (
            f"prefill_len {prefill_len} < prompt_len {prompt_len}"
        )
        self.prefill_len.np[req_idx] = prefill_len
        self.prefill_token_ids.stage_write(req_idx, 0, prefill_token_ids)
        self.num_computed_prefill_tokens[req_idx] = num_computed_tokens
        self.num_computed_tokens.stage_write_elem(req_idx, num_computed_tokens)

    def apply_staged_writes(self) -> None:
        self.prefill_len.copy_to_uva()
        self.prefill_token_ids.apply_write()
        self.num_computed_tokens.apply_write()

    # 已阅
    def remove_request(self, req_id: str) -> None:
        req_idx = self.req_id_to_index.pop(req_id, None)
        if req_idx is None:
            # Request not found.
            return
        self.index_to_req_id.pop(req_idx, None)
        self.free_indices.append(req_idx)
