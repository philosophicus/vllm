# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import torch

from vllm import SamplingParams

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class MoveDirectionality(Enum):
    # One-way i1->i2 req move within batch
    UNIDIRECTIONAL = auto()
    # Two-way i1<->i2 req swap within batch
    SWAP = auto()


# Batch indices of any removed requests.
RemovedRequest = int

# (index, params, prompt_tok_ids, output_tok_ids) tuples for new
# requests added to the batch.
# 说明：Note that the output token ids list grows with each engine step, 
# and this growth is visible to the logits processor because output 
# token ids are passed by reference. This is important for 
# LogitsProcessors that take into account the tokens generated so far.
AddedRequest = tuple[int, SamplingParams, list[int] | None, list[int]]

# (index 1, index 2, directionality) tuples representing
# one-way moves or two-way swaps of requests in batch
MovedRequest = tuple[int, int, MoveDirectionality]


# 已阅
@dataclass(frozen=True)
class BatchUpdate:
    """Persistent batch state change info for logitsprocs"""

    batch_size: int  # Current num reqs in batch

    # Metadata for requests added to, removed from, and moved
    # within the persistent batch.
    #
    # Key assumption: the `output_tok_ids` list (which is an element of each
    # tuple in `added`) is a reference to the request's running output tokens
    # list; via this reference, the logits processors always see the latest
    # list of generated output tokens.
    #
    # NOTE:
    # * Added or moved requests may replace existing requests with the same
    #   index.
    # * Operations should be processed in the following order:
    #   - removed, added, moved
    removed: Sequence[RemovedRequest]
    added: Sequence[AddedRequest]
    moved: Sequence[MovedRequest]


# 已阅
# 说明：所有 LogitsProcessor 的抽象基类
# 系统有内置的子类，也支持自定义
class LogitsProcessor(ABC):
    # 说明：抛异常后调用方（API 内）会捕获并处理，返回错误信息给用户
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        """Validate sampling params for this logits processor.

        Raise ValueError for invalid ones.
        """
        return None

    @abstractmethod
    def __init__(
        self, vllm_config: "VllmConfig", device: torch.device, is_pin_memory: bool
    ) -> None:
        raise NotImplementedError

    # 说明：logits 的 shape 是 (num_requests, vocab_size)
    @abstractmethod
    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply LogitsProcessor to batch logits tensor.

        The updated tensor must be returned but may be
        modified in-place.
        """
        raise NotImplementedError

    @abstractmethod
    def is_argmax_invariant(self) -> bool:
        """True if logits processor has no impact on the
        argmax computation in greedy sampling.
        NOTE: may or may not have the same value for all
        instances of a given LogitsProcessor subclass,
        depending on subclass implementation.
        """
        raise NotImplementedError

    
    # 说明：注意处理顺序：removed, added, moved
    @abstractmethod
    def update_state(
        self,
        batch_update: "BatchUpdate | None",
    ) -> None:
        """Called when there are new output tokens, prior
        to each forward pass.

        Args:
            batch_update: Non-None iff there have been changes
                to the batch makeup.
        """
        raise NotImplementedError
