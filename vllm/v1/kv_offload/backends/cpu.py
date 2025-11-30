# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from collections.abc import Iterable

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import LoadStoreSpec
from vllm.v1.kv_offload.backend import Backend, BlockStatus
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec


# 已阅
class CPUBlockStatus(BlockStatus):
    _fields_ = BlockStatus._fields_ + [("block_id", ctypes.c_int64)]  # type: ignore

    def __init__(self, block_id: int):
        super().__init__()
        self.block_id = block_id


# 已阅
class CPUBackend(Backend):
    def __init__(self, block_size: int, num_blocks: int):
        super().__init__(block_size=block_size, medium=CPULoadStoreSpec.medium())

        # 说明：总块数
        self.num_blocks: int = num_blocks
        # 说明：分配过的块的数量，其中部分块可能已被释放，但仍参与计数；目前在用的块数通过
        # num_allocated_blocks - len(allocated_blocks_free_list) 计算
        # 说明：单调递增，用来作为新分配块的 block_id
        self.num_allocated_blocks: int = 0
        # 说明：分配过的块的空闲列表
        self.allocated_blocks_free_list: list[int] = []

    def get_num_free_blocks(self):
        # 说明：可分配的块数 = 总块数 - 在用块数 = 总块数 - （分配过的块数 - 分配过但当前空闲的块数）
        # = len(allocated_blocks_free_list) + (num_blocks - num_allocated_blocks)
        return (
            len(self.allocated_blocks_free_list)
            + self.num_blocks
            - self.num_allocated_blocks
        )

    def allocate_blocks(self, block_hashes: list[BlockHash]) -> list[BlockStatus]:
        # 说明：优先分配新块，然后再分配已释放的块
        # 新块的数量 = min(请求块数, 可分配新块数)
        num_fresh_blocks = min(
            len(block_hashes), self.num_blocks - self.num_allocated_blocks
        )
        num_reused_blocks = len(block_hashes) - num_fresh_blocks
        assert len(self.allocated_blocks_free_list) >= num_reused_blocks

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh_blocks):
            blocks.append(CPUBlockStatus(self.num_allocated_blocks))
            self.num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused_blocks):
            block_id = self.allocated_blocks_free_list.pop()
            blocks.append(CPUBlockStatus(block_id))

        return blocks

    def free(self, block: BlockStatus):
        assert isinstance(block, CPUBlockStatus)
        self.allocated_blocks_free_list.append(block.block_id)

    def get_load_store_spec(
        self, block_hashes: Iterable[BlockHash], blocks: Iterable[BlockStatus]
    ) -> LoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])
