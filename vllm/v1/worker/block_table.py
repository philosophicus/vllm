# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cp_utils import get_total_cp_world_size

logger = init_logger(__name__)


# 已阅
# 说明：V1 版的 block table 实现
# 重要：处理了 allocation block size 和 kernel block size 不同的情况；
# 内部 block_size 使用了 kernel_block_size，所以 block IDs 对应的是基于 kernel block IDs；
# 对外接口则使用 allocation block size 的 block IDs
class BlockTable:
    def __init__(
        self,
        block_size: int,
        max_num_reqs: int,
        max_num_blocks_per_req: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        kernel_block_size: int,
        cp_kv_cache_interleave_size: int,
    ):
        """
        Args:
            block_size: Block size used for KV cache memory allocation
            max_num_reqs: Maximum number of concurrent requests supported.
            max_num_blocks_per_req: Maximum number of blocks per request.
            max_num_batched_tokens: Maximum number of tokens in a batch.
            pin_memory: Whether to pin memory for faster GPU transfers.
            device: Target device for the block table.
            kernel_block_size: The block_size of underlying attention kernel.
                Will be the same as `block_size` if `block_size` is supported
                by the attention kernel.
        """
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.pin_memory = pin_memory
        self.device = device

        if kernel_block_size == block_size:
            # Standard case: allocation and computation use same block size
            # No block splitting needed, direct mapping
            self.block_size = block_size
            self.blocks_per_kv_block = 1
            self.use_hybrid_blocks = False
        else:
            # Hybrid case: allocation block size differs from kernel block size
            # Memory blocks are subdivided to match kernel requirements
            # Example: 32-token memory blocks with 16-token kernel blocks
            # → Each memory block corresponds to 2 kernel blocks
            if block_size % kernel_block_size != 0:
                raise ValueError(
                    f"kernel_block_size {kernel_block_size} must divide "
                    f"kv_manager_block_size size {block_size} evenly"
                )

            self.block_size = kernel_block_size
            self.blocks_per_kv_block = block_size // kernel_block_size
            self.use_hybrid_blocks = True

        # 说明：max_num_blocks_per_req 是基于 max(
        #   cdiv(max_model_len, block_size * total_cp_world_size), 
        #   1 + num_speculative_tokens
        # ) 计算得到的，是 allocation block size 的数量
        # 补充：Adjust max_num_blocks_per_req for kernel blocks
        # self.blocks_per_kv_block 表示每个 KV cache block 包含多少个 kernel blocks
        self.max_num_blocks_per_req = max_num_blocks_per_req * self.blocks_per_kv_block

        # 补充：size 为 (请求数, 每请求最大 kernel block 数) 的二维张量，
        # 保存的是 kernel block IDs
        self.block_table = self._make_buffer(
            self.max_num_reqs, self.max_num_blocks_per_req, dtype=torch.int32
        )
        # 补充：记录每个请求实际使用的 kernel block 数
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)

        # 补充：size 为 (每批次最大 token 数,) 的一维张量，记录每个 token 的 slot 物理地址（offset）
        self.slot_mapping = self._make_buffer(
            self.max_num_batched_tokens, dtype=torch.int64
        )

        if self.use_hybrid_blocks:
            # 补充：预计算 kernel block 偏移数组，用于快速映射，如 blocks_per_kv_block = 2 时，
            # _kernel_block_arange = [[0, 1]]
            self._kernel_block_arange = np.arange(0, self.blocks_per_kv_block).reshape(
                1, -1
            )
        else:
            self._kernel_block_arange = None

        try:
            self.pcp_world_size = get_pcp_group().world_size
            self.pcp_rank = get_pcp_group().rank_in_group
        except AssertionError:
            # PCP might not be initialized in testing
            self.pcp_world_size = 1
            self.pcp_rank = 0
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        # 补充：每个 CP rank 上连续存储的 token 数量
        self.cp_kv_cache_interleave_size = cp_kv_cache_interleave_size

    # 已阅
    # 补充：Append block IDs to the specified row (request)
    def append_row(
        self,
        block_ids: list[int],
        row_idx: int,
    ) -> None:
        if not block_ids:
            return

        if self.use_hybrid_blocks:
            block_ids = self.map_to_kernel_blocks(
                np.array(block_ids), self.blocks_per_kv_block, self._kernel_block_arange
            )

        num_blocks = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        # 补充：使用 numpy.ndarray 应该是考虑了性能，以及下面有场景使用了 ravel(), where() 等方法
        self.block_table.np[row_idx, start : start + num_blocks] = block_ids

    # 已阅
    # 补充：Add block IDs to the specified row (request), overwriting existing entries
    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    # 已阅
    # 补充：Move a row from src to tgt，keep src unchanged, overwrite tgt
    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        block_table_np = self.block_table.np
        block_table_np[tgt, :num_blocks] = block_table_np[src, :num_blocks]
        self.num_blocks_per_row[tgt] = num_blocks

    # 已阅
    def swap_row(self, src: int, tgt: int) -> None:
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]

    # 已阅
    # 说明：req_indices 和 positions 都是一维 numpy 数组
    # req_indices: 每个 token 对应的请求 ID
    # positions: 每个 token 在对应请求中的位置（token index）
    # 计算得到每个 token 在当前 rank 上的 physical slot 映射（在 kv cache tensor 中的 index）
    # 说明：这里计算 slot mapping，是为了后续将 kv cache 拷贝到 kv cache tensor 中的对应位置，
    # 参考 start_load_kv
    # 说明：所有请求的相同层 kv cache 都是存储在同一个 kv cache tensor 中的，所以可以看到 block indices
    # 的计算方式为 req_indices * self.max_num_blocks_per_req + positions // virtual_block_size
    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
        # where K is the max_num_blocks_per_req and the block size is 2.
        # NOTE(woosuk): We can't simply use `token_indices // block_size`
        # here because M (max_model_len) is not necessarily divisible by
        # block_size.
        # 说明：上方注释中描述的是 token index 映射到 block index 的过程
        total_cp_world_size = self.pcp_world_size * self.dcp_world_size
        total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
        if total_cp_world_size > 1:
            # Note(hc): The DCP implement store kvcache with an interleave
            # style, the kvcache for the token whose token_idx is i is
            # always stored on the GPU whose dcp_rank equals i % cp_world_size:
            # 补充：按照上面的说法，此时 cp_kv_cache_interleave_size 的值为 1

            # 补充：关注这里 virtual block 的概念
            # block_size 是 kernel block 的大小
            # virtual_block_size > block_size >= cp_kv_cache_interleave_size
            # Use a "virtual block" which equals to world_size * block_size
            # for block_table_indices calculation.
            virtual_block_size = self.block_size * total_cp_world_size
            # 说明：确定每个 token 对应的 block index，参考上方注释中的 K 计算方式
            block_table_indices = (
                # 补充：max_num_blocks_per_req 是基于 kernel block 计算的
                # 个人理解：max_num_blocks_per_req 是单机的最大 block 数；
                # 对于每个请求来说，总的 block 数就应该是 self.max_num_blocks_per_req * total_cp_world_size
                # 这里 req_indices * self.max_num_blocks_per_req 是确定每个请求在单机的起始 block index
                # positions // virtual_block_size 是确定每个 position 在单机上的 block index
                req_indices * self.max_num_blocks_per_req
                + positions // virtual_block_size
            )

            # 补充：根据 block indices 获取每个 token 对应的 block numbers IDs (最后用来计算物理地址，所以应该是绝对 ID)
            # 此时不考虑目标位置是否在当前 rank 上
            block_numbers = self.block_table.np.ravel()[block_table_indices]
            # Use virtual_block_size for mask calculation, which marks local
            # tokens.
            # 补充：计算每个 token 在 virtual block 内的偏移量
            # 需要根据 virtual_block_offsets 来判断当前 token 是否属于本 rank
            # 因为 virtual_block_size > block_size >= cp_kv_cache_interleave_size，
            # 这里取 offsets 之后不影响后续再对其他值取 offset
            # 取模的意思是只看最后一个 block
            virtual_block_offsets = positions % virtual_block_size
            # 补充：先对 virtual_block_offsets 按照 cp_kv_cache_interleave_size 大小做分组（连续存储的最小单位），
            # 然后以组为单位进行 world_size 取模（效果即 interleave），判断当前 token 是否属于本 rank
            mask = (
                virtual_block_offsets
                // self.cp_kv_cache_interleave_size
                % total_cp_world_size
                == total_cp_rank
            )
            # Calculate local block_offsets
            # 补充：virtual_block_offsets // (total_cp_world_size * self.cp_kv_cache_interleave_size) 
            # 是看最后一个 block 内有多少个完整的分组（分组大小为 cp_kv_cache_interleave_size）存在于所有 rank 上 
            # 完整分组数 * self.cp_kv_cache_interleave_size (组大小) + 组内偏移量 (virtual_block_offsets % self.cp_kv_cache_interleave_size)
            # 计算出本 rank 上 block 内的 offset
            block_offsets = (
                virtual_block_offsets
                // (total_cp_world_size * self.cp_kv_cache_interleave_size)
                * self.cp_kv_cache_interleave_size
                + virtual_block_offsets % self.cp_kv_cache_interleave_size
            )
            # Calculate slot_mapping
            # 补充：kernel block IDs * kernel_block_size + block 内偏移量 = token 在本 rank 上的 slot
            slot_mapping = block_numbers * self.block_size + block_offsets
            # Write final slots, use -1 for not-local
            # 补充：记录属于本 rank 的 token 的 slot，其他 token 记为 -1
            self.slot_mapping.np[: req_indices.shape[0]] = np.where(
                mask, slot_mapping, -1
            )
        else:
            block_table_indices = (
                req_indices * self.max_num_blocks_per_req + positions // self.block_size
            )

            block_numbers = self.block_table.np.ravel()[block_table_indices]
            block_offsets = positions % self.block_size
            np.add(
                block_numbers * self.block_size,
                block_offsets,
                out=self.slot_mapping.np[: req_indices.shape[0]],
            )

    def commit_block_table(self, num_reqs: int) -> None:
        self.block_table.copy_to_gpu(num_reqs)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        self.slot_mapping.copy_to_gpu(num_tokens)

    def clear(self) -> None:
        self.block_table.gpu.fill_(0)
        self.block_table.cpu.fill_(0)

    # 已阅
    @staticmethod
    def map_to_kernel_blocks(
        kv_manager_block_ids: np.ndarray,
        blocks_per_kv_block: int,
        kernel_block_arange: np.ndarray,
    ) -> np.ndarray:
        """Convert kv_manager_block_id IDs to kernel block IDs.

        Example:
            # kv_manager_block_ids: 32 tokens,
            # Kernel block size: 16 tokens
            # blocks_per_kv_block = 2
            >>> kv_manager_block_ids = np.array([0, 1, 2])
            >>> Result: [0, 1, 2, 3, 4, 5]

            # Each kv_manager_block_id maps to 2 kernel block id:
            # kv_manager_block_id 0 → kernel block id [0, 1]
            # kv_manager_block_id 1 → kernel block id [2, 3]
            # kv_manager_block_id 2 → kernel block id [4, 5]
        """
        if blocks_per_kv_block == 1:
            return kv_manager_block_ids

        kernel_block_ids = (
            kv_manager_block_ids.reshape(-1, 1) * blocks_per_kv_block
            + kernel_block_arange
        )

        return kernel_block_ids.reshape(-1)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        """Returns the device tensor of the block table."""
        return self.block_table.gpu[:num_reqs]

    def get_cpu_tensor(self) -> torch.Tensor:
        """Returns the CPU tensor of the block table."""
        return self.block_table.cpu

    def get_numpy_array(self) -> np.ndarray:
        """Returns the numpy array of the block table."""
        return self.block_table.np

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size, dtype=dtype, device=self.device, pin_memory=self.pin_memory
        )


# 已阅
# 说明：每个 KVCacheGroupSpec 会在内部对应一个 BlockTable 实例，所有实例组成列表，保持在 block_tables 属性中
class MultiGroupBlockTable:
    """The BlockTables for each KV cache group."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        pin_memory: bool,
        device: torch.device,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        max_num_blocks: list[int] | None = None,
        cp_kv_cache_interleave_size: int = 1,
    ) -> None:
        if len(kernel_block_sizes) != len(block_sizes):
            raise ValueError(
                f"kernel_block_sizes length ({len(kernel_block_sizes)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )
        if max_num_blocks is None:
            # Note(hc): each dcp rank only store
            # (max_model_len//dcp_world_size) tokens in kvcache,
            # so the block_size which used for calc max_num_blocks_per_req
            # must be multiplied by dcp_world_size.
            total_cp_world_size = get_total_cp_world_size()
            max_num_blocks = [
                cdiv(max_model_len, block_size * total_cp_world_size)
                for block_size in block_sizes
            ]

        if len(max_num_blocks) != len(block_sizes):
            raise ValueError(
                f"max_num_blocks length ({len(max_num_blocks)}) "
                f"must match block_sizes length ({len(block_sizes)})"
            )

        self.block_tables = [
            BlockTable(
                block_size,
                max_num_reqs,
                max_num_blocks_per_req,
                max_num_batched_tokens,
                pin_memory,
                device,
                kernel_block_size,
                cp_kv_cache_interleave_size,
            )
            for block_size, kernel_block_size, max_num_blocks_per_req in zip(
                block_sizes, kernel_block_sizes, max_num_blocks
            )
        ]

    def append_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.append_row(block_ids[i], row_idx)

    def add_row(self, block_ids: tuple[list[int], ...], row_idx: int) -> None:
        for i, block_table in enumerate(self.block_tables):
            block_table.add_row(block_ids[i], row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.move_row(src, tgt)

    def swap_row(self, src: int, tgt: int) -> None:
        for block_table in self.block_tables:
            block_table.swap_row(src, tgt)

    def compute_slot_mapping(
        self, req_indices: np.ndarray, positions: np.ndarray
    ) -> None:
        for block_table in self.block_tables:
            block_table.compute_slot_mapping(req_indices, positions)

    def commit_block_table(self, num_reqs: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_block_table(num_reqs)

    def commit_slot_mapping(self, num_tokens: int) -> None:
        for block_table in self.block_tables:
            block_table.commit_slot_mapping(num_tokens)

    def clear(self) -> None:
        for block_table in self.block_tables:
            block_table.clear()

    def __getitem__(self, idx: int) -> "BlockTable":
        """Returns the BlockTable for the i-th KV cache group."""
        return self.block_tables[idx]
