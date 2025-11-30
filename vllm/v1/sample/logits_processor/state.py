# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from itertools import chain
from typing import TYPE_CHECKING

from vllm.v1.sample.logits_processor.interface import (
    AddedRequest,
    BatchUpdate,
    MovedRequest,
    RemovedRequest,
)

if TYPE_CHECKING:
    from vllm.v1.sample.logits_processor.interface import LogitsProcessor


# 已阅
# 根据缓存的 Batch 的状态变更（persistent batch state changes），构建 BatchUpdate 对象
class BatchUpdateBuilder:
    """Helps track persistent batch state changes and build
    a batch update data structure for logitsprocs
    Assumptions:
    * All information about requests removed from persistent batch
      during a step is aggregated in self._removed through calls to
      self.removed_append() at the beginning of a step. This must happen
      before the first time that self.removed, self.pop_removed()
      or self.peek_removed() are invoked in a given step
    * After the first time that self.removed, self.pop_removed()
      or self.peek_removed() are read in a step, no new removals
      are registered using self.removed_append()
    * Elements of self._removed are never directly modified, added or
      removed (i.e. modification is only via self.removed_append() and
      self.pop_removed())
    Guarantees under above assumptions:
    * self.removed is always sorted in descending order
    * self.pop_removed() and self.peek_removed() both return
      the lowest removed request index in the current step
    """

    _removed: list[RemovedRequest]
    _is_removed_sorted: bool
    added: list[AddedRequest]
    moved: list[MovedRequest]

    def __init__(
        self,
        removed: list[RemovedRequest] | None = None,
        added: list[AddedRequest] | None = None,
        moved: list[MovedRequest] | None = None,
    ) -> None:
        self._removed = removed or []
        self.added = added or []
        self.moved = moved or []
        self._is_removed_sorted = False

        # Used to track changes in the pooling case
        # where we don't populate the added list.
        self.batch_changed = False

    # 已阅
    def _ensure_removed_sorted(self) -> None:
        """Sort removed request indices in
        descending order.
        Idempotent after first call in a
        given step, until reset.
        """
        if not self._is_removed_sorted:
            self._removed.sort(reverse=True)
            self._is_removed_sorted = True

    @property
    def removed(self) -> list[RemovedRequest]:
        """Removed request indices sorted in
        descending order"""
        self._ensure_removed_sorted()
        return self._removed

    # 设计：
    # All information about requests removed from persistent batch
    # during a step is aggregated in self._removed through calls to
    # self.removed_append() at the beginning of a step. This must happen
    # before the first time that self.removed, self.pop_removed()
    # or self.peek_removed() are invoked in a given step
    # 理解：还没有整理好 removed 列表，当然不能访问
    def removed_append(self, index: int) -> None:
        """Register the removal of a request from the persistent batch.

        Must not be called after the first time self.removed,
        self.pop_removed() or self.peek_removed() are invoked.

        Args:
          index: request index
        """
        if self._is_removed_sorted:
            raise RuntimeError(
                "Cannot register new removed request after self.removed has been read."
            )
        self._removed.append(index)
        self.batch_changed = True

    def has_removed(self) -> bool:
        return bool(self._removed)

    def peek_removed(self) -> int | None:
        """Return lowest removed request index"""
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed[-1]
        return None

    # 已阅
    # 说明：降序排列，弹出已删除的最小的请求索引（注意是 index 不是 id）
    def pop_removed(self) -> int | None:
        """Pop lowest removed request index"""
        if self.has_removed():
            self._ensure_removed_sorted()
            return self._removed.pop()
        return None

    def reset(self) -> bool:
        """Returns True if there were any changes to the batch."""
        self._is_removed_sorted = False
        self._removed.clear()
        self.added.clear()
        self.moved.clear()
        batch_changed = self.batch_changed
        self.batch_changed = False
        return batch_changed

    def get_and_reset(self, batch_size: int) -> BatchUpdate | None:
        """Generate a logitsprocs batch update data structure and reset
        internal batch update builder state.

        Args:
          batch_size: current persistent batch size

        Returns:
          Frozen logitsprocs batch update instance; `None` if no updates
        """
        # Reset removal-sorting logic
        self._is_removed_sorted = False
        self.batch_changed = False
        if not any((self._removed, self.moved, self.added)):
            # No update; short-circuit
            return None
        # Build batch state update
        batch_update = BatchUpdate(
            batch_size=batch_size,
            removed=self._removed,
            moved=self.moved,
            added=self.added,
        )
        self._removed = []
        self.moved = []
        self.added = []
        return batch_update


# 已阅
# 根据 LogitsProcessor 是否在贪婪采样中对 argmax 计算有影响，将其分组存储，并支持返回合并后的分组
class LogitsProcessors:
    """Encapsulates initialized logitsproc objects."""

    def __init__(self, logitsprocs: Iterator["LogitsProcessor"] | None = None) -> None:
        self.argmax_invariant: list[LogitsProcessor] = []
        self.non_argmax_invariant: list[LogitsProcessor] = []
        if logitsprocs:
            for logitproc in logitsprocs:
                (
                    self.argmax_invariant
                    if logitproc.is_argmax_invariant()
                    else self.non_argmax_invariant
                ).append(logitproc)

    @property
    def all(self) -> Iterator["LogitsProcessor"]:
        """Iterator over all logits processors."""
        return chain(self.argmax_invariant, self.non_argmax_invariant)
