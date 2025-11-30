# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import abstractmethod

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

logger = init_logger(__name__)


# 说明：重点是 maybe_make_prepare_finalize 和 select_gemm_impl 方法，
# 分别用来创建 PrepareAndFinalize 实例和选择合适的 GEMM 实现，最终创建出 FusedMoeModularKernel 实例；
# 并且，只有当 prepare_finalize 不为 None 时，FusedMoEModularMethod 才会被使用
class FusedMoEMethodBase(QuantizeMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.moe: FusedMoEConfig = moe
        self.moe_quant_config: FusedMoEQuantConfig | None = None

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    # 说明：weight_scale_2 模式是指 FP4 量化中使用的两级缩放因子模式，
    # 其中第二级用于 Tensor 级别的全局缩放 
    def uses_weight_scale_2_pattern(self) -> bool:
        """
        Returns True if this quantization method uses 'weight_scale_2' pattern
        for per-tensor weight scales (e.g., FP4 variants), False otherwise.

        This method should be overridden by subclasses that use the
        'weight_scale_2' pattern instead of the standard 'weight_scale' pattern.
        """
        return False

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> FusedMoEPrepareAndFinalize | None:
        from .all2all_utils import maybe_make_prepare_finalize

        return maybe_make_prepare_finalize(
            self.moe, self.moe_quant_config, routing_tables
        )

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        # based on the all2all implementation, select the appropriate
        # gemm implementation
        raise NotImplementedError(
            f"{self.__class__.__name__} must select appropriate gemm "
            "implementation based on the prepare_finalize"
        )

    def prepare_dp_allgather_tensor(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Hook to prepare tensors and extra tensors for DP allgather + EP dispatch."""
        raise NotImplementedError(
            "Method 'prepare_dp_allgather_tensor' is not implemented in "
            f"{self.__class__.__name__}."
        )

    @abstractmethod
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        raise NotImplementedError

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    @property
    def supports_eplb(self) -> bool:
        return False

    @property
    def allow_inplace(self) -> bool:
        return False

    @property
    def method_name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def apply(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
