# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch

from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.scalar_type import ScalarType


# 已阅
@dataclass
class MPLinearLayerConfig:
    full_weight_shape: tuple[int, int]  # [in, out]
    partition_weight_shape: tuple[int, int]
    weight_type: ScalarType
    act_type: torch.dtype
    group_size: int
    # 说明：true 表示包含 zero points，即使用非对称量化
    zero_points: bool
    # 说明：true 表示包含 g_idx，用于激活重排序
    has_g_idx: bool
    out_type: torch.dtype | None = None


# 已阅
# 说明：混合精度（Mixed Precision）线性算子的抽象基类
# 说明：各参数的具体 shape 可以在 LinearBaseMethod 的子类中查看，
# LinearBaseMethod 各子类中会创建 MPLinearKernel 中子类实例
class MPLinearKernel(ABC):
    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        raise NotImplementedError

    def __init__(
        self,
        c: MPLinearLayerConfig,
        # 说明：量化权重对应的参数名称
        w_q_param_name: str,
        # 说明：量化缩放因子对应的参数名称
        w_s_param_name: str,
        # 说明：量化 zero points 对应的参数名称
        w_zp_param_name: str | None = None,
        # 说明：权重 group_idx 对应的参数名称
        w_gidx_param_name: str | None = None,
    ) -> None:
        assert self.can_implement(c)
        self.config = c
        self.w_q_name = w_q_param_name
        self.w_s_name = w_s_param_name
        if c.zero_points:
            assert w_zp_param_name is not None
        if c.has_g_idx:
            assert w_gidx_param_name is not None
        self.w_zp_name = w_zp_param_name
        self.w_gidx_name = w_gidx_param_name

    # 说明：LinearBaseMethod 实例负责加载权重，MPLinearKernel 实例负责处理权重
    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _transform_param(
        self, layer: torch.nn.Module, name: str | None, fn: Callable
    ) -> None:
        if name is not None and getattr(layer, name, None) is not None:
            old_param = getattr(layer, name)
            new_param = fn(old_param)
            # replace the parameter with torch.nn.Parameter for TorchDynamo
            # compatibility
            replace_parameter(
                layer, name, torch.nn.Parameter(new_param.data, requires_grad=False)
            )

    def _get_weight_params(
        self, layer: torch.nn.Module
    ) -> tuple[
        torch.Tensor,  # w_q
        torch.Tensor,  # w_s
        torch.Tensor | None,  # w_zp,
        torch.Tensor | None,  # w_gidx
    ]:
        return (
            getattr(layer, self.w_q_name),
            getattr(layer, self.w_s_name),
            getattr(layer, self.w_zp_name or "", None),
            getattr(layer, self.w_gidx_name or "", None),
        )
