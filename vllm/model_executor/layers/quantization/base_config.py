# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods
    from vllm.model_executor.models.utils import WeightsMapper
else:
    QuantizationMethods = str


# 说明：负责为层 create weights，并对输入张量 apply weights 的抽象基类；
# 理解：命名为 QuantizeMethodBase 应该是因为 weights 一般是需要量化的
# 说明：子类包括 LinearMethodBase、FusedMoEMethodBase、BaseKVMethodBase、UnquantizedEmbeddingMethod
# - LinearMethodBase 负责执行 matmul(A, B) + C，子类有 UnquantizedLinearMethod，
#   以及每种量化方法对应的 Method，如 GPTQLinearMethod, GPTQMarlinLinearMethod, AWQMarlinLinearMethod 等
# - FusedMoEMethodBase 的说明见类定义处
class QuantizeMethodBase(ABC):
    """Base class for different quantized methods."""

    @abstractmethod
    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        """Create weights for a layer.

        The weights will be set as attributes of the layer."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError

    # Not required functions
    def embedding(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Gather embeddings in the layer based on indices in the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Process the weight after loading.

        This can be used for example, to transpose weights for computation.
        """
        return


def method_has_implemented_embedding(method_class: type[QuantizeMethodBase]) -> bool:
    """
    Not all quant methods have embedding implemented, so we need to check that
    it exists for our given method. We check this by making sure the function
    has been changed from the base implementation.
    """
    base_embedding = inspect.getattr_static(QuantizeMethodBase, "embedding", None)
    class_embedding = inspect.getattr_static(method_class, "embedding", None)

    return class_embedding is not None and class_embedding is not base_embedding


class QuantizationConfig(ABC):
    """Base class for quantization configs."""

    def __init__(self):
        super().__init__()
        # mapping is updated by models as they initialize
        self.packed_modules_mapping: dict[str, list[str]] = dict()

    @abstractmethod
    def get_name(self) -> QuantizationMethods:
        """Name of the quantization method."""
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        """List of supported activation dtypes."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """Minimum GPU capability to support the quantization method.

        E.g., 70 for Volta, 75 for Turing, 80 for Ampere.
        This requirement is due to the custom CUDA kernels used by the
        quantization method.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> list[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> "QuantizationConfig":
        """Create a config class from the model's quantization config."""
        raise NotImplementedError

    # 说明：量化方法对应的实现类根据 hf_quant_config 中的 checkpoint 格式信息和用户指定的量化方法，
    # 来判断能否覆盖用户指定的量化方法
    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> QuantizationMethods | None:
        """
        Detects if this quantization method can support a given checkpoint
        format by overriding the user specified quantization method --
        this method should only be overwritten by subclasses in exceptional
        circumstances
        """
        return None

    @staticmethod
    def get_from_keys(config: dict[str, Any], keys: list[str]) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(
            f"Cannot find any of {keys} in the model's quantization config."
        )

    @staticmethod
    def get_from_keys_or(config: dict[str, Any], keys: list[str], default: Any) -> Any:
        """Get an optional value from the model's quantization config."""
        try:
            return QuantizationConfig.get_from_keys(config, keys)
        except ValueError:
            return default

    @abstractmethod
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        """Get the quantize method to use for the quantized layer.

        Args:
            layer: The layer for the quant method.
            prefix: The full name of the layer in the state dict
        Returns:
            The quantize method. None if the given layer doesn't support quant
            method.
        """
        raise NotImplementedError

    def get_cache_scale(self, name: str) -> str | None:
        return None

    # 说明：根据 WeightsMapper 中保存的 hf -> vllm 名称映射关系（子串、前缀、后缀），
    # 更新量化配置中的模块名称
    def apply_vllm_mapper(  # noqa: B027
        self, hf_to_vllm_mapper: "WeightsMapper"
    ):
        """
        Interface for models to update module names referenced in
        quantization configs in order to reflect the vllm model structure

        :param hf_to_vllm_mapper: maps from hf model structure (the assumed
            structure of the qconfig) to vllm model structure
        """
        # TODO (@kylesayrs): add implementations for all subclasses
        pass

    # 说明：配置初始化后调用
    def maybe_update_config(self, model_name: str):  # noqa: B027
        """
        Interface to update values after config initialization.
        """
        pass
