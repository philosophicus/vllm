# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from itertools import product

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.logger import init_logger

logger = init_logger(__name__)


class CudagraphDispatcher:
    """
    Runtime cudagraph dispatcher to dispatch keys for multiple set of
    cudagraphs.

    The dispatcher stores two sets of dispatch keys, one for PIECEWISE and one
    for FULL cudagraph runtime mode. The keys are initialized depending on
    attention support and what cudagraph mode is set in CompilationConfig. The
    keys stored in dispatcher are the only source of truth for valid
    cudagraphs that can be dispatched at runtime.

    At runtime, the dispatch method generates the runtime cudagraph mode (FULL,
    PIECEWISE, or NONE for no cudagraph) and the valid key (batch descriptor)
    based on the input key. After dispatching (communicated via forward
    context), the cudagraph wrappers will trust the dispatch key to either
    capture or replay (if the mode matches), or pass through to the underlying
    runnable without cudagraph (if the mode does not match or mode is NONE).
    """

    def __init__(self, vllm_config: VllmConfig):
        self.vllm_config = vllm_config
        self.compilation_config = vllm_config.compilation_config
        self.uniform_decode_query_len = (
            1
            if not self.vllm_config.speculative_config
            else 1 + self.vllm_config.speculative_config.num_speculative_tokens
        )

        # Dict to store valid cudagraph dispatching keys.
        self.cudagraph_keys: dict[CUDAGraphMode, set[BatchDescriptor]] = {
            CUDAGraphMode.PIECEWISE: set(),
            CUDAGraphMode.FULL: set(),
        }

        assert (
            not self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            or self.compilation_config.is_attention_compiled_piecewise()
        ), (
            "Compilation mode should be CompilationMode.VLLM_COMPILE when "
            "cudagraph_mode piecewise cudagraphs is used, "
            "and attention should be in splitting_ops or "
            "inductor splitting should be used. "
            f"cudagraph_mode={self.compilation_config.cudagraph_mode}, "
            f"compilation_mode={self.compilation_config.mode}, "
            f"splitting_ops={self.compilation_config.splitting_ops}"
        )

        self.keys_initialized = False

    # 已阅
    # 理解：uniform_decode 表示是否是均匀解码场景。均匀解码场景下，
    # 每个序列的解码长度相同，等于 uniform_decode_query_len
    def _create_padded_batch_descriptor(
        self, num_tokens: int, uniform_decode: bool, has_lora: bool
    ) -> BatchDescriptor:
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        uniform_decode_query_len = self.uniform_decode_query_len
        # 说明：获得 num_tokens 对应的 padded cudagraph capture size
        num_tokens_padded = self.vllm_config.pad_for_cudagraph(num_tokens)

        if uniform_decode and self.cudagraph_mode.has_mode(CUDAGraphMode.FULL):
            num_reqs = num_tokens_padded // uniform_decode_query_len
            assert num_tokens_padded % uniform_decode_query_len == 0
        else:
            uniform_decode = False
            num_reqs = min(num_tokens_padded, max_num_seqs)

        return BatchDescriptor(
            num_tokens=num_tokens_padded,
            num_reqs=num_reqs,
            uniform=uniform_decode,
            has_lora=has_lora,
        )

    # 已阅
    def add_cudagraph_key(
        self, runtime_mode: CUDAGraphMode, batch_descriptor: BatchDescriptor
    ):
        assert runtime_mode in [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL], (
            f"Invalid cudagraph runtime mode for keys: {runtime_mode}"
        )
        self.cudagraph_keys[runtime_mode].add(batch_descriptor)

    # 已阅
    # 问题：cudagraph keys 是做什么用的
    def initialize_cudagraph_keys(
        self, cudagraph_mode: CUDAGraphMode, uniform_decode_query_len: int
    ):
        # This should be called only after attention backend is initialized. So we can
        # get the correct cudagraph mode after backend support is resolved.
        self.cudagraph_mode = cudagraph_mode

        # LoRA activation cases to specialize the cuda graphs on
        if self.vllm_config.lora_config:
            # 说明：如果启用了 cudagraph_specialize_lora，则
            # 需要同时考虑有无 LoRA 的情况
            if self.compilation_config.cudagraph_specialize_lora:
                lora_cases = [True, False]
            else:
                lora_cases = [True]
        else:
            lora_cases = [False]

        # Note: we create all valid keys for cudagraph here but do not
        # guarantee all keys would be used. For example, if we allow lazy
        # capturing in future PR, some keys may never be triggered.
        if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
            for bs, has_lora in product(
                self.compilation_config.cudagraph_capture_sizes, lora_cases
            ):
                self.add_cudagraph_key(
                    cudagraph_mode.mixed_mode(),
                    # 问题：此时 mixed mode 为 FULL 或 PIECEWISE
                    # 如何理解下面传 uniform_decode=False？mixed mode 如何理解？
                    # mixed mode 指同时存在 prefill 和 decode 时的 cudagraph mode？
                    # 而同时存在 prefill 和 decode，此时为非均匀解码场景？
                    self._create_padded_batch_descriptor(
                        bs, False, has_lora
                    ).relax_for_mixed_batch_cudagraphs(),
                )

        # if decode cudagraph mode is FULL, and we don't already have mixed
        # mode full cudagraphs then add them here.
        if (
            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
            # 说明：此时 mixed mode 为 NONE 或 PIECEWISE
            # 问题：mixed mode 为 NONE 或 PIECEWISE 对请求数量无限制？
            # 所以用 uniform_decode=True 来计算 key？
            and cudagraph_mode.separate_routine()
        ):
            # 说明：使用 uniform_decode_query_len 和 max_num_seqs 计算出最大值
            max_num_tokens = (
                uniform_decode_query_len
                * self.vllm_config.scheduler_config.max_num_seqs
            )
            # 说明：过滤出适合 decode 的 cudagraph capture sizes，范围在
            # [uniform_decode_query_len, max_num_tokens]，即对应的请求数量在
            # [1, max_num_seqs] 范围内
            cudagraph_capture_sizes_for_decode = [
                x
                for x in self.compilation_config.cudagraph_capture_sizes
                if x <= max_num_tokens and x >= uniform_decode_query_len
            ]
            for bs, has_lora in product(cudagraph_capture_sizes_for_decode, lora_cases):
                self.add_cudagraph_key(
                    CUDAGraphMode.FULL,
                    self._create_padded_batch_descriptor(bs, True, has_lora),
                )

        self.keys_initialized = True

    def dispatch(
        self,
        num_tokens: int,
        uniform_decode: bool,
        has_lora: bool,
        disable_full: bool = False,
    ) -> tuple[CUDAGraphMode, BatchDescriptor]:
        """
        Given conditions(e.g.,batch descriptor and if using cascade attention),
        dispatch to a cudagraph runtime mode and the valid batch descriptor.
        A new batch descriptor is returned as we might dispatch a uniform batch
        to a graph that supports a more general batch (uniform to non-uniform).
        """
        if (
            not self.keys_initialized
            or self.cudagraph_mode == CUDAGraphMode.NONE
            or num_tokens > self.compilation_config.max_cudagraph_capture_size
        ):
            return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)

        batch_desc = self._create_padded_batch_descriptor(
            num_tokens, uniform_decode, has_lora
        )
        relaxed_batch_desc = batch_desc.relax_for_mixed_batch_cudagraphs()

        if not disable_full:
            # check if key exists for full cudagraph
            if batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, batch_desc

            # otherwise, check if the relaxed key exists
            if relaxed_batch_desc in self.cudagraph_keys[CUDAGraphMode.FULL]:
                return CUDAGraphMode.FULL, relaxed_batch_desc

        # also check if the relaxed key exists for more "general"
        # piecewise cudagraph
        if relaxed_batch_desc in self.cudagraph_keys[CUDAGraphMode.PIECEWISE]:
            return CUDAGraphMode.PIECEWISE, relaxed_batch_desc

        # finally, just return no cudagraphs and a trivial batch descriptor
        return CUDAGraphMode.NONE, BatchDescriptor(num_tokens)
