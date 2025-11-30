# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm._custom_ops as ops
import vllm.model_executor.layers.fused_moe.modular_kernel as mk


# 说明：需要 PrepareAndFinalize 实现类选择适当的 weight + reduce 方法，
# 而不是调用这里未实现的 apply 方法
class TopKWeightAndReduceDelegate(mk.TopKWeightAndReduce):
    """
    Useful in the case when some FusedMoEPermuteExpertsUnpermute
    implementation does not perform weight application and reduction
    but cannot address the needs of all the compatible PrepareAndFinalize
    implementations.
    For example, BatchedTritonExperts is compatible with both
    PplxPrepareAndFinalize and BatchedPrepareAndFinalize. PplxPrepareAndFinalize
    does the weight-application + reduction as part of the pplx combine kernel.
    But the BatchedPrepareAndFinalize needs an implementation. To facilitate
    this case, the BatchedTritonExperts could use TopKWeightAndReduceDelegate
    so the PrepareAndFinalize implementations could choose how to
    weight + reduce.
    """

    def __eq__(self, other):
        return isinstance(other, TopKWeightAndReduceDelegate)

    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        raise RuntimeError(
            "The caller is expected to choose an appropriate "
            "TopKWeightAndReduce implementation."
        )


# 已阅
class TopKWeightAndReduceNoOP(mk.TopKWeightAndReduce):
    """
    The fused_experts outputs have already been weight applied and reduced.
    This implementation is a no-op.
    """

    def __eq__(self, other):
        return isinstance(other, TopKWeightAndReduceNoOP)

    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        # Weight application and reduction operations are already done.
        if output is None:
            return fused_expert_output

        # MoEPrepareAndFinalizeNoEP needs the output to be in the `output`
        # tensor.
        assert output.size() == fused_expert_output.size(), (
            "output shape is expected to match the fused_expert_output shape. "
            f"But got output={output.size()}, "
            f"used_expert_output={fused_expert_output.size()}"
        )
        output.copy_(fused_expert_output, non_blocking=True)
        return output


# 已阅
# 说明：weight 表示将专家输出乘以路由权重；reduce 表示对 topk 个专家的输出求和
class TopKWeightAndReduceContiguous(mk.TopKWeightAndReduce):
    """
    TopKWeightAndReduce implementation for a fused_experts output
    of shape (m, topk, K)
    """

    def __eq__(self, other):
        return isinstance(other, TopKWeightAndReduceContiguous)

    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        # 说明：m 表示 token 数量，k 表示每个专家的输出维度 
        m, num_topk = topk_ids.size()
        k = fused_expert_output.size(-1)
        if fused_expert_output.ndim == 2:
            fused_expert_output = fused_expert_output.view(m, num_topk, k)

        assert fused_expert_output.size() == (m, num_topk, k), (
            f"Expected fused_expert_output size {(m, num_topk, k)}. But got "
            f"{fused_expert_output.size()}"
        )

        # 理解：true 表示对输入乘以路由权重；false 表示对输出乘以路由权重
        if not apply_router_weight_on_input:
            fused_expert_output.mul_(topk_weights.view(m, -1, 1))

        if output is None:
            output = torch.empty(
                (m, k),
                device=fused_expert_output.device,
                dtype=fused_expert_output.dtype,
            )
        assert output.size() == (m, k), (
            f"Expected output size {(m, k)}. But got {output.size()}"
        )

        ops.moe_sum(fused_expert_output, output)
        return output


# 已阅
# 说明：名为 batch 是因为 "the activations/tokens that subscribe to the same expert are batched together".
class TopKWeightAndReduceNaiveBatched(mk.TopKWeightAndReduce):
    """
    TopKWeightAndReduce implementation for a fused_experts output
    of shape (num_experts, batch_size, K)
    """

    def __init__(self, rank: int):
        self.rank = rank

    def __eq__(self, other):
        return isinstance(other, TopKWeightAndReduceNaiveBatched) and (
            other.rank == self.rank
        )

    def apply(
        self,
        output: torch.Tensor | None,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        # 说明：fused_expert_output 的 shape 为 (num_experts, batch_size, K)
        # 说明：batch_size 表示 max_tokens 数量, K 表示每个专家的输出维度
        assert fused_expert_output.ndim == 3
        # 说明：topk_ids 的 shape 为 (num_tokens, topk)
        num_tokens = topk_ids.size(0)
        num_local_experts = fused_expert_output.size(0)
        K = fused_expert_output.size(-1)

        if output is None:
            output = torch.zeros(
                (num_tokens, K),
                device=fused_expert_output.device,
                dtype=fused_expert_output.dtype,
            )
        else:
            output.fill_(0)

        assert output.size() == (num_tokens, K), (
            f"Expected output size {(num_tokens, K)}, but got {output.size()}"
        )

        first_expert = num_local_experts * self.rank
        last_expert = first_expert + num_local_experts

        for expert_id in range(first_expert, last_expert):
            # 说明：matching_tokens 的 shape 为 (num_tokens, topk)
            matching_tokens = topk_ids == expert_id
            # 说明：topks 的 shape 为 (num_tokens, )，表示每个 token 是否分配给了当前 expert
            topks = torch.any(matching_tokens, dim=1).flatten()
            # 说明：计算出有多少个 token 分配给了当前 expert
            rows = torch.count_nonzero(topks)
            # 说明：rhs = right-hand side
            # rhs 的 shape 为 (num_assigned_tokens, K)
            rhs = fused_expert_output[expert_id - first_expert, :rows, :]
            if not apply_router_weight_on_input:
                # 说明：布尔索引返回一维张量，需要获得 reshape 为 (num_assigned_tokens, 1) 的 view
                rhs.mul_(topk_weights[matching_tokens].view(rhs.size(0), 1))
            output[topks] = output[topks] + rhs

        return output
