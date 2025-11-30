# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

_SMALLEST_LOGIT = float("-inf")


# 已阅
# 说明：将 logits 中对应 bad words 的最后一个 token id 的 logit 设置为 -inf，
# 从而避免生成这些 bad words
def _apply_bad_words_single_batch(
    logits: torch.Tensor,
    bad_words_token_ids: list[list[int]],
    past_tokens_ids: list[int],
) -> None:
    for bad_word_ids in bad_words_token_ids:
        if len(bad_word_ids) > len(past_tokens_ids) + 1:
            continue

        prefix_length = len(bad_word_ids) - 1
        last_token_id = bad_word_ids[-1]
        # 说明：past_tokens_ids 中的倒数 prefix_length 个 token ids
        actual_prefix = past_tokens_ids[-prefix_length:] if prefix_length > 0 else []
        # 说明：bad_word_ids 中除最后一个 token id 外的前面 prefix_length 个 token ids
        expected_prefix = bad_word_ids[:prefix_length]

        assert len(actual_prefix) == len(expected_prefix)

        if actual_prefix == expected_prefix:
            logits[last_token_id] = _SMALLEST_LOGIT


# 已阅
# 说明：对 bad_words_token_ids 的相关请求（keys 为 req_index）逐一应用 bad words 过滤
def apply_bad_words(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    # 说明：每个请求已经生成的 token ids 列表，output_token_ids 或者 output_token_ids + spec_token_ids
    past_tokens_ids: list[list[int]],
) -> None:
    for i, bad_words_ids in bad_words_token_ids.items():
        _apply_bad_words_single_batch(logits[i], bad_words_ids, past_tokens_ids[i])


# 已阅
# 说明：针对每个请求的草稿 token，逐 token 应用 bad words 过滤
def apply_bad_words_with_drafts(
    logits: torch.Tensor,
    bad_words_token_ids: dict[int, list[list[int]]],
    # 说明：每个 token 位置对应的已经生成的 token ids 列表，包含草稿和正式采样的 token ids
    past_tokens_ids: list[list[int]],
    # 说明：每个请求的草稿 token 数量列表
    num_draft_tokens: list[int],
) -> None:
    start_idx = 0
    remaining = len(bad_words_token_ids)
    for i, n in enumerate(num_draft_tokens):
        if (bad_words_ids := bad_words_token_ids.get(i)) is not None:
            for draft_idx in range(start_idx, start_idx + n):
                _apply_bad_words_single_batch(
                    logits[draft_idx],
                    bad_words_ids,
                    past_tokens_ids[draft_idx],
                )
            remaining -= 1
            # 说明：没有 bad words 需要处理了，提前退出
            if not remaining:
                break
        start_idx += n
