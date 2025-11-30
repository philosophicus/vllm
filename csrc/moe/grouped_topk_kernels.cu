/*
 * Adapted from
 * https://github.com/NVIDIA/TensorRT-LLM/blob/v0.21.0/cpp/tensorrt_llm/kernels/noAuxTcKernels.cu
 * Copyright (c) 2025, The vLLM team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/std/limits>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

namespace vllm {
namespace moe {

constexpr unsigned FULL_WARP_MASK = 0xffffffff;
constexpr int32_t WARP_SIZE = 32;

namespace warp_topk {

// 已阅
template <int size, typename T>
__host__ __device__ constexpr T round_up_to_multiple_of(T len) {
  if (len == 0) {
    return 0;
  }
  return ((len - 1) / size + 1) * size;
}

// 已阅
template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
  return (v && !(v & (v - 1)));
}

// 已阅
template <bool greater, typename T>
__forceinline__ __device__ bool is_better_than(T val, T baseline) {
  return (val > baseline && greater) || (val < baseline && !greater);
}

// 已阅
template <bool greater, typename T, typename idxT>
__forceinline__ __device__ bool is_better_than(T val, T baseline, idxT index,
                                               idxT baseline_index) {
  bool res = (val > baseline && greater) || (val < baseline && !greater);
  if (val == baseline) {
    // 优化：这里可以直接用 index < baseline_index 代替
    res = (index < baseline_index && greater) ||
          (index < baseline_index && !greater);
  }
  return res;
}

// 已阅
// 说明：每个线程维护 size / WARP_SIZE 个元素，在线程内进行排序，直到 size = 32 时，调用特化版本
// 线程内元素是交错存储的，即线程 i 处理的元素索引为 i, i + WARP_SIZE, 
// i + 2 * WARP_SIZE, i + 3 * WARP_SIZE, ...，且一定是成对出现的
template <int size, bool ascending, bool reverse, typename T, typename idxT,
          bool is_stable>
struct BitonicMerge {
  // input should be a bitonic sequence, and sort it to be a monotonic sequence
  __device__ static void merge(T* __restrict__ val_arr,
                               idxT* __restrict__ idx_arr) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    // 说明：arr_len 表示每个线程处理的元素个数，且 arr_len >= 2，当 size = 32 时，会调用特化版本
    constexpr int arr_len = size / WARP_SIZE;

    constexpr int stride = arr_len / 2;
    for (int i = 0; i < stride; ++i) {
      int const other_i = i + stride;
      T& val = val_arr[i];
      T& other_val = val_arr[other_i];
      bool is_better;
      if constexpr (is_stable) {
        // 说明：ascending=true，值大索引小的更好，放到后面（i -> i + stride）
        // ascending=false，值小索引小的更好，放到后面（i -> i + stride）
        // 总结：ascending=true，值小索引大的靠前；ascending=false，值大索引大的靠前
        is_better = is_better_than<ascending>(val, other_val, idx_arr[i],
                                              idx_arr[other_i]);
      } else {
        is_better = is_better_than<ascending>(val, other_val);
      }

      if (is_better) {
        T tmp = val;
        val = other_val;
        other_val = tmp;

        idxT tmp2 = idx_arr[i];
        idx_arr[i] = idx_arr[other_i];
        idx_arr[other_i] = tmp2;
      }
    }

    // 说明：整个处理范围就是 arr_len 个元素，递归处理前后两半部分
    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr, idx_arr);
    BitonicMerge<size / 2, ascending, reverse, T, idxT, is_stable>::merge(
        val_arr + arr_len / 2, idx_arr + arr_len / 2);
  }
};

// 已阅
template <int size, bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort {
  __device__ static void sort(T* __restrict__ val_arr,
                              idxT* __restrict__ idx_arr) {
    static_assert(isPowerOf2(size));
    static_assert(size >= 2 * WARP_SIZE);
    constexpr int arr_len = size / WARP_SIZE;

    // 说明：升序 Bitonic Sequence
    BitonicSort<size / 2, true, T, idxT, is_stable>::sort(val_arr, idx_arr);
    // 说明：降序 Bitonic Sequence
    BitonicSort<size / 2, false, T, idxT, is_stable>::sort(
        val_arr + arr_len / 2, idx_arr + arr_len / 2);
    BitonicMerge<size, ascending, ascending, T, idxT, is_stable>::merge(
        val_arr, idx_arr);
  }
};

// 已阅
template <bool ascending, typename T, typename idxT, bool is_stable>
struct BitonicSort<32, ascending, T, idxT, is_stable> {
  __device__ static void sort(T* __restrict__ val_arr,
                              idxT* __restrict__ idx_arr) {
    int const lane = threadIdx.x % WARP_SIZE;

    // ascending doesn't matter before merging since all we need is a bitonic
    // sequence
    // 说明：size == WARP_SIZE == 32 == 2^5 时，需要遍历的 stage 共 4 个；
    // 每个 stage 内部需要处理的 stride 分别从 1，2，4，8 开始，对应的区间长度为 2，4，8，16；
    // 先产生两个长度为 16 的 Bitonic Sequence（所以区间长度最大是 16），
    // 最后合并得到长度为 32 的 Bitonic Sequence
    // 说明：通过两层循环来实现长度从 16 -> 8 -> 4 -> 2 的 BitonicSort 递归 & BitonicMerge 递归过程
    // 第一轮：stage = 0，stride = [1]，处理区间长度为 [2] 的 BitonicMerge
    // 第二轮：stage = 1，stride = [2，1]，处理区间长度为 [4, 2] 的 BitonicMerge
    // 第三轮：stage = 2，stride = [4，2，1]，处理区间长度为 [8, 4, 2] 的 BitonicMerge
    // 第四轮：stage = 3，stride = [8，4，2，1]，处理区间长度为 [16, 8, 4, 2] 的 BitonicMerge
    for (int stage = 0; stage < 4; ++stage) {
      for (int stride = (1 << stage); stride > 0; stride /= 2) {
        // 说明：判断 lane 的二进制表示中，第 stage + 1 位（从 0 开始计数，由右向左）是否为 1
        // stage = 0 时，判断 lane 的第 1 位（2^1）是否为 1，结果为 true 的 lane 有 2-3，6-7, 10-11, ...
        // stage = 1 时，判断 lane 的第 2 位（2^2）是否为 1，结果为 true 的 lane 有 4-7, 12-15, ...
        // stage = 2 时，判断 lane 的第 3 位（2^3）是否为 1，结果为 true 的 lane 有 8-15，24-31, ...
        // stage = 3 时，判断 lane 的第 4 位（2^4）是否为 1，结果为 true 的 lane 有 16-31
        // 这些都是降序 Bitonic Sequence 的区间（两个 Sequence 第一个升序，第二个降序）
        bool reverse = (lane >> stage) & 2;
        // 说明：i 和 i ^ stride 两个元素，判断当前线程处理的元素是否是后面那个
        bool is_second = lane & stride;

        T other = __shfl_xor_sync(FULL_WARP_MASK, *val_arr, stride);
        idxT other_idx = __shfl_xor_sync(FULL_WARP_MASK, *idx_arr, stride);

        bool is_better;
        if constexpr (is_stable) {
          // 理解：ascending 控制的是全局排序的顺序，true 表示升序，值小索引大的靠前；false 表示降序，值大索引大的靠前
          // reverse 控制的是当前处理的区间是否与 ascending 相反，true 表示相反，false 表示相同
          if constexpr (ascending) {
            // 说明：大前提为 ascending = true，升序排列
            // 说明：列举所有 is_better = true 的情况
            // 当前值较大或值相等索引较小（true）；降序排列（reverse=true），当前处理的是第二个元素（is_second=true）（false）
            // 当前值较大或值相等索引较小（true）；升序排列（reverse=false），当前处理的是第一个元素（is_second=false）（false）
            // 当前值较小或值相等索引较大（false）；降序排列（reverse=true），当前处理的是第一个元素（is_second=false）（true）
            // 当前值较小或值相等索引较大（false）；升序排列（reverse=false），当前处理的是第二个元素（is_second=true）（true）
            // 总结：整体升序，升序区间，值小索引大的靠前；降序区间，值大索引小的靠前
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr < other_idx))) !=
                        (reverse != is_second);
          } else {
            // 说明：大前提为 ascending = false，降序排列
            // 说明：列举所有 is_better = true 的情况
            // 当前值较大或值相等索引较大（true）；降序排列（reverse=true），当前处理的是第二个元素（is_second=true）（false）
            // 当前值较大或值相等索引较大（true）；升序排列（reverse=false），当前处理的是第一个元素（is_second=false）（false）
            // 当前值较小或值相等索引较小（false）；降序排列（reverse=true），当前处理的是第一个元素（is_second=false）（true）
            // 当前值较小或值相等索引较小（false）；升序排列（reverse=false），当前处理的是第二个元素（is_second=true）（true）
            // 总结：整体降序，升序区间，值小索引小的靠前；降序区间，值大索引大的靠前
            is_better = ((*val_arr > other) ||
                         ((*val_arr == other) && (*idx_arr > other_idx))) !=
                        (reverse != is_second);
          }
        } else {
          // 说明：is_stable 为 false 时，不考虑索引
          // 说明：列举所有等于 true 的情况
          // 当前值较大（true）；降序排列（reverse=true），当前处理的是第二个元素（is_second=true）（false）
          // 当前值较大（true）；升序排列（reverse=false），当前处理的是第一个元素（is_second=false）（false）
          // 当前值较小（false）；降序排列（reverse=true），当前处理的是第一个元素（is_second=false）（true）
          // 当前值较小（false）；升序排列（reverse=false），当前处理的是第二个元素（is_second=true）（true）
          // 总结：升序区间，值小的靠前；降序区间，值大的靠前
          is_better = (*val_arr != other &&
                       (*val_arr > other) != (reverse != is_second));
        }
        if (is_better) {
          *val_arr = other;
          *idx_arr = other_idx;
        }
      }
    }

    // 说明：前面的逻辑已经完成了长度为 16 的 Bitonic Sequence 的构建，最后一步是进行
    // 长度为 32 的 BitonicMerge，得到最终的单调序列
    BitonicMerge<32, ascending, ascending, T, idxT, is_stable>::merge(val_arr,
                                                                      idx_arr);
  }
};

// 已阅
// 说明：双调归并算法，size = 32 时的特化版本 (specialization)
// 说明：跨线程归并
// 问题：对线程内的元素顺序有什么影响
template <bool ascending, bool reverse, typename T, typename idxT,
          bool is_stable>
struct BitonicMerge<32, ascending, reverse, T, idxT, is_stable> {
  __device__ static void merge(T* __restrict__ val_arr,
                               idxT* __restrict__ idx_arr) {
    int const lane = threadIdx.x % WARP_SIZE;
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) {
      // 说明：i 和 i ^ stride 两个元素，判断当前线程处理的元素是否是后面那个
      bool is_second = lane & stride;
      T& val = *val_arr;
      // 说明：通过 shuffle 指令获取另一个元素的值
      T other = __shfl_xor_sync(FULL_WARP_MASK, val, stride);
      idxT& idx = *idx_arr;
      // 说明：通过 shuffle 指令获取另一个元素的索引
      idxT other_idx = __shfl_xor_sync(FULL_WARP_MASK, idx, stride);

      // 说明：判断是否需要互换（is_better = true），两边线程会得到相同的 is_better 结果
      bool is_better;
      if constexpr (is_stable) {
        if constexpr (ascending) {
          // 说明：大前提为 ascending = true，升序排列
          // 说明：在 BitonicSort 算法中，实际传参时，reverse = ascending，这里的 reverse 看成 ascending 即可
          // 说明：列举所有等于 true 的情况
          // 当前值较大或值相等索引较小（true）；升序排列（ascending=true），当前处理的是第一个元素（is_second=false）（true）
          // 当前值较小或值相等索引较大（false）；升序排列（ascending=true），当前处理的是第二个元素（is_second=true）（false）
          // 总结：升序排列，值小索引大的靠前
          // 说明：在 WarpSort 和 WarpSelect 中，reverse 为 !ascending，此时列举所有等于 true 的情况
          // 当前值较大或值相等索引较小（true）；升序排列（reverse=false），当前处理的是第二个元素（is_second=true）（true）
          // 当前值较小或值相等索引较大（false）；升序排列（reverse=false），当前处理的是第一个元素（is_second=false）（false）
          // 总结：ascending=true, reverse=false, 值大索引小的靠前，对于这样的数据，is_better_than<true> 返回 true
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr < other_idx))) ==
                      (reverse != is_second);  // for min
        } else {
          // 说明：大前提为 ascending = false，降序排列
          // 说明：实际传参时，reverse = ascending，所以这里的 reverse 是多余的，看成 ascending 即可
          // 说明：列举所有等于 true 的情况
          // 当前值较大或值相等索引较大（true）；降序排列（ascending=false），当前处理的是第二个元素（is_second=true）（true）
          // 当前值较小或值相等索引较小（false）；降序排列（ascending=false），当前处理的是第一个元素（is_second=false）（true）
          // 总结：降序排列，值大索引大的靠前
          // 说明：在 WarpSort 和 WarpSelect 中，reverse 为 !ascending，此时列举所有等于 true 的情况
          // 当前值较大或值相等索引较大（true）；降序排列（reverse=true），当前处理的是第一个元素（is_second=false）（true）
          // 当前值较小或值相等索引较小（false）；降序排列（reverse=true），当前处理的是第二个元素（is_second=true）（false）
          // 总结：ascending=false, reverse=true, 值小索引小的靠前，对于这样的数据，is_better_than<false> 返回 true
          is_better = ((*val_arr > other) ||
                       ((*val_arr == other) && (*idx_arr > other_idx))) ==
                      (reverse != is_second);  // for max
        }
      } else {
        // 说明：is_stable 为 false 时，不考虑索引
        // 说明：列举所有等于 true 的情况
        // 当前值较大（true）；升序排列（ascending=true），当前处理的是第一个元素（is_second=false）（true）
        // 当前值较大（true）；降序排列（ascending=false），当前处理的是第二个元素（is_second=true）（true）
        // 当前值较小（false）；降序排列（ascending=false），当前处理的是第一个元素（is_second=false）（false）
        // 当前值较小（false）；升序排列（ascending=true），当前处理的是第二个元素（is_second=true）（false）
        // 总结：升序排列，值小的靠前；降序排列，值大的靠前
        // ascending=true，值小的靠前；ascending=false，值大的靠前
        is_better =
            (val != other && ((val > other) == (ascending != is_second)));
      }

      if (is_better) {
        val = other;
        idx = other_idx;
      }
    }
  }
};

template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSort {
 public:
  __device__ WarpSort(idxT k, T dummy)
      : lane_(threadIdx.x % WARP_SIZE), k_(k), dummy_(dummy) {
    static_assert(capacity >= WARP_SIZE && isPowerOf2(capacity));

    for (int i = 0; i < max_arr_len_; ++i) {
      val_arr_[i] = dummy_;
      idx_arr_[i] = 0;
    }
  }

  // 说明：没有调用，不看
  // load and merge k sorted values
  __device__ void load_sorted(T const* __restrict__ in,
                              idxT const* __restrict__ in_idx, idxT start) {
    idxT idx = start + WARP_SIZE - 1 - lane_;
    // 说明：每个线程处理 max_arr_len_ 个间隔为 WARP_SIZE 的元素
    for (int i = max_arr_len_ - 1; i >= 0; --i, idx += WARP_SIZE) {
      if (idx < start + k_) {
        T t = in[idx];
        bool is_better;
        if constexpr (is_stable) {
          is_better =
              is_better_than<greater>(t, val_arr_[i], in_idx[idx], idx_arr_[i]);
        } else {
          is_better = is_better_than<greater>(t, val_arr_[i]);
        }
        // 说明：is_better 表示当前值更好
        // 更好的定义：对于 greater = true，表示值大索引小更好；对于 greater = false，表示值小索引小更好
        if (is_better) {
          val_arr_[i] = t;
          idx_arr_[i] = in_idx[idx];
        }
      }
    }

    // 说明：reverse 传入 !greater，实现 greater = true，值大索引小的更靠前；greater = false，值小索引小的更靠前
    BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(
        val_arr_, idx_arr_);
  }

  // 无调用
  __device__ void dump(T* __restrict__ out, idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) {
        out[out_i] = val_arr_[i];
        out_idx[out_i] = idx_arr_[i];
      }
    }
  }

  // 说明：将 topk 的索引输出到 out_idx 中
  __device__ void dumpIdx(idxT* __restrict__ out_idx) const {
    for (int i = 0; i < max_arr_len_; ++i) {
      // 说明：整体的顺序是按照（线程内位置，线程号）依次输出的
      idxT out_i = i * WARP_SIZE + lane_;
      if (out_i < k_) {
        out_idx[out_i] = idx_arr_[i];
      }
    }
  }

  // Accessors for per-lane selected value/index.
  // NOTE: For the common case `capacity == WARP_SIZE`, `max_arr_len_ == 1`
  // and callers should use `i == 0`.
  __device__ __forceinline__ idxT get_idx(int i = 0) const {
    return idx_arr_[i];
  }

  __device__ __forceinline__ T get_val(int i = 0) const { return val_arr_[i]; }

 protected:
  static constexpr int max_arr_len_ = capacity / WARP_SIZE;

  // 说明：每个线程维护 capacity / WARP_SIZE 个元素
  T val_arr_[max_arr_len_];
  idxT idx_arr_[max_arr_len_];

  int const lane_;
  idxT const k_;
  T const dummy_;

};  // end class WarpSort

template <int capacity, bool greater, typename T, typename idxT, bool is_stable>
class WarpSelect : public WarpSort<capacity, greater, T, idxT, is_stable> {
 public:
  __device__ WarpSelect(idxT k, T dummy)
      : WarpSort<capacity, greater, T, idxT, is_stable>(k, dummy),
        k_th_(dummy),
        k_th_idx_(0),
        k_th_lane_((k - 1) % WARP_SIZE) {
    extern __shared__ char smem_buf[];  // extern __shared__ T smem_buf[];

    // 说明：blockDim.x 已经被设置为 BLOCK_SIZE
    int const num_of_warp = blockDim.x / WARP_SIZE;
    int const warp_id = threadIdx.x / WARP_SIZE;
    // 说明：这里的 smem_buf 是前半部分存储 values，后半部分存储 indices
    val_smem_ = reinterpret_cast<T*>(smem_buf);
    // 说明：当前 warp 的值存储起始位置
    val_smem_ += warp_id * WARP_SIZE;
    idx_smem_ = reinterpret_cast<idxT*>(
        smem_buf +
        // 理解：每个线程产生一个值，值的大小为 sizeof(T)，所以 offset 为 warp_id * WARP_SIZE * sizeof(T)
        round_up_to_multiple_of<256>(num_of_warp * sizeof(T) * WARP_SIZE));
    idx_smem_ += warp_id * WARP_SIZE;
  }

  // 无调用
  __device__ void add(T const* in, idxT start, idxT end) {
    // 说明：从 start 开始，取 WARP_SIZE 的整数倍个数进行处理
    idxT const end_for_fullwarp =
        round_up_to_multiple_of<WARP_SIZE>(end - start) + start;
    // 说明：lane_ 在 WarpSort 的构造函数中已经被初始化为 threadIdx.x % WARP_SIZE
    // dummy_ 在 WarpSelect 的构造函数中传入，已经被设置为负无穷
    for (idxT i = start + lane_; i < end_for_fullwarp; i += WARP_SIZE) {
      T val = (i < end) ? in[i] : dummy_;
      add(val, i);
    }
  }

  // 已阅
  // 说明：idx 表示全局索引
  __device__ void add(T val, idxT idx) {
    bool do_add;
    if constexpr (is_stable) {
      // 说明：is_stable 为 true 时，需要同时比较值和索引
      // 说明：首次比较时，k_th_ 值为 dummy，k_th_idx_ 值为 0；
      // val 有可能是 dummy，此时 do_add = false
      do_add = is_better_than<greater>(val, k_th_, idx, k_th_idx_);
    } else {
      do_add = is_better_than<greater>(val, k_th_);
    }

    uint32_t mask = __ballot_sync(FULL_WARP_MASK, do_add);
    if (mask == 0) {
      // 说明：没有线程需要添加，直接返回
      return;
    }

    // 说明：__popc(mask & ((0x1u << lane_) - 1)) 表示在当前 lane 之前的线程中需要添加元素的线程数
    int pos = smem_buf_len_ + __popc(mask & ((0x1u << lane_) - 1));
    if (do_add && pos < WARP_SIZE) {
      // 说明：直接覆盖旧值
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
      do_add = false;
    }
    smem_buf_len_ += __popc(mask);
    if (smem_buf_len_ >= WARP_SIZE) {
      __syncwarp();
      // 说明：对 val_smem_ 和 idx_smem_ 中的 WARP_SIZE 个元素进行归并排序；
      // 对于排序后的结果，每个线程只保留自己对应的位置上的元素，
      // 有条件地替换 val_arr_ 和 idx_arr_ 中的最后一个位置（最差元素），然后再对 val_arr_ 和 idx_arr_ 进行排序
      merge_buf_(val_smem_[lane_], idx_smem_[lane_]);
      smem_buf_len_ -= WARP_SIZE;
    }
    if (do_add) {
      // 说明：之前因为 pos >= WARP_SIZE 而没有添加成功，现在 pos 肯定小于 WARP_SIZE
      pos -= WARP_SIZE;
      // 说明：直接覆盖旧值
      val_smem_[pos] = val;
      idx_smem_[pos] = idx;
    }
    __syncwarp();
  }

  // 已阅
  __device__ void done() {
    if (smem_buf_len_) {
      T val = (lane_ < smem_buf_len_) ? val_smem_[lane_] : dummy_;
      idxT idx = (lane_ < smem_buf_len_) ? idx_smem_[lane_] : 0;
      merge_buf_(val, idx);
    }
  }

 private:
  // 已阅
  __device__ void set_k_th_() {
    // 说明：广播第 k_th_lane_ 个线程的第 max_arr_len_ - 1 个元素的值和索引（线程中最大的元素）
    // 作为当前的 k_th_ 和 k_th_idx_
    k_th_ = __shfl_sync(FULL_WARP_MASK, val_arr_[max_arr_len_ - 1], k_th_lane_);
    if constexpr (is_stable) {
      k_th_idx_ =
          __shfl_sync(FULL_WARP_MASK, idx_arr_[max_arr_len_ - 1], k_th_lane_);
    }
  }

  // 已阅
  // 说明：
  // 1. BitonicSort 对完全乱序的 WARP_SIZE 个元素进行排序，ascending = greater = true，值小索引大的靠前；
  // 2. 每个线程取排序后自己对应位置的元素 val 和 idx，与 val_arr_ 和 idx_arr_ 中的最后一个元素比较，
  //    如果新值更好则进行替换；对于 max_arr_len_ = 1 的情况，替换之后，所有线程的 val_arr_ 和 idx_arr_ 构成
  //    长度为 32 的 Bionic Sequence
  // 3. 然后对 val_arr_ 和 idx_arr_（WARP_SIZE 个元素）进行排序，值大索引小的靠前
  //    （ascending = greater = true，reverse = !greater = false）；
  // 4. 最后调用 set_k_th_() 更新 k_th_ 和 k_th_idx_
  __device__ void merge_buf_(T val, idxT idx) {
    // 说明：WARP 内排序
    // ascending = greater = true, 值小索引大的靠前
    // 排序后 val 和 idx 变为有序数列中当前线程对应位置的值和索引
    BitonicSort<WARP_SIZE, greater, T, idxT, is_stable>::sort(&val, &idx);

    // 说明：当前线程中的最差值
    T& old = val_arr_[max_arr_len_ - 1];

    bool is_better;
    if constexpr (is_stable) {
      is_better =
          is_better_than<greater>(val, old, idx, idx_arr_[max_arr_len_ - 1]);
    } else {
      is_better = is_better_than<greater>(val, old);
    }

    // 说明：当前值比最后一个值更好，进行替换
    if (is_better) {
      old = val;
      idx_arr_[max_arr_len_ - 1] = idx;
    }

    // 说明：对 val_arr_ 和 idx_arr_ 进行排序，长度为 capacity / WARP_SIZE = max_arr_len_
    // ascending = greater = true，reverse = !greater = false，实现值大索引小的靠前，
    // 即排序后 max_arr_len_ - 1 位置的元素为最小值
    // 说明：capacity 实际传值为 32，此时 max_arr_len_ = 1
    // 疑似 Bug：线程内排序是升序，跨线程排序是降序，是无法实现全局降序排列的（因为 
    // max_arr_len_ = 1，所以并不会真的执行线程内排序，但程序逻辑上是不对的，无法扩展）
    BitonicMerge<capacity, greater, !greater, T, idxT, is_stable>::merge(
        val_arr_, idx_arr_);

    set_k_th_();
  }

  using WarpSort<capacity, greater, T, idxT, is_stable>::max_arr_len_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::val_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::idx_arr_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::lane_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::k_;
  using WarpSort<capacity, greater, T, idxT, is_stable>::dummy_;

  T* val_smem_;
  idxT* idx_smem_;
  int smem_buf_len_ = 0;

  T k_th_;
  idxT k_th_idx_;
  int const k_th_lane_;
};  // end class WarpSelect
}  // namespace warp_topk

// 已阅
// 说明：第一个为输出类型，第二个为输入类型
template <typename T_OUT, typename T_IN>
__device__ inline T_OUT cuda_cast(T_IN val) {
  return val;
}

// 已阅
template <>
__device__ inline float cuda_cast<float, __nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

// 已阅
template <typename T>
__device__ inline T neg_inf() {
  // cuda::std::numeric_limits<T>::infinity() returns `0` for [T=bf16 or fp16]
  // so we need to cast from fp32
  return cuda_cast<T, float>(-cuda::std::numeric_limits<float>::infinity());
}

// 已阅
template <typename T>
__device__ inline bool is_finite(const T val) {
#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120800)
  return cuda::std::isfinite(val);
#else
  return isfinite(cuda_cast<float, T>(val));
#endif
}

// 已阅
// Scoring function enums
enum ScoringFunc {
  SCORING_NONE = 0,    // no activation function
  SCORING_SIGMOID = 1  // apply sigmoid
};

// 已阅
// Efficient sigmoid approximation from TensorRT-LLM
__device__ inline float sigmoid_accurate(float x) {
  return 0.5f * tanhf(0.5f * x) + 0.5f;
}

// 已阅
template <typename T>
__device__ inline T apply_sigmoid(T val) {
  float f = cuda_cast<float, T>(val);
  return cuda_cast<T, float>(sigmoid_accurate(f));
}

// 已阅
template <ScoringFunc SF, typename T>
__device__ inline T apply_scoring(T val) {
  if constexpr (SF == SCORING_NONE) {
    return val;
  } else if constexpr (SF == SCORING_SIGMOID) {
    return apply_sigmoid(val);
  } else {
    static_assert(SF == SCORING_NONE || SF == SCORING_SIGMOID,
                  "Unsupported ScoringFunc in apply_scoring");
    return val;
  }
}

// 已阅
// 说明：计算组内 top2 之和
template <typename T, typename BiasT, ScoringFunc SF>
__device__ void topk_with_k2(T* output, T const* input, BiasT const* bias,
                             cg::thread_block_tile<32> const& tile,
                             int32_t const lane_id,
                             int const num_experts_per_group) {
  // Get the top2 per thread
  T largest = neg_inf<T>();
  T second_largest = neg_inf<T>();

  if (num_experts_per_group > WARP_SIZE) {
    // 说明：组内专家数量大于 warp 线程数，每个线程处理多个专家
    // lane_id 表示当前线程在 warp 内的索引
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      T value = apply_scoring<SF>(input[i]);
      value = value + static_cast<T>(bias[i]);

      if (value > largest) {
        second_largest = largest;
        largest = value;
      } else if (value > second_largest) {
        second_largest = value;
      }
    }
  } else {
    for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE) {
      T value = apply_scoring<SF>(input[i]);
      value = value + static_cast<T>(bias[i]);
      largest = value;
    }
  }
  // Get the top2 warpwise
  T max1 = cg::reduce(tile, largest, cg::greater<T>());

  T max2 = max1;
  bool equal_to_max1 = (max1 == largest);

  // 说明：__ballot_sync 返回一个位掩码，表示 warp 内每个线程的 equal_to_max1 结果
  // __popc 计算位掩码中值为 1 的位数，即有多少线程的 largest 等于 max1
  int count_max1 = __popc(__ballot_sync(FULL_WARP_MASK, equal_to_max1));

  if (count_max1 == 1) {
    largest = (largest == max1) ? second_largest : largest;
    max2 = cg::reduce(tile, largest, cg::greater<T>());
  }

  if (lane_id == 0) {
    *output = max1 + max2;
  }
}

// 已阅
template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
__global__ void grouped_topk_fused_kernel(
    T* scores, float* topk_values, IdxT* topk_indices, BiasT const* bias,
    int64_t const num_tokens, int64_t const num_experts, int64_t const n_group,
    int64_t const topk_group, int64_t const topk, bool renormalize,
    double routed_scaling_factor) {
  // 说明：one block per token
  int32_t const token_id = static_cast<int32_t>(blockIdx.x);
  if (token_id >= num_tokens) {
    return;
  }

  int32_t const warp_id = threadIdx.x / WARP_SIZE;
  int32_t const lane_id = threadIdx.x % WARP_SIZE;

  int32_t const n_group_i32 = static_cast<int32_t>(n_group);
  int32_t const topk_group_i32 = static_cast<int32_t>(topk_group);
  int32_t const topk_i32 = static_cast<int32_t>(topk);
  int32_t const num_experts_i32 = static_cast<int32_t>(num_experts);

  // 说明：one warp per group
  int32_t const num_warps = blockDim.x / WARP_SIZE;
  if (warp_id >= n_group_i32 || num_warps < n_group_i32) {
    return;
  }

  int32_t const num_experts_per_group = num_experts_i32 / n_group_i32;

  T* scores_token = scores + static_cast<int64_t>(token_id) * num_experts;

  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

  extern __shared__ char smem_buf[];
  // warpSelect internal staging buffer layout
  size_t const val_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(T);
  size_t const val_bytes_aligned =
      warp_topk::round_up_to_multiple_of<256>(val_bytes);
  size_t const idx_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(int32_t);
  size_t const internal_bytes = val_bytes_aligned + idx_bytes;

  // user-managed shared memory starts after warpSelect internal staging.
  uintptr_t ptr_u = reinterpret_cast<uintptr_t>(smem_buf + internal_bytes);
  ptr_u = (ptr_u + 15) & ~static_cast<uintptr_t>(15);  // align to 16B
  T* s_group_scores = reinterpret_cast<T*>(ptr_u);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");  // I think all prolog can be put before
                                         // acqbulk because it's ptr arithmetic
#endif

  // phase 1: per-group scan
  int32_t const group_offset = warp_id * num_experts_per_group;
  topk_with_k2<T, BiasT, SF>(s_group_scores + warp_id,
                             scores_token + group_offset, bias + group_offset,
                             tile, lane_id, num_experts_per_group);

  __syncthreads();

  // phase 2: warp0 selects groups + merges candidates to final topk
  if (warp_id != 0) {
    return;
  }

  topk_values += static_cast<int64_t>(token_id) * topk;
  topk_indices += static_cast<int64_t>(token_id) * topk;

  // 说明：greater == true，表示值小的靠前，值大的靠后；is_stable == true，表示值相等时索引大的靠前，索引小的靠后
  // select topk_group groups by group score
  warp_topk::WarpSelect</*capability*/ WARP_SIZE, /*greater*/ true, T, int32_t,
                        /* is_stable */ true>
      group_sel(static_cast<int32_t>(topk_group_i32), neg_inf<T>());

  // all lanes must participate in WarpSelect::add().
  T gscore = (lane_id < n_group_i32) ? s_group_scores[lane_id] : neg_inf<T>();
  group_sel.add(gscore, lane_id);
  group_sel.done();

  // proceed only if the k-th selected group score is not -inf
  bool proceed = false;
  if (topk_group_i32 > 0) {
    int const kth_lane = topk_group_i32 - 1;
    // broadcast the k-th selected group score to all lanes
    T kth_val = __shfl_sync(FULL_WARP_MASK, group_sel.get_val(0), kth_lane);
    proceed = (kth_val != neg_inf<T>());
  }

  // 说明：如果不满足条件，按顺序取前 topk 个专家，值均为 1.0 / topk
  if (!proceed) {
    for (int i = lane_id; i < topk_i32; i += WARP_SIZE) {
      topk_indices[i] = static_cast<IdxT>(i);
      topk_values[i] = 1.0f / static_cast<float>(topk_i32);
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    return;
  }

  // merge per-group topk candidates for selected groups, then select topk
  warp_topk::WarpSelect</*capability*/ WARP_SIZE, /*greater*/ true, T, int32_t,
                        /* is_stable */ true>
      expert_sel(static_cast<int32_t>(topk_i32), neg_inf<T>());

  // 说明：当前线程的 group idx
  // selected group ids reside in lanes [0, topk_group)
  int32_t sel_gid_lane = (lane_id < topk_group_i32) ? group_sel.get_idx(0) : 0;

  // add candidates from selected groups to expert_sel
  for (int32_t g = 0; g < topk_group_i32; ++g) {
    // 说明：广播 g 线程选中的 group id 到所有线程
    int32_t gid = __shfl_sync(FULL_WARP_MASK, sel_gid_lane, g);
    int32_t const offset = gid * num_experts_per_group;
    int32_t const align_num_experts_per_group =
        warp_topk::round_up_to_multiple_of<WARP_SIZE>(num_experts_per_group);
    for (int32_t i = lane_id; i < align_num_experts_per_group; i += WARP_SIZE) {
      // all lanes must call `add()` the same number of times.
      T cand = neg_inf<T>();
      int32_t idx = 0;
      if (i < num_experts_per_group) {
        idx = offset + i;
        T input = scores_token[idx];
        if (is_finite(input)) {
          T score = apply_scoring<SF>(input);
          cand = score + static_cast<T>(bias[idx]);
        }
      }
      expert_sel.add(cand, idx);
    }
  }
  expert_sel.done();

  // compute unbiased routing weights + optional renorm.
  float lane_unbiased = 0.0f;
  IdxT lane_idx = 0;
  if (lane_id < topk_i32) {
    lane_idx = static_cast<IdxT>(expert_sel.get_idx(0));
    T in = scores_token[static_cast<int32_t>(lane_idx)];
    lane_unbiased = cuda_cast<float, T>(apply_scoring<SF>(in));
  }

  float topk_sum = 1e-20f;
  if (renormalize) {
    topk_sum += cg::reduce(tile, lane_unbiased, cg::plus<float>());
  }

  float scale = static_cast<float>(routed_scaling_factor);
  if (renormalize) {
    scale /= topk_sum;
  }

  if (lane_id < topk_i32) {
    topk_indices[lane_id] = lane_idx;
    topk_values[lane_id] = lane_unbiased * scale;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, typename BiasT, typename IdxT>
void invokeNoAuxTc(T* scores, float* topk_values, IdxT* topk_indices,
                   BiasT const* bias, int64_t const num_tokens,
                   int64_t const num_experts, int64_t const n_group,
                   int64_t const topk_group, int64_t const topk,
                   bool const renormalize, double const routed_scaling_factor,
                   int const scoring_func, bool enable_pdl = false,
                   cudaStream_t const stream = 0) {
  cudaLaunchConfig_t config;
  // One block per token; one warp per group.
  config.gridDim = static_cast<uint32_t>(num_tokens);
  config.blockDim = static_cast<uint32_t>(n_group) * WARP_SIZE;
  // Dynamic shared memory: WarpSelect staging + per-group topk buffers.
  int32_t const num_warps = static_cast<int32_t>(n_group);
  size_t const val_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(T);
  size_t const val_bytes_aligned =
      warp_topk::round_up_to_multiple_of<256>(val_bytes);
  size_t const idx_bytes =
      static_cast<size_t>(num_warps) * WARP_SIZE * sizeof(int32_t);
  size_t const internal_bytes = val_bytes_aligned + idx_bytes;
  // 说明：16 用于给 16B 对齐留出足够空间；n_group * sizeof(T) 用于存储每个 group 的 score
  size_t const extra_bytes = 16 + static_cast<size_t>(n_group) * sizeof(T);
  config.dynamicSmemBytes = internal_bytes + extra_bytes;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  auto const sf = static_cast<ScoringFunc>(scoring_func);
  switch (sf) {
    case SCORING_NONE: {
      auto* kernel_instance =
          &grouped_topk_fused_kernel<T, BiasT, IdxT, SCORING_NONE>;
      cudaLaunchKernelEx(&config, kernel_instance, scores, topk_values,
                         topk_indices, bias, num_tokens, num_experts, n_group,
                         topk_group, topk, renormalize, routed_scaling_factor);
      return;
    }
    case SCORING_SIGMOID: {
      auto* kernel_instance =
          &grouped_topk_fused_kernel<T, BiasT, IdxT, SCORING_SIGMOID>;
      cudaLaunchKernelEx(&config, kernel_instance, scores, topk_values,
                         topk_indices, bias, num_tokens, num_experts, n_group,
                         topk_group, topk, renormalize, routed_scaling_factor);
      return;
    }
    default:
      // should be guarded by higher level checks.
      TORCH_CHECK(false, "Unsupported scoring_func in invokeNoAuxTc");
  }
}

#define INSTANTIATE_NOAUX_TC(T, BiasT, IdxT)                                 \
  template void invokeNoAuxTc<T, BiasT, IdxT>(                               \
      T * scores, float* topk_values, IdxT* topk_indices, BiasT const* bias, \
      int64_t const num_tokens, int64_t const num_experts,                   \
      int64_t const n_group, int64_t const topk_group, int64_t const topk,   \
      bool const renormalize, double const routed_scaling_factor,            \
      int const scoring_func, bool enable_pdl, cudaStream_t const stream);

INSTANTIATE_NOAUX_TC(float, float, int32_t);
INSTANTIATE_NOAUX_TC(float, half, int32_t);
INSTANTIATE_NOAUX_TC(float, __nv_bfloat16, int32_t);
INSTANTIATE_NOAUX_TC(half, float, int32_t);
INSTANTIATE_NOAUX_TC(half, half, int32_t);
INSTANTIATE_NOAUX_TC(half, __nv_bfloat16, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, float, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, half, int32_t);
INSTANTIATE_NOAUX_TC(__nv_bfloat16, __nv_bfloat16, int32_t);
}  // end namespace moe
}  // namespace vllm

// 已阅
std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    torch::Tensor const& bias, int64_t scoring_func = 0) {
  auto data_type = scores.scalar_type();
  auto bias_type = bias.scalar_type();
  auto input_size = scores.sizes();
  int64_t num_tokens = input_size[0];
  int64_t num_experts = input_size[1];
  TORCH_CHECK(input_size.size() == 2, "scores must be a 2D Tensor");
  TORCH_CHECK(n_group > 0, "n_group must be positive");
  TORCH_CHECK(topk > 0, "topk must be positive");
  TORCH_CHECK(topk_group > 0, "topk_group must be positive");
  TORCH_CHECK(topk_group <= n_group, "topk_group must be <= n_group");
  TORCH_CHECK(num_experts % n_group == 0,
              "num_experts should be divisible by n_group");
  TORCH_CHECK(n_group <= 32,
              "n_group should be smaller than or equal to 32 for now");
  TORCH_CHECK(topk <= 32, "topk should be smaller than or equal to 32 for now");
  TORCH_CHECK(topk <= topk_group * (num_experts / n_group),
              "topk must be <= topk_group * (num_experts / n_group)");
  TORCH_CHECK(scoring_func == vllm::moe::SCORING_NONE ||
                  scoring_func == vllm::moe::SCORING_SIGMOID,
              "scoring_func must be SCORING_NONE (0) or SCORING_SIGMOID (1)");

  // Always output float32 for topk_values (eliminates Python-side conversion)
  torch::Tensor topk_values = torch::empty(
      {num_tokens, topk}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
  torch::Tensor topk_indices = torch::empty(
      {num_tokens, topk}, torch::dtype(torch::kInt32).device(torch::kCUDA));

  auto stream = c10::cuda::getCurrentCUDAStream(scores.get_device());

#define LAUNCH_KERNEL(T, IdxT)                                               \
  do {                                                                       \
    switch (bias_type) {                                                     \
      case torch::kFloat16:                                                  \
        vllm::moe::invokeNoAuxTc<T, half, IdxT>(                             \
            reinterpret_cast<T*>(scores.mutable_data_ptr()),                 \
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),        \
            reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),        \
            reinterpret_cast<half const*>(bias.data_ptr()), num_tokens,      \
            num_experts, n_group, topk_group, topk, renormalize,             \
            routed_scaling_factor, static_cast<int>(scoring_func), false,    \
            stream);                                                         \
        break;                                                               \
      case torch::kFloat32:                                                  \
        vllm::moe::invokeNoAuxTc<T, float, IdxT>(                            \
            reinterpret_cast<T*>(scores.mutable_data_ptr()),                 \
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),        \
            reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),        \
            reinterpret_cast<float const*>(bias.data_ptr()), num_tokens,     \
            num_experts, n_group, topk_group, topk, renormalize,             \
            routed_scaling_factor, static_cast<int>(scoring_func), false,    \
            stream);                                                         \
        break;                                                               \
      case torch::kBFloat16:                                                 \
        vllm::moe::invokeNoAuxTc<T, __nv_bfloat16, IdxT>(                    \
            reinterpret_cast<T*>(scores.mutable_data_ptr()),                 \
            reinterpret_cast<float*>(topk_values.mutable_data_ptr()),        \
            reinterpret_cast<IdxT*>(topk_indices.mutable_data_ptr()),        \
            reinterpret_cast<__nv_bfloat16 const*>(bias.data_ptr()),         \
            num_tokens, num_experts, n_group, topk_group, topk, renormalize, \
            routed_scaling_factor, static_cast<int>(scoring_func), false,    \
            stream);                                                         \
        break;                                                               \
      default:                                                               \
        throw std::invalid_argument(                                         \
            "Invalid bias dtype, only supports float16, float32, and "       \
            "bfloat16");                                                     \
        break;                                                               \
    }                                                                        \
  } while (0)

  switch (data_type) {
    case torch::kFloat16:
      // Handle Float16
      LAUNCH_KERNEL(half, int32_t);
      break;
    case torch::kFloat32:
      // Handle Float32
      LAUNCH_KERNEL(float, int32_t);
      break;
    case torch::kBFloat16:
      // Handle BFloat16
      LAUNCH_KERNEL(__nv_bfloat16, int32_t);
      break;
    default:
      // Handle other data types
      throw std::invalid_argument(
          "Invalid dtype, only supports float16, float32, and bfloat16");
      break;
  }
#undef LAUNCH_KERNEL
  return {topk_values, topk_indices};
}
