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
constexpr int32_t BLOCK_SIZE = 512;
constexpr int32_t NUM_WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

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
// 说明：一个 block 中所有 warp 共享的 smem 大小
template <typename T, typename idxT>
int calc_smem_size_for_block_wide(int num_of_warp, int64_t k) {
  // 说明：每个 warp 处理一个 token，一个 token 需要存储 k 个值和 k 个索引
  int64_t cache_topk = (sizeof(T) + sizeof(idxT)) * num_of_warp * k;
  // 问题：这里的逻辑不是很理解，可能是启发式的经验值
  int64_t n = std::max<int>(num_of_warp / 2 * k, num_of_warp * WARP_SIZE);
  return max(cache_topk,
             round_up_to_multiple_of<256>(n * sizeof(T)) + n * sizeof(idxT));
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
      // 说明：首次比较时，k_th_ 值为 dummy，k_th_idx_ 未初始化；
      // val 有可能是 dummy，如 num_experts_per_group <= lane_id < align_num_experts_per_group 时，
      // 但这样并不会影响结果，因为 dummy 值最终不会进入 topk 中
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

    // after done(), smem is used for merging results among warps
    __syncthreads();
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
template <typename T, typename BiasT, ScoringFunc SF>
__global__ void topk_with_k2_kernel(T* output, T* input, BiasT const* bias,
                                    int64_t const num_tokens,
                                    int64_t const num_cases,
                                    int64_t const n_group,
                                    int64_t const num_experts_per_group) {
  int32_t warp_id = threadIdx.x / WARP_SIZE;
  int32_t lane_id = threadIdx.x % WARP_SIZE;

  // 说明：一个 warp 处理一个 case，一个 case 对应一个 token 的一个 expert group
  int32_t case_id = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id;
  if (case_id < num_cases) {
    input += case_id * num_experts_per_group;
    // bias is per expert group, offset to current group
    int32_t group_id = case_id % n_group;
    // 说明：group_bias 指向当前 expert group 的 bias 起始位置
    // bias 的 shape 是 [num_experts]，跳过前面 group_id * num_experts_per_group 个 experts
    // 得到起始位置
    BiasT const* group_bias = bias + group_id * num_experts_per_group;
    output += case_id;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;");
#endif
    topk_with_k2<T, BiasT, SF>(output, input, group_bias, tile, lane_id,
                               num_experts_per_group);
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// 已阅
template <typename T, typename BiasT, typename IdxT, ScoringFunc SF,
          int NGroup = -1>
__global__ void group_idx_and_topk_idx_kernel(
    T* scores, T const* group_scores, float* topk_values, IdxT* topk_indices,
    BiasT const* bias, int64_t const num_tokens, int64_t const n_group,
    int64_t const topk_group, int64_t const topk, int64_t const num_experts,
    int64_t const num_experts_per_group, bool renormalize,
    double routed_scaling_factor) {
  int32_t warp_id = threadIdx.x / WARP_SIZE;
  int32_t lane_id = threadIdx.x % WARP_SIZE;
  // 说明：一个 warp 处理一个 token
  int32_t case_id =
      blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id;  // one per token
  scores += case_id * num_experts;
  group_scores += case_id * n_group;
  topk_values += case_id * topk;
  topk_indices += case_id * topk;

  constexpr bool kUseStaticNGroup = (NGroup > 0);
  // use int32 to avoid implicit conversion
  // 说明：如果 NGroup 是编译时常量，则直接使用 NGroup，否则将 n_group 转换为 int32_t
  int32_t const n_group_i32 =
      kUseStaticNGroup ? NGroup : static_cast<int32_t>(n_group);

  int32_t align_num_experts_per_group =
      warp_topk::round_up_to_multiple_of<WARP_SIZE>(num_experts_per_group);

  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

  extern __shared__ char smem_buf[];  // NOTE: reuse the shared memory here to
                                      // store the target topk idx
  int32_t* s_topk_idx = reinterpret_cast<int32_t*>(smem_buf);
  // 说明：smem_buf 前半部分存储 topk indices，后半部分存储 topk values
  T* s_topk_value =
      reinterpret_cast<T*>(s_topk_idx + NUM_WARPS_PER_BLOCK * topk) +
      warp_id * topk;
  // 说明：当前 warp 的 topk idx 起始位置
  s_topk_idx += warp_id * topk;

  T value = neg_inf<T>();
  // 说明：终值是第 topk_group 大的 group score（值相等时独立计数）
  T topk_group_value = neg_inf<T>();
  // 说明：记录与 topk_group_value 相等的 group score 数量
  int32_t num_equalto_topkth_group;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");  // I think all prolog can be put before
                                         // acqbulk because it's ptr arithmetic
#endif

  if (case_id < num_tokens) {
    // calculate group_idx
    // 说明：WARP_SIZE - n_group_i32 表示当前 warp 内没有被分配 expert group 的线程数，
    // 这些线程的 value 初始值就是负无穷，因此初始时已经有 WARP_SIZE - n_group_i32 个线程的
    // group score 等于负无穷；下面逻辑中将 count_equal_to_top_value 初始化为 WARP_SIZE - n_group_i32
    // 就是这个原因
    int32_t target_num_min =
        WARP_SIZE - n_group_i32 + static_cast<int32_t>(topk_group);
    // The check is necessary to avoid abnormal input
    // 说明：线程处理 expert group 的分数
    if (lane_id < n_group_i32 && is_finite(group_scores[lane_id])) {
      value = group_scores[lane_id];
    }

    int count_equal_to_top_value = WARP_SIZE - n_group_i32;
    int pre_count_equal_to_top_value = 0;
    // Use loop to find the largset top_group
    while (count_equal_to_top_value < target_num_min) {
      topk_group_value = cg::reduce(tile, value, cg::greater<T>());
      if (value == topk_group_value) {
        // 说明：当前线程的 group score 是最大的，重置为负无穷，并在后续循环中保持不变
        value = neg_inf<T>();
      }
      pre_count_equal_to_top_value = count_equal_to_top_value;
      // 说明：group_score 是最大时，value 会被设置为负无穷，
      // 统计当前 warp 内有多少线程的 group score 等于负无穷
      // 包括：未进入分支的线程 + 之前轮次被设置为负无穷的线程 + 当前轮次被设置为负无穷的线程
      // 每执行一轮 while，count_equal_to_top_value 应该都会因为有线程的 value 被设置为负无穷而增加，
      // 直到达到 target_num_min 为止；即最多执行 topk_group 轮 while
      count_equal_to_top_value =
          __popc(__ballot_sync(FULL_WARP_MASK, (value == neg_inf<T>())));
    }
    // 说明：pre_count_equal_to_top_value 表示退出 while 前已经统计到的属于 topk_group 的线程数，
    // 退出循环说明已经找到了至少 topk_group 个最大的 group score，
    // num_equalto_topkth_group 表示至少还需要多少个线程的 group score 来凑足 topk_group 个组，
    // 同时说明最后这些线程的 group score 都相等
    num_equalto_topkth_group = target_num_min - pre_count_equal_to_top_value;
  }
  __syncthreads();

  // 说明：dummy 初始化为负无穷
  // 说明：greater = true，表示值小的靠前，值大的靠后；is_stable = true，表示值相同时索引大的靠前，索引小的靠后
  warp_topk::WarpSelect</*capability*/ WARP_SIZE, /*greater*/ true, T, int32_t,
                        /* is_stable */ true>
      queue((int32_t)topk, neg_inf<T>());

  int count_equalto_topkth_group = 0;
  // 说明：topk_group_value 此时记录的是第 topk_group 大的 group score
  // 为 false 说明没有足够的 expert group 可供选择
  bool if_proceed_next_topk = topk_group_value != neg_inf<T>();
  if (case_id < num_tokens && if_proceed_next_topk) {
    auto process_group = [&](int i_group) {
      if ((group_scores[i_group] > topk_group_value) ||
          ((group_scores[i_group] == topk_group_value) &&
           (count_equalto_topkth_group < num_equalto_topkth_group))) {
        int32_t offset = i_group * num_experts_per_group;
        for (int32_t i = lane_id; i < align_num_experts_per_group;
             i += WARP_SIZE) {
          T candidates = neg_inf<T>();
          if (i < num_experts_per_group) {
            // apply scoring function (if any) and add bias
            // 说明：获取 expert 的 score 值，并施加 scoring function 和 bias
            T input = scores[offset + i];
            if (is_finite(input)) {
              T score = apply_scoring<SF>(input);
              candidates = score + static_cast<T>(bias[offset + i]);
            }
          }
          // 说明：每个线程处理组内多个专家，添加到 WarpSelect 中进行 topk 选择
          // i >= num_experts_per_group 时，candidates 为负无穷
          queue.add(candidates, offset + i);
        }
        if (group_scores[i_group] == topk_group_value) {
          count_equalto_topkth_group++;
        }
      }
    };

    // 说明：选择 topk_group 个 expert group 进行处理，将组内专家的 score 添加到 WarpSelect 中进行 topk 选择
    // group_score 相同时，优先选择索引小的 group
    if constexpr (kUseStaticNGroup) {
#pragma unroll
      for (int i_group = 0; i_group < NGroup; ++i_group) {
        process_group(i_group);
      }
    } else {
      for (int i_group = 0; i_group < n_group_i32; ++i_group) {
        process_group(i_group);
      }
    }
    queue.done();
    // Get the topk_idx
    // 说明：将 topk 的索引输出到 s_topk_idx 中
    queue.dumpIdx(s_topk_idx);
  }

  // Load the valid score value
  // Calculate the summation
  float topk_sum = 1e-20;
  if (case_id < num_tokens && if_proceed_next_topk) {
    for (int i = lane_id;
         i < warp_topk::round_up_to_multiple_of<WARP_SIZE>(topk);
         i += WARP_SIZE) {
      T value = cuda_cast<T, float>(0.0f);
      if (i < topk) {
        // Load the score value (without bias) for normalization
        T input = scores[s_topk_idx[i]];
        value = apply_scoring<SF>(input);
        s_topk_value[i] = value;
      }
      if (renormalize) {
        topk_sum +=
            cg::reduce(tile, cuda_cast<float, T>(value), cg::plus<float>());
      }
    }
  }

  __syncthreads();

  if (case_id < num_tokens) {
    if (if_proceed_next_topk) {
      float scale = routed_scaling_factor;
      if (renormalize) {
        scale /= topk_sum;
      }
      for (int i = lane_id; i < topk; i += WARP_SIZE) {
        float base = cuda_cast<float, T>(s_topk_value[i]);
        float value = base * scale;
        topk_indices[i] = s_topk_idx[i];
        topk_values[i] = value;
      }
    } else {
      for (int i = lane_id; i < topk; i += WARP_SIZE) {
        topk_indices[i] = i;
        topk_values[i] = 1.0f / topk;
      }
    }
    // Note: when if_proceed_next_topk==false, choose the first 8 experts as the
    // default result.
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// 已阅
template <typename T, typename BiasT, typename IdxT, ScoringFunc SF>
inline void launch_group_idx_and_topk_kernel(
    cudaLaunchConfig_t const& config, T* scores, T* group_scores,
    float* topk_values, IdxT* topk_indices, BiasT const* bias,
    int64_t const num_tokens, int64_t const n_group, int64_t const topk_group,
    int64_t const topk, int64_t const num_experts,
    int64_t const num_experts_per_group, bool const renormalize,
    double const routed_scaling_factor) {
  auto launch = [&](auto* kernel_instance2) {
    cudaLaunchKernelEx(&config, kernel_instance2, scores, group_scores,
                       topk_values, topk_indices, bias, num_tokens, n_group,
                       topk_group, topk, num_experts, num_experts_per_group,
                       renormalize, routed_scaling_factor);
  };

  switch (n_group) {
    case 4: {
      launch(&group_idx_and_topk_idx_kernel<T, BiasT, IdxT, SF, 4>);
      break;
    }
    case 8: {
      launch(&group_idx_and_topk_idx_kernel<T, BiasT, IdxT, SF, 8>);
      break;
    }
    case 16: {
      launch(&group_idx_and_topk_idx_kernel<T, BiasT, IdxT, SF, 16>);
      break;
    }
    case 32: {
      launch(&group_idx_and_topk_idx_kernel<T, BiasT, IdxT, SF, 32>);
      break;
    }
    default: {
      launch(&group_idx_and_topk_idx_kernel<T, BiasT, IdxT, SF>);
      break;
    }
  }
}

template <typename T, typename BiasT, typename IdxT>
void invokeNoAuxTc(T* scores, T* group_scores, float* topk_values,
                   IdxT* topk_indices, BiasT const* bias,
                   int64_t const num_tokens, int64_t const num_experts,
                   int64_t const n_group, int64_t const topk_group,
                   int64_t const topk, bool const renormalize,
                   double const routed_scaling_factor, int const scoring_func,
                   bool enable_pdl = false, cudaStream_t const stream = 0) {
  int64_t num_cases = num_tokens * n_group;
  // 说明：计算线程块的数量，说明每个 warp 处理一个 case
  // k2 指每个 expert group 选 top2，用于计算 group score
  int64_t topk_with_k2_num_blocks = (num_cases - 1) / NUM_WARPS_PER_BLOCK + 1;
  cudaLaunchConfig_t config;
  config.gridDim = topk_with_k2_num_blocks;
  config.blockDim = BLOCK_SIZE;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  auto const sf = static_cast<ScoringFunc>(scoring_func);
  int64_t const num_experts_per_group = num_experts / n_group;
  // 理解：定义一个 lambda 函数，用于启动 topk_with_k2_kernel 核函数
  auto launch_topk_with_k2 = [&](auto* kernel_instance1) {
    cudaLaunchKernelEx(&config, kernel_instance1, group_scores, scores, bias,
                       num_tokens, num_cases, n_group, num_experts_per_group);
  };
  switch (sf) {
    case SCORING_NONE: {
      auto* kernel_instance1 = &topk_with_k2_kernel<T, BiasT, SCORING_NONE>;
      launch_topk_with_k2(kernel_instance1);
      break;
    }
    case SCORING_SIGMOID: {
      auto* kernel_instance1 = &topk_with_k2_kernel<T, BiasT, SCORING_SIGMOID>;
      launch_topk_with_k2(kernel_instance1);
      break;
    }
    default:
      // should be guarded by higher level checks.
      TORCH_CHECK(false, "Unsupported scoring_func in invokeNoAuxTc");
  }

  // 说明：计算线程块的数量，说明每个 warp 处理一个 token
  int64_t topk_with_k_group_num_blocks =
      (num_tokens - 1) / NUM_WARPS_PER_BLOCK + 1;
  size_t dynamic_smem_in_bytes =
      warp_topk::calc_smem_size_for_block_wide<T, int32_t>(NUM_WARPS_PER_BLOCK,
                                                           topk);
  config.gridDim = topk_with_k_group_num_blocks;
  config.blockDim = BLOCK_SIZE;
  config.dynamicSmemBytes = dynamic_smem_in_bytes;
  config.stream = stream;
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  switch (sf) {
    case SCORING_NONE: {
      launch_group_idx_and_topk_kernel<T, BiasT, IdxT, SCORING_NONE>(
          config, scores, group_scores, topk_values, topk_indices, bias,
          num_tokens, n_group, topk_group, topk, num_experts,
          num_experts_per_group, renormalize, routed_scaling_factor);
      break;
    }
    case SCORING_SIGMOID: {
      launch_group_idx_and_topk_kernel<T, BiasT, IdxT, SCORING_SIGMOID>(
          config, scores, group_scores, topk_values, topk_indices, bias,
          num_tokens, n_group, topk_group, topk, num_experts,
          num_experts_per_group, renormalize, routed_scaling_factor);
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported scoring_func in invokeNoAuxTc");
  }
}

#define INSTANTIATE_NOAUX_TC(T, BiasT, IdxT)                                  \
  template void invokeNoAuxTc<T, BiasT, IdxT>(                                \
      T * scores, T * group_scores, float* topk_values, IdxT* topk_indices,   \
      BiasT const* bias, int64_t const num_tokens, int64_t const num_experts, \
      int64_t const n_group, int64_t const topk_group, int64_t const topk,    \
      bool const renormalize, double const routed_scaling_factor,             \
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
  TORCH_CHECK(num_experts % n_group == 0,
              "num_experts should be divisible by n_group");
  TORCH_CHECK(n_group <= 32,
              "n_group should be smaller than or equal to 32 for now");
  TORCH_CHECK(topk <= 32, "topk should be smaller than or equal to 32 for now");
  TORCH_CHECK(scoring_func == vllm::moe::SCORING_NONE ||
                  scoring_func == vllm::moe::SCORING_SIGMOID,
              "scoring_func must be SCORING_NONE (0) or SCORING_SIGMOID (1)");

  torch::Tensor group_scores = torch::empty(
      {num_tokens, n_group}, torch::dtype(data_type).device(torch::kCUDA));
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
            reinterpret_cast<T*>(group_scores.mutable_data_ptr()),           \
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
            reinterpret_cast<T*>(group_scores.mutable_data_ptr()),           \
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
            reinterpret_cast<T*>(group_scores.mutable_data_ptr()),           \
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
