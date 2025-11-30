#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Optional.h>

#include "cuda_utils.h"
#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "quantization/vectorization_utils.cuh"

#ifdef USE_ROCM
  #include "quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

#include <algorithm>
#include <cassert>
#include <cfloat>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#endif

#if defined(__gfx942__)
constexpr float kFp8ScaleDivisor = 224.f;
#else
// 说明：标准 E4M3 格式的数值范围通常为 ‌[-448, 448]‌
constexpr float kFp8ScaleDivisor = 448.f;
#endif

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 int64_t block_size_in_bytes,
                 const torch::Tensor& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(src_device.index() == dst_device.index(),
                "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  // NOTE(youkaichao): keep in mind that `block_mapping` should be
  // a cpu tensor, otherwise every `item` call will require a gpu-cpu
  // synchronization.
  TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());

  const at::cuda::OptionalCUDAGuard device_guard(
      src_device.is_cuda() ? src_device : dst_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  const int64_t num_blocks = block_mapping.size(0);
  for (size_t i = 0; i < num_blocks; i++) {
    int64_t src_block_number = block_mapping[i][0].item<int64_t>();
    int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    cudaMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset,
                    block_size_in_bytes, memcpy_type, stream);
  }
}

namespace vllm {

// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_kernel(int64_t* key_cache_ptrs,
                                   int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ block_mapping,
                                   const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache =
      reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

// Kernel for MLA, which works on a single joint kv_cache
// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_mla_kernel(
    int64_t* cache_ptrs, const int64_t* __restrict__ block_mapping,
    const int mem_footprint_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;
  scalar_t* cache = reinterpret_cast<scalar_t*>(cache_ptrs[layer_idx]);
  int64_t src_block = block_mapping[2 * pair_idx];
  int64_t dst_block = block_mapping[2 * pair_idx + 1];
  int64_t src_offset = src_block * mem_footprint_per_block;
  int64_t dst_offset = dst_block * mem_footprint_per_block;
  for (int i = threadIdx.x; i < mem_footprint_per_block; i += blockDim.x) {
    cache[dst_offset + i] = cache[src_offset + i];
  }
}

}  // namespace vllm

namespace vllm {

// Used to copy/convert one element
template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
  float scale;

  __device__ __forceinline__ void operator()(OutT& dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);
    } else {
      dst = fp8::scaled_convert<OutT, InT, kv_dt>(src, scale);
    }
  }
};

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x,
                                         // block_size, x]
    cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size,
                                         // block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float* k_scale, const float* v_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int h_block_count = head_size / x;  // head_size//x

  const int h_block_idx = threadIdx.x;
  if (h_block_idx >= num_heads * h_block_count) {
    return;
  }

  const int head_idx = h_block_idx / h_block_count;
  const int h_block = h_block_idx % h_block_count;

  const scalar_t* __restrict__ key_src =
      key + token_idx * key_stride + head_idx * head_size + h_block * x;
  const int64_t src_value_start =
      token_idx * value_stride + head_idx * head_size + h_block * x;

  cache_t* __restrict__ key_dst =
      key_cache + block_idx * num_heads * h_block_count * block_size * x +
      head_idx * h_block_count * block_size * x + h_block * block_size * x +
      block_offset * x;
  const int64_t tgt_value_start =
      block_idx * num_heads * h_block_count * x * block_size +
      head_idx * h_block_count * x * block_size + h_block * x * block_size +
      block_offset;

  constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;
  float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
  float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

  vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, x, 0, 1, k_op);

  const scalar_t* __restrict__ value_src = value + src_value_start;
  cache_t* __restrict__ value_dst = value_cache + tgt_value_start;
#pragma unroll
  for (int i = 0; i < x; i++) {
    v_op(value_dst[i * block_size], value_src[i]);
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // NHD or HND, shape see comments below
    cache_t* __restrict__ value_cache,   // same above
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t block_stride, const int64_t page_stride,
    const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int block_size, const float* k_scale, const float* v_scale,
    const int kv_scale_stride) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n_elems = num_heads * head_size;

  // pointers to the beginning of the source row for this token.
  const scalar_t* __restrict__ key_src = key + token_idx * key_stride;
  const scalar_t* __restrict__ value_src = value + token_idx * value_stride;

  // find the start position inside the kv-cache for this token.
  cache_t* __restrict__ key_dst =
      key_cache + block_idx * block_stride + block_offset * page_stride;
  cache_t* __restrict__ value_dst =
      value_cache + block_idx * block_stride + block_offset * page_stride;

  // this is true for the NHD layout where `head_stride == head_size`
  const bool is_contiguous_heads = (head_stride == head_size);

  constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;

  if (is_contiguous_heads && kv_scale_stride == 0) {
    // NHD layout and k/v_scales are [1] (i.e. single scale for all heads)
    // kv cache: [num_blocks, block_size, num_heads, head_size]
    float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
    float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;

    CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
    CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

    vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, n_elems, threadIdx.x,
                                       blockDim.x, k_op);
    vectorize_with_alignment<VEC_SIZE>(value_src, value_dst, n_elems,
                                       threadIdx.x, blockDim.x, v_op);
  } else {
    // HND layout OR k/v_scales are [num_heads] (i.e. per-attn-head)
    // HND layout: heads are strided, but each head_size segment is contiguous
    // kv cache: [num_blocks, num_heads, block_size, head_size]
    const int lane = threadIdx.x & 31;     // 0..31 within warp
    const int warp_id = threadIdx.x >> 5;  // warp index within block
    const int warps_per_block = blockDim.x >> 5;

    for (int head = warp_id; head < num_heads; head += warps_per_block) {
      const scalar_t* __restrict__ k_src_h = key_src + head * head_size;
      const scalar_t* __restrict__ v_src_h = value_src + head * head_size;

      cache_t* __restrict__ k_dst_h =
          key_dst + static_cast<int64_t>(head) * head_stride;
      cache_t* __restrict__ v_dst_h =
          value_dst + static_cast<int64_t>(head) * head_stride;

      float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto)
                              ? 0.f
                              : k_scale[head * kv_scale_stride];
      float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto)
                              ? 0.f
                              : v_scale[head * kv_scale_stride];

      CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
      CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

      // within each head, let the 32 threads of the warp perform the vector
      // copy
      vectorize_with_alignment<VEC_SIZE>(k_src_h, k_dst_h, head_size, lane, 32,
                                         k_op);

      vectorize_with_alignment<VEC_SIZE>(v_src_h, v_dst_h, head_size, lane, 32,
                                         v_op);
    }
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  auto copy = [&](const scalar_t* __restrict__ src, cache_t* __restrict__ dst,
                  int src_stride, int dst_stride, int size, int offset) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      const int64_t src_idx = token_idx * src_stride + i;
      const int64_t dst_idx =
          block_idx * block_stride + block_offset * entry_stride + i + offset;
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        dst[dst_idx] = src[src_idx];
      } else {
        dst[dst_idx] =
            fp8::scaled_convert<cache_t, scalar_t, kv_dt>(src[src_idx], *scale);
      }
    }
  };

  copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
  copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_ds_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int64_t dst_idx_start =
      block_idx * block_stride + block_offset * entry_stride;

  // For the NoPE part, each tile of 128 elements is handled by half of one warp
  // (16 threads). There are 4 total tiles, so 2 warps (64 threads).
  // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
  // The RoPE part (last 64 elements) is handled by another 1 warp (32 threads).
  // So in total, we use 3 warps (96 threads) per block.

  // Cast kv_cache to 16_bit for RoPE values
  scalar_t* kv_cache_16bit =
      reinterpret_cast<scalar_t*>(&kv_cache[dst_idx_start]);

  // The last warp handles the RoPE part
  if (threadIdx.x >= 64) {
    // Each thread handles two elements of RoPE
    const int8_t pe_idx_start = (threadIdx.x - 64) * 2;
    const int64_t src_idx = token_idx * k_pe_stride + pe_idx_start;
    // Vectorized load of two 16-bit values, performed as one 32-bit load
    const int32_t vals = *reinterpret_cast<const int32_t*>(&k_pe[src_idx]);
    // RoPE values start after the packed 8-bit NoPE values and the
    // 32-bit scales
    const int64_t dst_idx = kv_lora_rank / 2 + 8 + pe_idx_start;
    // Vectorized store of two 16-bit values, performed as one 32-bit store
    *reinterpret_cast<int32_t*>(&kv_cache_16bit[dst_idx]) = vals;
    return;
  }

  // The first two warps handle the NoPE part
  const int8_t warp_idx = threadIdx.x >> 5;
  const int8_t lane_idx = threadIdx.x & 31;
  const int8_t tile_idx = warp_idx * 2 + (lane_idx >> 4);

  // Each thread handles 8 elements of NoPE
  // Load the NoPE elements for this thread into registers
  const int64_t src_idx_start = token_idx * kv_c_stride + (threadIdx.x * 8);
  // Vectorized load of eight 16-bit values, performed as an int4 load
  const int4 vals_i4 = *reinterpret_cast<const int4*>(&kv_c[src_idx_start]);
  const scalar_t* vals = reinterpret_cast<const scalar_t*>(&vals_i4);

  // Max absolute value of this thread's elements
  float max_abs = fmaxf(fmaxf(fmaxf(fabsf(vals[0]), fabsf(vals[1])),
                              fmaxf(fabsf(vals[2]), fabsf(vals[3]))),
                        fmaxf(fmaxf(fabsf(vals[4]), fabsf(vals[5])),
                              fmaxf(fabsf(vals[6]), fabsf(vals[7]))));

  // Warp-level reduction to find the max absolute value in each half-warp
#pragma unroll
  for (int offset = 8; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, VLLM_SHFL_XOR_SYNC_WIDTH(max_abs, offset, 16));
  }

  // Compute the scale for the tile
  float tile_scale = fmaxf(max_abs / kFp8ScaleDivisor, FLT_MIN);

  // The first lane of each half-warp writes the scale to kv_cache
  if ((lane_idx == 0) || (lane_idx == 16)) {
    float* kv_cache_32bit = reinterpret_cast<float*>(&kv_cache[dst_idx_start]);
    const uint64_t dst_idx = kv_lora_rank / 4 + tile_idx;
    kv_cache_32bit[dst_idx] = tile_scale;
  }

  // Now all threads in the block scale and write their elements
  // NoPE data is packed in the first kv_lora_rank/2 bytes (first 256 bytes)
  const int64_t dst_idx_base = dst_idx_start + (threadIdx.x * 8);

  uint8_t result[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    result[i] =
        fp8::scaled_convert<uint8_t, scalar_t, Fp8KVCacheDataType::kFp8E4M3>(
            vals[i], tile_scale);
  }

  // Store as aligned 64-bit writes
  *reinterpret_cast<uint64_t*>(&kv_cache[dst_idx_base]) =
      *reinterpret_cast<const uint64_t*>(result);
}

// 说明：cache_t 表示输出类型，scalar_t 表示输入类型
// 说明：将 k 张量中的值进行量化，并存储到 kv_cache 中；还会计算并存储 scale 值，存储位置在 k 张量量化值之后
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void indexer_k_quant_and_cache_kernel(
    const scalar_t* __restrict__ k,  // [num_tokens, head_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, cache_stride]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int head_dim,                        // dimension of each head
    const int quant_block_size,                // quantization block size
    // 说明：一个 block 包含多少个 token 的 kv_cache
    const int cache_block_size,                // cache block size
    const int cache_stride,  // stride for each token in kv_cache

    const bool use_ue8m0  // use ue8m0 scale format
) {
  constexpr int VEC_SIZE = 4;
  const int64_t token_idx = blockIdx.x;
  // 说明：blockIdx.y 表示一个 token 中的第几个 block；blockDim.y * blockDim.x 表示一个 block 的全部线程数；
  // blockIdx.y * blockDim.y * blockDim.x 表示相同 token 前面几个 block 负责处理的维度数；
  // blockDim.x = 32, blockDim.y = VEC_SIZE = 4，threadIdx.y * blockDim.x + threadIdx.x 表示当前线程负责处理的维度偏移量，
  // 分配顺序为（threadIdx.y, threadIdx.x），即外层为 y，内层为 x；
  // 说明：每个线程负责处理 VEC_SIZE 个连续的维度，所以最后乘以 VEC_SIZE
  const int64_t head_dim_idx = (blockIdx.y * blockDim.y * blockDim.x +
                                threadIdx.y * blockDim.x + threadIdx.x) *
                               VEC_SIZE;
  const int64_t slot_idx = slot_mapping[token_idx];
  const int64_t block_idx = slot_idx / cache_block_size;
  const int64_t block_offset = slot_idx % cache_block_size;

  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0 || (head_dim_idx >= head_dim)) {
    return;
  }

  // 说明：float2 对应两个 float32 元素，占用 8 字节
  // token_idx * head_dim + head_dim_idx 表示当前线程负责处理的第一个维度在整个 k 张量中的偏移量；
  // 理解：每个线程处理 VEC_SIZE 个值，同时 8 个字节包含 VEC_SIZE 个元素？通过下面方式得到当前线程对应的 float2 值？
  float2 k_val = (reinterpret_cast<const float2*>(
      k))[(token_idx * head_dim + head_dim_idx) / VEC_SIZE];
  // 说明：将地址转换为 scalar_t*，后续按元素真实类型访问
  scalar_t* k_val_ptr = reinterpret_cast<scalar_t*>(&k_val);
  // 说明：计算线程内 VEC_SIZE 个元素的 AbsMax
  float amax = 0.0f;
  for (int i = 0; i < VEC_SIZE; i++) {
    amax = fmaxf(amax, fabsf(float(k_val_ptr[i])));
  }

  // 说明：使用 warp shuffle 指令，计算整个 warp 内的最大值
  // Reduced amax
  for (int mask = 16; mask > 0; mask /= 2) {
#ifdef USE_ROCM
    amax = fmaxf(amax, __shfl_xor_sync(uint64_t(-1), amax, mask));
#else
    amax = fmaxf(amax, __shfl_xor_sync(unsigned(-1), amax, mask));
#endif
  }

  float scale = fmaxf(amax, 1e-4) / kFp8ScaleDivisor;

  if (use_ue8m0) {
    // 说明：UE8M0 格式要求 scale 必须是 2 的幂次方 [2^(-127), 2^128]
    scale = exp2f(ceilf(log2f(scale)));
  }

  const int64_t dst_offset = block_idx * cache_block_size * cache_stride +
                             block_offset * head_dim + head_dim_idx;
  for (int i = 0; i < VEC_SIZE; i++) {
    kv_cache[dst_offset + i] =
        fp8::scaled_convert<cache_t, scalar_t, kv_dt>(k_val_ptr[i], scale);
  }
  if (threadIdx.x == 0) {
    // 说明：存储 scale 值，位置在 kv_cache 的后面部分
    const int64_t dst_scale_idx =
        block_idx * cache_block_size * cache_stride +
        // 说明：cache_block_size * head_dim 表示 block 内所有 token 的 quantized k 值存储完之后的位置
        cache_block_size * head_dim +
        // 说明：block_offset * head_dim + head_dim_idx 表示当前线程负责处理的第一个维度在 block 内的位置，
        // 除以 quant_block_size 是因为每 quant_block_size 个维度共用一个 scale 值，
        // 乘以 4 是因为每个 scale 值占用 4 字节（float32）
        (block_offset * head_dim + head_dim_idx) * 4 / quant_block_size;
    // 说明：将 kv_cache 转换为 float*，然后根据计算的索引存储 scale 值
    reinterpret_cast<float*>(kv_cache)[dst_scale_idx / 4] = scale;
  }
}

// 已阅
// 说明：从 kv_cache 中将量化的 k 值和 scale 值提取出来，存储到 dst_k 和 dst_scale 中
// 说明：一个 block 处理 BLOCK_Y_SIZE 个 token，每个 token 有 8 * VEC_SIZE 个维度；
// 一个 block 包含 8 * BLOCK_Y_SIZE 个线程，每个线程处理 VEC_SIZE 个连续的维度
template <int BLOCK_Y_SIZE>
__global__ void cp_gather_indexer_k_quant_cache_kernel(
    const char* __restrict__ kv_cache,  // [num_blocks, block_size,
                                        // cache_stride]
    char* __restrict__ dst_k,           // [num_tokens, head_dim]
    char* __restrict__ dst_scale,  // [num_tokens, head_dim / quant_block_size *
                                   // 4]
    const int* __restrict__ block_table,  // [batch_size, num_blocks]
    const int* __restrict__ cu_seq_lens,  // [batch_size + 1]
    const int batch_size,                 // batch size
    // 说明：dst_k.stride(0)
    const int64_t token_stride,           // stride for each token in dst_k
    const int64_t head_dim,               // dimension of each head
    const int64_t block_stride,           // stride for each block in kv_cache
    const int64_t cache_token_stride,     // stride for each token in kv_cache
    const int64_t cache_block_size,  // num_tokens for each block in kv_cache
    const int num_blocks,            // number of blocks
    const int num_tokens,            // number of tokens
    const int quant_block_size       // quantization block size
) {
  // 说明：一个线程处理一个 token 的 VEC_SIZE 个连续的维度，每个维度占用 1 字节
  constexpr int VEC_SIZE = sizeof(float4) / sizeof(char);
  // blockIdx.x 表示第几组 token，每组 BLOCK_Y_SIZE 个 token；
  // blockDim.y = BLOCK_Y_SIZE，对应 token 维度，threadIdx.y 维度表示当前 block 内第几个 token
  const int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
  // 说明：blockIdx.y 表示 head_dim 的第几组维度，每组 8（blockDim.x） * 16 (VEC_SIZE) = 128 个维度；
  // 每个线程处理 VEC_SIZE 个维度，8 * VEC_SIZE 分为 8 组，每个线程对应一组，threadIdx.x 表示在 8 组中的第几个
  const int head_idx = (blockIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;
  // 说明：__shared__ 变量在同一个 block 内的所有线程共享
  // Find batch index within a block
  __shared__ int batch_idx[BLOCK_Y_SIZE];
  // 说明：计算当前线程负责处理的 token 在整个 batch 中属于哪个序列
  for (int iter = 0; iter < cuda_utils::ceil_div(batch_size, int(blockDim.x));
       iter++) {
    int tid = iter * blockDim.x + threadIdx.x;
    if (tid < batch_size) {
      const int seq_start = cu_seq_lens[tid];
      const int seq_end = cu_seq_lens[tid + 1];
      if (token_idx >= seq_start && token_idx < seq_end) {
        batch_idx[threadIdx.y] = tid;
      }
    }
  }

#ifndef USE_ROCM
  __syncwarp();
#endif

  if (head_idx >= head_dim || token_idx >= num_tokens) {
    return;
  }
  // 说明：计算当前 token 在其所属序列中的位置
  const int inbatch_seq_idx = token_idx - cu_seq_lens[batch_idx[threadIdx.y]];
  const int block_idx = block_table[batch_idx[threadIdx.y] * num_blocks +
                                    inbatch_seq_idx / cache_block_size];
  const int64_t src_block_offset = block_idx * block_stride;
  // 说明：block 内前面 token 的完整维度 + 当前 token 的 head_idx 偏移量
  const int64_t cache_inblock_offset =
      (inbatch_seq_idx % cache_block_size) * head_dim + head_idx;
  const int64_t src_inblock_offset = src_block_offset + cache_inblock_offset;
  const int64_t dst_inblock_offset = token_idx * token_stride + head_idx;

  // 说明：一次赋值 16 个字节
  reinterpret_cast<float4*>(dst_k)[dst_inblock_offset / VEC_SIZE] =
      reinterpret_cast<const float4*>(kv_cache)[src_inblock_offset / VEC_SIZE];
  ;
  if (threadIdx.x == 0) {
    const int64_t src_scale_offset =
        // 说明：block 初始位置的 offset + 完整量化值部分的大小（fp8 量化）+ 量化值组数 * 量化值大小（float，4 字节）
        src_block_offset + cache_block_size * head_dim +
        cache_inblock_offset * 4 / quant_block_size;
    reinterpret_cast<float*>(dst_scale)[dst_inblock_offset / quant_block_size] =
        reinterpret_cast<const float*>(kv_cache)[src_scale_offset / 4];
  }
}

}  // namespace vllm

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)               \
  vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>             \
      <<<grid, block, 0, stream>>>(                                   \
          reinterpret_cast<KV_T*>(key.data_ptr()),                    \
          reinterpret_cast<KV_T*>(value.data_ptr()),                  \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),           \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),         \
          slot_mapping.data_ptr<int64_t>(), key_stride, value_stride, \
          num_heads, head_size, block_size, x,                        \
          reinterpret_cast<const float*>(k_scale.data_ptr()),         \
          reinterpret_cast<const float*>(v_scale.data_ptr()));

void reshape_and_cache(
    torch::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);
  int head_div_x = head_size / x;

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_div_x, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE);
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)             \
  vllm::reshape_and_cache_flash_kernel<KV_T, CACHE_T, KV_DTYPE>           \
      <<<grid, block, 0, stream>>>(                                       \
          reinterpret_cast<KV_T*>(key.data_ptr()),                        \
          reinterpret_cast<KV_T*>(value.data_ptr()),                      \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),               \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),             \
          slot_mapping.data_ptr<int64_t>(), block_stride, page_stride,    \
          head_stride, key_stride, value_stride, num_heads, head_size,    \
          block_size, reinterpret_cast<const float*>(k_scale.data_ptr()), \
          reinterpret_cast<const float*>(v_scale.data_ptr()),             \
          kv_scale_stride);

void reshape_and_cache_flash(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,    // [1] or [num_heads]
    torch::Tensor& v_scale) {  // [1] or [num_heads]
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  TORCH_CHECK(k_scale.sizes() == v_scale.sizes(),
              "k_scale and v_scale must have the same shape");
  TORCH_CHECK(k_scale.numel() == 1 || k_scale.numel() == num_heads,
              "k_scale and v_scale must be of shape [1] or [num_heads]");
  int kv_scale_stride = (k_scale.numel() > 1) ? 1 : 0;

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE_FLASH);
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, KV_DTYPE)              \
  vllm::concat_and_cache_mla_kernel<KV_T, CACHE_T, KV_DTYPE>            \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                     \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                     \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,   \
          reinterpret_cast<const float*>(scale.data_ptr()));

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_DS_MLA(KV_T, CACHE_T, KV_DTYPE)           \
  vllm::concat_and_cache_ds_mla_kernel<KV_T, CACHE_T, KV_DTYPE>         \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                     \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                     \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,   \
          reinterpret_cast<const float*>(scale.data_ptr()));

void concat_and_cache_mla(
    torch::Tensor& kv_c,          // [num_tokens, kv_lora_rank]
    torch::Tensor& k_pe,          // [num_tokens, pe_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, (kv_lora_rank +
                                  // pe_dim)]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(1);
  int pe_dim = k_pe.size(1);
  int block_size = kv_cache.size(1);

  if (kv_cache_dtype == "fp8_ds_mla") {
    TORCH_CHECK(kv_lora_rank == 512, "kv_lora_rank must be 512 for fp8_ds_mla");
    TORCH_CHECK(pe_dim == 64, "pe_dim must be 64 for fp8_ds_mla");
    TORCH_CHECK(kv_cache.size(2) == 656 / kv_cache.itemsize(),
                "kv_cache.size(2) must be 656 bytes for fp8_ds_mla");
    TORCH_CHECK(kv_c.itemsize() == 2,
                "kv_c.itemsize() must be 2 for fp8_ds_mla");
    TORCH_CHECK(k_pe.itemsize() == 2,
                "k_pe.itemsize() must be 2 for fp8_ds_mla");
  } else {
    TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);
  }

  int kv_c_stride = kv_c.stride(0);
  int k_pe_stride = k_pe.stride(0);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_c));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (kv_cache_dtype == "fp8_ds_mla") {
    dim3 grid(num_tokens);
    // For the NoPE part, each tile of 128 elements is handled by half of one
    // warp (16 threads). There are 4 total tiles, so 2 warps (64 threads).
    // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
    // The RoPE part (last 64 elements) is handled by another 1 warp (32
    // threads). So in total, we use 3 warps (96 threads) per block.
    dim3 block(96);
    DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                               CALL_CONCAT_AND_CACHE_DS_MLA);
  } else {
    dim3 grid(num_tokens);
    dim3 block(std::min(kv_lora_rank, 512));
    DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                               CALL_CONCAT_AND_CACHE_MLA);
  }
}

namespace vllm {

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__global__ void convert_fp8_kernel(const Tin* __restrict__ src_cache,
                                   Tout* __restrict__ dst_cache,
                                   const float scale,
                                   const int64_t block_stride) {
  const int64_t block_idx = blockIdx.x;
  for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
    int64_t idx = block_idx * block_stride + i;
    dst_cache[idx] =
        fp8::scaled_convert<Tout, Tin, kv_dt>(src_cache[idx], scale);
  }
}

}  // namespace vllm

#define CALL_CONVERT_FP8(Tout, Tin, KV_DTYPE)                                \
  vllm::convert_fp8_kernel<Tout, Tin, KV_DTYPE><<<grid, block, 0, stream>>>( \
      reinterpret_cast<Tin*>(src_cache.data_ptr()),                          \
      reinterpret_cast<Tout*>(dst_cache.data_ptr()), scale, block_stride);

// Only for testing.
void convert_fp8(torch::Tensor& dst_cache, torch::Tensor& src_cache,
                 const double scale, const std::string& kv_cache_dtype) {
  torch::Device src_device = src_cache.device();
  torch::Device dst_device = dst_cache.device();
  TORCH_CHECK(src_device.is_cuda(), "src must be on a GPU")
  TORCH_CHECK(dst_device.is_cuda(), "dst must be on a GPU")
  TORCH_CHECK(src_device.index() == dst_device.index(),
              "src and dst must be on the same GPU");
  at::cuda::OptionalCUDAGuard device_guard(src_device);

  int64_t num_blocks = src_cache.size(0);
  int64_t block_stride = src_cache.stride(0);

  dim3 grid(num_blocks);
  dim3 block(std::min(block_stride, int64_t(512)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (kv_cache_dtype == "auto") {
    if (src_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(uint8_t, float, vllm::Fp8KVCacheDataType::kAuto);
    } else if (src_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint8_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(uint8_t, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(float, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    }
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    if (src_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(uint8_t, float, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (src_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint8_t, uint16_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(uint8_t, __nv_bfloat16,
                       vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(__nv_bfloat16, uint8_t,
                       vllm::Fp8KVCacheDataType::kFp8E4M3);
    }
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", kv_cache_dtype);
  }
}

namespace vllm {

// grid is launched with dimensions (batch, num_splits)
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          int ENTRY_SIZE, int CTA_SIZE>
__global__ void gather_and_maybe_dequant_cache(
    const cache_t* __restrict__ src_cache,     // [NUM_BLOCKS, BLOCK_SIZE,
                                               // ENTRIES...]
    scalar_t* __restrict__ dst,                // [TOT_TOKENS, ENTRIES...]
    const int32_t* __restrict__ block_table,   // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,   // [BATCH+1]
    const int32_t* __restrict__ token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNK]
    const int32_t num_tokens, const int32_t block_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const float* __restrict__ scale,
    const int32_t* __restrict__ seq_starts) {  // Optional: starting offsets per
                                               // batch
  constexpr int vec_size = sizeof(float4) / sizeof(scalar_t);
  using ltype = vllm::vec_n_t<cache_t, vec_size>;
  using stype = vllm::vec_n_t<scalar_t, vec_size>;
  // We are adding this for code readability which will be optimized out when
  // build in release.
  assert(CTA_SIZE == blockDim.x);

#pragma unroll
  for (int token_id = blockIdx.x; token_id < num_tokens;
       token_id += gridDim.x) {
    int64_t batch_id = token_to_seq[token_id];
    int64_t batch_start = cu_seq_lens[batch_id];
    int64_t batch_end = cu_seq_lens[batch_id + 1];
    int32_t batch_offset = token_id - batch_start;

    if (token_id >= batch_end) return;
    int32_t offset = 0;
    if (seq_starts != nullptr) {
      offset = seq_starts[batch_id];
    }
    batch_offset += offset;
    int32_t block_table_id = batch_offset / block_size;
    int32_t slot_id = batch_offset % block_size;
    int32_t block_table_offset = batch_id * block_table_stride + block_table_id;
    int32_t block_id = block_table[block_table_offset];
    int64_t cache_offset =
        block_id * cache_block_stride + slot_id * cache_entry_stride;
    constexpr int32_t vec_iter_cnt = ENTRY_SIZE / vec_size;
    scalar_t* dst_ = dst + token_id * dst_entry_stride;
    cache_t* src_ = const_cast<cache_t*>(src_cache) + cache_offset;

#pragma unroll
    for (int idx = threadIdx.x; idx < vec_iter_cnt; idx += CTA_SIZE) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        reinterpret_cast<stype*>(dst_)[idx] =
            static_cast<stype>(reinterpret_cast<ltype*>(src_)[idx]);
      } else {
        ltype loaded_val = reinterpret_cast<ltype*>(src_)[idx];
        stype store_val;
#pragma unroll
        for (int j = 0; j < vec_size; ++j) {
          store_val.val[j] = fp8::scaled_convert<scalar_t, cache_t, kv_dt>(
              loaded_val.val[j], *scale);
        }
        reinterpret_cast<stype*>(dst_)[idx] = store_val;
      }
    }
    // process tail
    constexpr int32_t tail_cnt = ENTRY_SIZE % vec_size;
    dst_ = dst_ + ENTRY_SIZE - tail_cnt;
    src_ = src_ + ENTRY_SIZE - tail_cnt;
#pragma unroll
    for (int idx = threadIdx.x; idx < tail_cnt; idx += CTA_SIZE) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        dst_[idx] = static_cast<scalar_t>(src_[idx]);
      } else {
        dst_[idx] =
            fp8::scaled_convert<scalar_t, cache_t, kv_dt>(src_[idx], *scale);
      }
    }
  }
}

}  // namespace vllm

// Macro to dispatch the kernel based on the data type.
// SCALAR_T is the data type of the destination tensor.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE)                        \
  vllm::gather_and_maybe_dequant_cache<SCALAR_T, CACHE_T, KV_DTYPE, 576,      \
                                       thread_block_size>                     \
      <<<grid, block, 0, stream>>>(                                           \
          reinterpret_cast<CACHE_T*>(src_cache.data_ptr()),                   \
          reinterpret_cast<SCALAR_T*>(dst.data_ptr()),                        \
          block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(),   \
          token_to_seq.data_ptr<int32_t>(), num_tokens, block_size,           \
          block_table_stride, cache_block_stride, cache_entry_stride,         \
          dst_entry_stride, reinterpret_cast<const float*>(scale.data_ptr()), \
          seq_starts_ptr);

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - token_to_seq contains the back mapping from token_id to batch_id
//  - Optionally, seq_starts (if provided) offsets the starting block index by
//  (seq_starts[bid] / page_size)
void gather_and_maybe_dequant_cache(
    torch::Tensor const& src_cache,     // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,           // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,   // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,   // [BATCH+1]
    torch::Tensor const& token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNKS]
    int64_t num_tokens, const std::string& kv_cache_dtype,
    torch::Tensor const& scale,
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t head_dim = dst.size(-1);

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt32,
              "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    TORCH_CHECK(seq_starts.value().dtype() == torch::kInt32,
                "seq_starts must be int32");
  }
  TORCH_CHECK(head_dim == 576,
              "gather_and_maybe_dequant_cache only support the head_dim to 576 "
              "for better performance")

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == cu_seq_lens.device(),
              "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  constexpr int32_t thread_block_size = 64;
  dim3 grid(num_tokens);
  dim3 block(thread_block_size);

  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  DISPATCH_BY_KV_CACHE_DTYPE(dst.dtype(), kv_cache_dtype, CALL_GATHER_CACHE);
}

namespace vllm {

// Gather and upconvert FP8 KV cache tokens to BF16 workspace
// Similar to cp_gather_cache but specifically for FP8->BF16 conversion
__global__ void cp_gather_and_upconvert_fp8_kv_cache(
    const uint8_t* __restrict__ src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, 656]
    __nv_bfloat16* __restrict__ dst,          // [TOT_TOKENS, 576]
    const int32_t* __restrict__ block_table,  // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ seq_lens,     // [BATCH]
    const int32_t* __restrict__ workspace_starts,  // [BATCH]
    const int32_t block_size, const int32_t head_dim,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride) {
  const int64_t bid = blockIdx.x;  // Batch ID
  const int32_t num_splits = gridDim.y;
  const int32_t split = blockIdx.y;
  const int32_t seq_start = workspace_starts[bid];
  const int32_t seq_len = seq_lens[bid];
  const int32_t tot_slots = seq_len;
  const int32_t split_slots = cuda_utils::ceil_div(tot_slots, num_splits);

  const int32_t split_start = split * split_slots;
  const int32_t split_end = min((split + 1) * split_slots, tot_slots);

  const bool is_active_split = (split_start < tot_slots);

  if (!is_active_split) return;

  // Adjust the pointer for the block_table for this batch
  const int32_t batch_offset = bid * block_table_stride;
  int32_t offset = split_start;
  int32_t offset_div = offset / block_size;
  offset = offset % block_size;
  const int32_t* batch_block_table = block_table + batch_offset;

  // Adjust dst pointer based on the cumulative sequence lengths
  dst += seq_start * dst_entry_stride;

  const int tid = threadIdx.x;

  // Process each token in this split
  for (int pid = split_start; pid < split_end; ++pid) {
    auto block_id = batch_block_table[offset_div];
    const uint8_t* token_ptr =
        src_cache + block_id * cache_block_stride + offset * cache_entry_stride;
    __nv_bfloat16* dst_ptr = dst + pid * dst_entry_stride;

    // FP8 format: 512 bytes fp8 + 16 bytes scales + 128 bytes rope (64 bf16)
    const uint8_t* no_pe_ptr = token_ptr;
    const float* scales_ptr = reinterpret_cast<const float*>(token_ptr + 512);
    const __nv_bfloat16* rope_ptr =
        reinterpret_cast<const __nv_bfloat16*>(token_ptr + 512 + 16);

    // Parallelize fp8 dequant (512 elements) and rope copy (64 elements)
    if (tid < 512) {
      // FP8 dequantization
      const int tile = tid >> 7;  // each tile is 128 elements
      const float scale = scales_ptr[tile];
      const uint8_t val = no_pe_ptr[tid];
      dst_ptr[tid] =
          fp8::scaled_convert<__nv_bfloat16, uint8_t,
                              vllm::Fp8KVCacheDataType::kFp8E4M3>(val, scale);
    } else if (tid < 576) {
      // Rope copy (64 bf16 elements)
      const int rope_idx = tid - 512;
      dst_ptr[512 + rope_idx] = rope_ptr[rope_idx];
    }

    // Move to next token
    offset += 1;
    if (offset == block_size) {
      offset_div += 1;
      offset = 0;
    }
  }
}

template <typename scalar_t>
// Note(hc): The cp_gather_cache allows seq_starts to no longer be divisible by
// block_size.
__global__ void cp_gather_cache(
    const scalar_t* __restrict__ src_cache,   // [NUM_BLOCKS, BLOCK_SIZE,
                                              // ENTRY_SIZE]
    scalar_t* __restrict__ dst,               // [TOT_TOKENS, ENTRY_SIZE]
    const int32_t* __restrict__ block_table,  // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,  // [BATCH+1]
    const int32_t block_size, const int32_t entry_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const int32_t* __restrict__ seq_starts  // Optional: starting offsets per
                                            // batch
) {
  const int64_t bid = blockIdx.x;  // Batch ID
  const int32_t num_splits = gridDim.y;
  const int32_t split = blockIdx.y;
  const int32_t seq_start = cu_seq_lens[bid];
  const int32_t seq_end = cu_seq_lens[bid + 1];
  const int32_t seq_len = seq_end - seq_start;
  const int32_t tot_slots = seq_len;
  const int32_t split_slots = cuda_utils::ceil_div(tot_slots, num_splits);

  const int32_t split_start = split * split_slots;
  const int32_t split_end = min((split + 1) * split_slots, tot_slots);

  const bool is_active_split = (split_start < tot_slots);

  if (!is_active_split) return;

  // Adjust the pointer for the block_table for this batch.
  // If seq_starts is provided, compute an offset based on it
  const int32_t batch_offset = bid * block_table_stride;
  int32_t offset = split_start;
  if (seq_starts != nullptr) {
    offset += seq_starts[bid];
  }
  int32_t offset_div = offset / block_size;
  offset = offset % block_size;
  const int32_t* batch_block_table = block_table + batch_offset;

  // Adjust dst pointer based on the cumulative sequence lengths.
  dst += seq_start * dst_entry_stride;

  auto copy_entry = [&](const scalar_t* __restrict__ _src,
                        scalar_t* __restrict__ _dst) {
    for (int i = threadIdx.x; i < entry_size; i += blockDim.x)
      _dst[i] = _src[i];
  };

  for (int pid = split_start; pid < split_end; ++pid) {
    auto block_id = batch_block_table[offset_div];
    auto block_start_ptr = src_cache + block_id * cache_block_stride;
    auto block_dst_ptr = dst + pid * dst_entry_stride;
    copy_entry(block_start_ptr + offset * cache_entry_stride, block_dst_ptr);
    offset += 1;
    // bump to next block
    if (offset == block_size) {
      offset_div += 1;
      offset = 0;
    }
  }
}
}  // namespace vllm

// Macro to dispatch the kernel based on the data type.
#define CALL_CP_GATHER_CACHE(CPY_DTYPE)                                 \
  vllm::cp_gather_cache<CPY_DTYPE><<<grid, block, 0, stream>>>(         \
      reinterpret_cast<CPY_DTYPE*>(src_cache.data_ptr()),               \
      reinterpret_cast<CPY_DTYPE*>(dst.data_ptr()),                     \
      block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(), \
      block_size, entry_size, block_table_stride, cache_block_stride,   \
      cache_entry_stride, dst_entry_stride, seq_starts_ptr);

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - Optionally, seq_starts (if provided) offsets the starting slot index by
//  seq_starts[bid]
void cp_gather_cache(
    torch::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,          // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,  // [BATCH+1]
    int64_t batch_size,
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t entry_size = src_cache.flatten(2, -1).size(2);

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt32,
              "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    TORCH_CHECK(seq_starts.value().dtype() == torch::kInt32,
                "seq_starts must be int32");
  }

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == cu_seq_lens.device(),
              "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  // Decide on the number of splits based on the batch size.
  int num_splits = batch_size > 128 ? 2 : batch_size > 64 ? 4 : 16;
  dim3 grid(batch_size, num_splits);
  dim3 block(1024);

  TORCH_CHECK(src_cache.dtype() == dst.dtype(),
              "src_cache and dst must have the same dtype");

  const int dtype_bits = src_cache.element_size() * 8;
  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  if (dtype_bits == 32) {
    CALL_CP_GATHER_CACHE(uint32_t);
  } else if (dtype_bits == 16) {
    CALL_CP_GATHER_CACHE(uint16_t);
  } else if (dtype_bits == 8) {
    CALL_CP_GATHER_CACHE(uint8_t);
  } else {
    TORCH_CHECK(false, "Unsupported data type width: ", dtype_bits);
  }
}

void cp_gather_and_upconvert_fp8_kv_cache(
    torch::Tensor const& src_cache,         // [NUM_BLOCKS, BLOCK_SIZE, 656]
    torch::Tensor const& dst,               // [TOT_TOKENS, 576]
    torch::Tensor const& block_table,       // [BATCH, BLOCK_INDICES]
    torch::Tensor const& seq_lens,          // [BATCH]
    torch::Tensor const& workspace_starts,  // [BATCH]
    int64_t batch_size) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t head_dim = dst.size(1);

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(seq_lens.dtype() == torch::kInt32, "seq_lens must be int32");
  TORCH_CHECK(workspace_starts.dtype() == torch::kInt32,
              "workspace_starts must be int32");

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == seq_lens.device(),
              "src_cache and seq_lens must be on the same device");
  TORCH_CHECK(src_cache.device() == workspace_starts.device(),
              "src_cache and workspace_starts must be on the same device");
  auto dtype = src_cache.scalar_type();
  TORCH_CHECK(
      dtype == at::ScalarType::Byte ||               // uint8
          dtype == at::ScalarType::Float8_e4m3fn ||  // fp8 e4m3
          dtype == at::ScalarType::Float8_e5m2,      // fp8 e5m2
      "src_cache must be uint8, float8_e4m3fn, or float8_e5m2, but got ",
      src_cache.dtype());
  TORCH_CHECK(dst.dtype() == torch::kBFloat16, "dst must be bfloat16");
  TORCH_CHECK(head_dim == 576, "head_dim must be 576 for MLA");

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  const uint8_t* src_ptr = nullptr;
  if (dtype == at::ScalarType::Byte) {
    src_ptr = src_cache.data_ptr<uint8_t>();
  } else {
    // float8_e4m3fn or float8_e5m2
    src_ptr = reinterpret_cast<const uint8_t*>(src_cache.data_ptr());
  }

  // Decide on the number of splits based on the batch size
  int num_splits = batch_size > 128 ? 2 : batch_size > 64 ? 4 : 16;
  dim3 grid(batch_size, num_splits);
  dim3 block(576);

  vllm::cp_gather_and_upconvert_fp8_kv_cache<<<grid, block, 0, stream>>>(
      src_ptr, reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
      block_table.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
      workspace_starts.data_ptr<int32_t>(), block_size, head_dim,
      block_table_stride, cache_block_stride, cache_entry_stride,
      dst_entry_stride);
}

// 已阅
// Macro to dispatch the kernel based on the data type.
#define CALL_INDEXER_K_QUANT_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)         \
  vllm::indexer_k_quant_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>       \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(k.data_ptr()),                        \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), head_dim, quant_block_size, \
          cache_block_size, cache_stride, use_ue8m0);

// 已阅
void indexer_k_quant_and_cache(
    torch::Tensor& k,             // [num_tokens, head_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, cache_stride]
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t quant_block_size,     // quantization block size
    const std::string& scale_fmt) {
  int num_tokens = k.size(0);
  int head_dim = k.size(1);
  // 说明：一个 block 包含多少个 token 的 kv_cache
  int cache_block_size = kv_cache.size(1);
  int cache_stride = kv_cache.size(2);
  bool use_ue8m0 = scale_fmt == "ue8m0";

  TORCH_CHECK(k.device() == kv_cache.device(),
              "k and kv_cache must be on the same device");
  TORCH_CHECK(k.device() == slot_mapping.device(),
              "k and slot_mapping must be on the same device");
  TORCH_CHECK(head_dim % quant_block_size == 0,
              "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 4;
  // 说明：一个 block 处理一个 token 中的 quant_block_size * vec_size 个维度
  dim3 grid(num_tokens, (head_dim + quant_block_size * vec_size - 1) /
                            (quant_block_size * vec_size));
  dim3 block(32, vec_size);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // 说明：kv_cache 的 dtype 是 fp8_e4m3，所以 cache_t 是 uint8_t；
  // 实际存储的数据是 fp8 格式的，只是用 uint8_t 来存储，参考 scaled_vec_conversion<uint8_t, xxx>
  DISPATCH_BY_KV_CACHE_DTYPE(k.dtype(), "fp8_e4m3",
                             CALL_INDEXER_K_QUANT_AND_CACHE);
}

// 说明：gridDim.x = (num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE，一个 block 处理 BLOCK_Y_SIZE 个 token；
// gridDim.y = (head_dim + 8 * vec_size - 1) / (8 * vec_size)，一个 block 处理 head_dim 中的 8 * vec_size 个维度，实际值为 8 * 16 = 128；
// blockDim.x = 8，blockDim.y = BLOCK_Y_SIZE；
// Macro to dispatch the kernel based on the data amount.
#define CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(BLOCK_Y_SIZE)                  \
  vllm::cp_gather_indexer_k_quant_cache_kernel<BLOCK_Y_SIZE>                \
      <<<dim3((num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE,               \
              (head_dim + 8 * vec_size - 1) / (8 * vec_size)),              \
         dim3(8, BLOCK_Y_SIZE), 0, stream>>>(                               \
          reinterpret_cast<char*>(kv_cache.data_ptr()),                     \
          reinterpret_cast<char*>(dst_k.data_ptr()),                        \
          reinterpret_cast<char*>(dst_scale.data_ptr()),                    \
          block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(), \
          batch_size, dst_k.stride(0), dst_k.size(1), kv_cache.stride(0),   \
          kv_cache.stride(1), kv_cache.size(1), block_table.size(1),        \
          num_tokens, quant_block_size);

void cp_gather_indexer_k_quant_cache(
    const torch::Tensor& kv_cache,  // [num_blocks, block_size, cache_stride]
    torch::Tensor& dst_k,           // [num_tokens, head_dim]
    // 说明：quant_block_size 个维度对应一个 scale，每个 scale 占 4 个字节
    torch::Tensor& dst_scale,  // [num_tokens, head_dim / quant_block_size * 4]
    const torch::Tensor& block_table,  // [batch_size, num_blocks]
    const torch::Tensor& cu_seq_lens   // [batch_size + 1]
) {
  int batch_size = block_table.size(0);
  int num_tokens = dst_k.size(0);
  int head_dim = dst_k.size(1);
  int quant_block_size = head_dim * 4 / dst_scale.size(1);

  TORCH_CHECK(kv_cache.device() == dst_k.device(),
              "kv_cache and dst_k must be on the same device");
  TORCH_CHECK(kv_cache.device() == dst_scale.device(),
              "kv_cache and dst_scale must be on the same device");
  TORCH_CHECK(kv_cache.device() == block_table.device(),
              "kv_cache and block_table must be on the same device");
  TORCH_CHECK(kv_cache.device() == cu_seq_lens.device(),
              "kv_cache and cu_seq_lens must be on the same device");
  TORCH_CHECK(head_dim % quant_block_size == 0,
              "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 16;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (num_tokens < 32) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(1);
  } else if (num_tokens < 64) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(2);
  } else if (num_tokens < 128) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(4);
  } else if (num_tokens < 256) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(8);
  } else if (num_tokens < 512) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(16);
  } else {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(32);
  }
}
