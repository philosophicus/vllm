#pragma once

#include "../../../../attention/attention_dtypes.h"
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <type_traits>

namespace vllm {
#ifndef USE_ROCM

namespace fp8 {
  #ifdef ENABLE_FP8

template <typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(
    const Tin& x, const __nv_fp8_interpretation_t fp8_type = __NV_E4M3) {
  return x;
}

// float -> c10::Float8_e4m3fn
template <>
__inline__ __device__ c10::Float8_e4m3fn
vec_conversion<c10::Float8_e4m3fn, float>(
    const float& a, const __nv_fp8_interpretation_t fp8_type) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return static_cast<c10::Float8_e4m3fn>(a);
    #else
  return c10::Float8_e4m3fn(__nv_cvt_float_to_fp8(a, __NV_SATFINITE, fp8_type),
                            c10::Float8_e4m3fn::from_bits());
    #endif
}

    #if 0  // Disable the following code to reduce the binary size.
// fp8 -> half
template <>
__inline__ __device__ uint16_t vec_conversion<uint16_t, uint8_t>(
    const uint8_t &a, const __nv_fp8_interpretation_t fp8_type) {
  __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  return res.x;
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t vec_conversion<uint32_t, uint16_t>(
    const uint16_t &a, const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint16_t u16[2];
    uint32_t u32;
  } tmp;
  __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, fp8_type);
  tmp.u16[0] = res.x;
  tmp.u16[1] = res.y;
  return tmp.u32;
}

// fp8x4 -> half2x2
template <>
__inline__ __device__ uint2 vec_conversion<uint2, uint32_t>(
    const uint32_t &a, const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] = vec_conversion<uint32_t, uint16_t>((uint16_t)a, fp8_type);
  tmp.u32[1] =
      vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U), fp8_type);
  return tmp.u32x2;
}

// fp8x8 -> half2x4
template <>
__inline__ __device__ uint4 vec_conversion<uint4, uint2>(
    const uint2 &a, const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = vec_conversion<uint2, uint32_t>(a.x, fp8_type);
  tmp.u64[1] = vec_conversion<uint2, uint32_t>(a.y, fp8_type);
  return tmp.u64x2;
}

// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16 vec_conversion<__nv_bfloat16, uint8_t>(
    const uint8_t &a, const __nv_fp8_interpretation_t fp8_type) {
  // Note there is no direct convert function from fp8 to bf16.
  // fp8 -> half
  __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  // half -> float -> bf16
  float tmp = half_to_float(res.x);
  return __float2bfloat16(tmp);
}

// fp8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, uint16_t>(
    const uint16_t &a, const __nv_fp8_interpretation_t fp8_type) {
  __nv_bfloat162 res;
  res.x = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a, fp8_type);
  res.y = vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U), fp8_type);
  return res;
}

// fp8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, uint32_t>(
    const uint32_t &a, const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t res;
  res.x = vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a, fp8_type);
  res.y =
      vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U), fp8_type);
  return res;
}

// fp8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, uint2>(
    const uint2 &a, const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t tmp1, tmp2;
  tmp1 = vec_conversion<bf16_4_t, uint32_t>(a.x, fp8_type);
  tmp2 = vec_conversion<bf16_4_t, uint32_t>(a.y, fp8_type);
  bf16_8_t res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// fp8 -> float
template <>
__inline__ __device__ float
vec_conversion<float, uint8_t>(const uint8_t &a,
                               const __nv_fp8_interpretation_t fp8_type) {
  // fp8 -> half
  uint16_t tmp = vec_conversion<uint16_t, uint8_t>(a, fp8_type);
  // half -> float
  return half_to_float(tmp);
}

// fp8x2 -> float2
template <>
__inline__ __device__ float2 vec_conversion<float2, uint16_t>(
    const uint16_t &a, const __nv_fp8_interpretation_t fp8_type) {
  // fp8x2 -> half2
  uint32_t tmp = vec_conversion<uint32_t, uint16_t>(a, fp8_type);
  // half2 -> float2
  return half2_to_float2(tmp);
}

// fp8x4 -> float4
template <>
__inline__ __device__ Float4_ vec_conversion<Float4_, uint32_t>(
    const uint32_t &a, const __nv_fp8_interpretation_t fp8_type) {
  Float4_ res;
  res.x = vec_conversion<float2, uint16_t>((uint16_t)a, fp8_type);
  res.y = vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U), fp8_type);
  return res;
}

// fp8x8 -> float8
template <>
__inline__ __device__ Float8_ vec_conversion<Float8_, uint2>(
    const uint2 &a, const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp1, tmp2;
  tmp1 = vec_conversion<Float4_, uint32_t>(a.x, fp8_type);
  tmp2 = vec_conversion<Float4_, uint32_t>(a.y, fp8_type);
  Float8_ res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// half -> fp8
template <>
__inline__ __device__ uint8_t vec_conversion<uint8_t, uint16_t>(
    const uint16_t &a, const __nv_fp8_interpretation_t fp8_type) {
  __half_raw tmp;
  tmp.x = a;
  __nv_fp8_storage_t res =
      __nv_cvt_halfraw_to_fp8(tmp, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}

// bf16 -> fp8
template <>
__inline__ __device__ uint8_t vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16 &a, const __nv_fp8_interpretation_t fp8_type) {
      #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
      #else
  __nv_fp8_storage_t res = __nv_cvt_bfloat16raw_to_fp8(
      __nv_bfloat16_raw(a), __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
      #endif
}

// float -> fp8
template <>
__inline__ __device__ uint8_t vec_conversion<uint8_t, float>(
    const float &a, const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(a, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}

// fp8x4 -> float4
template <>
__inline__ __device__ float4 vec_conversion<float4, uint32_t>(
    const uint32_t &a, const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp = vec_conversion<Float4_, uint32_t>(a, fp8_type);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}

template <>
__inline__ __device__ uint32_t vec_conversion<uint32_t, float2>(
    const float2 &a, const __nv_fp8_interpretation_t fp8_type) {
  union {
    half2 float16;
    uint32_t uint32;
  };

  float16 = __float22half2_rn(a);
  return uint32;
}

template <>
__inline__ __device__ uint2 vec_conversion<uint2, Float4_>(
    const Float4_ &a, const __nv_fp8_interpretation_t fp8_type) {
  uint2 b;
  float2 val;
  val.x = a.x.x;
  val.y = a.x.y;
  b.x = vec_conversion<uint32_t, float2>(val, fp8_type);

  val.x = a.y.x;
  val.y = a.y.y;
  b.y = vec_conversion<uint32_t, float2>(val, fp8_type);

  return b;
}

template <>
__inline__ __device__ float4 vec_conversion<float4, Float4_>(
    const Float4_ &a, const __nv_fp8_interpretation_t fp8_type) {
  float4 b;
  b.x = a.x.x;
  b.y = a.x.y;
  b.z = a.y.x;
  b.w = a.y.y;
  return b;
}

template <>
__inline__ __device__ uint4 vec_conversion<uint4, Float8_>(
    const Float8_ &a, const __nv_fp8_interpretation_t fp8_type) {
  uint4 b;
  b.x = vec_conversion<uint32_t, float2>(a.x, fp8_type);
  b.y = vec_conversion<uint32_t, float2>(a.y, fp8_type);
  b.z = vec_conversion<uint32_t, float2>(a.z, fp8_type);
  b.w = vec_conversion<uint32_t, float2>(a.w, fp8_type);
  return b;
}

template <>
__inline__ __device__ __nv_bfloat162 vec_conversion<__nv_bfloat162, float2>(
    const float2 &a, const __nv_fp8_interpretation_t fp8_type) {
  __nv_bfloat162 b;
  from_float(b, a);
  return b;
}

template <>
__inline__ __device__ bf16_4_t vec_conversion<bf16_4_t, Float4_>(
    const Float4_ &a, const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t b;
  from_float(b, a);
  return b;
}

template <>
__inline__ __device__ bf16_8_t vec_conversion<bf16_8_t, Float8_>(
    const Float8_ &a, const __nv_fp8_interpretation_t fp8_type) {
  bf16_8_t b;
  from_float(b, a);
  return b;
}
    #endif

/* Scaled and vectorized conversions, for data exchange between high and low
   precision domains Convention of the scale in API, e.g: FP8_data =
   Quantization( High_Precision_data / scale ) s.t. Quantize(HP / scale) => FP8
     Dequant(FP8) * scale =>  HP
 */

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(
    const Tin& x, const float scale, const __nv_fp8_interpretation_t fp8_type) {
  return x;
}

// fp8 -> half
template <>
__inline__ __device__ uint16_t scaled_vec_conversion<uint16_t, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __half_raw tmp = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  return float_to_half(half_to_float(tmp.x) * scale);
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t scaled_vec_conversion<uint32_t, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint16_t u16[2];
    uint32_t u32;
  } tmp;
  __half2_raw res = __nv_cvt_fp8x2_to_halfraw2(a, fp8_type);
  tmp.u16[0] = float_to_half(half_to_float(res.x) * scale);
  tmp.u16[1] = float_to_half(half_to_float(res.y) * scale);
  return tmp.u32;
}

// fp8x4 -> half2x2
template <>
__inline__ __device__ uint2 scaled_vec_conversion<uint2, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] =
      scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)a, scale, fp8_type);
  tmp.u32[1] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U),
                                                         scale, fp8_type);
  return tmp.u32x2;
}

// fp8x8 -> half2x4
template <>
__inline__ __device__ uint4
scaled_vec_conversion<uint4, uint2>(const uint2& a, const float scale,
                                    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = scaled_vec_conversion<uint2, uint32_t>(a.x, scale, fp8_type);
  tmp.u64[1] = scaled_vec_conversion<uint2, uint32_t>(a.y, scale, fp8_type);
  return tmp.u64x2;
}

// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16
scaled_vec_conversion<__nv_bfloat16, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  // Note there is no direct convert function from fp8 to bf16.
  // fp8 -> half
  __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  // half -> float -> bf16
  float tmp = half_to_float(res.x);
  return __float2bfloat16(tmp * scale);
}

// fp8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162
scaled_vec_conversion<__nv_bfloat162, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_bfloat162 res;
  res.x = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a, scale,
                                                        fp8_type);
  res.y = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U),
                                                        scale, fp8_type);
  return res;
}

// fp8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t scaled_vec_conversion<bf16_4_t, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t res;
  res.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a, scale,
                                                          fp8_type);
  res.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U),
                                                          scale, fp8_type);
  return res;
}

// fp8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t scaled_vec_conversion<bf16_8_t, uint2>(
    const uint2& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t tmp1, tmp2;
  tmp1 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.x, scale, fp8_type);
  tmp2 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.y, scale, fp8_type);
  bf16_8_t res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// fp8 -> float
template <>
__inline__ __device__ float scaled_vec_conversion<float, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  // fp8 -> half
  __half_raw res = __nv_cvt_fp8_to_halfraw(a, fp8_type);
  uint16_t tmp = res.x;

  // half -> float
  return half_to_float(tmp) * scale;
}

// fp8x2 -> float2
template <>
__inline__ __device__ float2 scaled_vec_conversion<float2, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  // fp8x2 -> half2
  uint32_t tmp = scaled_vec_conversion<uint32_t, uint16_t>(a, scale, fp8_type);
  // half2 -> float2
  return half2_to_float2(tmp);
}

// fp8x4 -> float4
template <>
__inline__ __device__ Float4_ scaled_vec_conversion<Float4_, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ res;
  res.x = scaled_vec_conversion<float2, uint16_t>((uint16_t)a, scale, fp8_type);
  res.y = scaled_vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U), scale,
                                                  fp8_type);
  return res;
}

// fp8x8 -> float8
template <>
__inline__ __device__ Float8_ scaled_vec_conversion<Float8_, uint2>(
    const uint2& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp1, tmp2;
  tmp1 = scaled_vec_conversion<Float4_, uint32_t>(a.x, scale, fp8_type);
  tmp2 = scaled_vec_conversion<Float4_, uint32_t>(a.y, scale, fp8_type);
  Float8_ res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// 说明：__NV_SATFINITE 表示饱和转换（Saturate） + 有限值约束（Finite）
// 饱和转换：当源数据超出目标数据格式的有效数值范围时，不进行溢出报错，而是将其「裁剪」（饱和）到目标格式的边界值。
//  - 对比无饱和转换：若不启用 SAT，超出范围的数值可能产生随机溢出值，导致后续计算异常。
// 有限值约束：确保转换结果是有限的有效浮点数，严格排除特殊浮点值：+Infinity（正无穷）、-Infinity（负无穷）、NaN（非数值，如 0/0、sqrt (-1)）。
//  - 若源数据是 ±Infinity：转换为目标格式的最大 / 最小有效值（同饱和转换的边界值）；
//  - 若源数据是 NaN：通常归一化为目标格式的 0 或最大有效值（具体由 CUDA 版本和目标格式决定，优先保证结果可计算）；
// half -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res =
      __nv_cvt_float_to_fp8(half_to_float(a) / scale, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}

// 已阅
// bf16 -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
    #else
  __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(__bfloat162float(a) / scale,
                                                 __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
    #endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

// 已阅
// float -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, float>(
    const float& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res =
      __nv_cvt_float_to_fp8(a / scale, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}

// fp8x4 -> float4
template <>
__inline__ __device__ float4 scaled_vec_conversion<float4, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp = scaled_vec_conversion<Float4_, uint32_t>(a, scale, fp8_type);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}
  #endif  // ENABLE_FP8

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout convert(const Tin& x) {
  #if 0  // Disable the following code to reduce the binary size.
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return vec_conversion<Tout, Tin>(x, __NV_E4M3);
  } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return vec_conversion<Tout, Tin>(x, __NV_E5M2);
  }
  #endif
  assert(false);
  __builtin_unreachable();  // Suppress missing return statement warning
}

// 已阅
template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  #ifdef ENABLE_FP8
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E4M3);
  } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E5M2);
  }
  #endif
  assert(false);
  __builtin_unreachable();  // Suppress missing return statement warning
}

  // 已阅
  // 说明：cache_t 表示输出类型，scalar_t 表示输入类型；
  // 输出类型的确定与 scalar_t (SRC_DTYPE) 和 kv_dt (KV_DTYPE) 有关
  // The following macro is used to dispatch the conversion function based on
  // the data type of the key and value cache. The FN is a macro that calls a
  // function with template<typename scalar_t, typename cache_t,
  // Fp8KVCacheDataType kv_dt>.
  #define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                  \
    if (KV_DTYPE == "auto") {                                                  \
      if (SRC_DTYPE == at::ScalarType::Float) {                                \
        FN(float, float, vllm::Fp8KVCacheDataType::kAuto);                     \
      } else if (SRC_DTYPE == at::ScalarType::Half) {                          \
        FN(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);               \
      } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                      \
        FN(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);     \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE); \
      }                                                                        \
    } else {                                                                   \
      if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {                       \
        if (SRC_DTYPE == at::ScalarType::Float) {                              \
          FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);              \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
          FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);           \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
          FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);      \
        } else {                                                               \
          TORCH_CHECK(false,                                                   \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);      \
        }                                                                      \
      } else if (KV_DTYPE == "fp8_e5m2") {                                     \
        if (SRC_DTYPE == at::ScalarType::Float) {                              \
          FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);              \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
          FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);           \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
          FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);      \
        } else {                                                               \
          TORCH_CHECK(false,                                                   \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);      \
        }                                                                      \
      } else if (KV_DTYPE == "fp8_ds_mla") {                                   \
        if (SRC_DTYPE == at::ScalarType::Float) {                              \
          FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);              \
        } else if (SRC_DTYPE == at::ScalarType::Half) {                        \
          FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);           \
        } else if (SRC_DTYPE == at::ScalarType::BFloat16) {                    \
          FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);      \
        } else {                                                               \
          TORCH_CHECK(false,                                                   \
                      "Unsupported input type of kv cache: ", SRC_DTYPE);      \
        }                                                                      \
      } else {                                                                 \
        TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);   \
      }                                                                        \
    }

}  // namespace fp8
#endif  // not USE_ROCM
}  // namespace vllm
