// clang-format off
// adapted from https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_kernel.cuh
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "selective_scan.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#ifdef USE_ROCM
    #include <c10/hip/HIPException.h>  // For C10_HIP_CHECK and C10_HIP_KERNEL_LAUNCH_CHECK
#else
    #include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#endif

#ifndef USE_ROCM
    // 说明：cub 库提供了高性能的并行算法实现，如并行扫描（scan）和块加载/存储（block load/store）等
    // https://nvidia.github.io/cccl/unstable/cub/api_docs/thread_level.html
    // https://nvidia.github.io/cccl/unstable/cub/api_docs/warp_wide.html
    // https://nvidia.github.io/cccl/unstable/cub/api_docs/block_wide.html
    // https://nvidia.github.io/cccl/unstable/cub/api_docs/device_wide.html
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_scan.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif

#include "selective_scan.h"
#include "static_switch.h"

template<int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_,
         bool kIsVariableB_, bool kIsVariableC_,
         bool kHasZ_, bool kVarlen_, typename input_t_, typename weight_t_, typename state_t_>
struct Selective_Scan_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    using state_t = state_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    // 说明：每个线程处理 kNItems 个 token
    static constexpr int kNItems = kNItems_;
    // 理解：kNRows 相当于 Mamba2 论文中的 P (head dimension)，每个线程处理 kNItems 个 token，每个 token 包含 kNRows 个 dimension
    static constexpr int kNRows = kNRows_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    // 说明：kNBytes 为 4 字节时，输入类型为 float，此时每次加载 4 个元素；
    // kNBytes 为 2 字节时，输入类型为 half 或 bfloat16，此时每次加载 8 个元素。
    static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    // 说明：每个线程需要进行的加载次数，因为每次加载 kNElts 个元素，所以总的加载次数是 kNItems / kNElts
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsEvenLen = kVarlen_ ? false : kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kHasZ = kHasZ_;
    static constexpr bool kVarlen = kVarlen_;

    static constexpr bool kDirectIO = kVarlen_ ? false : kIsEvenLen && kNLoads == 1;
    static constexpr int kNLoadsIndex = kNItems / 4;
    // 说明：向量化加载和存储的类型，vec_t 是一个包含 kNElts 个 input_t 的向量类型，
    // 这样每次加载或存储就可以处理 kNElts 个元素，从而提高内存访问效率。
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = float2;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, kNItems , cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads ,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    // 说明：文档见 https://nvidia.github.io/cccl/unstable/cub/api_docs/block_wide.html
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    // 说明：加载输入、权重和存储输出所需的共享内存大小
    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 // 说明：kIsVariableB 和 kIsVariableC 为 true 时，才需要使用 TempStorage 来加载 B 和 C 的值
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    // 说明：kSmemIOSize + Scan 所需的内存总大小，共享内存依次存储：
    // 1. 加载输入和权重所需的临时存储空间，大小为 kSmemIOSize；
    // 2. 扫描操作所需的临时存储空间，大小为 sizeof(typename BlockScanT::TempStorage)；
    // 3. 扫描过程中需要保存的每个 state 的 running prefix，大小为 kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t)
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

// 已阅
template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params) {
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr bool kVarlen = Ktraits::kVarlen;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    // cast to lvalue reference of expected type
    // char *smem_loadstorescan = smem_ + 2 * MAX_DSTATE * sizeof(weight_t);
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_ + 2 * MAX_DSTATE * sizeof(weight_t));
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_loadstorescan);
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    // 问题：这里使用的是 BlockLoadWeightT::TempStorage 而不是 BlockLoadWeightVecT::TempStorage，
    // 是否因为 BlockLoadWeightT::TempStorage 占用的空间可能会更大一些，否则如果 BlockLoadWeightVecT::TempStorage 占用的空间更大一些，
    // 那么可能产生数据的 overlap ?
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    // weight_t *smem_a = reinterpret_cast<weight_t *>(smem_ + smem_loadstorescan_size);
    // weight_t *smem_bc = reinterpret_cast<weight_t *>(smem_a + MAX_DSTATE);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    int seqlen = params.seqlen;
    int sequence_start_index = batch_id;
    if constexpr (kVarlen){
        int *query_start_loc = reinterpret_cast<int *>(params.query_start_loc_ptr);
        sequence_start_index = query_start_loc[batch_id];
        seqlen = query_start_loc[batch_id + 1] - sequence_start_index;
    }
    const bool has_initial_state = params.has_initial_state_ptr == nullptr ? false
        : reinterpret_cast<bool *>(params.has_initial_state_ptr)[batch_id];

    const int* cache_indices = params.cache_indices_ptr == nullptr ? nullptr
        : reinterpret_cast<int *>(params.cache_indices_ptr);
    const int cache_index = cache_indices == nullptr ? batch_id : cache_indices[batch_id]; 
    // cache_index == params.pad_slot_id is defined as padding, so we exit early
    if (cache_index == params.pad_slot_id){
        return;
    }
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + sequence_start_index * params.u_batch_stride
        + dim_id * kNRows * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + sequence_start_index * params.delta_batch_stride
        + dim_id * kNRows * params.delta_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
    // 说明：B, C 实际 shape 为 [1, dstate, total_length]
    // (ngroups, dstate, total_length) for varlen or (batch, ngroups, dstate, seqlen)
    // 说明：B, C 缺 ngroups, dstate 和 total_length 三个维度，因为 params.B_d_stride 和 params.C_d_stride 在 varlen 时为 0
    // Bvar 和 Cvar 缺 dstate 维度
    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + sequence_start_index * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + sequence_start_index * params.C_batch_stride + group_id * params.C_group_stride;

    typename Ktraits::state_t *ssm_states;
    if (params.cache_enabled) {
        // APC mode: ssm_states points to the base, we'll use absolute cache slots later
        ssm_states = reinterpret_cast<typename Ktraits::state_t *>(params.ssm_states_ptr) +
            dim_id * kNRows * params.ssm_states_dim_stride;
    } else {
        // Non-APC mode: offset by cache_index as before
        ssm_states = reinterpret_cast<typename Ktraits::state_t *>(params.ssm_states_ptr) +
            cache_index * params.ssm_states_batch_stride +
            dim_id * kNRows * params.ssm_states_dim_stride;
    }
    
    float D_val[kNRows] = {0};
    if (params.D_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            D_val[r] = reinterpret_cast<float *>(params.D_ptr)[dim_id * kNRows + r];
        }
    }
    float delta_bias[kNRows] = {0};
    if (params.delta_bias_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            delta_bias[r] = reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id * kNRows + r];
        }
    }


    // for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {
    //     smem_a[state_idx] = A[state_idx * params.A_dstate_stride];
    //     smem_bc[state_idx] = B[state_idx * params.B_dstate_stride] * C[state_idx * params.C_dstate_stride];
    // }

    // 说明：kNThreads 个线程，每个线程处理 kNItems 个 token，组成一个 chunk；
    // kChunkSize 默认情况是 (128, 16) = 2048, 在 params.cache_enabled && params.block_size == 1024 时是 (64, 16) = 1024,
    // 否则就是大于 seqlen 的值，这样就相当于不进行 chunk 划分，直接在一个 chunk 中处理所有 token
    constexpr int kChunkSize = kNThreads * kNItems;

    // 说明：params.block_size 使用的是 mamba_block_size，这是一个可配置的参数；
    // Use block_size for chunking when APC is enabled, otherwise use 2048 for backwards compatibility
    const int iteration_chunk_size = params.cache_enabled ? params.block_size : 2048;
    const int n_chunks = (seqlen + iteration_chunk_size - 1) / iteration_chunk_size;

    const int* batch_cache_indices = cache_indices != nullptr ?
                                     cache_indices + batch_id * params.cache_indices_stride : nullptr;
    const int* block_idx_first_scheduled = params.block_idx_first_scheduled_token_ptr != nullptr ?
                                           reinterpret_cast<const int*>(params.block_idx_first_scheduled_token_ptr) : nullptr;
    const int* block_idx_last_scheduled = params.block_idx_last_scheduled_token_ptr != nullptr ?
                                          reinterpret_cast<const int*>(params.block_idx_last_scheduled_token_ptr) : nullptr;
    const int* initial_state_idx = params.initial_state_idx_ptr != nullptr ?
                                   reinterpret_cast<const int*>(params.initial_state_idx_ptr) : nullptr;

    // 说明：initial ssm state 的 index
    const size_t load_cache_slot = params.cache_enabled && batch_cache_indices != nullptr ? batch_cache_indices[initial_state_idx[batch_id]] : cache_index;

    // 优化：n_chunks 是根据 iteration_chunk_size 计算出来的，而 chunk_size 使用的是 kChunkSize，
    // 在 params.cache_block = false 时，使用 iteration_chunk_size 和 kChunkSize 计算分块的结果都一样，要么都不分块，要么都根据 2048 长度进行分块；
    // 在 params.cache_block = true && params.block_size != 1024 时，kChunkSize 根据序列长度不同会有不同的值，而 params.block_size 的值是固定的，
    // 如果 params.block_size 的值小于 kChunkSize，那么实际的块数会比分出的块数少，相当于会多执行几次无效循环
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];

        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, seqlen - chunk * kChunkSize);
            if constexpr (!kDirectIO) { __syncthreads(); }
            load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, seqlen - chunk * kChunkSize);
        }
        u += kChunkSize;
        delta += kChunkSize;
    
        float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float u_val = float(u_vals[r][i]);
                delta_vals[r][i] = float(delta_vals_load[r][i]) + delta_bias[r];
                if (params.delta_softplus) {
                    // 说明：当 delta 值 <= 20.f 时，使用 \ln(1 + \exp(x)) 计算；当 delta 值较大时，由于误差较小，直接使用 delta 值；
                    delta_vals[r][i] = delta_vals[r][i] <= 20.f ? log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
                }
                // 说明：delta_u_vals[r][i] 存储了 delta_vals[r][i] 和 u_val 的乘积，
                // 因为在后续计算中会多次使用到这个乘积值，这样可以避免重复计算，提高效率；
                delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                // 说明：out_vals[r][i] 存储了 D_val[r] 和 u_val 的乘积，
                // 如果 D_ 不为 nullptr，则 D_val[r] 是 D 中对应元素的值；
                // 如果 D_ 为 nullptr，则 D_val[r] 默认为 0，这样 out_vals[r][i] 的初始值就是 0；
                // 说明：在 S4 论文中有相关介绍 (https://arxiv.org/pdf/2111.00396)，
                // the term D*u can be viewed as a skip connection
                out_vals[r][i] = D_val[r] * u_val;
            }
        }

        __syncthreads();
        // 说明：dstate 对应 python 代码中的 ssm_state_size，对应论文中的 N，即 state dimension
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            weight_t A_val[kNRows];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                A_val[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
                // Multiply the real part of A with LOG2E so we can use exp2f instead of expf.
                constexpr float kLog2e = M_LOG2E;
                A_val[r] *= kLog2e;
            }
            // 说明：launch 中提到 kIsVariableB, kIsVariableC and kHasZ are all set to True to reduce binary size 
            // This variable holds B * C if both B and C are constant across seqlen. If only B varies
            // across seqlen, this holds C. If only C varies across seqlen, this holds B.
            // If both B and C vary, this is unused.
            weight_t BC_val[kNRows];
            weight_t B_vals[kNItems], C_vals[kNItems];
            if constexpr (kIsVariableB) {
                // 说明：Bvar 缺 dstate 维度，这里补充之后加载数据
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    // 说明：乘 1 是因为值为实数而非复数，否则乘 2
                    // 完整代码见 https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_kernel.cuh
                    smem_load_weight, (seqlen - chunk * kChunkSize) * (1));
                if constexpr (!kIsVariableC) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                    }
                }
            }
            if constexpr (kIsVariableC) {
                // 说明：避免 shared memory bank conflict，当 B 和 C 都变化时，B 的加载使用 smem_load_weight，C 的加载使用 smem_load_weight1
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, (seqlen - chunk * kChunkSize) * (1));
                if constexpr (!kIsVariableB) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
                    }
                }
            }
            if constexpr (!kIsVariableB && !kIsVariableC) {
                #pragma unroll
                for (int r = 0; r < kNRows; ++r) {
                    BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride] * C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                }
            }

            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                if (r > 0) { __syncthreads(); }  // Scan could be using the same smem
                scan_t thread_data[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    // 说明：下面的值都是在固定 dim 和 state_idx 后得到的，r == 0
                    // 2^(delta_vals[r][i] * A_val[r]) 对应 Mamba-1 论文中的 zero-order hold 等式中的 exp(delta * A)，
                    // A_val[r] 在上面已经乘了 LOG2E，所以这里使用 exp2f 可以得到相同的结果；
                    // B_vals[i] * delta_u_vals[r][i] 的值是 delta * B * u，相当于 \bar{B} * u 的后半部分，即
                    // \bar{B} * u = (delta * A)^{-1} (\bar{A} - I) \cdot (delta * B) * u
                    // 说明：在 Mamba-3 的论文中提到了，这里使用的是 Euler's discretization（一阶近似），
                    // 即 h_t = \exp^{\Delta_t A_t} h_{t−1} + \Delta_t B_t x_t
                    thread_data[i] = make_float2(exp2f(delta_vals[r][i] * A_val[r]),
                                                 !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i]);
                    if (seqlen % (kNItems * kNThreads) != 0) {  // So that the last state is correct
                        // 说明：threadIdx.x * kNItems 表示前面的线程已经处理的连续 token 的数量，i 是当前线程处理的位置，
                        // seqlen - chunk * kChunkSize 是排除前面的 chunk 之后剩余的 token 数量，
                        // 如果当前线程处理的位置超过了剩余的 token 数量（说明处于最后一个 chunk），那么就将 thread_data[i] 的值设置为 (1.f, 0.f)，
                        // \bar{A} 的值为 1 让 h 保持不变，delta * B * u 为 0 让输出不受到这个位置的影响
                        if (threadIdx.x * kNItems + i >= seqlen - chunk * kChunkSize) {
                            thread_data[i] = make_float2(1.f, 0.f);
                        }
                    }
                }
                // Initialize running total
                scan_t running_prefix;
                if (chunk > 0) {
                    // 说明：smem_running_prefix 指向的区域长度为 kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t)；
                    // 说明：chunk 外循环，state_idx 内循环，每次内循环结束后都会将当前 state_idx 对应的 running prefix 存储到 
                    // smem_running_prefix 中，供下一个 chunk 处理时加载使用；
                    // 这里就是直接加载上一个 chunk 处理同一个 state_idx 时存储的 running prefix 的值，作为当前 chunk 处理时的初始状态；
                    running_prefix = smem_running_prefix[state_idx + r * MAX_DSTATE];
                } else {
                    // 说明：chunk == 0 时，从 ssm_states 中加载初始状态
                    // Load initial state
                    if (params.cache_enabled && has_initial_state && batch_cache_indices != nullptr) {
                        // 说明：load_cache_slot 对应的是 initial ssm state 的 index
                        size_t state_offset = load_cache_slot * params.ssm_states_batch_stride +
                                             r * params.ssm_states_dim_stride +
                                             state_idx * params.ssm_states_dstate_stride;
                        running_prefix = make_float2(1.0, float(ssm_states[state_offset]));
                    } else if (has_initial_state) {
                        // Non-APC mode: load from current batch position
                        running_prefix = make_float2(1.0, float(ssm_states[state_idx * params.ssm_states_dstate_stride]));
                    } else {
                        // No initial state
                        running_prefix = make_float2(1.0, 0.0);
                    }
                }

                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                // 说明：InclusiveScan 的文档见
                // https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockScan.html#_CPPv4I_i00EN3cub9BlockScan13InclusiveScanEvRA16ITEMS_PER_THREAD_1TRA16ITEMS_PER_THREAD_1T6ScanOpR21BlockPrefixCallbackOp
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                // There's a syncthreads in the scan op, so we don't need to sync here.
                // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
                if (threadIdx.x == 0) {
                    // 说明：每个 chunk 处理完后，Block 内的线程 0 会将当前 state_idx 对应的 running prefix
                    // 存储到 smem_running_prefix 中，供下一个 chunk 处理时加载使用；
                    smem_running_prefix[state_idx + r * MAX_DSTATE] = prefix_op.running_prefix;

                    // Store state at the end of each chunk when cache is enabled
                    if (params.cache_enabled && batch_cache_indices != nullptr) {

                        size_t cache_slot;
                        if (chunk == n_chunks - 1) {
                            // 说明：最后一个 chunk 写入 block_idx_last，即从后面对齐
                            cache_slot = batch_cache_indices[block_idx_last_scheduled[batch_id]];
                        } else {
                            // 说明：非最后一个 chunk 写入 block_idx_first + chunk，即从前面对齐
                            cache_slot = batch_cache_indices[block_idx_first_scheduled[batch_id] + chunk];
                        }

                        size_t state_offset = cache_slot * params.ssm_states_batch_stride +
                                             r * params.ssm_states_dim_stride +
                                             state_idx * params.ssm_states_dstate_stride;

                        // 说明：ssm_states 保存的是 y 值
                        // 理解：不存储 x 值，是因为初始的 state 为 0（Mamba2 论文的 B.1 Problem Definition 中有提到），
                        // 所以 [x, y] * [0, 1] 的结果就是 y 的值，直接存储 y 就可以了
                        ssm_states[state_offset] = typename Ktraits::state_t(prefix_op.running_prefix.y);
                    } else if (!params.cache_enabled && chunk == n_chunks - 1) {
                        // Non-APC mode: store only final state at current batch position
                        ssm_states[state_idx * params.ssm_states_dstate_stride] = typename Ktraits::state_t(prefix_op.running_prefix.y);
                    }
                }
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const weight_t C_val = !kIsVariableC
                        ? BC_val[r]
                        : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
                    // 说明：原来存储的是 D * u，现在加上 C * h
                    out_vals[r][i] += thread_data[i].y * C_val;
                }
            }
        }
        // 说明：out 对应 delta
        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + sequence_start_index * params.out_batch_stride
            + dim_id * kNRows * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            store_output<Ktraits>(out + r * params.out_d_stride, out_vals[r], smem_store, seqlen - chunk * kChunkSize);
        }

        if constexpr (kHasZ) {
            // 说明：out_z 对应 z
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + sequence_start_index * params.z_batch_stride
                + dim_id * kNRows * params.z_d_stride + chunk * kChunkSize;
            input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + sequence_start_index * params.out_z_batch_stride
                + dim_id * kNRows * params.out_z_d_stride + chunk * kChunkSize;
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                input_t z_vals[kNItems];
                __syncthreads();
                load_input<Ktraits>(z + r * params.z_d_stride, z_vals, smem_load, seqlen - chunk * kChunkSize);
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float z_val = z_vals[i];
                    // 说明：Swish / SiLU
                    out_vals[r][i] *= z_val / (1 + expf(-z_val));
                }
                __syncthreads();
                store_output<Ktraits>(out_z + r * params.out_z_d_stride, out_vals[r], smem_store, seqlen - chunk * kChunkSize);
            }
        }

        Bvar += kChunkSize * 1;
        Cvar += kChunkSize * 1;
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t, typename state_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    // Only kNRows == 1 is tested for now, which ofc doesn't differ from previously when we had each block
    // processing 1 row.
    // 说明：kNRows 相当于 Mamba2 论文中的 P (head dimension)
    constexpr int kNRows = 1;
    // kIsVariableB, kIsVariableC and kHasZ are all set to True to reduce binary size
    constexpr bool kIsVariableB = true;
    constexpr bool kIsVariableC = true;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.z_ptr != nullptr , kHasZ, [&] {
            BOOL_SWITCH(params.query_start_loc_ptr != nullptr , kVarlen, [&] {
                using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kIsVariableB, kIsVariableC, kHasZ,  kVarlen, input_t, weight_t, state_t>;
                constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
                // 说明：每个 block 处理一个完整序列的一个 channel (kNRows = 1)
                dim3 grid(params.batch, params.dim / kNRows);
                auto kernel = &selective_scan_fwd_kernel<Ktraits>;
                if (kSmemSize >= 48 * 1024) {
#ifdef USE_ROCM
                    C10_HIP_CHECK(hipFuncSetAttribute(
                        reinterpret_cast<const void*>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#else
                    C10_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
#endif
                }
                kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
        });
    });
}

template<typename input_t, typename weight_t, typename state_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {

    #ifndef USE_ROCM
        if (params.cache_enabled && params.block_size == 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 128) {
            selective_scan_fwd_launch<32, 4, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 256) {
            selective_scan_fwd_launch<32, 8, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_fwd_launch<32, 16, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t, state_t>(params, stream);
        } else {
            selective_scan_fwd_launch<128, 16, input_t, weight_t, state_t>(params, stream);
        }
    #else
        if (params.cache_enabled && params.block_size == 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 256) {
            selective_scan_fwd_launch<64, 4, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_fwd_launch<64, 8, input_t, weight_t, state_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t, state_t>(params, stream);
        } else {
            selective_scan_fwd_launch<128, 16, input_t, weight_t, state_t>(params, stream);
        }
    #endif
}

template void selective_scan_fwd_cuda<at::BFloat16, float, at::BFloat16>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<at::BFloat16, float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<at::Half, float, at::Half>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<at::Half, float, float>(SSMParamsBase &params, cudaStream_t stream);
template void selective_scan_fwd_cuda<float, float, float>(SSMParamsBase &params, cudaStream_t stream);

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, STYPE, NAME, ...)       \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        using weight_t = float;                                                     \
        if (STYPE == at::ScalarType::Half) {                                        \
            using state_t = at::Half;                                               \
            __VA_ARGS__();                                                          \
        } else if (STYPE == at::ScalarType::Float) {                                \
            using state_t = float;                                                  \
            __VA_ARGS__();                                                          \
        } else {                                                                    \
            AT_ERROR(#NAME, " not implemented for state type '", toString(STYPE), "'"); \
        }                                                                           \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        using weight_t = float;                                                     \
        if (STYPE == at::ScalarType::BFloat16) {                                    \
            using state_t = at::BFloat16;                                           \
            __VA_ARGS__();                                                          \
        } else if (STYPE == at::ScalarType::Float) {                                \
            using state_t = float;                                                  \
            __VA_ARGS__();                                                          \
        } else {                                                                    \
            AT_ERROR(#NAME, " not implemented for state type '", toString(STYPE), "'"); \
        }                                                                           \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        using weight_t = float;                                                     \
        using state_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }


template<typename input_t, typename weight_t, typename state_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream);

void set_ssm_params_fwd(SSMParamsBase &params,
                        // sizes
                        const size_t batch,
                        const size_t dim,
                        const size_t seqlen,
                        const size_t dstate,
                        const size_t n_groups,
                        const bool is_variable_B,
                        const bool is_variable_C,
                        // device pointers
                        const torch::Tensor u,
                        const torch::Tensor delta,
                        const torch::Tensor A,
                        const torch::Tensor B,
                        const torch::Tensor C,
                        const torch::Tensor out,
                        const torch::Tensor z,
                        const torch::Tensor out_z,
                        const std::optional<at::Tensor>& D,
                        const std::optional<at::Tensor>& delta_bias,
                        const torch::Tensor ssm_states,
                        bool has_z,
                        bool delta_softplus,
                        const std::optional<at::Tensor>& query_start_loc,
                        const std::optional<at::Tensor>& cache_indices,
                        const std::optional<at::Tensor>& has_initial_state,
                        bool varlen,
                        int64_t pad_slot_id,
                        int64_t block_size,
                        const std::optional<torch::Tensor> &block_idx_first_scheduled_token,
                        const std::optional<torch::Tensor> &block_idx_last_scheduled_token,
                        const std::optional<torch::Tensor> &initial_state_idx) {

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.seqlen = seqlen;
    params.dstate = dstate;
    params.n_groups = n_groups;
    // 说明：每组多少个 dim
    params.dim_ngroups_ratio = dim / n_groups;
    params.pad_slot_id = pad_slot_id;

    params.delta_softplus = delta_softplus;

    params.is_variable_B = is_variable_B;
    params.is_variable_C = is_variable_C;

    // Set the pointers and strides.
    params.u_ptr = u.data_ptr();
    params.delta_ptr = delta.data_ptr();
    params.A_ptr = A.data_ptr();
    params.B_ptr = B.data_ptr();
    params.C_ptr = C.data_ptr();
    params.D_ptr = D.has_value() ? D.value().data_ptr() : nullptr;
    params.delta_bias_ptr = delta_bias.has_value() ? delta_bias.value().data_ptr() : nullptr;
    params.out_ptr = out.data_ptr();
    params.ssm_states_ptr = ssm_states.data_ptr();
    params.z_ptr = has_z ? z.data_ptr() : nullptr;
    params.out_z_ptr = has_z ? out_z.data_ptr() : nullptr;
    params.query_start_loc_ptr = query_start_loc.has_value() ? query_start_loc.value().data_ptr() : nullptr;
    params.cache_indices_ptr = cache_indices.has_value() ? cache_indices.value().data_ptr() : nullptr;
    params.has_initial_state_ptr = has_initial_state.has_value() ? has_initial_state.value().data_ptr() : nullptr;

    // 说明：这里判断 APC 模式的条件是 block_idx_first_scheduled_token 有值，
    // 在 conv1d 中判断 APC 模式的条件是 block_idx_last_scheduled_token 有值
    // Set cache parameters - cache is enabled if we have direct cache writing params
    params.cache_enabled = block_idx_first_scheduled_token.has_value();
    params.block_size = static_cast<int>(block_size);

    // Set direct cache writing pointers
    params.block_idx_first_scheduled_token_ptr = block_idx_first_scheduled_token.has_value() ? block_idx_first_scheduled_token.value().data_ptr() : nullptr;
    params.block_idx_last_scheduled_token_ptr = block_idx_last_scheduled_token.has_value() ? block_idx_last_scheduled_token.value().data_ptr() : nullptr;
    params.initial_state_idx_ptr = initial_state_idx.has_value() ? initial_state_idx.value().data_ptr() : nullptr;

    // All stride are in elements, not bytes.
    params.A_d_stride = A.stride(0);
    params.A_dstate_stride = A.stride(1);

    if (varlen){
        params.B_batch_stride = B.stride(2);
        params.B_group_stride = B.stride(0);
        params.B_dstate_stride = B.stride(1);
        params.C_batch_stride = C.stride(2);
        params.C_group_stride = C.stride(0);
        params.C_dstate_stride = C.stride(1);

        params.u_batch_stride = u.stride(1);
        params.u_d_stride = u.stride(0);
        params.delta_batch_stride = delta.stride(1);
        params.delta_d_stride = delta.stride(0);
        if (has_z) {
            params.z_batch_stride = z.stride(1);
            params.z_d_stride = z.stride(0);
            params.out_z_batch_stride = out_z.stride(1);
            params.out_z_d_stride = out_z.stride(0);
        }
        params.out_batch_stride = out.stride(1);
        params.out_d_stride = out.stride(0);

        params.ssm_states_batch_stride = ssm_states.stride(0);
        params.ssm_states_dim_stride = ssm_states.stride(1);
        params.ssm_states_dstate_stride = ssm_states.stride(2);

        params.cache_indices_stride = cache_indices.has_value() ? cache_indices.value().stride(0) : 0;

    }
    else{
        if (!is_variable_B) {
            params.B_d_stride = B.stride(0);
        } else {
            params.B_batch_stride = B.stride(0);
            params.B_group_stride = B.stride(1);
        }
        params.B_dstate_stride = !is_variable_B ? B.stride(1) : B.stride(2);
        if (!is_variable_C) {
            params.C_d_stride = C.stride(0);
        } else {
            params.C_batch_stride = C.stride(0);
            params.C_group_stride = C.stride(1);
        }
        params.C_dstate_stride = !is_variable_C ? C.stride(1) : C.stride(2);
        params.u_batch_stride = u.stride(0);
        params.u_d_stride = u.stride(1);
        params.delta_batch_stride = delta.stride(0);
        params.delta_d_stride = delta.stride(1);
        if (has_z) {
            params.z_batch_stride = z.stride(0);
            params.z_d_stride = z.stride(1);
            params.out_z_batch_stride = out_z.stride(0);
            params.out_z_d_stride = out_z.stride(1);
        }
        params.out_batch_stride = out.stride(0);
        params.out_d_stride = out.stride(1);
        
        params.ssm_states_batch_stride = ssm_states.stride(0);
        params.ssm_states_dim_stride = ssm_states.stride(1);
        params.ssm_states_dstate_stride = ssm_states.stride(2);

        params.cache_indices_stride = cache_indices.has_value() ? cache_indices.value().stride(0) : 0;
    }
}

void selective_scan_fwd(const torch::Tensor &u, const torch::Tensor &delta,
                  // 说明：B, C shape 为 [1, dstate, total_length]
                  const torch::Tensor &A, const torch::Tensor &B, const torch::Tensor &C,
                  const std::optional<torch::Tensor> &D_,
                  // 说明：实参为 gate_p
                  const std::optional<torch::Tensor> &z_,
                  const std::optional<torch::Tensor> &delta_bias_,
                  bool delta_softplus,
                  const std::optional<torch::Tensor> &query_start_loc,
                  const std::optional<torch::Tensor> &cache_indices,
                  const std::optional<torch::Tensor> &has_initial_state,
                  const torch::Tensor &ssm_states,
                  // used to identify padding entries if cache_indices provided
                  // in case of padding, the kernel will return early
                  int64_t pad_slot_id,
                  int64_t block_size,
                  const std::optional<torch::Tensor> &block_idx_first_scheduled_token,
                  const std::optional<torch::Tensor> &block_idx_last_scheduled_token,
                  const std::optional<torch::Tensor> &initial_state_idx) {
    auto input_type = u.scalar_type();
    auto weight_type = A.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(weight_type == at::ScalarType::Float);

    const bool is_variable_B = B.dim() >= 3;
    const bool is_variable_C = C.dim() >= 3;

    TORCH_CHECK(delta.scalar_type() == input_type);
    TORCH_CHECK(B.scalar_type() == (!is_variable_B ? weight_type : input_type));
    TORCH_CHECK(C.scalar_type() == (!is_variable_C ? weight_type : input_type));

    TORCH_CHECK(u.is_cuda());
    TORCH_CHECK(delta.is_cuda());
    TORCH_CHECK(A.is_cuda());
    TORCH_CHECK(B.is_cuda());
    TORCH_CHECK(C.is_cuda());

    TORCH_CHECK(u.stride(-1) == 1 || u.size(-1) == 1);
    TORCH_CHECK(delta.stride(-1) == 1 || delta.size(-1) == 1);

    const auto sizes = u.sizes();
    const bool varlen = query_start_loc.has_value();
    const int batch_size = varlen ? query_start_loc.value().sizes()[0] - 1 : sizes[0];
    const int dim = varlen ? sizes[0] : sizes[1];
    const int seqlen = varlen ? sizes[1] : sizes[2];
    const int dstate = A.size(1);
    const int n_groups = varlen ? B.size(0) : B.size(1);

    TORCH_CHECK(dstate <= 256, "selective_scan only supports state dimension <= 256");

    if (varlen) {
        CHECK_SHAPE(u, dim, seqlen);
        CHECK_SHAPE(delta, dim, seqlen);
    } else {
        CHECK_SHAPE(u, batch_size, dim, seqlen);
        CHECK_SHAPE(delta, batch_size, dim, seqlen);
    }
    CHECK_SHAPE(A, dim, dstate);
    TORCH_CHECK(is_variable_B, "is_variable_B = False is disabled in favor of reduced binary size")
    if (varlen) {
        CHECK_SHAPE(B, n_groups, dstate, seqlen);
    } else {
        CHECK_SHAPE(B, batch_size, n_groups, dstate, seqlen); 
    }
    TORCH_CHECK(B.stride(-1) == 1 || B.size(-1) == 1);

    TORCH_CHECK(is_variable_C, "is_variable_C = False is disabled in favor of reduced binary size")
    if (varlen) {
        CHECK_SHAPE(C, n_groups, dstate, seqlen);
    } else {
        CHECK_SHAPE(C, batch_size, n_groups, dstate, seqlen); 
    }
    TORCH_CHECK(C.stride(-1) == 1 || C.size(-1) == 1);

    if (D_.has_value()) {
        auto D = D_.value();
        TORCH_CHECK(D.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(D.is_cuda());
        TORCH_CHECK(D.stride(-1) == 1 || D.size(-1) == 1);
        CHECK_SHAPE(D, dim);
    }

    if (delta_bias_.has_value()) {
        auto delta_bias = delta_bias_.value();
        TORCH_CHECK(delta_bias.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(delta_bias.is_cuda());
        TORCH_CHECK(delta_bias.stride(-1) == 1 || delta_bias.size(-1) == 1);
        CHECK_SHAPE(delta_bias, dim);
    }


    if (has_initial_state.has_value()) {
        auto has_initial_state_ = has_initial_state.value();
        TORCH_CHECK(has_initial_state_.scalar_type() == at::ScalarType::Bool);
        TORCH_CHECK(has_initial_state_.is_cuda());
        CHECK_SHAPE(has_initial_state_, batch_size);
    }


    if (query_start_loc.has_value()) {
        auto query_start_loc_ = query_start_loc.value();
        TORCH_CHECK(query_start_loc_.scalar_type() == at::ScalarType::Int);
        TORCH_CHECK(query_start_loc_.is_cuda());
    }


    if (cache_indices.has_value()) {
        auto cache_indices_ = cache_indices.value();
        TORCH_CHECK(cache_indices_.scalar_type() == at::ScalarType::Int);
        TORCH_CHECK(cache_indices_.is_cuda());

        // cache_indices can be either 1D (batch_size,) for non-APC mode
        // or 2D (batch_size, max_positions) for APC mode
        const bool is_apc_mode = block_idx_first_scheduled_token.has_value();
        if (is_apc_mode) {
            TORCH_CHECK(cache_indices_.dim() == 2, "cache_indices must be 2D for APC mode");
            TORCH_CHECK(cache_indices_.size(0) == batch_size, "cache_indices first dimension must match batch_size");
        } else {
            CHECK_SHAPE(cache_indices_, batch_size);
        }
    }
   

    at::Tensor z, out_z;
    const bool has_z = z_.has_value();
    if (has_z) {
        z = z_.value();
        TORCH_CHECK(z.scalar_type() == input_type);
        TORCH_CHECK(z.is_cuda());
        TORCH_CHECK(z.stride(-1) == 1 || z.size(-1) == 1);
        if (varlen){
            CHECK_SHAPE(z, dim, seqlen);
        } else {
            CHECK_SHAPE(z, batch_size, dim, seqlen);
        }
        
        out_z = z;
    }

    // Right now u has BHL layout and delta has HBL layout, and we want out to have HBL layout
    at::Tensor out = delta;
    // ssm_states can now be either the same as input_type or float32
    auto state_type = ssm_states.scalar_type();
    TORCH_CHECK(state_type == input_type || state_type == at::ScalarType::Float);
    TORCH_CHECK(ssm_states.is_cuda());
    TORCH_CHECK(ssm_states.stride(-1) == 1);

    SSMParamsBase params;
    set_ssm_params_fwd(params, batch_size, dim, seqlen, dstate, n_groups, is_variable_B, is_variable_C,
                       u, delta, A, B, C, out, z, out_z,
                       D_,
                       delta_bias_,
                       ssm_states,
                       has_z,
                       delta_softplus,
                       query_start_loc,
                       cache_indices,
                       has_initial_state,
                       varlen,
                       pad_slot_id,
                       block_size,
                       block_idx_first_scheduled_token,
                       block_idx_last_scheduled_token,
                       initial_state_idx
                       );

    
    const at::cuda::OptionalCUDAGuard device_guard(device_of(u));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_WTYPE_ITYPE_FLOAT_AND_HALF_AND_BF16(u.scalar_type(), ssm_states.scalar_type(), "selective_scan_fwd", [&] {
        selective_scan_fwd_cuda<input_t, weight_t, state_t>(params, stream);
    });
}
