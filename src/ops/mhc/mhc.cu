// The SenseNova-Loop mHC residual streams: a two-stream router with a log-space Sinkhorn mix, the
// pre-gated branch, the mix-and-gate merge, and the enter/leave stream transitions. The contract is
// include/ninfer/ops/mhc.h; the formulas are the release `modeling_sensenova_loop.py`'s.

#include "ninfer/ops/mhc.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

using bf16 = __nv_bfloat16;

constexpr int kS       = kMhcStreams;
constexpr int kRows    = kMhcRouterRows;
constexpr int kThreads = 256;
constexpr int kWarps   = kThreads / 32;
constexpr std::int32_t kMaxStreamWidth = 32768;
constexpr std::int32_t kMaxIters       = 64;
// Feature tiles of an element-wise launch ride grid.y (at most 65,535); columns ride grid.x.
constexpr std::int32_t kMaxHidden      = 65535 * kThreads;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("mhc: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("mhc ") + what + ": " + cudaGetErrorString(error));
    }
}

bool contiguous_2d(const Tensor& t, DType dtype) {
    return t.data != nullptr && t.dtype == dtype && t.is_contiguous() && t.ne[2] == 1 && t.ne[3] == 1;
}

bool aligned16(const void* p) { return reinterpret_cast<std::uintptr_t>(p) % 16 == 0; }

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.data), b0 = reinterpret_cast<std::uintptr_t>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

__device__ __forceinline__ float sigmoidf(float x) { return 1.0F / (1.0F + expf(-x)); }

// log(sum(exp(a)) + log(sum(exp(b))) style reduction over a two-element axis, kept in log space so a
// long Sinkhorn run cannot underflow a stream that has already lost to the other one.
__device__ __forceinline__ void lse_subtract_pair(float& a, float& b) {
    const float peak = fmaxf(a, b);
    const float lse  = peak + logf(expf(a - peak) + expf(b - peak));
    a -= lse;
    b -= lse;
}

// The log-space Sinkhorn-Knopp projection of a 2x2 logit block, `iters` rounds of row then column
// normalisation, exponentiated at the end. Runs on one thread: the block is four values.
__device__ __forceinline__ void sinkhorn_2x2(float (&l)[kS][kS], int iters) {
    for (int round = 0; round < iters; ++round) {
        lse_subtract_pair(l[0][0], l[0][1]);
        lse_subtract_pair(l[1][0], l[1][1]);
        lse_subtract_pair(l[0][0], l[1][0]);
        lse_subtract_pair(l[0][1], l[1][1]);
    }
    for (int i = 0; i < kS; ++i) {
        for (int j = 0; j < kS; ++j) { l[i][j] = expf(l[i][j]); }
    }
}

// One block per column: the statistic over the whole K and all kRows projection rows in one pass over
// the column (each residual value is read once and meets every row), then the split and the log-space
// Sinkhorn on thread 0 (a 2 x 2 block), then the pre-gated branch over the H features.
__global__ void __launch_bounds__(kThreads)
router_kernel(const bf16* __restrict__ residual, const float* __restrict__ weight,
              const float* __restrict__ bias, int hidden, int iters, bf16* __restrict__ branch,
              float* __restrict__ post, float* __restrict__ mix) {
    const int t     = blockIdx.x;
    const int width = kS * hidden;
    const bf16* column = residual + static_cast<std::size_t>(t) * width;

    __shared__ float partial[kWarps][kRows + 1];
    __shared__ float total[kRows + 1];
    __shared__ float pre_bf16[kS];

    float acc[kRows];
    float sumsq = 0.0F;
#pragma unroll
    for (int i = 0; i < kRows; ++i) { acc[i] = 0.0F; }

    for (int c = threadIdx.x; c < width / 8; c += kThreads) {
        const uint4 raw = *reinterpret_cast<const uint4*>(column + 8 * c);
        const __nv_bfloat162 v0 = *reinterpret_cast<const __nv_bfloat162*>(&raw.x);
        const __nv_bfloat162 v1 = *reinterpret_cast<const __nv_bfloat162*>(&raw.y);
        const __nv_bfloat162 v2 = *reinterpret_cast<const __nv_bfloat162*>(&raw.z);
        const __nv_bfloat162 v3 = *reinterpret_cast<const __nv_bfloat162*>(&raw.w);
        const float v[8] = {__low2float(v0),  __high2float(v0), __low2float(v1),  __high2float(v1),
                            __low2float(v2),  __high2float(v2), __low2float(v3),  __high2float(v3)};
        sumsq += v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3] + v[4] * v[4] + v[5] * v[5] +
                 v[6] * v[6] + v[7] * v[7];
#pragma unroll
        for (int i = 0; i < kRows; ++i) {
            const float4 wa = *reinterpret_cast<const float4*>(weight + static_cast<std::size_t>(i) * width + 8 * c);
            const float4 wb = *reinterpret_cast<const float4*>(weight + static_cast<std::size_t>(i) * width + 8 * c + 4);
            acc[i] += wa.x * v[0] + wa.y * v[1] + wa.z * v[2] + wa.w * v[3] + wb.x * v[4] +
                      wb.y * v[5] + wb.z * v[6] + wb.w * v[7];
        }
    }

    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
    for (int i = 0; i <= kRows; ++i) {
        float value = i < kRows ? acc[i] : sumsq;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_xor_sync(0xFFFFFFFFU, value, offset);
        }
        if (lane == 0) { partial[warp][i] = value; }
    }
    __syncthreads();
    if (threadIdx.x <= kRows) {
        float sum = 0.0F;
        for (int w = 0; w < kWarps; ++w) { sum += partial[w][threadIdx.x]; }
        total[threadIdx.x] = sum;
    }
    __syncthreads();

    // The router's scalar stage: one normalisation, eight logits, the two sigmoids and the 2x2
    // Sinkhorn. Everything here is four values wide, so a single thread is the cheapest schedule.
    if (threadIdx.x == 0) {
        const float r = rsqrtf(total[kRows] / static_cast<float>(width) + kMhcNormEpsilon);
        float m[kRows];
        for (int i = 0; i < kRows; ++i) { m[i] = total[i] * r + bias[i]; }

        float l[kS][kS];
        for (int i = 0; i < kS; ++i) {
            pre_bf16[i] = __bfloat162float(__float2bfloat16_rn(sigmoidf(m[i])));
            post[static_cast<std::size_t>(t) * kS + i] = 2.0F * sigmoidf(m[kS + i]);
            for (int j = 0; j < kS; ++j) { l[i][j] = m[2 * kS + i * kS + j]; }
        }
        sinkhorn_2x2(l, iters);
        for (int i = 0; i < kS; ++i) {
            for (int j = 0; j < kS; ++j) {
                mix[static_cast<std::size_t>(t) * kS * kS + i * kS + j] = l[i][j];
            }
        }
    }
    __syncthreads();

    for (int d = threadIdx.x; d < hidden; d += kThreads) {
        float sum = 0.0F;
#pragma unroll
        for (int s = 0; s < kS; ++s) {
            sum += pre_bf16[s] * __bfloat162float(residual[static_cast<std::size_t>(t) * width +
                                                           static_cast<std::size_t>(s) * hidden + d]);
        }
        branch[static_cast<std::size_t>(t) * hidden + d] = __float2bfloat16_rn(sum);
    }
}

__global__ void __launch_bounds__(kThreads)
merge_kernel(const bf16* __restrict__ residual, const bf16* __restrict__ branch,
             const float* __restrict__ post, const float* __restrict__ mix, int hidden,
             bf16* __restrict__ out) {
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    const int t = blockIdx.x;
    if (d >= hidden) { return; }
    const std::size_t column = static_cast<std::size_t>(t) * kS * hidden;
    float old[kS];
#pragma unroll
    for (int j = 0; j < kS; ++j) { old[j] = __bfloat162float(residual[column + static_cast<std::size_t>(j) * hidden + d]); }
    const float bv = __bfloat162float(branch[static_cast<std::size_t>(t) * hidden + d]);
#pragma unroll
    for (int i = 0; i < kS; ++i) {
        float mixed = 0.0F;
#pragma unroll
        for (int j = 0; j < kS; ++j) {
            const float coefficient = __bfloat162float(__float2bfloat16_rn(
                mix[static_cast<std::size_t>(t) * kS * kS + i * kS + j]));
            mixed += coefficient * old[j];
        }
        const float mixed_bf16 = __bfloat162float(__float2bfloat16_rn(mixed));
        const float post_bf16  = __bfloat162float(
            __float2bfloat16_rn(post[static_cast<std::size_t>(t) * kS + i]));
        out[column + static_cast<std::size_t>(i) * hidden + d] =
            __float2bfloat16_rn(mixed_bf16 + post_bf16 * bv);
    }
}

__global__ void __launch_bounds__(kThreads)
enter_kernel(const bf16* __restrict__ x, int hidden, bf16* __restrict__ streams) {
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    const int t = blockIdx.x;
    if (d >= hidden) { return; }
    const bf16 v = x[static_cast<std::size_t>(t) * hidden + d];
    bf16* column = streams + static_cast<std::size_t>(t) * kS * hidden;
#pragma unroll
    for (int s = 0; s < kS; ++s) { column[static_cast<std::size_t>(s) * hidden + d] = v; }
}

__global__ void __launch_bounds__(kThreads)
leave_kernel(const bf16* __restrict__ streams, int hidden, bf16* __restrict__ x) {
    const int d = blockIdx.y * blockDim.x + threadIdx.x;
    const int t = blockIdx.x;
    if (d >= hidden) { return; }
    const bf16* column = streams + static_cast<std::size_t>(t) * kS * hidden;
    float sum = 0.0F;
#pragma unroll
    for (int s = 0; s < kS; ++s) { sum += __bfloat162float(column[static_cast<std::size_t>(s) * hidden + d]); }
    x[static_cast<std::size_t>(t) * hidden + d] =
        __float2bfloat16_rn(sum / static_cast<float>(kS));
}

dim3 element_grid(int columns, int hidden) {
    return dim3(columns, static_cast<unsigned>((hidden + kThreads - 1) / kThreads));
}

} // namespace

void mhc_router(const Tensor& residual, const Tensor& weight, const Tensor& bias,
                std::int32_t mix_iters, Tensor& branch, Tensor& post, Tensor& mix,
                cudaStream_t stream) {
    require(contiguous_2d(residual, DType::BF16) && contiguous_2d(weight, DType::FP32) &&
                contiguous_2d(bias, DType::FP32) && contiguous_2d(branch, DType::BF16) &&
                contiguous_2d(post, DType::FP32) && contiguous_2d(mix, DType::FP32),
            "router needs a contiguous BF16 residual and branch and FP32 weights and outputs");
    const int width = residual.ne[0], columns = residual.ne[1];
    require(width % 8 == 0 && width > 0 && width <= kMaxStreamWidth && columns > 0 &&
                width % kS == 0,
            "the residual width must be a multiple of 8 (and of S) up to 32,768");
    const int hidden = width / kS;
    require(weight.ne[0] == width && weight.ne[1] == kRows && bias.numel() == kRows,
            "weight or bias do not match the residual width");
    require(branch.ne[0] == hidden && branch.ne[1] == columns && post.ne[0] == kS &&
                post.ne[1] == columns && mix.ne[0] == kS * kS && mix.ne[1] == columns,
            "branch, post or mix have the wrong shape");
    require(aligned16(residual.data) && aligned16(weight.data),
            "the residual and weight must be 16-byte aligned");
    require(mix_iters >= 1 && mix_iters <= kMaxIters, "mix_iters out of range");
    for (const Tensor* out : {&branch, &post, &mix}) {
        for (const Tensor* in : {&residual, &weight, &bias}) {
            require(!overlaps(*out, *in), "an output overlaps an input");
        }
    }
    require(!overlaps(branch, post) && !overlaps(branch, mix) && !overlaps(post, mix),
            "outputs overlap");
    router_kernel<<<columns, kThreads, 0, stream>>>(
        static_cast<const bf16*>(residual.data), static_cast<const float*>(weight.data),
        static_cast<const float*>(bias.data), hidden, mix_iters,
        static_cast<bf16*>(branch.data), static_cast<float*>(post.data),
        static_cast<float*>(mix.data));
    check_launch("router");
}

void mhc_merge(const Tensor& residual, const Tensor& branch, const Tensor& post, const Tensor& mix,
               Tensor& out, cudaStream_t stream) {
    require(contiguous_2d(residual, DType::BF16) && contiguous_2d(branch, DType::BF16) &&
                contiguous_2d(post, DType::FP32) && contiguous_2d(mix, DType::FP32) &&
                contiguous_2d(out, DType::BF16),
            "merge needs a contiguous BF16 residual, branch and output and FP32 post and mix");
    const int width = residual.ne[0], columns = residual.ne[1];
    require(width % kS == 0 && width > 0 && width <= kMaxStreamWidth && columns > 0,
            "the residual width must be a multiple of S up to 32,768");
    const int hidden = width / kS;
    require(hidden <= kMaxHidden, "the hidden size exceeds the element-wise launch domain");
    require(branch.ne[0] == hidden && branch.ne[1] == columns && post.ne[0] == kS &&
                post.ne[1] == columns && mix.ne[0] == kS * kS && mix.ne[1] == columns &&
                out.ne[0] == width && out.ne[1] == columns,
            "merge shapes disagree");
    // out may alias residual: every output element reads only its own column's features.
    require(!overlaps(out, branch) && !overlaps(out, post) && !overlaps(out, mix),
            "the output overlaps an input");
    require(!overlaps(branch, post) && !overlaps(branch, mix) && !overlaps(post, mix),
            "inputs overlap");
    merge_kernel<<<element_grid(columns, hidden), kThreads, 0, stream>>>(
        static_cast<const bf16*>(residual.data), static_cast<const bf16*>(branch.data),
        static_cast<const float*>(post.data), static_cast<const float*>(mix.data), hidden,
        static_cast<bf16*>(out.data));
    check_launch("merge");
}

void mhc_enter(const Tensor& x, Tensor& streams, cudaStream_t stream) {
    require(contiguous_2d(x, DType::BF16) && contiguous_2d(streams, DType::BF16),
            "enter needs contiguous BF16 tensors");
    const int hidden = x.ne[0], columns = x.ne[1];
    require(hidden > 0 && hidden <= kMaxHidden && columns > 0 && streams.ne[0] == kS * hidden &&
                streams.ne[1] == columns,
            "enter shapes disagree");
    require(!overlaps(streams, x), "the streams overlap the input");
    enter_kernel<<<element_grid(columns, hidden), kThreads, 0, stream>>>(
        static_cast<const bf16*>(x.data), hidden, static_cast<bf16*>(streams.data));
    check_launch("enter");
}

void mhc_leave(const Tensor& streams, Tensor& x, cudaStream_t stream) {
    require(contiguous_2d(streams, DType::BF16) && contiguous_2d(x, DType::BF16),
            "leave needs contiguous BF16 tensors");
    const int width = streams.ne[0], columns = streams.ne[1];
    require(width % kS == 0 && width > 0 && width <= kMaxStreamWidth && columns > 0,
            "the streams width must be a multiple of S up to 32,768");
    const int hidden = width / kS;
    require(hidden <= kMaxHidden && x.ne[0] == hidden && x.ne[1] == columns,
            "leave shapes disagree");
    require(!overlaps(x, streams), "the output overlaps the input");
    leave_kernel<<<element_grid(columns, hidden), kThreads, 0, stream>>>(
        static_cast<const bf16*>(streams.data), hidden, static_cast<bf16*>(x.data));
    check_launch("leave");
}

} // namespace ninfer::ops
