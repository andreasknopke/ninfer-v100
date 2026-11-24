// YaRN-scaled split-half NeoX RoPE for the SenseNova-Loop release; the contract is
// include/ninfer/ops/rope_yarn.h. The rotation, the layout and the axis assignment are the generic
// RoPE route's, so they come from ops/kernel/rope_math.cuh; only the frequency derivation is new.
// The ramp, the blend and the phase follow the reference implementation's
// `find_correction_range` / `linear_ramp_mask` pair, evaluated in FP64.

#include "ninfer/ops/rope_yarn.h"

#include "ops/kernel/rope_math.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr double kPi = 3.14159265358979323846;

constexpr std::int32_t kTextHeadDim  = 256;
constexpr std::int32_t kMropeRotary  = 64;
constexpr std::int32_t kMaxRotaryDim = 256;
// Half of the largest rotary_dim in the domain: the per-token cache of cos/sin pairs.
constexpr int kMaxHalf = kMaxRotaryDim / 2;
constexpr int kThreads = 256;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("rope_yarn: ") + message); }
}

void check_launch(const char* what) {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string("rope_yarn ") + what + ": " + cudaGetErrorString(error));
    }
}

bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 0xFU) == 0; }

// Half-open byte ranges [a, a+an) and [b, b+bn).
bool overlaps(const void* a, std::size_t an, const void* b, std::size_t bn) {
    const auto pa = reinterpret_cast<std::uintptr_t>(a);
    const auto pb = reinterpret_cast<std::uintptr_t>(b);
    return pa < pb + bn && pb < pa + an;
}

// The reference implementation's `find_correction_dim`: the pair index at which a head completes
// `rotations` rotations over the base context.
double correction_dim(double rotary_dim, double theta, double original_max, double rotations) {
    return rotary_dim * std::log(original_max / (rotations * 2.0 * kPi)) / (2.0 * std::log(theta));
}

struct YarnKernelParameters {
    double factor;
    double low;
    double high;
    double attention_scale;
};

// The host half of the YaRN derivation: the ramp bounds and the attention factor. A non-positive
// recorded attention factor means "not recorded", and the reference implementation derives it.
YarnKernelParameters yarn_parameters(int rotary_dim, float theta, const YarnRopeConfig& yarn) {
    require(yarn.factor > 0.0F && std::isfinite(yarn.factor), "factor must be positive and finite");
    require(yarn.original_max_position_embeddings > 0.0F &&
                std::isfinite(yarn.original_max_position_embeddings),
            "original_max_position_embeddings must be positive and finite");
    require(yarn.beta_fast > 0.0F && std::isfinite(yarn.beta_fast),
            "beta_fast must be positive and finite");
    require(yarn.beta_slow > 0.0F && std::isfinite(yarn.beta_slow),
            "beta_slow must be positive and finite");

    const double rd = static_cast<double>(rotary_dim);
    const double th = static_cast<double>(theta);
    const double original_max = static_cast<double>(yarn.original_max_position_embeddings);
    double low  = correction_dim(rd, th, original_max, static_cast<double>(yarn.beta_fast));
    double high = correction_dim(rd, th, original_max, static_cast<double>(yarn.beta_slow));
    low  = std::max(low, 0.0);
    high = std::min(high, rd - 1.0);
    // A degenerate ramp would divide by zero in the kernel; the reference implementation widens it
    // by an epsilon instead, which pins every pair to one end of the ramp.
    if (low == high) { high = low + 0.001; }

    double attention_scale;
    if (yarn.attention_factor > 0.0F && std::isfinite(yarn.attention_factor)) {
        attention_scale = static_cast<double>(yarn.attention_factor);
    } else if (yarn.factor <= 1.0F) {
        attention_scale = 1.0;
    } else {
        attention_scale = 0.1 * std::log(static_cast<double>(yarn.factor)) + 1.0;
    }
    return {static_cast<double>(yarn.factor), low, high, attention_scale};
}

// One block per token. The first `rotary_dim/2` threads derive the token's scaled cos/sin once in
// FP64 and publish them to shared memory; the remaining threads then rotate the heads, one warp per
// head, with each lane owning one pair.
__global__ void __launch_bounds__(kThreads)
rope_yarn_kernel(const std::int32_t* __restrict__ positions, std::int32_t axes,
                 bf16* __restrict__ q, bf16* __restrict__ k, std::int32_t head_dim,
                 std::int32_t rotary_dim, float theta, std::int32_t q_heads, std::int32_t k_heads,
                 std::int32_t tokens, std::int64_t q_token_stride, std::int64_t k_token_stride,
                 double factor, double low, double high, double attention_scale) {
    __shared__ float cos_cache[kMaxHalf];
    __shared__ float sin_cache[kMaxHalf];

    const int token = static_cast<int>(blockIdx.x);
    const int half  = rotary_dim >> 1;

    if (static_cast<int>(threadIdx.x) < half) {
        const int pair = static_cast<int>(threadIdx.x);
        int axis       = 0;
        float exponent = 0.0F;
        generic_axis_frequency(axes, head_dim, rotary_dim, pair, &axis, &exponent);

        // The phase passes 2^18 radians at the release context limit, so the frequency, the blend
        // and the phase stay in FP64; only the published coefficients are FP32.
        const double base = pow(static_cast<double>(theta), static_cast<double>(exponent));
        const double ramp =
            fmin(fmax((static_cast<double>(pair) - low) / (high - low), 0.0), 1.0);
        const double frequency = base * (1.0 - ramp) + (base / factor) * ramp;
        const double angle =
            static_cast<double>(positions[static_cast<std::int64_t>(axis) * tokens + token]) *
            frequency;
        double sine = 0.0;
        double cosine = 0.0;
        sincos(angle, &sine, &cosine);
        sin_cache[pair] = static_cast<float>(sine * attention_scale);
        cos_cache[pair] = static_cast<float>(cosine * attention_scale);
    }
    __syncthreads();

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int block_warps = static_cast<int>(blockDim.x) >> 5;
    for (int combined_head = warp; combined_head < q_heads + k_heads;
         combined_head += block_warps) {
        const bool is_q             = combined_head < q_heads;
        const int head              = is_q ? combined_head : combined_head - q_heads;
        bf16* __restrict__ data     = is_q ? q : k;
        const std::int64_t stride_t = is_q ? q_token_stride : k_token_stride;
        if (lane >= half) { continue; }
        const std::int64_t base_index =
            static_cast<std::int64_t>(token) * stride_t + static_cast<std::int64_t>(head) * head_dim;
        const float c = cos_cache[lane];
        const float s = sin_cache[lane];
        const float first  = __bfloat162float(data[base_index + lane]);
        const float second = __bfloat162float(data[base_index + lane + half]);
        data[base_index + lane]       = __float2bfloat16_rn(rope_first(first, second, c, s));
        data[base_index + lane + half] = __float2bfloat16_rn(rope_second(first, second, c, s));
    }
}

// One block per token, single tensor.
__global__ void __launch_bounds__(kThreads)
rope_yarn_single_kernel(const std::int32_t* __restrict__ positions, std::int32_t axes,
                        bf16* __restrict__ x, std::int32_t head_dim, std::int32_t rotary_dim,
                        float theta, std::int32_t heads, std::int32_t tokens,
                        std::int64_t token_stride, double factor, double low, double high,
                        double attention_scale) {
    __shared__ float cos_cache[kMaxHalf];
    __shared__ float sin_cache[kMaxHalf];

    const int token = static_cast<int>(blockIdx.x);
    const int half  = rotary_dim >> 1;

    if (static_cast<int>(threadIdx.x) < half) {
        const int pair = static_cast<int>(threadIdx.x);
        int axis       = 0;
        float exponent = 0.0F;
        generic_axis_frequency(axes, head_dim, rotary_dim, pair, &axis, &exponent);
        const double base = pow(static_cast<double>(theta), static_cast<double>(exponent));
        const double ramp =
            fmin(fmax((static_cast<double>(pair) - low) / (high - low), 0.0), 1.0);
        const double frequency = base * (1.0 - ramp) + (base / factor) * ramp;
        const double angle =
            static_cast<double>(positions[static_cast<std::int64_t>(axis) * tokens + token]) *
            frequency;
        double sine = 0.0;
        double cosine = 0.0;
        sincos(angle, &sine, &cosine);
        sin_cache[pair] = static_cast<float>(sine * attention_scale);
        cos_cache[pair] = static_cast<float>(cosine * attention_scale);
    }
    __syncthreads();

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int block_warps = static_cast<int>(blockDim.x) >> 5;
    for (int head = warp; head < heads; head += block_warps) {
        if (lane >= half) { continue; }
        const std::int64_t base_index =
            static_cast<std::int64_t>(token) * token_stride + static_cast<std::int64_t>(head) * head_dim;
        const float c = cos_cache[lane];
        const float s = sin_cache[lane];
        const float first  = __bfloat162float(x[base_index + lane]);
        const float second = __bfloat162float(x[base_index + lane + half]);
        x[base_index + lane] = __float2bfloat16_rn(rope_first(first, second, c, s));
        x[base_index + lane + half] = __float2bfloat16_rn(rope_second(first, second, c, s));
    }
}

// Everything both entry points check before they look at a head tensor. The configuration is
// derived here so an unusable one is rejected before any pointer is inspected.
YarnKernelParameters validate_common(const Tensor& positions, int rotary_dim, float theta,
                                     const YarnRopeConfig& yarn, std::int32_t tokens) {
    require(positions.dtype == DType::I32, "positions must be I32");
    require(theta > 0.0F && std::isfinite(theta), "theta must be positive and finite");
    require(rotary_dim > 0 && (rotary_dim & 1) == 0, "rotary_dim must be positive and even");
    require(rotary_dim <= kMaxRotaryDim, "rotary_dim must be at most 256");
    require(tokens >= 1, "tokens must be positive");
    return yarn_parameters(rotary_dim, theta, yarn);
}

int validate_positions(const Tensor& positions, std::int32_t tokens) {
    require(positions.ne[0] == tokens && positions.ne[2] == 1 && positions.ne[3] == 1 &&
                (positions.ne[1] == 1 || positions.ne[1] == 3),
            "positions must have shape [T] or [T,3]");
    require(positions.is_contiguous(), "positions must be contiguous");
    require(positions.data != nullptr, "positions data must be non-null");
    return static_cast<int>(positions.ne[1]);
}

void validate_head_tensor(const Tensor& t, std::int32_t head_dim, std::int32_t heads,
                          std::int32_t tokens) {
    require(t.dtype == DType::BF16, "head tensors must be BF16");
    require(heads > 0, "head counts must be positive");
    require(t.ne[0] == head_dim && t.ne[1] == heads && t.ne[2] == tokens && t.ne[3] == 1,
            "invalid head tensor shape");
    constexpr std::int64_t elem = 2;
    require(t.nb[0] == elem && t.nb[1] == elem * head_dim &&
                t.nb[2] >= elem * static_cast<std::int64_t>(head_dim) * heads &&
                (t.nb[2] % elem) == 0,
            "invalid head tensor strides");
    require(t.data != nullptr, "head tensor data must be non-null");
    require(aligned16(t.data), "head tensor data must be 16-byte aligned");
}

} // namespace

void rope_yarn(const Tensor& positions, int rotary_dim, float theta, const YarnRopeConfig& yarn,
               Tensor& q, Tensor& k, cudaStream_t stream) {
    const std::int32_t tokens = q.ne[2];
    const YarnKernelParameters p = validate_common(positions, rotary_dim, theta, yarn, tokens);
    const int axes = validate_positions(positions, tokens);
    if (axes == 3) {
        require(rotary_dim == kMropeRotary, "3-D Text MRoPE requires rotary_dim=64");
    }
    const std::int32_t head_dim = q.ne[0];
    require(head_dim == kTextHeadDim, "Text mode requires head_dim=256");
    const std::int32_t q_heads = q.ne[1];
    const std::int32_t k_heads = k.ne[1];
    validate_head_tensor(q, head_dim, q_heads, tokens);
    validate_head_tensor(k, head_dim, k_heads, tokens);

    const std::size_t q_bytes = static_cast<std::size_t>(q.nb[2]) * static_cast<std::size_t>(tokens);
    const std::size_t k_bytes = static_cast<std::size_t>(k.nb[2]) * static_cast<std::size_t>(tokens);
    const std::size_t pos_bytes =
        static_cast<std::size_t>(axes) * static_cast<std::size_t>(tokens) * 4U;
    require(!overlaps(q.data, q_bytes, k.data, k_bytes), "q and k must not overlap");
    require(!overlaps(q.data, q_bytes, positions.data, pos_bytes),
            "q and positions must not overlap");
    require(!overlaps(k.data, k_bytes, positions.data, pos_bytes),
            "k and positions must not overlap");

    rope_yarn_kernel<<<tokens, kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), axes,
        static_cast<bf16*>(q.data), static_cast<bf16*>(k.data), head_dim, rotary_dim, theta,
        q_heads, k_heads, tokens, q.nb[2] / 2, k.nb[2] / 2, p.factor, p.low, p.high,
        p.attention_scale);
    check_launch("launch");
}

void rope_yarn(const Tensor& positions, int rotary_dim, float theta, const YarnRopeConfig& yarn,
               Tensor& x, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[2];
    const YarnKernelParameters p = validate_common(positions, rotary_dim, theta, yarn, tokens);
    const int axes = validate_positions(positions, tokens);
    if (axes == 3) {
        require(rotary_dim == kMropeRotary, "3-D Text MRoPE requires rotary_dim=64");
    }
    const std::int32_t head_dim = x.ne[0];
    require(head_dim == kTextHeadDim, "Text mode requires head_dim=256");
    const std::int32_t heads = x.ne[1];
    validate_head_tensor(x, head_dim, heads, tokens);

    const std::size_t x_bytes = static_cast<std::size_t>(x.nb[2]) * static_cast<std::size_t>(tokens);
    const std::size_t pos_bytes =
        static_cast<std::size_t>(axes) * static_cast<std::size_t>(tokens) * 4U;
    require(!overlaps(x.data, x_bytes, positions.data, pos_bytes),
            "x and positions must not overlap");

    rope_yarn_single_kernel<<<tokens, kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), axes, static_cast<bf16*>(x.data),
        head_dim, rotary_dim, theta, heads, tokens, x.nb[2] / 2, p.factor, p.low, p.high,
        p.attention_scale);
    check_launch("launch");
}

} // namespace ninfer::ops
