#pragma once

// The multi-stream residual arithmetic of the SenseNova-Loop decoder: two residual streams carried
// through the looped block, each sublayer gated by an FP32 router whose mixing matrix is a
// Sinkhorn-normalised 2x2 matrix evaluated in log space. It is not hc_sinkhorn.h's mixer: that one
// is DeepSeek-V4's four-stream form with an additive epsilon and a linear-space Sinkhorn, while this
// family has two streams, no epsilon, a log-space Sinkhorn, and BF16 cast boundaries between the
// router and its two matmuls. The formulas come from the release `modeling_sensenova_loop.py`.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/// The residual streams of the SenseNova-Loop mHC block (mhc_mult).
inline constexpr std::int32_t kMhcStreams = 2;

/// Rows of one router: S pre, S post and S * S mix logits.
inline constexpr std::int32_t kMhcRouterRows = 2 * kMhcStreams + kMhcStreams * kMhcStreams;

/// The RMSNorm epsilon the release model hard-codes into both routers.
inline constexpr float kMhcNormEpsilon = 1.0e-6F;

/**
 * Op: mhc_router (the pre, post and mix coefficients of one residual state, and the sublayer input)
 *
 * Math / indexing:
 *   The residual holds S streams of H features per column, stream-major: R[s*H + d, t] is feature d
 *   of stream s. For every column t independently, with K = S*H and n = kMhcRouterRows:
 *     r        = rsqrt(mean_k R[k, t]^2 + kMhcNormEpsilon)      (one statistic over the whole K)
 *     m[i]     = sum_k R[k, t] * r * w[i, k] + b[i]             for i < n
 *     pre[i]   = sigmoid(m[i])                                   for i < S
 *     post[i]  = 2 * sigmoid(m[S + i])                           for i < S
 *     L[i, j]  = m[2S + i*S + j]                                 for i, j < S
 *   The mix is a Sinkhorn-Knopp projection of L onto the doubly-stochastic matrices, run in log
 *   space. With LSE(X) over an axis defined as max + log(sum(exp(X - max))):
 *     repeat `mix_iters` times:
 *       L[i, :] = L[i, :] - LSE_row(L[i, :])
 *       L[:, j] = L[:, j] - LSE_col(L[:, j])
 *     mix[i, j] = exp(L[i, j])
 *   The branch is the pre-weighted collapse of the streams, and it is the only place `pre` is used:
 *     branch[d, t] = bf16( sum_s bf16(pre[s, t]) * R[s*H + d, t] )
 *   with no epsilon anywhere in the router.
 *
 * Logical shapes:
 *   residual BF16 [S*H, T]; weight FP32 [S*H, n] stored row-major per logit, w[i, k] at
 *   weight[i*K + k]; bias FP32 [n]; branch BF16 [H, T]; post FP32 [S, T] at post[t*S + i];
 *   mix FP32 [S*S, T] at mix[t*S*S + i*S + j]. S = kMhcStreams; T >= 1.
 *
 * Supported domain:
 *   Contiguous tensors; H = 4096 (so K = 8192) is the registered production geometry and any H with
 *   K a multiple of 8 and at most 32,768 is served; 16-byte aligned residual and weight; 1 <=
 *   mix_iters <= 64. T is any positive value (columns are independent).
 *
 * Numeric:
 *   FP32 arithmetic from the represented BF16 residual and FP32 router weights. The statistic, the
 *   projection and the log-space Sinkhorn are FP32. `pre` is rounded to BF16 before the branch
 *   sum, which is the reference model's cast boundary and is observable in `branch`. The oracle is
 *   the same formula in FP64 with the same BF16 cast boundaries.
 *
 * Effects:
 *   Writes branch, post and mix. residual, weight and bias are read only and no output may overlap
 *   an input or another output.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void mhc_router(const Tensor& residual, const Tensor& weight, const Tensor& bias,
                std::int32_t mix_iters, Tensor& branch, Tensor& post, Tensor& mix,
                cudaStream_t stream);

/**
 * Op: mhc_merge (a sublayer's branch folded back into the streams)
 *
 * Math / indexing:
 *   For every stream i, feature d and column t, with R the residual before the call:
 *     mixed[i, d, t]  = bf16( sum_j bf16(mix[i, j, t]) * R[j*H + d, t] )
 *     out[i*H + d, t] = bf16( mixed[i, d, t] + bf16(post[i, t]) * branch[d, t] )
 *   Both the residual mix and the post gate are rounded to BF16 before their products, matching the
 *   reference model's matmul and broadcast casts.
 *
 * Logical shapes:
 *   residual BF16 [S*H, T]; branch BF16 [H, T]; post FP32 [S, T] and mix FP32 [S*S, T] in
 *   mhc_router's layout; out BF16 [S*H, T]. S = kMhcStreams; T >= 1.
 *
 * Supported domain: contiguous tensors; any H with 1 <= H <= 16,776,960 and S*H at most 32,768.
 *
 * Numeric: FP32 products and sums from the represented values, with the two BF16 boundaries above
 *   and one final rounding. The oracle reproduces those boundaries in FP64.
 *
 * Effects:
 *   Writes out. residual, branch, post and mix are read only; out may alias residual, in which case
 *   every output element reads only its own column, so an in-place call is well defined. No other
 *   output may overlap an input.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void mhc_merge(const Tensor& residual, const Tensor& branch, const Tensor& post, const Tensor& mix,
               Tensor& out, cudaStream_t stream);

/**
 * Op: mhc_enter (one stream opened into two identical streams)
 *
 * Math / indexing:
 *   R[s*H + d, t] = x[d, t] for every s < S.
 *
 * Logical shapes: x BF16 [H, T]; streams BF16 [S*H, T]. S = kMhcStreams; T >= 1.
 *
 * Supported domain: contiguous tensors; any H with 1 <= H <= 16,776,960.
 *
 * Numeric: exact copy, no rounding.
 *
 * Effects: writes streams; x is read only and must not overlap it.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void mhc_enter(const Tensor& x, Tensor& streams, cudaStream_t stream);

/**
 * Op: mhc_leave (the streams averaged back into one)
 *
 * Math / indexing:
 *   x[d, t] = bf16( (1/S) * sum_s R[s*H + d, t] )
 *
 * Logical shapes: streams BF16 [S*H, T]; x BF16 [H, T]. S = kMhcStreams; T >= 1.
 *
 * Supported domain: contiguous tensors; any H with 1 <= H <= 16,776,960.
 *
 * Numeric: the sum and the division are FP32 from the represented values, rounded once to BF16.
 *
 * Effects: writes x; streams is read only and must not overlap it.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void mhc_leave(const Tensor& streams, Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
