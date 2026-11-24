#pragma once

// The YaRN-scaled form of the generic RoPE route. The SenseNova-Loop release asks for YaRN at every
// position of a partial MRoPE head, and the scaling changes only how a pair's frequency is derived:
// the pair's base frequency is blended towards base/factor across a linear ramp, and the resulting
// cosine and sine are scaled by an attention factor. The rotation itself, the tensor layout and the
// axis assignment are rope.h's unchanged, so this family shares rope_math.cuh with it rather than
// restating them. rope.h's own entry points stay unscaled; Qwen and Inkling heads never enter here.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * One YaRN rope configuration, as the release `rope_parameters` records it.
 *
 * `factor` is the YaRN scaling factor and must exceed zero; `original_max_position_embeddings` is
 * the context length the base model was trained on. `beta_fast` and `beta_slow` are the rotation
 * counts that bound the frequency ramp; the release leaves them out of the checkpoint and uses the
 * reference implementation's defaults 32 and 1.
 *
 * `attention_factor` is the multiplier applied to cosine and sine. A value that is not positive
 * means "not recorded", and the op derives it the way the reference implementation does: 1 for
 * `factor <= 1`, and 0.1 * ln(factor) + 1 otherwise.
 */
struct YarnRopeConfig {
    float factor;
    float original_max_position_embeddings;
    float beta_fast;
    float beta_slow;
    float attention_factor;
};

/**
 * Op: rope_yarn (YaRN-scaled split-half NeoX RoPE, applied in place)
 *
 * Math / indexing:
 *   rope.h's rotation and indexing, with a different frequency. For pair i in [0, R/2), where
 *   R = rotary_dim, and each head:
 *     ideal[i]              = x[i] * cos(phi) - x[i+R/2] * sin(phi)
 *     ideal[i+R/2]          = x[i+R/2] * cos(phi) + x[i] * sin(phi)
 *   Dimensions [R, head_dim) are unchanged. Supported modes are:
 *   - Text 1-D: positions I32 [T]; phi = position(t) * f(i).
 *   - Text MRoPE: positions I32 [T,3]; pair i uses axis i%3 with the same f(i). The release's
 *     `mrope_section` of [11,11,10] over R/2 = 32 pairs is exactly this assignment.
 *   The frequency f(i) is the YaRN blend of the base frequency g(i) = theta^(-2i/R) with the
 *   linearly-interpolated g(i)/factor:
 *     low  = R * ln(original_max_position_embeddings / (beta_fast * 2*pi)) / (2 * ln(theta))
 *     high = R * ln(original_max_position_embeddings / (beta_slow  * 2*pi)) / (2 * ln(theta))
 *     ramp = clamp((i - low) / (high - low), 0, 1)
 *     f(i) = g(i) * (1 - ramp) + (g(i) / factor) * ramp
 *   and phi's cosine and sine are each multiplied by the attention factor above.
 *
 * Logical shapes:
 *   positions I32 [T] or [T,3]; q BF16 [head_dim, q_heads, T]; k BF16 [head_dim, k_heads, T].
 *
 * Supported domain:
 *   head_dim = 256 with even 0 < rotary_dim <= 256; positive head counts; T >= 1; theta > 0 and
 *   finite; `YarnRopeConfig` with positive finite factor, original_max_position_embeddings,
 *   beta_fast and beta_slow. Contiguous tensors with contiguous head features and heads, an
 *   optional padded token stride, and 16-byte aligned q and k. positions is contiguous and must not
 *   overlap q or k; q and k must not overlap one another.
 *
 * Numeric:
 *   The ramp, the blended frequency, the phase and the attention factor are FP64, because the
 *   release context puts the phase above 2^18 radians and FP32 range reduction there drifts away
 *   from the naive value. Cosine and sine are rounded to FP32 once, after the attention factor, and
 *   the rotation of the represented BF16 inputs is FP32, so the result is one BF16 rounding of the
 *   FP32 contraction. The oracle is the same formula in FP64 from the represented inputs.
 *   Unrotated dimensions remain bit-exact. Private kernel arithmetic is implementation-defined.
 *
 * Effects:
 *   Mutates only dimensions [0, rotary_dim) of the supplied q and k storage. positions is read only.
 *
 * Workspace: none. Execution: stream-ordered; graph capturable.
 */
void rope_yarn(const Tensor& positions, int rotary_dim, float theta, const YarnRopeConfig& yarn,
               Tensor& q, Tensor& k, cudaStream_t stream);

// Single-tensor form with the same formula and storage contract. The head count comes directly
// from x; Q versus K role does not change the transformation.
void rope_yarn(const Tensor& positions, int rotary_dim, float theta, const YarnRopeConfig& yarn,
               Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
