// Qualifies rope_yarn against the documented formula evaluated in FP64, and against the unscaled
// rope op where the two must agree. The release configuration is the one SenseNova-Loop ships with,
// so its derived ramp bounds are pinned here as literals: a change to the derivation is a contract
// change and has to fail this test rather than silently move the ramp.

#include "core/device.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/rope_yarn.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr float kTheta = 1.0e7F;
constexpr int kHeadDim = 256;
constexpr int kRotary  = 64;
constexpr double kPairRtol = 6.9e-3;

// The release `rope_parameters` records only rope_type, factor, original_max_position_embeddings and
// partial_rotary_factor; beta_fast, beta_slow and attention_factor take the reference defaults.
constexpr ops::YarnRopeConfig kReleaseConfig{2.0F, 262144.0F, 32.0F, 1.0F, 0.0F};

// The ramp bounds the release configuration derives. Pairs below kLowKeep are left at the base
// frequency, pairs above kHighFull are divided by the factor in full, and the pairs in between ride
// the linear ramp.
constexpr double kExpectedLow  = 14.240960344108071;
constexpr double kExpectedHigh = 21.121645959284784;
constexpr double kExpectedAttention = 1.0693147180559945;

struct Geometry {
    const char* label;
    int rotary_dim;
    int axes;
    int tokens;
};

std::size_t dense_elements(int heads, int tokens) {
    return static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(heads) *
           static_cast<std::size_t>(tokens);
}

std::size_t dense_index(int heads, int token, int head, int dim) {
    return (static_cast<std::size_t>(token) * static_cast<std::size_t>(heads) +
            static_cast<std::size_t>(head)) *
               static_cast<std::size_t>(kHeadDim) +
           static_cast<std::size_t>(dim);
}

std::vector<float> make_bf16_input(std::size_t elements, std::uint32_t seed) {
    std::vector<float> input(elements);
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-4.0F, 4.0F);
    for (float& value : input) { value = bf16_to_f32(f32_to_bf16(distribution(generator))); }
    return input;
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(),
                   [](float value) { return f32_to_bf16(value); });
    return bits;
}

std::vector<int> make_positions(int axes, int tokens, int first_position) {
    std::vector<int> positions(static_cast<std::size_t>(axes) * tokens);
    for (int axis = 0; axis < axes; ++axis) {
        for (int token = 0; token < tokens; ++token) {
            positions[static_cast<std::size_t>(axis) * tokens + token] =
                first_position + 97 * axis + (2 * axis + 1) * token;
        }
    }
    return positions;
}

struct YarnBounds {
    double factor;
    double low;
    double high;
    double attention_scale;
};

// The documented host-side derivation, in FP64.
YarnBounds yarn_bounds(int rotary_dim, float theta, const ops::YarnRopeConfig& yarn) {
    const double rd = static_cast<double>(rotary_dim);
    const double th = static_cast<double>(theta);
    const double om = static_cast<double>(yarn.original_max_position_embeddings);
    const auto correction = [&](double rotations) {
        return rd * std::log(om / (rotations * 2.0 * 3.14159265358979323846)) / (2.0 * std::log(th));
    };
    double low  = std::max(correction(static_cast<double>(yarn.beta_fast)), 0.0);
    double high = std::min(correction(static_cast<double>(yarn.beta_slow)), rd - 1.0);
    if (low == high) { high = low + 0.001; }
    double attention_scale;
    if (yarn.attention_factor > 0.0F) {
        attention_scale = static_cast<double>(yarn.attention_factor);
    } else if (yarn.factor <= 1.0F) {
        attention_scale = 1.0;
    } else {
        attention_scale = 0.1 * std::log(static_cast<double>(yarn.factor)) + 1.0;
    }
    return {static_cast<double>(yarn.factor), low, high, attention_scale};
}

// The oracle: the documented rotation with the YaRN frequency, evaluated naively in FP64 from the
// represented BF16 inputs. It does not reproduce output storage rounding or kernel arithmetic.
std::vector<double> yarn_oracle(const std::vector<float>& input, const std::vector<int>& positions,
                                const Geometry& geometry, int heads, const YarnBounds& bounds) {
    std::vector<double> output(input.begin(), input.end());
    const int half = geometry.rotary_dim / 2;
    for (int token = 0; token < geometry.tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int pair = 0; pair < half; ++pair) {
                const int axis     = geometry.axes == 3 ? pair % 3 : 0;
                const double base  = std::pow(static_cast<double>(kTheta),
                                             -2.0 * static_cast<double>(pair) / geometry.rotary_dim);
                const double ramp  = std::min(
                    std::max((static_cast<double>(pair) - bounds.low) / (bounds.high - bounds.low),
                             0.0),
                    1.0);
                const double frequency =
                    base * (1.0 - ramp) + (base / bounds.factor) * ramp;
                const double phase =
                    static_cast<double>(
                        positions[static_cast<std::size_t>(axis) * geometry.tokens + token]) *
                    frequency;
                const double cosine = std::cos(phase) * bounds.attention_scale;
                const double sine   = std::sin(phase) * bounds.attention_scale;
                const std::size_t lo = dense_index(heads, token, head, pair);
                const std::size_t hi = dense_index(heads, token, head, pair + half);
                const double first   = static_cast<double>(input[lo]);
                const double second  = static_cast<double>(input[hi]);
                output[lo]           = first * cosine - second * sine;
                output[hi]           = second * cosine + first * sine;
            }
        }
    }
    return output;
}

std::vector<std::uint16_t> make_strided_storage(const std::vector<std::uint16_t>& dense,
                                                int dense_per_token, int token_stride, int tokens,
                                                std::uint16_t padding) {
    std::vector<std::uint16_t> storage(static_cast<std::size_t>(token_stride) * tokens, padding);
    for (int token = 0; token < tokens; ++token) {
        std::copy_n(dense.data() + static_cast<std::size_t>(token) * dense_per_token,
                    dense_per_token, storage.data() + static_cast<std::size_t>(token) * token_stride);
    }
    return storage;
}

std::vector<std::uint16_t> read_back(const GuardedDeviceBuffer& buffer, int dense_per_token,
                                     int token_stride, int tokens) {
    std::vector<std::uint16_t> storage(buffer.bytes() / sizeof(std::uint16_t));
    cuda_check(cudaMemcpy(storage.data(), buffer.data(), buffer.bytes(), cudaMemcpyDeviceToHost),
               "copy back");
    std::vector<std::uint16_t> dense(static_cast<std::size_t>(dense_per_token) * tokens);
    for (int token = 0; token < tokens; ++token) {
        std::copy_n(storage.data() + static_cast<std::size_t>(token) * token_stride,
                    dense_per_token,
                    dense.data() + static_cast<std::size_t>(token) * dense_per_token);
    }
    return dense;
}

// The house RoPE profile: each rotated output is bounded relative to the FP64 norm of its public
// input pair, which is scale-invariant under the cancellation a rotation can produce.
int verify_profile(const std::string& label, const std::vector<std::uint16_t>& got_bits,
                   const std::vector<double>& expected, const std::vector<float>& input,
                   const Geometry& geometry, int heads) {
    const int half = geometry.rotary_dim / 2;
    double worst   = 0.0;
    int violations = 0;
    for (int token = 0; token < geometry.tokens; ++token) {
        for (int head = 0; head < heads; ++head) {
            for (int pair = 0; pair < half; ++pair) {
                for (int member = 0; member < 2; ++member) {
                    const int dim           = member == 0 ? pair : pair + half;
                    const std::size_t index = dense_index(heads, token, head, dim);
                    const double got        = bf16_to_f32(got_bits[index]);
                    const double want       = expected[index];
                    if (!std::isfinite(got) || !std::isfinite(want)) {
                        std::cerr << label << ": non-finite output at index=" << index << '\n';
                        return 1;
                    }
                    const double pair_norm = std::hypot(
                        static_cast<double>(input[dense_index(heads, token, head, pair)]),
                        static_cast<double>(input[dense_index(heads, token, head, pair + half)]));
                    const double abs_error = std::abs(got - want);
                    const double required  = pair_norm == 0.0
                                                 ? (abs_error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                                                 : abs_error / pair_norm;
                    worst = std::max(worst, required);
                    if (required > kPairRtol) { ++violations; }
                }
            }
        }
    }
    if (violations != 0) {
        std::cerr << label << ": " << violations << " pointwise violations, worst pair-rtol="
                  << worst << '\n';
        return 1;
    }
    return 0;
}

// Dimensions [rotary_dim, head_dim) and the token padding must survive untouched, bit for bit.
int verify_untouched(const std::string& label, const std::vector<std::uint16_t>& before_storage,
                     const std::vector<std::uint16_t>& after_storage, int dense_per_token,
                     int token_stride, int tokens, int rotary_dim, int heads) {
    for (int token = 0; token < tokens; ++token) {
        for (int element = 0; element < token_stride; ++element) {
            const int dim = element % kHeadDim;
            const bool rotatable =
                element < dense_per_token && dim < rotary_dim;
            if (rotatable) { continue; }
            const std::size_t index = static_cast<std::size_t>(token) * token_stride + element;
            if (before_storage[index] != after_storage[index]) {
                std::cerr << label << ": untouched element changed at token=" << token
                          << " element=" << element << '\n';
                return 1;
            }
        }
    }
    (void)heads;
    return 0;
}

int run_pair_case(DeviceExecutionView execution, const Geometry& geometry,
                  const ops::YarnRopeConfig& config, int q_heads, int k_heads, int first_position,
                  int q_padding = 0, int k_padding = 0) {
    constexpr std::uint16_t kPadding = 0x3f81U;
    const int q_dense_per_token      = kHeadDim * q_heads;
    const int k_dense_per_token      = kHeadDim * k_heads;
    const int q_stride               = q_dense_per_token + q_padding;
    const int k_stride               = k_dense_per_token + k_padding;

    const auto q = make_bf16_input(dense_elements(q_heads, geometry.tokens), 0x1001U);
    const auto k = make_bf16_input(dense_elements(k_heads, geometry.tokens), 0x2001U);
    const auto q_before = to_bf16_bits(q);
    const auto k_before = to_bf16_bits(k);
    const auto q_storage =
        make_strided_storage(q_before, q_dense_per_token, q_stride, geometry.tokens, kPadding);
    const auto k_storage =
        make_strided_storage(k_before, k_dense_per_token, k_stride, geometry.tokens, kPadding);
    const auto positions = make_positions(geometry.axes, geometry.tokens, first_position);

    const YarnBounds bounds = yarn_bounds(geometry.rotary_dim, kTheta, config);
    const auto q_expected   = yarn_oracle(q, positions, geometry, q_heads, bounds);
    const auto k_expected   = yarn_oracle(k, positions, geometry, k_heads, bounds);

    GuardedDeviceBuffer q_device(q_storage.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer k_device(k_storage.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer position_device(positions.size() * sizeof(int));
    q_device.copy_from_host(q_storage.data(), q_device.bytes());
    k_device.copy_from_host(k_storage.data(), k_device.bytes());
    position_device.copy_from_host(positions.data(), position_device.bytes());

    Tensor position_tensor(position_device.data(), DType::I32,
                           {geometry.tokens, geometry.axes});
    Tensor q_tensor(q_device.data(), DType::BF16, {kHeadDim, q_heads, geometry.tokens});
    Tensor k_tensor(k_device.data(), DType::BF16, {kHeadDim, k_heads, geometry.tokens});
    q_tensor.nb[2] = static_cast<std::int64_t>(q_stride) * sizeof(std::uint16_t);
    k_tensor.nb[2] = static_cast<std::int64_t>(k_stride) * sizeof(std::uint16_t);

    ops::rope_yarn(position_tensor, geometry.rotary_dim, kTheta, config, q_tensor, k_tensor,
                   execution.stream);
    cuda_synchronize();

    int failures = 0;
    failures += verify_profile(std::string(geometry.label) + " q",
                               read_back(q_device, q_dense_per_token, q_stride, geometry.tokens),
                               q_expected, q, geometry, q_heads);
    failures += verify_profile(std::string(geometry.label) + " k",
                               read_back(k_device, k_dense_per_token, k_stride, geometry.tokens),
                               k_expected, k, geometry, k_heads);
    failures +=
        verify_untouched(std::string(geometry.label) + " q", q_storage,
                         [&] {
                             std::vector<std::uint16_t> s(q_device.bytes() / sizeof(std::uint16_t));
                             cuda_check(cudaMemcpy(s.data(), q_device.data(), q_device.bytes(),
                                                    cudaMemcpyDeviceToHost),
                                        "copy back");
                             return s;
                         }(),
                         q_dense_per_token, q_stride, geometry.tokens, geometry.rotary_dim, q_heads);
    failures +=
        verify_untouched(std::string(geometry.label) + " k", k_storage,
                         [&] {
                             std::vector<std::uint16_t> s(k_device.bytes() / sizeof(std::uint16_t));
                             cuda_check(cudaMemcpy(s.data(), k_device.data(), k_device.bytes(),
                                                    cudaMemcpyDeviceToHost),
                                        "copy back");
                             return s;
                         }(),
                         k_dense_per_token, k_stride, geometry.tokens, geometry.rotary_dim, k_heads);
    return failures;
}

// The single-tensor form must agree with the pair form on the same input, bit for bit. The pair
// form gets two separate tensors holding the same data, because q and k may not overlap.
int run_single_matches_pair(DeviceExecutionView execution) {
    const Geometry geometry{"single vs pair", kRotary, 3, 11};
    constexpr int heads = 4;

    const auto x = make_bf16_input(dense_elements(heads, geometry.tokens), 0x3001U);
    const auto bits = to_bf16_bits(x);
    const auto positions = make_positions(geometry.axes, geometry.tokens, 65'536);

    GuardedDeviceBuffer q_buffer(bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer k_buffer(bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer single_buffer(bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer position_device(positions.size() * sizeof(int));
    q_buffer.copy_from_host(bits.data(), q_buffer.bytes());
    k_buffer.copy_from_host(bits.data(), k_buffer.bytes());
    single_buffer.copy_from_host(bits.data(), single_buffer.bytes());
    position_device.copy_from_host(positions.data(), position_device.bytes());

    Tensor position_tensor(position_device.data(), DType::I32, {geometry.tokens, geometry.axes});
    Tensor q_tensor(q_buffer.data(), DType::BF16, {kHeadDim, heads, geometry.tokens});
    Tensor k_tensor(k_buffer.data(), DType::BF16, {kHeadDim, heads, geometry.tokens});
    Tensor single_tensor(single_buffer.data(), DType::BF16, {kHeadDim, heads, geometry.tokens});

    ops::rope_yarn(position_tensor, kRotary, kTheta, kReleaseConfig, q_tensor, k_tensor,
                   execution.stream);
    ops::rope_yarn(position_tensor, kRotary, kTheta, kReleaseConfig, single_tensor,
                   execution.stream);
    cuda_synchronize();

    std::vector<std::uint16_t> a(q_buffer.bytes() / sizeof(std::uint16_t));
    std::vector<std::uint16_t> b(single_buffer.bytes() / sizeof(std::uint16_t));
    cuda_check(cudaMemcpy(a.data(), q_buffer.data(), q_buffer.bytes(), cudaMemcpyDeviceToHost),
               "copy back a");
    cuda_check(cudaMemcpy(b.data(), single_buffer.data(), single_buffer.bytes(), cudaMemcpyDeviceToHost),
               "copy back b");
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            std::cerr << "single vs pair: bit mismatch at " << i << '\n';
            return 1;
        }
    }
    return 0;
}

// With factor 1 the blend is the base frequency and the attention factor is 1, so YaRN must reduce
// to the unscaled op exactly. This pins the rotation, the layout and the axis assignment to rope.h.
int run_factor_one_matches_plain_rope(DeviceExecutionView execution) {
    int failures = 0;
    for (int axes : {1, 3}) {
        for (int tokens : {1, 5, 40}) {
            const Geometry geometry{axes == 1 ? "factor 1 text" : "factor 1 mrope", kRotary, axes,
                                    tokens};
            constexpr ops::YarnRopeConfig identity{1.0F, 262144.0F, 32.0F, 1.0F, 0.0F};
            constexpr int q_heads = 16, k_heads = 4;

            const auto q = make_bf16_input(dense_elements(q_heads, tokens), 0x4001U);
            const auto k = make_bf16_input(dense_elements(k_heads, tokens), 0x5001U);
            const auto q_bits = to_bf16_bits(q);
            const auto k_bits = to_bf16_bits(k);
            const auto positions = make_positions(axes, tokens, 131'072);

            GuardedDeviceBuffer qa(q_bits.size() * sizeof(std::uint16_t));
            GuardedDeviceBuffer ka(k_bits.size() * sizeof(std::uint16_t));
            GuardedDeviceBuffer qb(q_bits.size() * sizeof(std::uint16_t));
            GuardedDeviceBuffer kb(k_bits.size() * sizeof(std::uint16_t));
            GuardedDeviceBuffer position_device(positions.size() * sizeof(int));
            qa.copy_from_host(q_bits.data(), qa.bytes());
            ka.copy_from_host(k_bits.data(), ka.bytes());
            qb.copy_from_host(q_bits.data(), qb.bytes());
            kb.copy_from_host(k_bits.data(), kb.bytes());
            position_device.copy_from_host(positions.data(), position_device.bytes());

            Tensor pos_a(position_device.data(), DType::I32, {tokens, axes});
            Tensor pos_b(position_device.data(), DType::I32, {tokens, axes});
            Tensor q_a(qa.data(), DType::BF16, {kHeadDim, q_heads, tokens});
            Tensor k_a(ka.data(), DType::BF16, {kHeadDim, k_heads, tokens});
            Tensor q_b(qb.data(), DType::BF16, {kHeadDim, q_heads, tokens});
            Tensor k_b(kb.data(), DType::BF16, {kHeadDim, k_heads, tokens});

            ops::rope_yarn(pos_a, kRotary, kTheta, identity, q_a, k_a, execution.stream);
            ops::rope(pos_b, kRotary, kTheta, q_b, k_b, execution);
            cuda_synchronize();

            const auto compare = [&](const GuardedDeviceBuffer& yarn_side,
                                     const GuardedDeviceBuffer& plain_side, const char* label) {
                std::vector<std::uint16_t> x(yarn_side.bytes() / sizeof(std::uint16_t));
                std::vector<std::uint16_t> y(plain_side.bytes() / sizeof(std::uint16_t));
                cuda_check(cudaMemcpy(x.data(), yarn_side.data(), yarn_side.bytes(),
                                      cudaMemcpyDeviceToHost),
                           "copy back x");
                cuda_check(cudaMemcpy(y.data(), plain_side.data(), plain_side.bytes(),
                                      cudaMemcpyDeviceToHost),
                           "copy back y");
                if (x != y) {
                    std::cerr << "factor 1 vs plain rope: " << label << " differs ("
                              << geometry.label << ", T=" << tokens << ")\n";
                    return 1;
                }
                return 0;
            };
            failures += compare(qa, qb, "q");
            failures += compare(ka, kb, "k");
        }
    }
    return failures;
}

// The ramp must actually be non-degenerate for the release configuration: the low pairs keep the
// base frequency and the high pairs are divided by the factor in full. Checked through the op by
// comparing against two oracles that pin the ramp at one end.
int run_ramp_endpoints(DeviceExecutionView execution) {
    const Geometry geometry{"ramp endpoints", kRotary, 1, 3};
    constexpr int heads = 2;
    const YarnBounds bounds = yarn_bounds(kRotary, kTheta, kReleaseConfig);

    if (std::abs(bounds.low - kExpectedLow) > 1e-9 ||
        std::abs(bounds.high - kExpectedHigh) > 1e-9 ||
        std::abs(bounds.attention_scale - kExpectedAttention) > 1e-12) {
        std::cerr << "ramp bounds drifted: low=" << bounds.low << " high=" << bounds.high
                  << " attention=" << bounds.attention_scale << '\n';
        return 1;
    }
    // The release ramp really does split the pairs, otherwise the two oracles below are the same test.
    if (!(bounds.low > 0.0 && bounds.high < (kRotary / 2 - 1))) {
        std::cerr << "ramp is degenerate, low=" << bounds.low << " high=" << bounds.high << '\n';
        return 1;
    }

    const auto x = make_bf16_input(dense_elements(heads, geometry.tokens), 0x6001U);
    const auto bits = to_bf16_bits(x);
    const auto positions = make_positions(1, geometry.tokens, 262'144);

    GuardedDeviceBuffer device(bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer position_device(positions.size() * sizeof(int));
    device.copy_from_host(bits.data(), device.bytes());
    position_device.copy_from_host(positions.data(), position_device.bytes());

    Tensor position_tensor(position_device.data(), DType::I32, {geometry.tokens});
    Tensor tensor(device.data(), DType::BF16, {kHeadDim, heads, geometry.tokens});
    ops::rope_yarn(position_tensor, kRotary, kTheta, kReleaseConfig, tensor, execution.stream);
    cuda_synchronize();

    std::vector<std::uint16_t> got(device.bytes() / sizeof(std::uint16_t));
    cuda_check(cudaMemcpy(got.data(), device.data(), device.bytes(), cudaMemcpyDeviceToHost),
               "copy back");

    const int half = kRotary / 2;
    int failures   = 0;
    for (int pair = 0; pair < half; ++pair) {
        const double pinned_ramp = pair <= std::floor(bounds.low) ? 0.0 : 1.0;
        const bool pinned        = pair <= std::floor(bounds.low) ||
                            pair >= std::ceil(bounds.high);
        if (!pinned) { continue; }
        const double base =
            std::pow(static_cast<double>(kTheta), -2.0 * static_cast<double>(pair) / kRotary);
        const double frequency =
            base * (1.0 - pinned_ramp) + (base / bounds.factor) * pinned_ramp;
        for (int token = 0; token < geometry.tokens; ++token) {
            for (int head = 0; head < heads; ++head) {
                const double phase =
                    static_cast<double>(positions[token]) * frequency;
                const double cosine = std::cos(phase) * bounds.attention_scale;
                const double sine   = std::sin(phase) * bounds.attention_scale;
                const std::size_t lo = dense_index(heads, token, head, pair);
                const std::size_t hi = dense_index(heads, token, head, pair + half);
                const double first   = static_cast<double>(x[lo]);
                const double second  = static_cast<double>(x[hi]);
                const double want_lo = first * cosine - second * sine;
                const double want_hi = second * cosine + first * sine;
                const double got_lo  = bf16_to_f32(got[lo]);
                const double got_hi  = bf16_to_f32(got[hi]);
                const double norm    = std::hypot(first, second);
                if (std::abs(got_lo - want_lo) > kPairRtol * norm ||
                    std::abs(got_hi - want_hi) > kPairRtol * norm) {
                    std::cerr << "ramp endpoint mismatch at pair=" << pair << " token=" << token
                              << " head=" << head << " (ramp pinned to " << pinned_ramp << ")\n";
                    ++failures;
                }
            }
        }
    }
    return failures;
}

int run_refusals(DeviceExecutionView execution) {
    constexpr int tokens = 2, heads = 2;
    GuardedDeviceBuffer x(static_cast<std::size_t>(kHeadDim) * heads * tokens * 2);
    GuardedDeviceBuffer positions(tokens * 3 * sizeof(int));
    x.fill(0);
    positions.fill(0);

    const auto refuses = [](auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            return true;
        } catch (...) {
            return false;
        }
        return false;
    };

    int failures = 0;
    const auto check = [&](bool refused, const char* what) {
        if (!refused) {
            std::cerr << "refusal missing: " << what << '\n';
            ++failures;
        }
    };

    const Tensor pos1(positions.data(), DType::I32, {tokens});
    const Tensor pos3(positions.data(), DType::I32, {tokens, 3});
    const Tensor pos2(positions.data(), DType::I32, {tokens, 2});
    Tensor x_t(x.data(), DType::BF16, {kHeadDim, heads, tokens});
    Tensor bad_dim(x.data(), DType::BF16, {128, heads, tokens});
    Tensor bad_dtype(x.data(), DType::FP32, {kHeadDim, heads, tokens});

    check(refuses([&] {
              ops::rope_yarn(pos2, kRotary, kTheta, kReleaseConfig, x_t, execution.stream);
          }),
          "a 2-axis positions tensor is refused");
    check(refuses([&] {
              ops::rope_yarn(pos3, 62, kTheta, kReleaseConfig, x_t, execution.stream);
          }),
          "3-axis MRoPE with rotary_dim != 64 is refused");
    check(refuses([&] {
              ops::rope_yarn(pos1, kRotary, kTheta, kReleaseConfig, bad_dim, execution.stream);
          }),
          "a non-256 head dimension is refused");
    check(refuses([&] {
              ops::rope_yarn(pos1, kRotary, kTheta, kReleaseConfig, bad_dtype, execution.stream);
          }),
          "a non-BF16 head tensor is refused");
    check(refuses([&] {
              ops::rope_yarn(pos1, 0, kTheta, kReleaseConfig, x_t, execution.stream);
          }),
          "rotary_dim 0 is refused");
    check(refuses([&] {
              ops::rope_yarn(pos1, kRotary, -1.0F, kReleaseConfig, x_t, execution.stream);
          }),
          "a non-positive theta is refused");
    check(refuses([&] {
              ops::rope_yarn(pos1, kRotary, kTheta, ops::YarnRopeConfig{0.0F, 262144.0F, 32.0F, 1.0F, 0.0F},
                             x_t, execution.stream);
          }),
          "a non-positive YaRN factor is refused");
    check(refuses([&] {
              ops::rope_yarn(pos1, kRotary, kTheta, ops::YarnRopeConfig{2.0F, 0.0F, 32.0F, 1.0F, 0.0F},
                             x_t, execution.stream);
          }),
          "a non-positive original context is refused");
    check(refuses([&] {
              ops::rope_yarn(pos1, kRotary, kTheta, ops::YarnRopeConfig{2.0F, 262144.0F, 0.0F, 1.0F, 0.0F},
                             x_t, execution.stream);
          }),
          "a non-positive beta_fast is refused");
    (void)execution;
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    DeviceContext device;
    const auto execution = device.execution_view().on_stream(nullptr);
    int failures         = 0;

    // The release geometry: 16 Q / 4 KV heads over a 256-wide head with 64 rotated dimensions.
    // Decode, short prefill and a padded-stride prefill, in both 1-D and 3-axis MRoPE.
    for (int axes : {1, 3}) {
        for (int tokens : {1, 2, 7, 128}) {
            failures += run_pair_case(execution,
                                      {axes == 1 ? "release text" : "release mrope", kRotary, axes,
                                       tokens},
                                      kReleaseConfig, 16, 4, 4096);
        }
    }
    failures += run_pair_case(execution, {"release padded stride", kRotary, 1, 8}, kReleaseConfig,
                              16, 4, 4096, 16, 8);

    // The documented position checkpoints, including the two that straddle the base context limit
    // and the native-context tail. These are where an FP32 phase would drift.
    for (int axes : {1, 3}) {
        for (int position : {0, 8192, 262143, 262144, 524281, 524287}) {
            failures += run_pair_case(
                execution,
                {axes == 1 ? "checkpoint text" : "checkpoint mrope", kRotary, axes, 1},
                kReleaseConfig, 16, 4, position);
        }
    }
    // A prefill that runs across the base-context boundary inside one call.
    failures += run_pair_case(execution, {"context boundary prefill", kRotary, 1, 64},
                              kReleaseConfig, 16, 4, 262'100);

    failures += run_single_matches_pair(execution);
    failures += run_factor_one_matches_plain_rope(execution);
    failures += run_ramp_endpoints(execution);
    failures += run_refusals(execution);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " rope_yarn correctness\n";
    return failures == 0 ? 0 : 1;
}
