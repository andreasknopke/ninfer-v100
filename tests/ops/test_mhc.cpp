// mhc_router, mhc_merge, mhc_enter and mhc_leave against FP64 oracles that evaluate the closed
// formulas of include/ninfer/ops/mhc.h on the represented inputs (the BF16 residual, the FP32 router
// weights): the statistic over the whole stream, the eight projection rows, the sigmoid pre and post,
// the log-space Sinkhorn mix, and the pre-gated branch; the mix-and-gate merge; the stream transitions.
//
// The oracles reproduce the contract's BF16 cast boundaries exactly (pre before the branch sum, mix
// before the residual mix, post before the gate), so what is left to measure is only the FP32
// accumulation against the FP64 reference.
//
// Cases: SenseNova-Loop-9B's geometry (S = 2, H = 4096, 20 mix iterations) at T = 1, 2, 3, 8, 17; a
// small H = 64 geometry with a different iteration count; H = 40 (a width the 8-wide router load does
// not tile evenly past); H = 37 for the element-wise Ops, which have no vector requirement; and a
// near-uniform router whose mix is close to 0.5 everywhere.
//
// Criteria (suite-owned): the FP32 outputs of the router meet |got - ref| <= 2e-5 |ref| + 2e-7 (an
// 8,192-term FP32 sum and the iterations in FP32; finite required); the BF16 outputs of branch, merge
// and leave meet |got - ref| <= 2^-8 |ref| + 2^-12 rms(ref) (one rounding of an FP32 sum from exact
// FP32 inputs). Also: each column bitwise independent of the others (T = 1 against its own column of a
// T = 8 call), a graph-captured call bitwise equal to the eager call, an in-place merge equal to the
// oracle on the pre-call residual, a mix that is doubly stochastic to the iteration's accuracy,
// refusals (misaligned residual and weight, overlapping outputs, a wrong shape, a non-contiguous bias,
// a width the router cannot tile, mix_iters out of range), and extents past the launch tiles.

#include "ninfer/ops/mhc.h"

#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace t = ninfer::test;
using ninfer::DType;
using ninfer::Tensor;

constexpr int kS    = ninfer::ops::kMhcStreams;
constexpr int kRows = ninfer::ops::kMhcRouterRows;
constexpr double kNormEps = ninfer::ops::kMhcNormEpsilon;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::uint32_t next(std::uint32_t& state) { return state = state * 1664525U + 1013904223U; }
float uniform(std::uint32_t& state) {
    return static_cast<float>(static_cast<std::int32_t>(next(state) >> 8) - (1 << 23)) / static_cast<float>(1 << 23);
}

struct Device {
    void* p = nullptr;
    explicit Device(std::size_t bytes) { t::cuda_check(cudaMalloc(&p, bytes ? bytes : 1), "cudaMalloc"); }
    ~Device() { cudaFree(p); }
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    template <class V>
    void put(const V& v) { t::cuda_check(cudaMemcpy(p, v.data(), v.size() * sizeof(typename V::value_type), cudaMemcpyHostToDevice), "put"); }
    template <class T>
    std::vector<T> get(std::size_t count) const {
        std::vector<T> out(count);
        t::cuda_check(cudaMemcpy(out.data(), p, count * sizeof(T), cudaMemcpyDeviceToHost), "get");
        return out;
    }
};

std::vector<std::uint16_t> bf16_values(std::size_t count, std::uint32_t seed, float sigma) {
    std::vector<std::uint16_t> v(count);
    std::uint32_t s = seed;
    for (auto& x : v) { x = t::f32_to_bf16(sigma * uniform(s) * 1.7F); }
    return v;
}

std::vector<float> f32_values(std::size_t count, std::uint32_t seed, float sigma) {
    std::vector<float> v(count);
    std::uint32_t s = seed;
    for (auto& x : v) { x = sigma * uniform(s); }
    return v;
}

// The reference model's cast boundaries, applied in the order the contract states them.
double bf16_round(double x) { return t::bf16_to_f32(t::f32_to_bf16(static_cast<float>(x))); }

// The log-space Sinkhorn-Knopp projection of a 2x2 logit block, exactly as the contract defines it.
void sinkhorn_oracle(double (&l)[kS][kS], int iters) {
    const auto lse_sub = [](double& a, double& b) {
        const double peak = std::max(a, b);
        const double lse  = peak + std::log(std::exp(a - peak) + std::exp(b - peak));
        a -= lse;
        b -= lse;
    };
    for (int round = 0; round < iters; ++round) {
        lse_sub(l[0][0], l[0][1]);
        lse_sub(l[1][0], l[1][1]);
        lse_sub(l[0][0], l[1][0]);
        lse_sub(l[0][1], l[1][1]);
    }
    for (int i = 0; i < kS; ++i) {
        for (int j = 0; j < kS; ++j) { l[i][j] = std::exp(l[i][j]); }
    }
}

struct Router {
    std::vector<double> post;   // [T][S]
    std::vector<double> mix;    // [T][S*S]
    std::vector<double> branch; // [T][H]
};

Router router_oracle(const std::vector<std::uint16_t>& r, const std::vector<float>& w,
                     const std::vector<float>& b, int hidden, int T, int iters) {
    const int width = kS * hidden;
    Router out;
    out.post.assign(static_cast<std::size_t>(T) * kS, 0);
    out.mix.assign(static_cast<std::size_t>(T) * kS * kS, 0);
    out.branch.assign(static_cast<std::size_t>(T) * hidden, 0);
    for (int c = 0; c < T; ++c) {
        const std::size_t col = static_cast<std::size_t>(c) * width;
        double ss = 0;
        for (int k = 0; k < width; ++k) { ss += std::pow(t::bf16_to_f32(r[col + k]), 2.0); }
        const double rs = 1.0 / std::sqrt(ss / width + kNormEps);
        double m[kRows];
        for (int i = 0; i < kRows; ++i) {
            double acc = 0;
            for (int k = 0; k < width; ++k) {
                acc += static_cast<double>(w[static_cast<std::size_t>(i) * width + k]) *
                       t::bf16_to_f32(r[col + k]);
            }
            m[i] = acc * rs + b[i];
        }
        const auto sig = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
        double pre[kS];
        for (int i = 0; i < kS; ++i) {
            pre[i]        = bf16_round(sig(m[i])); // the contract's cast before the branch sum
            out.post[static_cast<std::size_t>(c) * kS + i] = 2.0 * sig(m[kS + i]);
        }
        double l[kS][kS];
        for (int i = 0; i < kS; ++i) {
            for (int j = 0; j < kS; ++j) { l[i][j] = m[2 * kS + i * kS + j]; }
        }
        sinkhorn_oracle(l, iters);
        for (int i = 0; i < kS; ++i) {
            for (int j = 0; j < kS; ++j) {
                out.mix[static_cast<std::size_t>(c) * kS * kS + i * kS + j] = l[i][j];
            }
        }
        for (int d = 0; d < hidden; ++d) {
            double sum = 0;
            for (int s = 0; s < kS; ++s) {
                sum += pre[s] * t::bf16_to_f32(r[col + static_cast<std::size_t>(s) * hidden + d]);
            }
            out.branch[static_cast<std::size_t>(c) * hidden + d] = sum;
        }
    }
    return out;
}

std::vector<double> merge_oracle(const std::vector<std::uint16_t>& r,
                                 const std::vector<std::uint16_t>& branch,
                                 const std::vector<float>& post, const std::vector<float>& mix,
                                 int hidden, int T) {
    const int width = kS * hidden;
    std::vector<double> out(static_cast<std::size_t>(T) * width);
    for (int c = 0; c < T; ++c) {
        const std::size_t col = static_cast<std::size_t>(c) * width;
        for (int d = 0; d < hidden; ++d) {
            const double branch_value = t::bf16_to_f32(branch[static_cast<std::size_t>(c) * hidden + d]);
            double old[kS];
            for (int j = 0; j < kS; ++j) { old[j] = t::bf16_to_f32(r[col + static_cast<std::size_t>(j) * hidden + d]); }
            for (int i = 0; i < kS; ++i) {
                double mixed = 0;
                for (int j = 0; j < kS; ++j) {
                    mixed += bf16_round(mix[static_cast<std::size_t>(c) * kS * kS + i * kS + j]) * old[j];
                }
                mixed = bf16_round(mixed);
                const double gate = bf16_round(post[static_cast<std::size_t>(c) * kS + i]);
                out[col + static_cast<std::size_t>(i) * hidden + d] = mixed + gate * branch_value;
            }
        }
    }
    return out;
}

// The FP32-output criterion of mhc_router.
double fp32_error(const std::vector<float>& got, const std::vector<double>& ref) {
    double worst = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) { return 1e30; }
        worst = std::max(worst, std::fabs(got[i] - ref[i]) / (2e-5 * std::fabs(ref[i]) + 2e-7));
    }
    return worst; // <= 1 passes
}

// The BF16-output criterion of branch, merge and leave.
double bf16_error(const std::vector<std::uint16_t>& got, const std::vector<double>& ref) {
    double rms = 0;
    for (const double v : ref) { rms += v * v; }
    rms = std::sqrt(rms / static_cast<double>(ref.size()));
    double worst = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double g = t::bf16_to_f32(got[i]);
        if (!std::isfinite(g)) { return 1e30; }
        worst = std::max(worst, std::fabs(g - ref[i]) / (std::ldexp(1.0, -8) * std::fabs(ref[i]) + std::ldexp(1.0, -12) * rms));
    }
    return worst;
}

struct RouterRun {
    std::vector<std::uint16_t> branch;
    std::vector<float> post, mix;
};

RouterRun run_router(const std::vector<std::uint16_t>& r, const std::vector<float>& w,
                     const std::vector<float>& b, int hidden, int T, int iters, cudaStream_t stream,
                     bool graph = false) {
    const int width = kS * hidden;
    Device rd(r.size() * 2), wd(w.size() * 4), bd(b.size() * 4), bdv(T * hidden * 2),
        pd(T * kS * 4), md(static_cast<std::size_t>(T) * kS * kS * 4);
    rd.put(r);
    wd.put(w);
    bd.put(b);
    const Tensor R(rd.p, DType::BF16, {width, T}), W(wd.p, DType::FP32, {width, kRows}),
        B(bd.p, DType::FP32, {kRows});
    Tensor branch(bdv.p, DType::BF16, {hidden, T}), post(pd.p, DType::FP32, {kS, T}),
        mix(md.p, DType::FP32, {kS * kS, T});
    const auto call = [&] { ninfer::ops::mhc_router(R, W, B, iters, branch, post, mix, stream); };
    if (graph) {
        cudaGraph_t g        = nullptr;
        cudaGraphExec_t exec = nullptr;
        t::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        call();
        t::cuda_check(cudaStreamEndCapture(stream, &g), "end capture");
        t::cuda_check(cudaGraphInstantiate(&exec, g, 0), "instantiate");
        t::cuda_check(cudaGraphLaunch(exec, stream), "launch");
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(g);
    } else {
        call();
    }
    t::cuda_check(cudaStreamSynchronize(stream), "sync");
    return {bdv.get<std::uint16_t>(static_cast<std::size_t>(T) * hidden),
            pd.get<float>(T * kS), md.get<float>(static_cast<std::size_t>(T) * kS * kS)};
}

void test_router(int hidden, int T, float w_sigma, int iters, cudaStream_t stream) {
    const int width = kS * hidden;
    const auto r    = bf16_values(static_cast<std::size_t>(width) * T, 100U + T, 1.0F);
    const auto w    = f32_values(static_cast<std::size_t>(kRows) * width, 7U, w_sigma);
    const auto b    = f32_values(kRows, 8U, 0.4F);
    const RouterRun got  = run_router(r, w, b, hidden, T, iters, stream);
    const Router ref     = router_oracle(r, w, b, hidden, T, iters);
    const std::string label = "router H=" + std::to_string(hidden) + " T=" + std::to_string(T) +
                              " iters=" + std::to_string(iters);

    const double pe = fp32_error(got.post, ref.post), me = fp32_error(got.mix, ref.mix),
                 be = bf16_error(got.branch, ref.branch);
    check(pe <= 1.0 && me <= 1.0 && be <= 1.0,
          label + ": within the criteria (post " + std::to_string(pe) + ", mix " + std::to_string(me) +
              ", branch " + std::to_string(be) + ")");
    std::printf("%s: worst error / bound: post %.3f mix %.3f branch %.3f\n", label.c_str(), pe, me, be);

    // A converged mix is doubly stochastic, but only as far as the iteration budget actually
    // converges: a badly scaled matrix needs more Sinkhorn steps than the release asks for, and then
    // the FP64 reference is not doubly stochastic either (H=40 T=2 lands at 3.2e-4 after 20 steps).
    // What the op has to guarantee is that it is no further from doubly stochastic than the
    // reference is, with room for FP32 rounding on top.
    if (iters >= 8) {
        for (int c = 0; c < T; ++c) {
            for (int i = 0; i < kS; ++i) {
                double row = 0, column = 0, ref_row = 0, ref_column = 0;
                for (int j = 0; j < kS; ++j) {
                    const std::size_t base = static_cast<std::size_t>(c) * kS * kS;
                    row += got.mix[base + i * kS + j];
                    column += got.mix[base + j * kS + i];
                    ref_row += ref.mix[base + i * kS + j];
                    ref_column += ref.mix[base + j * kS + i];
                }
                const double got_dev = std::max(std::fabs(row - 1.0), std::fabs(column - 1.0));
                const double ref_dev = std::max(std::fabs(ref_row - 1.0), std::fabs(ref_column - 1.0));
                check(got_dev <= ref_dev + 1e-6,
                      label + ": mix row and column " + std::to_string(i) +
                          " are no less doubly stochastic than the reference (got " +
                          std::to_string(got_dev) + ", reference " + std::to_string(ref_dev) + ")");
            }
        }
    }
}

void test_column_independence_and_graph(cudaStream_t stream) {
    const int hidden = 4096, T = 8;
    const int width  = kS * hidden;
    const auto r     = bf16_values(static_cast<std::size_t>(width) * T, 55U, 1.0F);
    const auto w     = f32_values(static_cast<std::size_t>(kRows) * width, 9U, 0.02F);
    const auto b     = f32_values(kRows, 10U, 0.4F);
    const RouterRun all = run_router(r, w, b, hidden, T, 20, stream);
    for (const int c : {0, 3, 7}) {
        const std::vector<std::uint16_t> one(r.begin() + static_cast<std::ptrdiff_t>(c) * width,
                                             r.begin() + static_cast<std::ptrdiff_t>(c + 1) * width);
        const RouterRun alone = run_router(one, w, b, hidden, 1, 20, stream);
        check(std::memcmp(alone.post.data(), all.post.data() + c * kS, kS * 4) == 0 &&
                  std::memcmp(alone.mix.data(), all.mix.data() + c * kS * kS, kS * kS * 4) == 0 &&
                  std::memcmp(alone.branch.data(), all.branch.data() + static_cast<std::size_t>(c) * hidden,
                              static_cast<std::size_t>(hidden) * 2) == 0,
              "column " + std::to_string(c) + " alone is bitwise its column of the 8-column call");
    }
    const RouterRun graphed = run_router(r, w, b, hidden, T, 20, stream, true);
    check(graphed.post == all.post && graphed.mix == all.mix && graphed.branch == all.branch,
          "a graph-captured router call is bitwise the eager call");
}

void test_merge(int hidden, int T, cudaStream_t stream) {
    const int width = kS * hidden;
    const auto r    = bf16_values(static_cast<std::size_t>(width) * T, 300U + T, 1.0F);
    const auto br   = bf16_values(static_cast<std::size_t>(hidden) * T, 301U, 1.0F);
    const auto post = f32_values(static_cast<std::size_t>(T) * kS, 302U, 1.0F);
    const auto mix  = f32_values(static_cast<std::size_t>(T) * kS * kS, 303U, 0.5F);
    Device rd(r.size() * 2), brd(br.size() * 2), pd(post.size() * 4), md(mix.size() * 4),
        od(r.size() * 2);
    rd.put(r);
    brd.put(br);
    pd.put(post);
    md.put(mix);
    const std::string label = "merge H=" + std::to_string(hidden) + " T=" + std::to_string(T);

    Tensor R(rd.p, DType::BF16, {width, T});
    const Tensor BR(brd.p, DType::BF16, {hidden, T}), P(pd.p, DType::FP32, {kS, T}),
        M(md.p, DType::FP32, {kS * kS, T});
    Tensor O(od.p, DType::BF16, {width, T});
    ninfer::ops::mhc_merge(R, BR, P, M, O, stream);
    t::cuda_check(cudaStreamSynchronize(stream), "sync");
    const auto ref = merge_oracle(r, br, post, mix, hidden, T);
    const double oe = bf16_error(od.get<std::uint16_t>(ref.size()), ref);
    check(oe <= 1.0, label + " within the criterion (" + std::to_string(oe) + ")");

    // The same call with out aliasing residual, against the oracle on the pre-call residual.
    ninfer::ops::mhc_merge(R, BR, P, M, R, stream);
    t::cuda_check(cudaStreamSynchronize(stream), "sync");
    const double ie = bf16_error(rd.get<std::uint16_t>(ref.size()), ref);
    check(ie <= 1.0, label + " in place within the criterion (" + std::to_string(ie) + ")");
    std::printf("%s: worst error / bound %.3f (out-of-place), %.3f (in place)\n", label.c_str(), oe, ie);
}

void test_enter_leave(int hidden, int T, cudaStream_t stream) {
    const int width = kS * hidden;
    const auto x    = bf16_values(static_cast<std::size_t>(hidden) * T, 400U + T, 1.0F);
    Device xd(x.size() * 2), sd(static_cast<std::size_t>(width) * T * 2), od(x.size() * 2);
    xd.put(x);
    const std::string label = "enter/leave H=" + std::to_string(hidden) + " T=" + std::to_string(T);

    const Tensor X(xd.p, DType::BF16, {hidden, T});
    Tensor S_(sd.p, DType::BF16, {width, T}), O(od.p, DType::BF16, {hidden, T});
    ninfer::ops::mhc_enter(X, S_, stream);
    t::cuda_check(cudaStreamSynchronize(stream), "sync");
    const auto opened = sd.get<std::uint16_t>(static_cast<std::size_t>(width) * T);
    bool exact = true;
    for (int c = 0; c < T && exact; ++c) {
        for (int s = 0; s < kS && exact; ++s) {
            for (int d = 0; d < hidden; ++d) {
                if (opened[static_cast<std::size_t>(c) * width + static_cast<std::size_t>(s) * hidden + d] !=
                    x[static_cast<std::size_t>(c) * hidden + d]) {
                    exact = false;
                    break;
                }
            }
        }
    }
    check(exact, label + ": enter duplicates the stream bitwise");

    ninfer::ops::mhc_leave(S_, O, stream);
    t::cuda_check(cudaStreamSynchronize(stream), "sync");
    std::vector<double> averaged(x.size());
    for (int c = 0; c < T; ++c) {
        for (int d = 0; d < hidden; ++d) {
            double sum = 0;
            for (int s = 0; s < kS; ++s) {
                sum += t::bf16_to_f32(opened[static_cast<std::size_t>(c) * width +
                                             static_cast<std::size_t>(s) * hidden + d]);
            }
            averaged[static_cast<std::size_t>(c) * hidden + d] = sum / kS;
        }
    }
    const double ae = bf16_error(od.get<std::uint16_t>(averaged.size()), averaged);
    check(ae <= 1.0, label + " within the criterion (" + std::to_string(ae) + ")");
    // Averaging a stream with itself is exact, so leave(enter(x)) is x.
    const auto closed = od.get<std::uint16_t>(x.size());
    check(closed == x, label + ": leave undoes enter exactly");
    std::printf("%s: worst leave error / bound %.3f\n", label.c_str(), ae);
}

template <class F>
bool refuses(F&& f) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void test_refusals(cudaStream_t stream) {
    const int hidden = 128, T = 2, width = kS * hidden;
    const auto r    = bf16_values(static_cast<std::size_t>(width) * T + 8, 1U, 1.0F);
    const auto w    = f32_values(static_cast<std::size_t>(kRows) * width, 3U, 0.1F);
    const auto b    = f32_values(kRows, 4U, 0.1F);
    Device rd(r.size() * 2), wd(w.size() * 4), bd(b.size() * 4), bvd(T * hidden * 2), pd(T * kS * 4),
        md(static_cast<std::size_t>(T) * kS * kS * 4);
    rd.put(r);
    wd.put(w);
    bd.put(b);
    const Tensor R(rd.p, DType::BF16, {width, T}), W(wd.p, DType::FP32, {width, kRows}),
        B(bd.p, DType::FP32, {kRows});
    Tensor branch(bvd.p, DType::BF16, {hidden, T}), post(pd.p, DType::FP32, {kS, T}),
        mix(md.p, DType::FP32, {kS * kS, T});
    namespace o = ninfer::ops;
    check(!refuses([&] { o::mhc_router(R, W, B, 20, branch, post, mix, stream); }),
          "a valid router call is served");
    const Tensor misaligned(static_cast<char*>(rd.p) + 2, DType::BF16, {width, T});
    check(refuses([&] { o::mhc_router(misaligned, W, B, 20, branch, post, mix, stream); }),
          "a misaligned residual is refused");
    const Tensor misaligned_w(static_cast<char*>(wd.p) + 4, DType::FP32, {width, kRows});
    check(refuses([&] { o::mhc_router(R, misaligned_w, B, 20, branch, post, mix, stream); }),
          "a misaligned weight is refused");
    Tensor overlapping(pd.p, DType::FP32, {kS, T});
    check(refuses([&] { o::mhc_router(R, W, B, 20, branch, post, overlapping, stream); }),
          "overlapping outputs are refused");
    check(refuses([&] { o::mhc_router(R, W, B, 0, branch, post, mix, stream); }),
          "zero mix_iters are refused");
    check(refuses([&] { o::mhc_router(R, W, B, 65, branch, post, mix, stream); }),
          "mix_iters past 64 are refused");
    const Tensor wrong_w(wd.p, DType::FP32, {width, kRows - 1});
    check(refuses([&] { o::mhc_router(R, wrong_w, B, 20, branch, post, mix, stream); }),
          "a wrong weight shape is refused");
    const Tensor odd_r(rd.p, DType::BF16, {width - 2, T});
    check(refuses([&] { o::mhc_router(odd_r, W, B, 20, branch, post, mix, stream); }),
          "a width that is not a multiple of 8 is refused");
    const Tensor strided_b = B.view({kRows / 2, 2, 1, 1}).permute({1, 0, 2, 3});
    check(refuses([&] { o::mhc_router(R, W, strided_b, 20, branch, post, mix, stream); }),
          "a non-contiguous bias is refused");

    // merge: the output may alias the residual but nothing else.
    Tensor O(rd.p, DType::BF16, {width, T});
    check(!refuses([&] { o::mhc_merge(R, branch, post, mix, O, stream); }),
          "a valid merge call is served");
    Tensor alias_branch(rd.p, DType::BF16, {hidden, T});
    check(refuses([&] { o::mhc_merge(R, alias_branch, post, mix, O, stream); }),
          "a merge output overlapping the branch is refused");
    const Tensor wrong_branch(bvd.p, DType::BF16, {hidden + 1, T});
    check(refuses([&] { o::mhc_merge(R, wrong_branch, post, mix, O, stream); }),
          "a wrong merge branch shape is refused");

    // enter / leave shape agreement.
    Device sd(static_cast<std::size_t>(width) * T * 2), od2(static_cast<std::size_t>(hidden) * T * 2);
    const Tensor X(rd.p, DType::BF16, {hidden, T});
    check(refuses([&] {
        Tensor bad(sd.p, DType::BF16, {width + kS, T});
        o::mhc_enter(X, bad, stream);
    }), "a wrong enter streams shape is refused");
    check(refuses([&] {
        Tensor bad_streams(sd.p, DType::BF16, {width, T});
        Tensor bad_out(od2.p, DType::BF16, {hidden + 1, T});
        o::mhc_leave(bad_streams, bad_out, stream);
    }), "a wrong leave output shape is refused");
    t::cuda_check(cudaStreamSynchronize(stream), "sync");
}

// Extents past the launch tiles: more columns than one grid dimension is comfortable with, and the
// widest hidden the element-wise Ops admit.
void test_large_extents(cudaStream_t stream) {
    const int hidden = 128, width = kS * hidden, T = 65537;
    const auto r = bf16_values(static_cast<std::size_t>(width) * T, 77U, 1.0F);
    const auto w = f32_values(static_cast<std::size_t>(kRows) * width, 5U, 0.1F);
    const auto b = f32_values(kRows, 6U, 0.1F);
    const RouterRun all = run_router(r, w, b, hidden, T, 20, stream);
    for (const int c : {0, 65535, 65536}) {
        const std::vector<std::uint16_t> one(r.begin() + static_cast<std::ptrdiff_t>(c) * width,
                                             r.begin() + static_cast<std::ptrdiff_t>(c + 1) * width);
        const RouterRun alone = run_router(one, w, b, hidden, 1, 20, stream);
        check(std::memcmp(alone.mix.data(), all.mix.data() + static_cast<std::size_t>(c) * kS * kS,
                          kS * kS * 4) == 0 &&
                  std::memcmp(alone.post.data(), all.post.data() + static_cast<std::size_t>(c) * kS,
                              kS * 4) == 0,
              "router column " + std::to_string(c) + " of a 65,537-column call equals its own call");
    }
    test_merge(37, 5, stream); // a hidden size no vector width divides
    test_enter_leave(37, 5, stream);
}

} // namespace

int main() {
    try {
        if (t::cuda_unavailable()) {
            std::printf("SKIP: no CUDA device\n");
            return 77;
        }
        cudaStream_t stream = nullptr;
        t::cuda_check(cudaStreamCreate(&stream), "stream");
        for (const int T : {1, 2, 3, 8, 17}) { test_router(4096, T, 0.02F, 20, stream); }
        test_router(64, 3, 0.2F, 5, stream);
        test_router(40, 2, 0.2F, 20, stream);  // a width the 8-wide load tiles only partly
        test_router(4096, 2, 0.002F, 20, stream); // near-uniform logits
        test_column_independence_and_graph(stream);
        for (const int T : {1, 3, 8}) { test_merge(4096, T, stream); }
        test_enter_leave(4096, 1, stream);
        test_enter_leave(4096, 8, stream);
        test_refusals(stream);
        test_large_extents(stream);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
    if (failures != 0) {
        std::fprintf(stderr, "mhc: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("mhc: passed\n");
    return 0;
}
