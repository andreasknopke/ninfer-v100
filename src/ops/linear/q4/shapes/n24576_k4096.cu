#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {
// SenseNova-Loop-9B's fused gate/up on the rotated dense MLP: 24576 rows over a 4096 hidden. The
// ladder is n5120_k17408.cu's, the nearest registered shape, and is not separately measured here.
// T=1 is answered by volta_decode_launch in q4_dispatch.cpp before this table is consulted.
Q4Launch select_q4_n24576_k4096(std::int32_t tokens) {
    if (tokens <= 8) return launch_q4_ksplit<24576, 4096, 8>;
    if (tokens <= 24) return launch_q4_ksplit<24576, 4096, 24>;
    if (tokens <= 32) return launch_q4_a16_mma_r32_t32_k64_wr16_wt8_s4_a4_b2;
    if (tokens <= 64) return launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 128) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    if (tokens <= 160) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 192) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 224) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 256) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 320) return launch_q4_a16_mma_r64_t80;
    return launch_q4_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
