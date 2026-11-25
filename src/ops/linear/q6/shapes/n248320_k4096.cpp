#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

Q6Launch select_q6_n248320_k4096(std::int32_t tokens) {
    // SenseNova-Loop-9B's vocabulary head: the 248320-row head of the 35B over a 4096-wide hidden
    // state instead of 5120. Every route below reads K at runtime (the GEMV guards on
    // k % (kGroupK * kGroupsPerTile) == 0, i.e. k % 1024, and 4096 satisfies it), so the tier
    // boundaries carry over from the 5120-wide head unchanged. On sm_70 the first three tiers
    // intercept into the streaming GEMV, which is the measured-best route there anyway.
    if (tokens == 1) return launch_q6_gemv_t1;
    if (tokens == 2) return launch_q6_gemv_t2;
    if (tokens <= 8) return launch_q6_small_t_c8;
    if (tokens <= 16) return launch_q6_small_t_c16;
    if (tokens <= 32) return launch_q6_small_t_c32;
    if (tokens <= 48) return launch_q6_a16_mma_r64_t48_k128;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
