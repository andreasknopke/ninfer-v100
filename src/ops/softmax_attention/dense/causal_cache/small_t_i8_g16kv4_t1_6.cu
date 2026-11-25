// ninfer::ops - INT8-family small-T partial-kernel slice: SenseNova-Loop-9B geometry, T=1..6.
//
// CausalD256H16Kv4 has the same query-head count as the 35B-A3B's CausalD256H16Kv2 but a GQA group
// of four rather than eight, so every partial kernel instantiation differs. The tiles mirror the
// 35B-A3B's 1..6 range, which covers the widest verify chunk the sm_70 route resolves.
#include "ops/softmax_attention/dense/causal_cache/small_t_i8_launch.cuh"

namespace ninfer::ops::detail {

void causal_small_t_i8_g16kv4_t1(const SmallTI8SliceArgs& args) {
    small_t_i8_slice<CausalD256H16Kv4, 1>(args);
}

void causal_small_t_i8_g16kv4_t2(const SmallTI8SliceArgs& args) {
    small_t_i8_slice<CausalD256H16Kv4, 2>(args);
}

void causal_small_t_i8_g16kv4_t3(const SmallTI8SliceArgs& args) {
    small_t_i8_slice<CausalD256H16Kv4, 3>(args);
}

void causal_small_t_i8_g16kv4_t4(const SmallTI8SliceArgs& args) {
    small_t_i8_slice<CausalD256H16Kv4, 4>(args);
}

void causal_small_t_i8_g16kv4_t5(const SmallTI8SliceArgs& args) {
    small_t_i8_slice<CausalD256H16Kv4, 5>(args);
}

void causal_small_t_i8_g16kv4_t6(const SmallTI8SliceArgs& args) {
    small_t_i8_slice<CausalD256H16Kv4, 6>(args);
}

} // namespace ninfer::ops::detail
