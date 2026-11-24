# SenseNova-Loop mHC residual streams (two-stream router with a log-space Sinkhorn mix, the
# pre-gated branch, the mix-and-gate merge, and the enter/leave transitions). The contract is
# include/ninfer/ops/mhc.h.
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/mhc.cu")
