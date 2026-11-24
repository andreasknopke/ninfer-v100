# SenseNova-Loop-9B (sensenova_loop): the looped decoder's execution order. Host code only -- the
# plan is built from the config while loading and consulted whenever a per-layer buffer is sized, so
# it never reaches the GPU (docs/maintainer/sensenova-loop-9b-model.md).
target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/loop_plan.cpp"
)
