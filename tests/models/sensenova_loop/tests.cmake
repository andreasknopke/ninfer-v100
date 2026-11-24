# The looped decoder's execution order: the slot table, the pass numbers, and the state-plane counts
# derived from the release layer types (no GPU; the plan is host data built from the config).
ninfer_add_test(ninfer_sensenova_loop_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loop_plan.cpp"
  LIBRARIES ninfer_model_loading ninfer_core)
