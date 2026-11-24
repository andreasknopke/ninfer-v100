# The looped decoder's execution order: the slot table, the pass numbers, and the state-plane counts
# derived from the release layer types (no GPU; the plan is host data built from the config).
ninfer_add_test(ninfer_sensenova_loop_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loop_plan.cpp"
  LIBRARIES ninfer_model_loading ninfer_core)
# The release configuration: the field set it actually carries, the constraints the runtime below
# assumes, and the refusals that stop an unimplemented feature from being dropped silently (no GPU).
ninfer_add_test(ninfer_sensenova_loop_config_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_config.cpp"
  LIBRARIES ninfer_model_loading ninfer_artifact ninfer_core)
