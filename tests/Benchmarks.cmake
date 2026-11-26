# Time the guarded cold and warm render cohort through the staged product.
add_custom_target(JuicerRenderBenchmark
    COMMAND "${CMAKE_COMMAND}" -E time "$<TARGET_FILE:JuicerProductionPreparedRenderProbe>"
        "${JUICER_STAGE_PREFIX}/${JUICER_BUNDLE}/Contents/${JUICER_ARCH_DIR}/$<TARGET_FILE_NAME:juicer>"
        "${JUICER_STAGE_PREFIX}/${JUICER_BUNDLE}/Contents/Resources"
        "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/spectral-v1/ordinary-${JUICER_TEST_PRESET_NAME}.json"
        "${JUICER_TEST_PRESET_NAME}" --summary
    DEPENDS JuicerProductionPreparedRenderProbe
    USES_TERMINAL VERBATIM)
