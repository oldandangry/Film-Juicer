# Check the launch sites selected by the native route matrix in Debug.
file(WRITE "${SCANNER_TRACE}" "")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        "JUICER_DIAGNOSTICS=3" "JUICER_DIAGNOSTICS_PATH=${SCANNER_TRACE}"
        "${SCANNER_TEST}" --gtest_fail_if_no_test_selected
        --gtest_filter=ProductionScan.SourcePlanesAndFusedDirForBothPolarities:ProductionScan.MaterializedDirFiltersNonuniformDensity
    COMMAND_ERROR_IS_FATAL ANY)

file(STRINGS "${SCANNER_TRACE}" launches REGEX "^CUDA_SCAN [|] route=")
set(expected)
foreach(polarity RANGE 0 1)
    list(APPEND expected
        "CUDA_SCAN | route=direct input=source"
        "CUDA_SCAN | route=direct input=developed_planes"
        "CUDA_SCAN | route=direct input=fused_dir"
        "CUDA_SCAN | route=print input=source")
    foreach(white RANGE 1 3)
        list(APPEND expected "CUDA_SCAN | route=print input=developed_planes")
    endforeach()
    foreach(white RANGE 1 3)
        list(APPEND expected "CUDA_SCAN | route=print input=fused_dir")
    endforeach()
endforeach()
list(APPEND expected "CUDA_SCAN | route=direct input=developed_planes")
if(NOT launches STREQUAL expected)
    message(FATAL_ERROR "Unexpected scanner launch order in ${SCANNER_TRACE}\nExpected: ${expected}\nActual: ${launches}")
endif()
message(STATUS "Observed all 12 scanner selections and the nonuniform materialized DIR case at their CUDA launch sites")
execute_process(
    COMMAND "${SCANNER_TEST}" --gtest_fail_if_no_test_selected
        --gtest_filter=-ProductionScan.SourcePlanesAndFusedDirForBothPolarities:ProductionScan.MaterializedDirFiltersNonuniformDensity
    COMMAND_ERROR_IS_FATAL ANY)
