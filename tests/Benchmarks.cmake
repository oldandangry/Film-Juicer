# Explicit OFX measurements use discovery-only shipping loads and a nonshipping
# companion. Ordinary objects and shipping Rust selection remain unchanged.
add_executable(JuicerOfxMeasurement benchmarks/ofx_measurement.cpp
    ofx/instance_host.cpp ofx/production_images.cpp benchmarks/observation_contract.cpp)
target_include_directories(JuicerOfxMeasurement PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ofx")
target_link_libraries(JuicerOfxMeasurement PRIVATE
    juicer_settings juicer_ofx_headers Threads::Threads ${CMAKE_DL_LIBS}
    CUDA::cudart CUDA::cuda_driver)
set_target_properties(JuicerOfxMeasurement PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(WIN32)
    target_link_libraries(JuicerOfxMeasurement PRIVATE Psapi)
endif()

add_executable(JuicerScannerCostBenchmark benchmarks/scanner_cost.cpp
    ../src/Cuda/Film/JuicerCudaFilmPipeline.cu
    ../src/Cuda/Scan/JuicerCudaScanPipeline.cu)
target_include_directories(JuicerScannerCostBenchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks")
target_compile_definitions(JuicerScannerCostBenchmark PRIVATE JUICER_SCANNER_COST_TEST=1)
target_link_libraries(JuicerScannerCostBenchmark PRIVATE juicer_settings CUDA::cudart)
set_target_properties(JuicerScannerCostBenchmark PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")

add_library(juicer_measurement_objects OBJECT
    ../src/Cuda/JuicerCudaDeviceLedger.cpp ../src/Cuda/JuicerCudaResources.cpp)
target_link_libraries(juicer_measurement_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_measurement_objects PRIVATE JUICER_PERFORMANCE_OBSERVATION=1)
add_library(JuicerOfxMeasurementModule MODULE ../native/ofx_exports.cpp
    benchmarks/observation.cpp $<TARGET_OBJECTS:juicer_measurement_objects>)
target_link_libraries(JuicerOfxMeasurementModule PRIVATE
    juicer_core film_juicer_rust_test_support juicer_dependencies)
set_target_properties(JuicerOfxMeasurementModule PROPERTIES
    PREFIX "" SUFFIX ".ofx" LINKER_LANGUAGE CXX
    LIBRARY_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerOfxMeasurementModule juicer_test_resources)
if(WIN32)
    target_compile_definitions(JuicerOfxMeasurementModule PRIVATE _WINDLL)
    target_sources(JuicerOfxMeasurementModule PRIVATE benchmarks/measurement.def)
else()
    set_property(TARGET JuicerOfxMeasurementModule PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_options(JuicerOfxMeasurementModule PRIVATE
        -static-libstdc++ -static-libgcc
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/measurement.exports"
        "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
    set_property(TARGET JuicerOfxMeasurementModule APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/measurement.exports")
endif()

set(spectral_replay_contents "${CMAKE_CURRENT_BINARY_DIR}/spectral-replay/ordinary/Contents")
target_compile_definitions(JuicerHalationComparisonProbe PRIVATE JUICER_SPECTRAL_REPLAY_CAPTURE=1)
target_link_libraries(JuicerHalationComparisonProbe PRIVATE CUDA::cuda_driver)
add_library(JuicerOfxSpectralReplayModule MODULE
    ../native/ofx_exports.cpp benchmarks/spectral_replay_exports.cpp)
target_link_libraries(JuicerOfxSpectralReplayModule PRIVATE
    juicer_core film_juicer_rust_test_support juicer_dependencies)
set_target_properties(JuicerOfxSpectralReplayModule PROPERTIES
    PREFIX "" SUFFIX ".ofx" LINKER_LANGUAGE CXX OUTPUT_NAME "juicer-spectral-replay"
    LIBRARY_OUTPUT_DIRECTORY "${spectral_replay_contents}/${JUICER_ARCH_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${spectral_replay_contents}/${JUICER_ARCH_DIR}")
add_dependencies(JuicerOfxSpectralReplayModule bundle-archive)
# Refresh Resources even when the link is already up to date.
add_custom_target(juicer_spectral_replay_resources
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${JUICER_STAGE_PREFIX}/${JUICER_CONTENTS}/Resources" "${spectral_replay_contents}/Resources"
    DEPENDS bundle-archive VERBATIM)
add_dependencies(JuicerOfxSpectralReplayModule juicer_spectral_replay_resources)
if(WIN32)
    target_compile_definitions(JuicerOfxSpectralReplayModule PRIVATE _WINDLL)
    add_custom_command(TARGET JuicerOfxSpectralReplayModule POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${JUICER_cudart64_13.dll}" "${JUICER_cufft64_12.dll}" ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS}
            "${spectral_replay_contents}/${JUICER_ARCH_DIR}" VERBATIM)
else()
    set_property(TARGET JuicerOfxSpectralReplayModule PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_options(JuicerOfxSpectralReplayModule PRIVATE
        -static-libstdc++ -static-libgcc
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/spectral_replay.exports"
        "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
    set_property(TARGET JuicerOfxSpectralReplayModule APPEND PROPERTY LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/spectral_replay.exports")
endif()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/spectral-replay/artifacts.json" CONTENT
    "{\"variant\":\"ordinary\",\"module\":\"$<TARGET_FILE:JuicerOfxSpectralReplayModule>\",\"native_archive\":\"$<TARGET_FILE:juicer_core>\",\"rust_archive\":\"$<TARGET_FILE:film_juicer_rust_test_support>\",\"rust_features\":[\"test-support\"],\"resources\":\"${spectral_replay_contents}/Resources\"}\n")

# Explicit maintenance experiment; it never substitutes a production kernel.
add_executable(JuicerHighlightBoostPrecisionTests
    scatter_halation/highlight_precision.cpp
    ../src/Cuda/Film/JuicerCudaHighlightBoost.cu)
target_link_libraries(JuicerHighlightBoostPrecisionTests PRIVATE
    juicer_settings CUDA::cudart)
set_target_properties(JuicerHighlightBoostPrecisionTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
