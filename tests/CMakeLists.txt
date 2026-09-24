include(FetchContent)

find_package(Python3 3.13 EXACT REQUIRED COMPONENTS Interpreter)

set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
if(WIN32)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
    URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(googletest)

foreach(flags IN ITEMS CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE)
    if("${${flags}}" MATCHES "(^|[ \t\r\n])--?use_fast_math([= \t\r\n]|$)")
        message(FATAL_ERROR "Film-Juicer correctness tests forbid --use_fast_math")
    endif()
endforeach()

set(JUICER_TEST_BINARY_DIR "${CMAKE_BINARY_DIR}/tests/bin")
set(JUICER_TEST_RESOURCE_DIR "${CMAKE_BINARY_DIR}/tests/Resources")
string(TOLOWER "${CMAKE_BUILD_TYPE}" JUICER_TEST_CONFIGURATION)
if(WIN32)
    set(JUICER_TEST_PRESET_NAME "windows-clang-${JUICER_TEST_CONFIGURATION}")
else()
    set(JUICER_TEST_PRESET_NAME "linux-${JUICER_TEST_CONFIGURATION}")
endif()
set(JUICER_TEST_ARTIFACT_ROOT
    "${PROJECT_SOURCE_DIR}/out/validation/${JUICER_TEST_PRESET_NAME}")

set(JUICER_RUST_TEST_COMMAND
    "${JUICER_CARGO_EXECUTABLE}" test
    --locked
    --package film-juicer-core
    --target "${JUICER_RUST_TARGET}"
    --target-dir "${JUICER_RUST_TARGET_DIR}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND JUICER_RUST_TEST_COMMAND --release)
endif()
add_test(NAME Rust.Core
    COMMAND "${CMAKE_COMMAND}" -E env
        "RUSTFLAGS="
        "CARGO_BUILD_RUSTFLAGS="
        "CARGO_ENCODED_RUSTFLAGS="
        ${JUICER_RUST_TEST_COMMAND})
set_tests_properties(Rust.Core PROPERTIES
    LABELS "host"
    TIMEOUT 120
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

add_test(NAME Quality.Dispatcher
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/quality"
        -p test_check_quality.py
        -v)
set_tests_properties(Quality.Dispatcher PROPERTIES
    LABELS "host"
    TIMEOUT 60
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

file(GLOB_RECURSE JUICER_TEST_RESOURCE_FILES CONFIGURE_DEPENDS
    LIST_DIRECTORIES FALSE "${PROJECT_SOURCE_DIR}/Resources/*")
set(JUICER_TEST_RESOURCE_STAMP "${CMAKE_CURRENT_BINARY_DIR}/resources.stamp")
add_custom_command(
    OUTPUT "${JUICER_TEST_RESOURCE_STAMP}"
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${JUICER_TEST_RESOURCE_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${PROJECT_SOURCE_DIR}/Resources" "${JUICER_TEST_RESOURCE_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E touch "${JUICER_TEST_RESOURCE_STAMP}"
    DEPENDS ${JUICER_TEST_RESOURCE_FILES}
    COMMENT "Staging Film-Juicer test resources"
    VERBATIM)
add_custom_target(juicer_test_resources DEPENDS "${JUICER_TEST_RESOURCE_STAMP}")

add_executable(JuicerScatterHalationHostTests
    scatter_halation/recipe_descriptor_test.cpp)
target_link_libraries(JuicerScatterHalationHostTests PRIVATE
    juicer_core GTest::gtest)
set_target_properties(JuicerScatterHalationHostTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerScatterHalationHostTests juicer_test_resources)
if(UNIX)
    set_property(TARGET JuicerScatterHalationHostTests
        PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

include(GoogleTest)
add_executable(JuicerHashContractTests hash/hash_contract_test.cpp)
target_link_libraries(JuicerHashContractTests PRIVATE juicer_core GTest::gtest_main)
set_target_properties(JuicerHashContractTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerHashContractTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
gtest_discover_tests(JuicerHashContractTests
    TEST_PREFIX "Hash.Host."
    DISCOVERY_MODE POST_BUILD
    DISCOVERY_TIMEOUT 30
    PROPERTIES LABELS "host" TIMEOUT 120)

gtest_discover_tests(JuicerScatterHalationHostTests
    TEST_PREFIX "ScatterHalation.Host."
    DISCOVERY_MODE POST_BUILD
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "host"
        TIMEOUT 120
        ENVIRONMENT
            "JUICER_TEST_ARTIFACT_DIR=${JUICER_TEST_ARTIFACT_ROOT}/scatter-halation-host")

set(DIFFUSION_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/diffusion")
set(DIFFUSION_GENERATED_HEADER "${DIFFUSION_GENERATED_DIR}/reference_cases.h")
add_custom_command(
    OUTPUT "${DIFFUSION_GENERATED_HEADER}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${DIFFUSION_GENERATED_DIR}"
    COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/diffusion/generate_host_cases.py"
        --manifest
        "${CMAKE_CURRENT_SOURCE_DIR}/diffusion/fixtures/diffusion_reference_v1.json"
        --arrays
        "${CMAKE_CURRENT_SOURCE_DIR}/diffusion/fixtures/diffusion_reference_v1.npz"
        --output "${DIFFUSION_GENERATED_HEADER}"
    DEPENDS
        diffusion/generate_host_cases.py
        diffusion/fixtures/diffusion_reference_v1.json
        diffusion/fixtures/diffusion_reference_v1.npz
    COMMENT "Generating diffusion host reference cases"
    VERBATIM)
add_executable(JuicerDiffusionHostReferenceTest
    diffusion/host_reference_test.cpp
    "${DIFFUSION_GENERATED_HEADER}")
target_include_directories(JuicerDiffusionHostReferenceTest PRIVATE
    "${DIFFUSION_GENERATED_DIR}")
target_link_libraries(JuicerDiffusionHostReferenceTest PRIVATE juicer_core)
set_target_properties(JuicerDiffusionHostReferenceTest PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Diffusion.HostReference
    COMMAND JuicerDiffusionHostReferenceTest)
set_tests_properties(Diffusion.HostReference PROPERTIES
    LABELS "host;reference"
    TIMEOUT 120)

add_test(NAME Gamma.Comparator
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/gamma"
        -p test_compare.py
        -v)
set_tests_properties(Gamma.Comparator PROPERTIES
    LABELS "host"
    TIMEOUT 60
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME Ofx.SyntheticHostLifecycle
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/ofx/probe.py"
            "$<TARGET_FILE:juicer>"
            --expected-descriptor
            "${CMAKE_CURRENT_SOURCE_DIR}/ofx/descriptor_manifest.json")
    set_tests_properties(Ofx.SyntheticHostLifecycle PROPERTIES
        LABELS "host"
        TIMEOUT 120
        ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endif()

add_executable(JuicerProcessorReferenceProbe ofx/processor_reference_test.cpp)
target_link_libraries(JuicerProcessorReferenceProbe PRIVATE
    juicer_core juicer_ofxs CUDA::cuda_driver)
if(WIN32)
    set(JUICER_PROCESSOR_REFERENCE_FILE "processor_reference_pixels_windows.dat")
else()
    set(JUICER_PROCESSOR_REFERENCE_FILE "processor_reference_pixels_linux.dat")
endif()
target_compile_definitions(JuicerProcessorReferenceProbe PRIVATE
    JUICER_PROCESSOR_REFERENCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/${JUICER_PROCESSOR_REFERENCE_FILE}")
set_target_properties(JuicerProcessorReferenceProbe PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerProcessorReferenceProbe juicer_test_resources)
if(UNIX)
    set_property(TARGET JuicerProcessorReferenceProbe
        PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Ofx.Gpu.ProcessorReference COMMAND JuicerProcessorReferenceProbe)
set_tests_properties(Ofx.Gpu.ProcessorReference PROPERTIES
    LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerAdapterTraceProbe ofx/adapter_trace_test.cpp)
target_link_libraries(JuicerAdapterTraceProbe PRIVATE
    juicer_core juicer_ofxs CUDA::cuda_driver)
set_target_properties(JuicerAdapterTraceProbe PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerAdapterTraceProbe juicer_test_resources)
if(UNIX)
    set_property(TARGET JuicerAdapterTraceProbe
        PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Ofx.Gpu.AdapterTrace
    COMMAND JuicerAdapterTraceProbe
        --trace-output
        "${JUICER_TEST_ARTIFACT_ROOT}/ofx/adapter_trace.txt")
set_tests_properties(Ofx.Gpu.AdapterTrace PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerScatterHalationGpuTests
    scatter_halation/integration_test.cpp
    scatter_halation/integration_test_cases.cpp)
add_library(juicer_admission_test_state OBJECT ../src/JuicerState.cpp)
target_link_libraries(juicer_admission_test_state PRIVATE juicer_settings)
target_compile_definitions(juicer_admission_test_state PRIVATE JUICER_ADMISSION_TEST_HOOK=1)
target_compile_definitions(JuicerScatterHalationGpuTests PRIVATE JUICER_ADMISSION_TEST_HOOK=1)
target_sources(JuicerScatterHalationGpuTests PRIVATE
    $<TARGET_OBJECTS:juicer_admission_test_state>)
add_library(juicer_context_drain_test_root OBJECT ../src/ProcessRoot.cpp)
target_link_libraries(juicer_context_drain_test_root PRIVATE juicer_settings)
target_compile_definitions(juicer_context_drain_test_root PRIVATE
    JUICER_CONTEXT_DRAIN_TEST_HOOK=1)
target_compile_definitions(JuicerScatterHalationGpuTests PRIVATE
    JUICER_CONTEXT_DRAIN_TEST_HOOK=1)
target_sources(JuicerScatterHalationGpuTests PRIVATE
    $<TARGET_OBJECTS:juicer_context_drain_test_root>)
target_include_directories(JuicerScatterHalationGpuTests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/scatter_halation")
target_link_libraries(JuicerScatterHalationGpuTests PRIVATE
    juicer_core CUDA::cuda_driver)
set_target_properties(JuicerScatterHalationGpuTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerScatterHalationGpuTests juicer_test_resources)
if(UNIX)
    set_property(TARGET JuicerScatterHalationGpuTests
        PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

function(juicer_add_scatter_gpu_test test_name case_group timeout_seconds)
    add_test(NAME "ScatterHalation.Gpu.${test_name}"
        COMMAND JuicerScatterHalationGpuTests
            --case-group "${case_group}"
            --fixture-root
                "${CMAKE_CURRENT_SOURCE_DIR}/scatter_halation/fixtures"
            --resource-root "${JUICER_TEST_RESOURCE_DIR}"
            --scratch-root
                "${JUICER_TEST_ARTIFACT_ROOT}/scatter-halation-${case_group}")
    set_tests_properties("ScatterHalation.Gpu.${test_name}" PROPERTIES
        LABELS "gpu"
        TIMEOUT "${timeout_seconds}"
        RESOURCE_LOCK juicer_gpu)
endfunction()

juicer_add_scatter_gpu_test(HostContracts host-contracts 240)
juicer_add_scatter_gpu_test(PreparedFrame prepared-frame 240)
juicer_add_scatter_gpu_test(FocusedCudaReference focused-cuda-reference 600)
set_property(TEST ScatterHalation.Gpu.FocusedCudaReference
    PROPERTY LABELS "gpu;reference")
juicer_add_scatter_gpu_test(RouteBoundaries route-boundaries 600)
juicer_add_scatter_gpu_test(CapturedCarrier captured-carrier 600)
set_property(TEST ScatterHalation.Gpu.CapturedCarrier
    PROPERTY LABELS "gpu;reference")
juicer_add_scatter_gpu_test(ZeroWork zero-work 300)
juicer_add_scatter_gpu_test(Lifecycle lifecycle 600)

add_executable(JuicerGrainScratchTests grain/scratch_reuse_test.cpp)
target_link_libraries(JuicerGrainScratchTests PRIVATE
    juicer_core GTest::gtest CUDA::cuda_driver)
set_target_properties(JuicerGrainScratchTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerGrainScratchTests juicer_test_resources)
if(UNIX)
    set_property(TARGET JuicerGrainScratchTests
        PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Grain.Gpu.ScratchReuse COMMAND JuicerGrainScratchTests)
set_tests_properties(Grain.Gpu.ScratchReuse PROPERTIES
    LABELS "gpu" TIMEOUT 600 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerDirExposureCacheTests dir/exposure_cache_test.cpp)
target_link_libraries(JuicerDirExposureCacheTests PRIVATE
    juicer_core GTest::gtest CUDA::cuda_driver)
set_target_properties(JuicerDirExposureCacheTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerDirExposureCacheTests juicer_test_resources)
if(UNIX)
    set_property(TARGET JuicerDirExposureCacheTests
        PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Dir.Gpu.ExposureCache COMMAND JuicerDirExposureCacheTests)
set_tests_properties(Dir.Gpu.ExposureCache PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerGrainDeltaTests grain/delta_fusion_test.cu)
target_link_libraries(JuicerGrainDeltaTests PRIVATE juicer_core)
set_target_properties(JuicerGrainDeltaTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerGrainDeltaTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Grain.Gpu.DeltaFusion COMMAND JuicerGrainDeltaTests)
set_tests_properties(Grain.Gpu.DeltaFusion PROPERTIES
    LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerScannerOutputTests scanner/output_encoding_test.cpp)
target_link_libraries(JuicerScannerOutputTests PRIVATE
    juicer_core GTest::gtest CUDA::cuda_driver)
set_target_properties(JuicerScannerOutputTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerScannerOutputTests juicer_test_resources)
if(UNIX)
    set_property(TARGET JuicerScannerOutputTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Scanner.Gpu.OutputEncoding COMMAND JuicerScannerOutputTests)
set_tests_properties(Scanner.Gpu.OutputEncoding PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
