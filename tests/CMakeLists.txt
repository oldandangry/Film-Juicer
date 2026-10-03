include(FetchContent)

enable_language(C)

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

add_test(NAME Rust.Bridge
    COMMAND "${CMAKE_COMMAND}" -E env
        "RUSTFLAGS="
        "CARGO_BUILD_RUSTFLAGS="
        "CARGO_ENCODED_RUSTFLAGS="
        "${JUICER_CARGO_EXECUTABLE}" test
            --locked --package film-juicer-plugin
            --no-default-features ${JUICER_RUST_FEATURE_ARGUMENT}
            --target "${JUICER_RUST_TARGET}"
            --target-dir "${JUICER_RUST_TARGET_DIR}"
            ${JUICER_RUST_PROFILE_ARGUMENT})
set_tests_properties(Rust.Bridge PROPERTIES
    LABELS "host" TIMEOUT 120 WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

add_test(NAME Ffi.Host.NativeBoundary
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p "test_native_boundary.py" -v)
set_tests_properties(Ffi.Host.NativeBoundary PROPERTIES LABELS "host" TIMEOUT 30)

add_executable(JuicerCudaAbiTests ffi/cuda_abi_c.c ffi/cuda_abi_test.cpp)
set_target_properties(JuicerCudaAbiTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_include_directories(JuicerCudaAbiTests PRIVATE "${PROJECT_SOURCE_DIR}/native")
target_link_libraries(JuicerCudaAbiTests PRIVATE film_juicer_rust)
if(WIN32)
    target_compile_options(JuicerCudaAbiTests PRIVATE /W3 /WX)
else()
    target_compile_options(JuicerCudaAbiTests PRIVATE -Wall -Wextra -Werror -pedantic-errors)
endif()
add_test(NAME Ffi.Host.CudaAbi COMMAND JuicerCudaAbiTests)
set_tests_properties(Ffi.Host.CudaAbi PROPERTIES LABELS "host" TIMEOUT 30)

add_executable(JuicerProfileOwnerTests ffi/profile_abi_c.c ffi/profile_owner_test.cpp)
set_target_properties(JuicerProfileOwnerTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_include_directories(JuicerProfileOwnerTests PRIVATE "${PROJECT_SOURCE_DIR}/native")
target_link_libraries(JuicerProfileOwnerTests PRIVATE film_juicer_rust GTest::gtest)
if(WIN32)
    target_compile_options(JuicerProfileOwnerTests PRIVATE /W3 /WX)
else()
    target_compile_options(JuicerProfileOwnerTests PRIVATE -Wall -Wextra -Werror -pedantic-errors)
endif()
add_test(NAME Ffi.Host.ProfileOwner COMMAND JuicerProfileOwnerTests
    "${JUICER_TEST_ARTIFACT_ROOT}/ffi/profile-owner-resources")
set_tests_properties(Ffi.Host.ProfileOwner PROPERTIES LABELS "host" TIMEOUT 30)

add_executable(JuicerCudaOwnerTests ffi/cuda_owner_test.cpp)
target_link_libraries(JuicerCudaOwnerTests PRIVATE juicer_core)
set_target_properties(JuicerCudaOwnerTests PROPERTIES LINKER_LANGUAGE CXX)
if(UNIX)
    set_property(TARGET JuicerCudaOwnerTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Ffi.Host.CudaOwner COMMAND JuicerCudaOwnerTests)
set_tests_properties(Ffi.Host.CudaOwner PROPERTIES LABELS "host" TIMEOUT 30)

# Terminal failure hooks exist only in this replacement API object.
add_library(juicer_terminal_test_objects OBJECT ../native/juicer_cuda_api.cpp)
target_link_libraries(juicer_terminal_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_terminal_test_objects PRIVATE JUICER_CUDA_TERMINAL_TEST_HOOK=1)
add_executable(JuicerCudaTerminalTests ffi/cuda_terminal_test.cpp $<TARGET_OBJECTS:juicer_terminal_test_objects>)
target_link_libraries(JuicerCudaTerminalTests PRIVATE juicer_core)
target_compile_definitions(JuicerCudaTerminalTests PRIVATE JUICER_CUDA_TERMINAL_TEST_HOOK=1)
set_target_properties(JuicerCudaTerminalTests PROPERTIES LINKER_LANGUAGE CXX)
if(UNIX)
    set_property(TARGET JuicerCudaTerminalTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
foreach(terminal_case IN ITEMS success serialization allocation standard unknown typed wording runtime context-loss diagnostic-allocation accepting-failure
        unregistered unregistered-reentry unregistered-shutdown-lock unregistered-destroy-lock
        saved-shutdown-lock saved-destroy-lock accepting-shutdown-lock accepting-destroy-lock
        buffers malformed-destroy malformed-blocked-destroy shutdown-progress)
    add_test(NAME Ffi.Host.CudaTerminal.${terminal_case} COMMAND JuicerCudaTerminalTests ${terminal_case})
    set_tests_properties(Ffi.Host.CudaTerminal.${terminal_case} PROPERTIES LABELS "host" TIMEOUT 30)
endforeach()

# Compile the allocation algorithm with a bounded test entry; production has no hook.
add_library(juicer_allocation_retry_test_objects OBJECT
    ../src/Cuda/ResourceManager/JuicerCudaResourceManager.cpp)
target_link_libraries(juicer_allocation_retry_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_allocation_retry_test_objects PRIVATE
    JUICER_ALLOCATION_RETRY_TEST_HOOK=1)
add_executable(JuicerCudaFailureTests ffi/cuda_failure_test.cpp
    $<TARGET_OBJECTS:juicer_allocation_retry_test_objects>)
target_link_libraries(JuicerCudaFailureTests PRIVATE juicer_core GTest::gtest_main)
target_compile_definitions(JuicerCudaFailureTests PRIVATE JUICER_ALLOCATION_RETRY_TEST_HOOK=1)
set_target_properties(JuicerCudaFailureTests PROPERTIES LINKER_LANGUAGE CXX)
if(UNIX)
    set_property(TARGET JuicerCudaFailureTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Ffi.Host.CudaFailure COMMAND JuicerCudaFailureTests)
set_tests_properties(Ffi.Host.CudaFailure PROPERTIES LABELS "host" TIMEOUT 30)

# The staging probe runs the production upload algorithm with isolated operation seams.
add_library(juicer_pinned_upload_test_objects OBJECT ../src/Cuda/JuicerCudaResources.cpp)
target_link_libraries(juicer_pinned_upload_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_pinned_upload_test_objects PRIVATE JUICER_PINNED_UPLOAD_TEST_HOOK=1)
add_executable(JuicerPinnedUploadTests resource/pinned_upload_test.cpp
    $<TARGET_OBJECTS:juicer_pinned_upload_test_objects>)
target_link_libraries(JuicerPinnedUploadTests PRIVATE juicer_core GTest::gtest_main CUDA::cuda_driver)
target_compile_definitions(JuicerPinnedUploadTests PRIVATE JUICER_PINNED_UPLOAD_TEST_HOOK=1)
set_target_properties(JuicerPinnedUploadTests PROPERTIES LINKER_LANGUAGE CXX)
if(UNIX)
    set_property(TARGET JuicerPinnedUploadTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Resource.Gpu.PinnedUpload COMMAND JuicerPinnedUploadTests)
set_tests_properties(Resource.Gpu.PinnedUpload PROPERTIES LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)

# Deferred retirement executes the real drain with a bounded post-extraction seam.
add_library(juicer_deferred_destroy_test_objects OBJECT ../src/Cuda/JuicerCudaResources.cpp)
target_link_libraries(juicer_deferred_destroy_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_deferred_destroy_test_objects PRIVATE JUICER_DEFERRED_DESTROY_TEST_HOOK=1)
add_executable(JuicerDeferredDestroyTests resource/deferred_destroy_test.cpp
    $<TARGET_OBJECTS:juicer_deferred_destroy_test_objects>)
target_link_libraries(JuicerDeferredDestroyTests PRIVATE juicer_core CUDA::cuda_driver)
target_compile_definitions(JuicerDeferredDestroyTests PRIVATE JUICER_DEFERRED_DESTROY_TEST_HOOK=1)
set_target_properties(JuicerDeferredDestroyTests PROPERTIES LINKER_LANGUAGE CXX)
if(UNIX)
    set_property(TARGET JuicerDeferredDestroyTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
foreach(deferred_case IN ITEMS lock allocation storage ordinary control)
    add_test(NAME Resource.Gpu.DeferredDestroy.${deferred_case} COMMAND JuicerDeferredDestroyTests ${deferred_case})
    set_tests_properties(Resource.Gpu.DeferredDestroy.${deferred_case} PROPERTIES LABELS "gpu" TIMEOUT 60 RESOURCE_LOCK juicer_gpu)
endforeach()

add_executable(JuicerCudaInspectTests ffi/cuda_inspect_test.cpp)
target_link_libraries(JuicerCudaInspectTests PRIVATE juicer_core GTest::gtest_main)
set_target_properties(JuicerCudaInspectTests PROPERTIES LINKER_LANGUAGE CXX)
if(UNIX)
    set_property(TARGET JuicerCudaInspectTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Ffi.Host.CudaInspect COMMAND JuicerCudaInspectTests --gtest_filter=InspectHost.*)
set_tests_properties(Ffi.Host.CudaInspect PROPERTIES LABELS "host" TIMEOUT 30)
# Keep the host binary independent of the NVIDIA driver at load time.
add_executable(JuicerCudaInspectGpuTests ffi/cuda_inspect_test.cpp)
target_compile_definitions(JuicerCudaInspectGpuTests PRIVATE JUICER_CUDA_INSPECT_GPU_TEST=1)
target_link_libraries(JuicerCudaInspectGpuTests PRIVATE juicer_core GTest::gtest_main CUDA::cuda_driver)
set_target_properties(JuicerCudaInspectGpuTests PROPERTIES LINKER_LANGUAGE CXX)
if(UNIX)
    set_property(TARGET JuicerCudaInspectGpuTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Ffi.Gpu.CudaInspect COMMAND JuicerCudaInspectGpuTests --gtest_filter=InspectGpu.*)
set_tests_properties(Ffi.Gpu.CudaInspect PROPERTIES LABELS "gpu" TIMEOUT 60 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerCudaOwnerUnloadTests ofx/cuda_owner_unload_test.cpp ../src/main.cpp)
target_link_libraries(JuicerCudaOwnerUnloadTests PRIVATE juicer_core juicer_ofxs)
set_target_properties(JuicerCudaOwnerUnloadTests PROPERTIES LINKER_LANGUAGE CXX)
if(UNIX)
    set_property(TARGET JuicerCudaOwnerUnloadTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Ofx.Host.CudaOwnerUnload COMMAND JuicerCudaOwnerUnloadTests)
set_tests_properties(Ofx.Host.CudaOwnerUnload PROPERTIES LABELS "host" TIMEOUT 30)

add_executable(JuicerRouteBridgeTests
    route/route_bridge_test.cpp ../src/ScanRoute.cpp)
target_include_directories(JuicerRouteBridgeTests PRIVATE
    "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/native")
target_link_libraries(JuicerRouteBridgeTests PRIVATE film_juicer_rust)
if(WIN32)
    target_compile_options(JuicerRouteBridgeTests PRIVATE /W3 /WX)
else()
    target_compile_options(JuicerRouteBridgeTests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME Route.Host.Bridge COMMAND JuicerRouteBridgeTests)
set_tests_properties(Route.Host.Bridge PROPERTIES LABELS "host" TIMEOUT 30)

add_test(NAME Quality.Dispatcher
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/quality"
        -p test_check_quality.py
        -v)
set_tests_properties(Quality.Dispatcher PROPERTIES
    LABELS "host"
    TIMEOUT 60
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

add_test(NAME Quality.RustNaming
    COMMAND "${CMAKE_COMMAND}" -E env
        "JUICER_CARGO=${JUICER_CARGO_EXECUTABLE}"
        "JUICER_RUST_TARGET=${JUICER_RUST_TARGET}"
        "JUICER_TEST_ARTIFACT_DIR=${JUICER_TEST_ARTIFACT_ROOT}/quality/rust-naming"
        "PYTHONDONTWRITEBYTECODE=1"
        "${Python3_EXECUTABLE}" -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/quality"
            -p test_rust_naming.py -v)
set_tests_properties(Quality.RustNaming PROPERTIES
    LABELS "host" TIMEOUT 180)

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
add_executable(JuicerScratchRequestTests resource/scratch_request_test.cpp)
target_link_libraries(JuicerScratchRequestTests PRIVATE juicer_core GTest::gtest_main)
set_target_properties(JuicerScratchRequestTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerScratchRequestTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Resource.Host.ScratchRequest COMMAND JuicerScratchRequestTests)
set_tests_properties(Resource.Host.ScratchRequest PROPERTIES LABELS "host" TIMEOUT 30)

add_executable(JuicerPreparedProjectionTests
    ffi/prepared_projection_test.cpp ffi/prepared_span_test.cpp
    ffi/prepared_descriptors.cpp)
target_link_libraries(JuicerPreparedProjectionTests PRIVATE juicer_core GTest::gtest_main)
set_target_properties(JuicerPreparedProjectionTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerPreparedProjectionTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Ffi.Host.PreparedProjection COMMAND JuicerPreparedProjectionTests)
set_tests_properties(Ffi.Host.PreparedProjection PROPERTIES LABELS "host" TIMEOUT 30)

add_executable(JuicerProfileCudaSafetyTests profile/native_cuda_safety_test.cu)
target_link_libraries(JuicerProfileCudaSafetyTests PRIVATE juicer_core GTest::gtest_main)
set_target_properties(JuicerProfileCudaSafetyTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerProfileCudaSafetyTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Profile.Gpu.NativeSafety COMMAND JuicerProfileCudaSafetyTests)
set_tests_properties(Profile.Gpu.NativeSafety PROPERTIES
    LABELS "gpu" TIMEOUT 60 RESOURCE_LOCK juicer_gpu)

# These objects expose only the real private asset lookups and Hanatos formula.
add_library(juicer_asset_lookup_test_objects OBJECT ../src/RenderRecipe.cpp ../src/Scanner.cpp)
target_link_libraries(juicer_asset_lookup_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_asset_lookup_test_objects PRIVATE JUICER_ASSET_LOOKUP_TEST_HOOK=1)
add_executable(JuicerAssetLookupTests
    ffi/asset_lookup_test.cpp $<TARGET_OBJECTS:juicer_asset_lookup_test_objects>)
target_link_libraries(JuicerAssetLookupTests PRIVATE juicer_core GTest::gtest_main)
target_compile_definitions(JuicerAssetLookupTests PRIVATE JUICER_ASSET_LOOKUP_TEST_HOOK=1)
set_target_properties(JuicerAssetLookupTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerAssetLookupTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Assets.Host.LookupSafety COMMAND JuicerAssetLookupTests)
set_tests_properties(Assets.Host.LookupSafety PROPERTIES LABELS "host" TIMEOUT 30)

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

add_test(NAME Ofx.Gpu.SequentialOwners COMMAND JuicerProcessorReferenceProbe --sequential-owners)
set_tests_properties(Ofx.Gpu.SequentialOwners PROPERTIES LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)

add_test(NAME Ofx.Gpu.CutoverContract COMMAND JuicerProcessorReferenceProbe --cutover-contract)
set_tests_properties(Ofx.Gpu.CutoverContract PROPERTIES LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)

add_library(juicer_render_test_objects OBJECT
    ../native/juicer_cuda_api.cpp ../src/Cuda/JuicerCudaExecutor.cpp)
target_link_libraries(juicer_render_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_render_test_objects PRIVATE JUICER_CUDA_RENDER_TEST_HOOK=1)
add_executable(JuicerPreparedBoundaryProbe
    ofx/processor_reference_test.cpp
    ffi/prepared_boundary.cpp
    ffi/cuda_render_test.cpp
    $<TARGET_OBJECTS:juicer_render_test_objects>
    $<TARGET_OBJECTS:juicer_pinned_upload_test_objects>
    ffi/prepared_boundary.c
    ffi/prepared_descriptors.cpp)
target_link_libraries(JuicerPreparedBoundaryProbe PRIVATE juicer_core juicer_ofxs CUDA::cuda_driver)
if(MSVC)
    target_compile_options(JuicerPreparedBoundaryProbe PRIVATE $<$<COMPILE_LANGUAGE:C>:/W3;/WX>)
else()
    target_compile_options(JuicerPreparedBoundaryProbe PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Werror;-pedantic-errors>)
endif()
target_compile_definitions(JuicerPreparedBoundaryProbe PRIVATE
    JUICER_PINNED_UPLOAD_TEST_HOOK=1
    JUICER_CUDA_RENDER_TEST_HOOK=1
    JUICER_PREPARED_BOUNDARY_TEST=1
    JUICER_PROCESSOR_REFERENCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/${JUICER_PROCESSOR_REFERENCE_FILE}")
set_target_properties(JuicerPreparedBoundaryProbe PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerPreparedBoundaryProbe juicer_test_resources)
if(UNIX)
    set_property(TARGET JuicerPreparedBoundaryProbe PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
foreach(prepared_case IN ITEMS negative-direct negative-print positive-direct positive-print
        combined-print glare-plus-zero glare-minus-zero)
    add_test(NAME "Ffi.Gpu.PreparedBoundary.${prepared_case}"
        COMMAND JuicerPreparedBoundaryProbe --prepared-case "${prepared_case}")
    set_tests_properties("Ffi.Gpu.PreparedBoundary.${prepared_case}" PROPERTIES
        LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
endforeach()

foreach(render_case IN ITEMS negative-direct negative-print)
    add_test(NAME "Ffi.Gpu.CudaRender.${render_case}"
        COMMAND JuicerPreparedBoundaryProbe --render-contract "${render_case}")
    set_tests_properties("Ffi.Gpu.CudaRender.${render_case}" PROPERTIES
        LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
endforeach()

# Isolated copies keep failure injection and observations out of juicer_core.
add_library(juicer_executor_failure_test_objects OBJECT
    ../src/mainProcessing.cpp
    ../src/Cuda/JuicerCudaExecutor.cpp
    ../src/Cuda/JuicerCudaResources.cpp
    ../src/ProcessRoot.cpp)
target_link_libraries(juicer_executor_failure_test_objects PRIVATE juicer_settings juicer_ofx_headers)
target_compile_definitions(juicer_executor_failure_test_objects PRIVATE
    JUICER_EXECUTOR_FAILURE_TEST_HOOK=1)
add_executable(JuicerExecutorFailureOrderProbe ofx/processor_reference_test.cpp
    $<TARGET_OBJECTS:juicer_executor_failure_test_objects>)
target_link_libraries(JuicerExecutorFailureOrderProbe PRIVATE
    juicer_core juicer_ofxs CUDA::cuda_driver)
target_compile_definitions(JuicerExecutorFailureOrderProbe PRIVATE
    JUICER_EXECUTOR_FAILURE_TEST_HOOK=1
    JUICER_PROCESSOR_REFERENCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/${JUICER_PROCESSOR_REFERENCE_FILE}")
set_target_properties(JuicerExecutorFailureOrderProbe PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerExecutorFailureOrderProbe juicer_test_resources)
if(UNIX)
    set_property(TARGET JuicerExecutorFailureOrderProbe
        PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Ofx.Gpu.ExecutorFailureOrder
    COMMAND JuicerExecutorFailureOrderProbe --failure-order)
set_tests_properties(Ofx.Gpu.ExecutorFailureOrder PROPERTIES
    LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)
foreach(resource_failure IN ITEMS DefectFence GrainUpload)
    add_test(NAME "Ofx.Gpu.ResourceFailure.${resource_failure}"
        COMMAND JuicerExecutorFailureOrderProbe --resource-failure "${resource_failure}")
    set_tests_properties("Ofx.Gpu.ResourceFailure.${resource_failure}" PROPERTIES
        LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)
endforeach()

add_custom_target(juicer_test_module
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${JUICER_TEST_BINARY_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:juicer>" "${JUICER_TEST_BINARY_DIR}/$<TARGET_FILE_NAME:juicer>"
    DEPENDS juicer juicer_test_resources)

add_executable(JuicerAdapterTraceProbe
    ofx/adapter_trace_test.cpp route/route_failure_adapter.cpp)
target_include_directories(JuicerAdapterTraceProbe PRIVATE "${PROJECT_SOURCE_DIR}/native")
target_link_libraries(JuicerAdapterTraceProbe PRIVATE
    juicer_core juicer_ofxs CUDA::cuda_driver)
set_target_properties(JuicerAdapterTraceProbe PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerAdapterTraceProbe juicer_test_resources juicer_test_module)
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
add_test(NAME Ofx.Gpu.StreamContract
    COMMAND JuicerAdapterTraceProbe --stream-contract "${JUICER_TEST_BINARY_DIR}/$<TARGET_FILE_NAME:juicer>"
        --trace-output "${JUICER_TEST_ARTIFACT_ROOT}/ofx/stream-contract.log")
set_tests_properties(Ofx.Gpu.StreamContract PROPERTIES LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)

add_library(juicer_admission_test_state OBJECT ../src/JuicerState.cpp)
target_link_libraries(juicer_admission_test_state PRIVATE juicer_settings juicer_ofx_headers)
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
juicer_add_scatter_gpu_test(TerminalRetention terminal-retention 120)
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
