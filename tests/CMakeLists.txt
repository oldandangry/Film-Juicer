include(FetchContent)

if(NOT DEFINED JUICER_TEST_ROOT)
    set(JUICER_TEST_ROOT "${PROJECT_SOURCE_DIR}")
endif()
if(NOT DEFINED JUICER_PRESET_BUILD_DIR)
    set(JUICER_PRESET_BUILD_DIR "${CMAKE_BINARY_DIR}")
endif()

set(JUICER_TEST_BINARY_DIR "${CMAKE_BINARY_DIR}/tests/bin")
set(JUICER_TEST_RESOURCE_DIR "${JUICER_TEST_ROOT}/Resources")
set(JUICER_TEST_RESOURCE_STAGE_DIR "${JUICER_PRESET_BUILD_DIR}/tests/Resources")
string(TOLOWER "${CMAKE_BUILD_TYPE}" JUICER_TEST_CONFIGURATION)
if(WIN32)
    set(JUICER_TEST_PRESET_NAME "windows-clang-${JUICER_TEST_CONFIGURATION}")
else()
    set(JUICER_TEST_PRESET_NAME "linux-${JUICER_TEST_CONFIGURATION}")
endif()
set(JUICER_TEST_ARTIFACT_ROOT
    "${JUICER_TEST_ROOT}/out/validation/${JUICER_TEST_PRESET_NAME}")
include("${CMAKE_CURRENT_SOURCE_DIR}/ResourceStaging.cmake")
juicer_stage_test_resources("${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_RESOURCE_STAGE_DIR}"
    "${JUICER_PRESET_BUILD_DIR}/tests/resources.stamp")

add_custom_target(juicer_test_module
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${JUICER_TEST_BINARY_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "$<TARGET_FILE:juicer>" "${JUICER_TEST_BINARY_DIR}/$<TARGET_FILE_NAME:juicer>"
    DEPENDS juicer juicer_test_resources)

if(JUICER_BUILD_BENCHMARKS)
    include(Benchmarks.cmake)
endif()
if(NOT BUILD_TESTING)
    return()
endif()

find_package(Python3 3.13 EXACT REQUIRED COMPONENTS Interpreter)

enable_language(C)
string(REGEX REPLACE "(^| )[/-]DNDEBUG( |$)" " "
    CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_RELEASE}")
if(NOT WIN32)
    string(APPEND CMAKE_C_FLAGS_DEBUG " -g")
endif()

function(juicer_add_host_test name target)
    add_test(NAME "${name}" COMMAND ${target} ${ARGN})
    set_tests_properties("${name}" PROPERTIES LABELS "host" TIMEOUT 300)
endfunction()

add_executable(JuicerOfxAbiTests ffi/ofx_abi_c.c)
set_target_properties(JuicerOfxAbiTests PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_include_directories(JuicerOfxAbiTests PRIVATE "${JUICER_TEST_ROOT}/native")
target_link_libraries(JuicerOfxAbiTests PRIVATE juicer_ofx_headers)
if(WIN32)
    target_compile_options(JuicerOfxAbiTests PRIVATE /W4 /WX)
else()
    target_compile_options(JuicerOfxAbiTests PRIVATE -Wall -Wextra -Werror -pedantic-errors)
endif()
juicer_add_host_test(Ffi.Host.OfxAbi JuicerOfxAbiTests)

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

add_library(juicer_host_gtest_main OBJECT ffi/host_gtest_main.cpp)
target_compile_features(juicer_host_gtest_main PRIVATE cxx_std_20)
target_include_directories(juicer_host_gtest_main PRIVATE "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(juicer_host_gtest_main PRIVATE GTest::gtest)


foreach(flags IN ITEMS CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE)
    if("${${flags}}" MATCHES "(^|[ \t\r\n])--?use_fast_math([= \t\r\n]|$)")
        message(FATAL_ERROR "Film-Juicer correctness tests forbid --use_fast_math")
    endif()
endforeach()

include(GoogleTest)
add_executable(JuicerHashContractTests hash/hash_contract_test.cpp
    "${JUICER_TEST_ROOT}/src/Logging.cpp")
target_link_libraries(JuicerHashContractTests PRIVATE juicer_settings GTest::gtest juicer_host_gtest_main)
set_target_properties(JuicerHashContractTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
gtest_add_tests(TARGET JuicerHashContractTests EXTRA_ARGS --gtest_fail_if_no_test_selected
    TEST_PREFIX "Hash.Host." TEST_LIST juicer_hash_tests)
set_tests_properties(${juicer_hash_tests} PROPERTIES LABELS "host" TIMEOUT 300)

add_executable(JuicerRenderAssertionTests ofx/render_assertions_test.cpp ofx/render_assertions.cpp)
target_compile_features(JuicerRenderAssertionTests PRIVATE cxx_std_20)
target_include_directories(JuicerRenderAssertionTests PRIVATE "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerRenderAssertionTests PRIVATE GTest::gtest juicer_host_gtest_main)
add_test(NAME Ofx.Host.RenderAssertions COMMAND JuicerRenderAssertionTests)
set_tests_properties(Ofx.Host.RenderAssertions PROPERTIES LABELS "host" TIMEOUT 300)

add_executable(JuicerCudaInspectTests ffi/cuda_inspect_test.cpp)
target_link_libraries(JuicerCudaInspectTests PRIVATE juicer_core film_juicer_rust juicer_cuda_resource_operations juicer_dependencies GTest::gtest juicer_host_gtest_main)
set_target_properties(JuicerCudaInspectTests PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME Ffi.Host.CudaInspect COMMAND JuicerCudaInspectTests --gtest_fail_if_no_test_selected --gtest_filter=InspectHost.*)
set_tests_properties(Ffi.Host.CudaInspect PROPERTIES LABELS "host" TIMEOUT 300)
# Keep the host binary independent of the NVIDIA driver at load time.
add_executable(JuicerCudaInspectGpuTests ffi/cuda_inspect_test.cpp)
target_compile_definitions(JuicerCudaInspectGpuTests PRIVATE JUICER_CUDA_INSPECT_GPU_TEST=1)
target_link_libraries(JuicerCudaInspectGpuTests PRIVATE juicer_core film_juicer_rust juicer_cuda_resource_operations juicer_dependencies GTest::gtest juicer_host_gtest_main CUDA::cuda_driver)
set_target_properties(JuicerCudaInspectGpuTests PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME Ffi.Gpu.CudaInspect COMMAND JuicerCudaInspectGpuTests
    --gtest_fail_if_no_test_selected --gtest_filter=InspectGpu.*-InspectGpu.DeviceMismatchWhenTwoDevicesAreAvailable)
set_tests_properties(Ffi.Gpu.CudaInspect PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
# Keep the two-device requirement visible and independently reportable. Its skip
# remains incomplete evidence on single-device hosts, never an accepted pass.
add_test(NAME Ffi.Gpu.CudaInspect.MultiDevice COMMAND JuicerCudaInspectGpuTests
    --gtest_fail_if_no_test_selected --gtest_filter=InspectGpu.DeviceMismatchWhenTwoDevicesAreAvailable)
set_tests_properties(Ffi.Gpu.CudaInspect.MultiDevice PROPERTIES LABELS "gpu;multi-gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerOfxModuleTests ofx/module_test.cpp ofx/module_host.cpp ../native/ofx_exports.cpp)
target_compile_features(JuicerOfxModuleTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxModuleTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxModuleTests PRIVATE juicer_core film_juicer_rust juicer_cuda_resource_operations juicer_dependencies juicer_ofx_headers)
set_target_properties(JuicerOfxModuleTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
juicer_add_host_test(Ofx.Host.Module JuicerOfxModuleTests
    "${CMAKE_CURRENT_SOURCE_DIR}/ofx/descriptor_manifest.json")

add_executable(JuicerOfxContextTests ofx/context_test.cpp ofx/context_host.cpp ../native/ofx_exports.cpp)
target_compile_features(JuicerOfxContextTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxContextTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxContextTests PRIVATE juicer_core film_juicer_rust juicer_cuda_resource_operations juicer_dependencies juicer_ofx_headers)
set_target_properties(JuicerOfxContextTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerOfxContextTests juicer_test_resources)
juicer_add_host_test(Ofx.Host.Context JuicerOfxContextTests --reference "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/s6-d/context.json")
set_tests_properties(Ofx.Host.Context PROPERTIES LABELS "host;boundary")

add_executable(JuicerOfxInstanceTests ofx/instance_test.cpp ofx/instance_host.cpp ../native/ofx_exports.cpp)
target_compile_features(JuicerOfxInstanceTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxInstanceTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxInstanceTests PRIVATE juicer_core film_juicer_rust juicer_cuda_resource_operations juicer_dependencies juicer_ofx_headers)
set_target_properties(JuicerOfxInstanceTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerOfxInstanceTests juicer_test_resources)
juicer_add_host_test(Ofx.Host.Instance JuicerOfxInstanceTests
    --reference "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/s6-d/instance.json")
juicer_add_host_test(Ofx.Host.FilmV1Instance JuicerOfxInstanceTests
    --film-v1 --reference "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/s6-d/instance.json")

add_executable(JuicerOfxActionTests ofx/action_test.cpp ofx/instance_host.cpp ../native/ofx_exports.cpp)
target_compile_features(JuicerOfxActionTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxActionTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxActionTests PRIVATE juicer_core film_juicer_rust juicer_cuda_resource_operations juicer_dependencies juicer_ofx_headers)
set_target_properties(JuicerOfxActionTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerOfxActionTests juicer_test_resources)
juicer_add_host_test(Ofx.Host.Actions JuicerOfxActionTests --reference "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/s6-d/actions.json")
set_tests_properties(Ofx.Host.Actions PROPERTIES LABELS "host;boundary")

add_executable(JuicerOfxRenderTests ofx/render_test.cpp ofx/instance_host.cpp ../native/ofx_exports.cpp)
target_compile_features(JuicerOfxRenderTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxRenderTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxRenderTests PRIVATE juicer_core film_juicer_rust juicer_cuda_resource_operations juicer_dependencies juicer_ofx_headers)
set_target_properties(JuicerOfxRenderTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerOfxRenderTests juicer_test_resources)
juicer_add_host_test(Ofx.Host.Render JuicerOfxRenderTests
    --reference "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/s6-d/render.json")

foreach(reference_target IN ITEMS JuicerOfxContextTests JuicerOfxInstanceTests JuicerOfxActionTests JuicerOfxRenderTests)
    target_sources(${reference_target} PRIVATE ofx/host_reference.cpp)
    target_compile_definitions(${reference_target} PRIVATE
        JUICER_TEST_PRESET_NAME="${JUICER_TEST_PRESET_NAME}")
endforeach()

add_executable(JuicerScratchRequestTests resource/scratch_request_test.cpp)
target_include_directories(JuicerScratchRequestTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ffi")
target_link_libraries(JuicerScratchRequestTests PRIVATE juicer_core film_juicer_rust GTest::gtest juicer_host_gtest_main)
set_target_properties(JuicerScratchRequestTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Resource.Host.ScratchRequest COMMAND JuicerScratchRequestTests)
set_tests_properties(Resource.Host.ScratchRequest PROPERTIES LABELS "host" TIMEOUT 300)

add_executable(JuicerProfileCudaSafetyTests profile/native_cuda_safety_test.cu)
target_link_libraries(JuicerProfileCudaSafetyTests PRIVATE juicer_core film_juicer_rust GTest::gtest juicer_host_gtest_main)
if(WIN32)
    target_link_libraries(JuicerProfileCudaSafetyTests PRIVATE CUDA::cudart)
else()
    target_link_libraries(JuicerProfileCudaSafetyTests PRIVATE CUDA::cudart_static)
endif()
set_target_properties(JuicerProfileCudaSafetyTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Profile.Gpu.NativeSafety COMMAND JuicerProfileCudaSafetyTests)
set_tests_properties(Profile.Gpu.NativeSafety PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_test(NAME Ffi.Gpu.ProductionPreparedRender
    COMMAND JuicerProductionPreparedRenderProbe
        "${JUICER_STAGE_PREFIX}/${JUICER_BUNDLE}/Contents/${JUICER_ARCH_DIR}/$<TARGET_FILE_NAME:juicer>"
        "${JUICER_STAGE_PREFIX}/${JUICER_BUNDLE}/Contents/Resources"
        "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/spectral-v1/ordinary-${JUICER_TEST_PRESET_NAME}.json"
        "${JUICER_TEST_PRESET_NAME}" --summary)
set_tests_properties(Ffi.Gpu.ProductionPreparedRender PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
add_test(NAME Ofx.Gpu.FrameAndPrintEdits
    COMMAND JuicerProductionPreparedRenderProbe --frame-and-print-edits
        "${JUICER_STAGE_PREFIX}/${JUICER_BUNDLE}/Contents/${JUICER_ARCH_DIR}/$<TARGET_FILE_NAME:juicer>"
        "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/spectral-v1/ordinary-${JUICER_TEST_PRESET_NAME}.json"
        "${CMAKE_CURRENT_SOURCE_DIR}/print/preflash-fixtures"
        "${JUICER_TEST_ARTIFACT_ROOT}"
        "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ofx.Gpu.FrameAndPrintEdits PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
add_test(NAME Ffi.Gpu.FilmV1Staged
    COMMAND JuicerProductionPreparedRenderProbe --film-v1-witness
        "${JUICER_STAGE_PREFIX}/${JUICER_BUNDLE}/Contents/${JUICER_ARCH_DIR}/$<TARGET_FILE_NAME:juicer>")
set_tests_properties(Ffi.Gpu.FilmV1Staged PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_test(NAME Ofx.SyntheticHostLifecycle
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/ofx/probe.py"
            "$<TARGET_FILE:juicer>"
            --expected-descriptor
            "${CMAKE_CURRENT_SOURCE_DIR}/ofx/descriptor_manifest.json")
    set_tests_properties(Ofx.SyntheticHostLifecycle PROPERTIES
        LABELS "host"
        TIMEOUT 300
        ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endif()

add_executable(JuicerGrainDeltaTests grain/delta_fusion_test.cu)
target_link_libraries(JuicerGrainDeltaTests PRIVATE juicer_core film_juicer_rust)
if(WIN32)
    target_link_libraries(JuicerGrainDeltaTests PRIVATE CUDA::cudart)
else()
    target_link_libraries(JuicerGrainDeltaTests PRIVATE CUDA::cudart_static)
endif()
set_target_properties(JuicerGrainDeltaTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Grain.Gpu.DeltaFusion COMMAND JuicerGrainDeltaTests)
set_tests_properties(Grain.Gpu.DeltaFusion PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_dependencies(JuicerProductionOfxProbe juicer_test_module)
add_test(NAME Ffi.Host.ProductionOfx COMMAND JuicerProductionOfxProbe
    "${JUICER_TEST_BINARY_DIR}/$<TARGET_FILE_NAME:juicer>")
add_test(NAME Ffi.Host.ProductionCudaAbi COMMAND JuicerProductionCudaAbiProbe)
set_tests_properties(Ffi.Host.ProductionOfx Ffi.Host.ProductionCudaAbi PROPERTIES
    LABELS "host" TIMEOUT 300 ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1")

add_executable(JuicerScannerBindingTests scanner/binding_test.cpp)
target_include_directories(JuicerScannerBindingTests PRIVATE "${JUICER_TEST_ROOT}/native")
target_compile_definitions(JuicerScannerBindingTests PRIVATE
    FJ_SCANNER_BINDING_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/scanner/color-fixtures/scanner-binding-v1.json")
target_link_libraries(JuicerScannerBindingTests PRIVATE
    juicer_core film_juicer_rust juicer_cuda_resource_operations juicer_dependencies GTest::gtest juicer_host_gtest_main)
juicer_add_host_test(Scanner.Host.Binding JuicerScannerBindingTests --gtest_fail_if_no_test_selected)

add_executable(JuicerScannerSpectralTests scanner/spectral_xyz_test.cpp scanner/spectral_xyz.cu)
target_link_libraries(JuicerScannerSpectralTests PRIVATE
    juicer_core film_juicer_rust juicer_cuda_resource_operations juicer_dependencies GTest::gtest juicer_host_gtest_main)
set_property(TARGET JuicerScannerSpectralTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME Scanner.Gpu.PreGlareXyz COMMAND JuicerScannerSpectralTests --gtest_fail_if_no_test_selected)
set_tests_properties(Scanner.Gpu.PreGlareXyz PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerScannerRouteTests
    scanner/routes_test.cpp resource/allocation_fault.cpp resource/observed_operations.cpp
    ofx/cancellation_test.cpp ofx/instance_host.cpp ofx/production_images.cpp)
target_compile_definitions(JuicerScannerRouteTests PRIVATE
    JUICER_DIFFUSION_FRAME_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/diffusion/frame-fixtures/native-payload.json")
target_link_libraries(JuicerScannerRouteTests PRIVATE
    juicer_core film_juicer_rust juicer_dependencies juicer_ofx_headers CUDA::cuda_driver
    GTest::gtest juicer_host_gtest_main)
set_target_properties(JuicerScannerRouteTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerScannerRouteTests juicer_test_resources)
set_property(TARGET JuicerScannerRouteTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
if(JUICER_TEST_CONFIGURATION STREQUAL "debug")
    add_test(NAME Scanner.Gpu.Routes
        COMMAND "${CMAKE_COMMAND}"
            "-DSCANNER_TEST=$<TARGET_FILE:JuicerScannerRouteTests>"
            "-DSCANNER_TRACE=${CMAKE_CURRENT_BINARY_DIR}/scanner-routes.log"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/scanner/check_routes.cmake")
else()
    add_test(NAME Scanner.Gpu.Routes COMMAND JuicerScannerRouteTests --gtest_fail_if_no_test_selected)
endif()
set_tests_properties(Scanner.Gpu.Routes PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerPinnedUploadTests
    resource/pinned_upload_test.cpp resource/observed_operations.cpp resource/allocation_fault.cpp)
target_compile_definitions(JuicerPinnedUploadTests PRIVATE
    JUICER_TEST_SEAMS=1
    JUICER_TEST_PRESET_NAME="${JUICER_TEST_PRESET_NAME}"
    JUICER_PREFLASH_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/print/preflash-fixtures"
    JUICER_DIR_BOUNDARY_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/dir/boundary-fixtures/${JUICER_TEST_PRESET_NAME}.json")
target_link_libraries(JuicerPinnedUploadTests PRIVATE
    juicer_core film_juicer_rust juicer_dependencies CUDA::cuda_driver GTest::gtest juicer_host_gtest_main)
set_property(TARGET JuicerPinnedUploadTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME Resource.Gpu.PinnedUpload COMMAND JuicerPinnedUploadTests --gtest_fail_if_no_test_selected)
set_tests_properties(Resource.Gpu.PinnedUpload PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerDeferredDestroyTests resource/deferred_destroy_test.cpp)
target_compile_definitions(JuicerDeferredDestroyTests PRIVATE JUICER_TEST_SEAMS=1)
target_link_libraries(JuicerDeferredDestroyTests PRIVATE
    juicer_core film_juicer_rust juicer_dependencies CUDA::cuda_driver)
set_property(TARGET JuicerDeferredDestroyTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME Resource.Gpu.DeferredDestroy COMMAND JuicerDeferredDestroyTests)
set_tests_properties(Resource.Gpu.DeferredDestroy PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerGrainOfxApertureTests
    grain/film_apertures_test.cpp grain/film_bank.cpp grain/film_oracle.cpp)
target_compile_definitions(JuicerGrainOfxApertureTests PRIVATE
    JUICER_GRAIN_TEST_BANK="${JUICER_TEST_RESOURCE_DIR}/grain/portra-400"
    JUICER_GRAIN_TEST_TABLES="${CMAKE_CURRENT_SOURCE_DIR}/grain/portra-table-fixtures"
    JUICER_GRAIN_APERTURE_WIRE="${CMAKE_CURRENT_SOURCE_DIR}/grain/film-aperture-geometry.txt")
target_link_libraries(JuicerGrainOfxApertureTests PRIVATE
    juicer_core film_juicer_rust juicer_cuda_resource_operations juicer_dependencies
    CUDA::cuda_driver GTest::gtest juicer_host_gtest_main)
set_property(TARGET JuicerGrainOfxApertureTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_test(NAME Grain.Gpu.FilmOfxApertures COMMAND JuicerGrainOfxApertureTests --gtest_fail_if_no_test_selected)
set_tests_properties(Grain.Gpu.FilmOfxApertures PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

foreach(target IN ITEMS JuicerScannerBindingTests JuicerCudaInspectTests JuicerCudaInspectGpuTests
        JuicerOfxModuleTests JuicerOfxContextTests JuicerOfxInstanceTests JuicerOfxActionTests JuicerOfxRenderTests)
    set_property(TARGET ${target} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endforeach()
