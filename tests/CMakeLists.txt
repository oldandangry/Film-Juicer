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

add_executable(JuicerCudaAbiTests ffi/cuda_abi_c.c ffi/cuda_abi_test.cpp)
set_target_properties(JuicerCudaAbiTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_include_directories(JuicerCudaAbiTests PRIVATE "${JUICER_TEST_ROOT}/native")
target_link_libraries(JuicerCudaAbiTests PRIVATE film_juicer_rust_test_support)
if(WIN32)
    target_compile_options(JuicerCudaAbiTests PRIVATE /W3 /WX)
else()
    target_compile_options(JuicerCudaAbiTests PRIVATE -Wall -Wextra -Werror -pedantic-errors)
endif()
add_test(NAME Ffi.Host.CudaAbi COMMAND JuicerCudaAbiTests)
set_tests_properties(Ffi.Host.CudaAbi PROPERTIES LABELS "host" TIMEOUT 300)

add_executable(JuicerOfxAbiTests ffi/ofx_abi_c.c ffi/ofx_abi_test.cpp)
set_target_properties(JuicerOfxAbiTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_include_directories(JuicerOfxAbiTests PRIVATE "${JUICER_TEST_ROOT}/native")
target_link_libraries(JuicerOfxAbiTests PRIVATE film_juicer_rust_test_support juicer_ofx_headers)
if(WIN32)
    target_compile_options(JuicerOfxAbiTests PRIVATE /W3 /WX)
else()
    target_compile_options(JuicerOfxAbiTests PRIVATE -Wall -Wextra -Werror -pedantic-errors)
endif()
add_test(NAME Ffi.Host.OfxAbi COMMAND JuicerOfxAbiTests)
set_tests_properties(Ffi.Host.OfxAbi PROPERTIES LABELS "host;boundary" TIMEOUT 300)

add_executable(JuicerProfileOwnerTests ffi/profile_abi_c.c ffi/profile_owner_test.cpp)
set_target_properties(JuicerProfileOwnerTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_include_directories(JuicerProfileOwnerTests PRIVATE "${JUICER_TEST_ROOT}/native")
target_link_libraries(JuicerProfileOwnerTests PRIVATE film_juicer_rust_test_support GTest::gtest)
if(WIN32)
    target_compile_options(JuicerProfileOwnerTests PRIVATE /W3 /WX)
else()
    target_compile_options(JuicerProfileOwnerTests PRIVATE -Wall -Wextra -Werror -pedantic-errors)
endif()
add_test(NAME Ffi.Host.ProfileOwner COMMAND JuicerProfileOwnerTests
    "${JUICER_TEST_ARTIFACT_ROOT}/ffi/profile-owner-resources")
set_tests_properties(Ffi.Host.ProfileOwner PROPERTIES LABELS "host" TIMEOUT 300)

add_executable(JuicerSpectralOwnerTests ffi/spectral_abi_c.c ffi/spectral_owner_test.cpp)
set_target_properties(JuicerSpectralOwnerTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_include_directories(JuicerSpectralOwnerTests PRIVATE "${JUICER_TEST_ROOT}/native")
target_link_libraries(JuicerSpectralOwnerTests PRIVATE film_juicer_rust_test_support GTest::gtest)
if(WIN32)
    target_compile_options(JuicerSpectralOwnerTests PRIVATE /W3 /WX)
else()
    target_compile_options(JuicerSpectralOwnerTests PRIVATE -Wall -Wextra -Werror -pedantic-errors)
endif()
add_test(NAME Ffi.Host.SpectralOwner COMMAND JuicerSpectralOwnerTests
    "${JUICER_TEST_RESOURCE_DIR}"
    "${JUICER_TEST_ARTIFACT_ROOT}/trusted-resource-decoding/C8/slice-3/spectral-owner")
set_tests_properties(Ffi.Host.SpectralOwner PROPERTIES LABELS "host" TIMEOUT 300)

# One explicit development owner; linkage determines dependencies, not test relevance.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/route/development.json")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/route/development.json" route_development)
string(JSON route_development_target GET "${route_development}" target)
string(JSON route_development_test GET "${route_development}" test)
string(JSON route_development_source GET "${route_development}" source)
add_executable(${route_development_target}
    route/route_bridge_test.cpp "${JUICER_TEST_ROOT}/${route_development_source}")
target_include_directories(JuicerRouteBridgeTests PRIVATE
    "${JUICER_TEST_ROOT}/src" "${JUICER_TEST_ROOT}/native")
target_link_libraries(JuicerRouteBridgeTests PRIVATE film_juicer_rust_test_support)
if(WIN32)
    target_compile_options(JuicerRouteBridgeTests PRIVATE /W3 /WX)
else()
    target_compile_options(JuicerRouteBridgeTests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ${route_development_test} COMMAND ${route_development_target})
set_tests_properties(Route.Host.Bridge PROPERTIES LABELS "host" TIMEOUT 300)

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

# Standalone foreign-owner conveniences have no production consumers. Their
# archive stays Rust-neutral; each final executable selects its actual variant.
add_library(juicer_asset_test_support STATIC
    ffi/capture_identity.cpp ffi/native_path.cpp route/route_fixture.cpp
    ffi/color_fixture.cpp ffi/gamut_fixture.cpp ffi/scanner_fixture.cpp ffi/frame_fixture.cpp
    ffi/dir_fixture.cpp ffi/effects_fixture.cpp ffi/diffusion_fixture.cpp ffi/halation_fixture.cpp
    ffi/owner_fixture.cpp
    ffi/test_asset_bridge.cpp ffi/test_asset_library.cpp)
target_link_libraries(juicer_asset_test_support PRIVATE juicer_settings juicer_ofx_headers)

# Keep the ordinary archive and instrumented replacements; share only the vendor
# FFT runtime between test executables. Product links select their own libraries.
add_library(juicer_test_core INTERFACE)
target_link_libraries(juicer_test_core INTERFACE
    juicer_core juicer_asset_test_support Threads::Threads ${CMAKE_DL_LIBS} CUDA::cufft)
if(WIN32)
    target_link_libraries(juicer_test_core INTERFACE CUDA::cudart)
else()
    target_link_libraries(juicer_test_core INTERFACE CUDA::cudart_static)
endif()

# Preserve the selected production/test-support Rust variant; the native core
# supplies its actual runtime dependency without introducing substitute symbols.
foreach(target IN ITEMS JuicerCudaAbiTests JuicerOfxAbiTests JuicerProfileOwnerTests
        JuicerSpectralOwnerTests JuicerRouteBridgeTests)
    target_link_libraries(${target} PRIVATE juicer_test_core)
endforeach()

add_library(juicer_noise_test_objects OBJECT ../native/juicer_cuda_api.cpp)
target_link_libraries(juicer_noise_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_noise_test_objects PRIVATE JUICER_NATIVE_CALL_TEST_HOOK=1)
add_executable(JuicerNoiseOwnerTests ffi/noise_owner_test.cpp $<TARGET_OBJECTS:juicer_noise_test_objects>)
target_link_libraries(JuicerNoiseOwnerTests PRIVATE juicer_test_core film_juicer_rust_test_support GTest::gtest)
target_compile_definitions(JuicerNoiseOwnerTests PRIVATE JUICER_NATIVE_CALL_TEST_HOOK=1)
set_target_properties(JuicerNoiseOwnerTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Ffi.Host.NoiseOwner COMMAND JuicerNoiseOwnerTests "${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_ARTIFACT_ROOT}/trusted-resource-decoding/C8/slice-5/noise-owner")
set_tests_properties(Ffi.Host.NoiseOwner PROPERTIES LABELS "host" TIMEOUT 300)

add_executable(JuicerCatalogOwnerTests ffi/catalog_abi_c.c ffi/catalog_owner_test.cpp
    ofx/context_host.cpp ../native/ofx_exports.cpp)
target_link_libraries(JuicerCatalogOwnerTests PRIVATE juicer_test_core film_juicer_rust_test_support juicer_ofx_headers)
add_dependencies(JuicerCatalogOwnerTests juicer_test_resources)
target_compile_definitions(JuicerCatalogOwnerTests PRIVATE
    JUICER_TEST_RESOURCE_DIR="${JUICER_TEST_RESOURCE_DIR}")
set_target_properties(JuicerCatalogOwnerTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Ffi.Host.CatalogOwner COMMAND JuicerCatalogOwnerTests
    "${JUICER_TEST_ARTIFACT_ROOT}/trusted-resource-decoding/C8/slice-2/catalog"
    "${JUICER_TEST_RESOURCE_DIR}")
set_tests_properties(Ffi.Host.CatalogOwner PROPERTIES LABELS "host" TIMEOUT 300)

add_executable(JuicerProductionProfileOwnerTests ffi/production_profile_abi_c.c ffi/production_profile_owner_test.cpp)
target_link_libraries(JuicerProductionProfileOwnerTests PRIVATE juicer_test_core film_juicer_rust_test_support GTest::gtest)

add_executable(JuicerFilmInputTests ffi/film_input_abi_c.c ffi/film_input_test.cpp)
target_link_libraries(JuicerFilmInputTests PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerFilmInputTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(WIN32)
    set(JUICER_FILM_INPUT_FIXTURE "windows.json")
else()
    set(JUICER_FILM_INPUT_FIXTURE "linux.json")
endif()
add_test(NAME Ffi.Host.FilmInput COMMAND JuicerFilmInputTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/input-fixtures/${JUICER_FILM_INPUT_FIXTURE}")
set_tests_properties(Ffi.Host.FilmInput PROPERTIES LABELS "host" TIMEOUT 300)
add_test(NAME Film.Host.InputProducts COMMAND JuicerFilmInputTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/input-fixtures/${JUICER_FILM_INPUT_FIXTURE}" products)
set_tests_properties(Film.Host.InputProducts PROPERTIES LABELS "host" TIMEOUT 300)

add_library(juicer_scanner_test_objects OBJECT ffi/scanner_fixture.cpp)
target_link_libraries(juicer_scanner_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_scanner_test_objects PRIVATE JUICER_SCANNER_TEST_HOOK=1)
add_library(juicer_gaussian_test_objects OBJECT ../src/Cuda/JuicerCudaResources.cpp)
target_link_libraries(juicer_gaussian_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_gaussian_test_objects PRIVATE JUICER_GAUSSIAN_TEST_HOOK=1)
add_library(juicer_dir_boundary_test_objects OBJECT ../src/Cuda/JuicerCudaResources.cpp)
target_link_libraries(juicer_dir_boundary_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_dir_boundary_test_objects PRIVATE JUICER_DIR_BOUNDARY_TEST_HOOK=1)
add_executable(JuicerFilmRecipeTests ffi/film_recipe_abi_c.c ffi/film_recipe_test.cpp  ffi/spatial_optics_fixture.cpp dir/boundary_test_hooks.cpp
    ffi/prepared_frame_fixture.cpp
    $<TARGET_OBJECTS:juicer_dir_boundary_test_objects>
    $<TARGET_OBJECTS:juicer_scanner_test_objects>)
target_link_libraries(JuicerFilmRecipeTests PRIVATE juicer_test_core film_juicer_rust_test_support)
target_compile_definitions(JuicerFilmRecipeTests PRIVATE
    JUICER_SCANNER_TEST_HOOK=1)
set_target_properties(JuicerFilmRecipeTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
juicer_add_host_test(Ffi.Host.EffectsFrame JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/effects-frame-fixtures/manifest.json" effects-frame "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Film.Host.EffectsFrameProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/effects-frame-fixtures/manifest.json" effects-frame-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Ffi.Host.GrainFrame JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/grain/frame-fixtures/manifest.json" grain-frame "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Film.Host.GrainFrameProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/grain/frame-fixtures/manifest.json" grain-frame-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Ffi.Host.DirBoundary JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/dir/boundary-fixtures/manifest.json" dir-boundary "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Film.Host.DirBoundaryProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/dir/boundary-fixtures/manifest.json" dir-boundary-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Ffi.Host.DirFrame JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/dir/frame-fixtures/manifest.json" dir-frame "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Film.Host.DirFrameProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/dir/frame-fixtures/manifest.json" dir-frame-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Ffi.Host.DiffusionFrame JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/diffusion/frame-fixtures/manifest.json" diffusion-frame "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Optics.Host.DiffusionFrameProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/diffusion/frame-fixtures/manifest.json" diffusion-frame-products "${JUICER_TEST_PRESET_NAME}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Keep the full numerical fixture separate at the preset's four-worker limit.
    set_tests_properties(Optics.Host.DiffusionFrameProducts PROPERTIES PROCESSORS 4)
endif()
juicer_add_host_test(Ffi.Host.HalationFrame JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scatter_halation/frame-fixtures/manifest.json" halation-frame "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Optics.Host.HalationFrameProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scatter_halation/frame-fixtures/manifest.json" halation-frame-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Ffi.Host.FrameGeometry JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/frame/geometry-fixtures/manifest.json" frame "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Frame.Host.GeometryProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/frame/geometry-fixtures/manifest.json" frame-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Ffi.Host.GaussianKernel JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/optics/gaussian-fixtures/manifest.json" gaussian "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Optics.Host.GaussianKernelProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/optics/gaussian-fixtures/manifest.json" gaussian-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Ffi.Host.ScannerPostEffects JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scanner/post-effects-fixtures/manifest.json" scanner-post "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Scanner.Host.PostEffectsProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scanner/post-effects-fixtures/manifest.json" scanner-post-products "${JUICER_TEST_PRESET_NAME}")
if(WIN32)
    set(JUICER_FILM_DIR_FIXTURE windows)
else()
    set(JUICER_FILM_DIR_FIXTURE linux)
endif()
juicer_add_host_test(Ffi.Host.FilmRecipes JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/dir-fixtures/${JUICER_FILM_DIR_FIXTURE}.json")
juicer_add_host_test(Ffi.Host.FilmReference JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/reference-fixtures/manifest.json" film-reference "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Film.Host.ReferenceProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/reference-fixtures/manifest.json" film-reference-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Film.Host.DirDensityProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/dir-fixtures/${JUICER_FILM_DIR_FIXTURE}.json" products)
juicer_add_host_test(Ffi.Host.VisualGrain JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/grain-fixtures/manifest.json" grain "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Film.Host.VisualGrainProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/grain-fixtures/manifest.json" grain-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Ffi.Host.PhysicalEffects JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/effects-fixtures/manifest.json" effects "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Film.Host.PhysicalEffectsProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/effects-fixtures/manifest.json" effects-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Ffi.Host.SpatialOptics JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/optics-fixtures/manifest.json" optics "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Film.Host.SpatialOpticsProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/optics-fixtures/manifest.json" optics-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Ffi.Host.PrintIlluminant JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/illuminant-fixtures/rust-manifest.json" print-illuminant "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Print.Host.IlluminantProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/illuminant-fixtures/rust-manifest.json" print-illuminant-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Ffi.Host.PrintPreflash JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/preflash-fixtures/manifest.json" print-preflash "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Print.Host.PreflashProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/preflash-fixtures/manifest.json" print-preflash-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Ffi.Host.PrintBalance JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/balance-fixtures/manifest.json" print-balance "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Print.Host.BalanceProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/balance-fixtures/manifest.json" print-balance-products "${JUICER_TEST_PRESET_NAME}")
juicer_add_host_test(Ffi.Host.ScannerCorrection JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scanner/correction-fixtures/manifest.json" scanner-correction "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.ScannerCorrection PROPERTIES
    ENVIRONMENT "JUICER_DIAGNOSTICS=2;JUICER_DIAGNOSTICS_PATH=${JUICER_TEST_ARTIFACT_ROOT}/scanner/correction-boundary-trace.log")
juicer_add_host_test(Scanner.Host.CorrectionProducts JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scanner/correction-fixtures/manifest.json" scanner-correction-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Scanner.Host.CorrectionProducts PROPERTIES
    ENVIRONMENT "JUICER_DIAGNOSTICS=2;JUICER_DIAGNOSTICS_PATH=${JUICER_TEST_ARTIFACT_ROOT}/scanner/correction-product-trace.log")
add_test(NAME Print.Gpu.BalanceCache COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/balance-fixtures/manifest.json" print-balance-cache "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Print.Gpu.BalanceCache PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
add_test(NAME Print.Gpu.PreflashCache COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/preflash-fixtures/manifest.json" print-preflash-cache "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Print.Gpu.PreflashCache PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
add_executable(JuicerFilmDevelopmentTests ffi/film_development_abi_c.c ffi/film_development_test.cpp)
target_link_libraries(JuicerFilmDevelopmentTests PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerFilmDevelopmentTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Ffi.Host.FilmDevelopment COMMAND JuicerFilmDevelopmentTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/fixtures/${JUICER_TEST_PRESET_NAME}.json")
set_tests_properties(Ffi.Host.FilmDevelopment PROPERTIES LABELS "host" TIMEOUT 300 PROCESSORS 2)
add_test(NAME Film.Host.DevelopmentProducts COMMAND JuicerFilmDevelopmentTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/fixtures/${JUICER_TEST_PRESET_NAME}.json" products)
set_tests_properties(Film.Host.DevelopmentProducts PROPERTIES LABELS "host" TIMEOUT 300 PROCESSORS 2)
set_target_properties(JuicerProductionProfileOwnerTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Ffi.Host.ProductionProfileOwner COMMAND JuicerProductionProfileOwnerTests
    "${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_ROOT}/tests/profile/fixtures"
    "${JUICER_TEST_ARTIFACT_ROOT}/trusted-resource-decoding/C8/slice-2/profile-owner")
set_tests_properties(Ffi.Host.ProductionProfileOwner PROPERTIES LABELS "host" TIMEOUT 300)

add_executable(JuicerCudaOwnerTests ffi/cuda_owner_test.cpp ffi/runtime_c.c)
target_link_libraries(JuicerCudaOwnerTests PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerCudaOwnerTests PROPERTIES LINKER_LANGUAGE CXX
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
add_test(NAME Ffi.Host.CudaOwner COMMAND JuicerCudaOwnerTests)
set_tests_properties(Ffi.Host.CudaOwner PROPERTIES LABELS "host" TIMEOUT 300)

# Terminal failure hooks exist only in this replacement API object.
add_library(juicer_terminal_test_objects OBJECT ../native/juicer_cuda_api.cpp)
target_link_libraries(juicer_terminal_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_terminal_test_objects PRIVATE JUICER_CUDA_TERMINAL_TEST_HOOK=1)
add_executable(JuicerCudaTerminalTests ffi/cuda_terminal_test.cpp $<TARGET_OBJECTS:juicer_terminal_test_objects>)
target_link_libraries(JuicerCudaTerminalTests PRIVATE juicer_test_core film_juicer_rust_test_support)
target_compile_definitions(JuicerCudaTerminalTests PRIVATE JUICER_CUDA_TERMINAL_TEST_HOOK=1
    JUICER_TEST_RESOURCE_DIR="${JUICER_TEST_RESOURCE_DIR}")
set_target_properties(JuicerCudaTerminalTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
foreach(terminal_case IN ITEMS host-cache-failure host-cache-native-precedence partial-library construction catalog-retention registered-reentry success serialization allocation standard unknown typed wording runtime context-loss diagnostic-allocation accepting-failure
        unregistered unregistered-reentry unregistered-shutdown-lock unregistered-destroy-lock
        saved-shutdown-lock saved-destroy-lock accepting-shutdown-lock accepting-destroy-lock
        buffers malformed-destroy malformed-blocked-destroy shutdown-progress)
    add_test(NAME Ffi.Host.CudaTerminal.${terminal_case} COMMAND JuicerCudaTerminalTests ${terminal_case})
    set_tests_properties(Ffi.Host.CudaTerminal.${terminal_case} PROPERTIES LABELS "host" TIMEOUT 300)
endforeach()

# Compile the allocation algorithm with a bounded test entry; production has no hook.
add_library(juicer_allocation_retry_test_objects OBJECT
    ../src/Cuda/ResourceManager/JuicerCudaResourceManager.cpp)
target_link_libraries(juicer_allocation_retry_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_allocation_retry_test_objects PRIVATE
    JUICER_ALLOCATION_RETRY_TEST_HOOK=1)
add_executable(JuicerCudaFailureTests ffi/cuda_failure_test.cpp
    $<TARGET_OBJECTS:juicer_allocation_retry_test_objects>)
target_link_libraries(JuicerCudaFailureTests PRIVATE juicer_test_core film_juicer_rust_test_support GTest::gtest juicer_host_gtest_main)
target_compile_definitions(JuicerCudaFailureTests PRIVATE JUICER_ALLOCATION_RETRY_TEST_HOOK=1)
set_target_properties(JuicerCudaFailureTests PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME Ffi.Host.CudaFailure COMMAND JuicerCudaFailureTests)
set_tests_properties(Ffi.Host.CudaFailure PROPERTIES LABELS "host" TIMEOUT 300)

# The staging probe runs the production upload algorithm with isolated operation seams.
add_library(juicer_pinned_upload_test_objects OBJECT ../src/Cuda/JuicerCudaResources.cpp)
target_link_libraries(juicer_pinned_upload_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_pinned_upload_test_objects PRIVATE JUICER_PINNED_UPLOAD_TEST_HOOK=1)
add_executable(JuicerPinnedUploadTests resource/pinned_upload_test.cpp
    $<TARGET_OBJECTS:juicer_pinned_upload_test_objects>)
target_link_libraries(JuicerPinnedUploadTests PRIVATE juicer_test_core film_juicer_rust_test_support GTest::gtest juicer_host_gtest_main CUDA::cuda_driver)
target_compile_definitions(JuicerPinnedUploadTests PRIVATE JUICER_PINNED_UPLOAD_TEST_HOOK=1
    JUICER_TEST_RESOURCE_DIR="${JUICER_TEST_RESOURCE_DIR}")
set_target_properties(JuicerPinnedUploadTests PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME Resource.Gpu.PinnedUpload COMMAND JuicerPinnedUploadTests)
set_tests_properties(Resource.Gpu.PinnedUpload PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

# Gaussian cache failure and collision cases use the real context-owned miss path.
add_executable(JuicerGaussianKernelTests resource/gaussian_kernel_test.cpp
    resource/gaussian_kernel_test_hooks.cpp ffi/spectral_allocation_probe.cpp
    $<TARGET_OBJECTS:juicer_gaussian_test_objects>)
target_link_libraries(JuicerGaussianKernelTests PRIVATE juicer_test_core film_juicer_rust_test_support CUDA::cuda_driver)
target_compile_definitions(JuicerGaussianKernelTests PRIVATE JUICER_GAUSSIAN_TEST_HOOK=1)
set_target_properties(JuicerGaussianKernelTests PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME Resource.Gpu.GaussianKernel COMMAND JuicerGaussianKernelTests
    "${CMAKE_CURRENT_SOURCE_DIR}/optics/gaussian-fixtures/${JUICER_TEST_PRESET_NAME}.json")
set_tests_properties(Resource.Gpu.GaussianKernel PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

# Deferred retirement executes the real drain with a bounded post-extraction seam.
add_library(juicer_deferred_destroy_test_objects OBJECT ../src/Cuda/JuicerCudaResources.cpp)
target_link_libraries(juicer_deferred_destroy_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_deferred_destroy_test_objects PRIVATE JUICER_DEFERRED_DESTROY_TEST_HOOK=1)
add_executable(JuicerDeferredDestroyTests resource/deferred_destroy_test.cpp
    $<TARGET_OBJECTS:juicer_deferred_destroy_test_objects>)
target_link_libraries(JuicerDeferredDestroyTests PRIVATE juicer_test_core film_juicer_rust_test_support CUDA::cuda_driver)
target_compile_definitions(JuicerDeferredDestroyTests PRIVATE JUICER_DEFERRED_DESTROY_TEST_HOOK=1)
set_target_properties(JuicerDeferredDestroyTests PROPERTIES LINKER_LANGUAGE CXX)
foreach(deferred_case IN ITEMS lock allocation storage ordinary control)
    add_test(NAME Resource.Gpu.DeferredDestroy.${deferred_case} COMMAND JuicerDeferredDestroyTests ${deferred_case})
    set_tests_properties(Resource.Gpu.DeferredDestroy.${deferred_case} PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
endforeach()

add_executable(JuicerCudaInspectTests ffi/cuda_inspect_test.cpp)
target_link_libraries(JuicerCudaInspectTests PRIVATE juicer_test_core film_juicer_rust_test_support GTest::gtest juicer_host_gtest_main)
set_target_properties(JuicerCudaInspectTests PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME Ffi.Host.CudaInspect COMMAND JuicerCudaInspectTests --gtest_fail_if_no_test_selected --gtest_filter=InspectHost.*)
set_tests_properties(Ffi.Host.CudaInspect PROPERTIES LABELS "host" TIMEOUT 300)
# Keep the host binary independent of the NVIDIA driver at load time.
add_executable(JuicerCudaInspectGpuTests ffi/cuda_inspect_test.cpp)
target_compile_definitions(JuicerCudaInspectGpuTests PRIVATE JUICER_CUDA_INSPECT_GPU_TEST=1)
target_link_libraries(JuicerCudaInspectGpuTests PRIVATE juicer_test_core film_juicer_rust_test_support GTest::gtest juicer_host_gtest_main CUDA::cuda_driver)
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
target_link_libraries(JuicerOfxModuleTests PRIVATE juicer_test_core film_juicer_rust juicer_ofx_headers)
set_target_properties(JuicerOfxModuleTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
juicer_add_host_test(Ofx.Host.Module JuicerOfxModuleTests
    "${CMAKE_CURRENT_SOURCE_DIR}/ofx/descriptor_manifest.json")

add_executable(JuicerOfxContextTests ofx/context_test.cpp ofx/context_host.cpp ../native/ofx_exports.cpp)
target_compile_features(JuicerOfxContextTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxContextTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxContextTests PRIVATE juicer_test_core film_juicer_rust juicer_ofx_headers)
set_target_properties(JuicerOfxContextTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerOfxContextTests juicer_test_resources)
juicer_add_host_test(Ofx.Host.Context JuicerOfxContextTests --reference "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/s6-d/context.json")
set_tests_properties(Ofx.Host.Context PROPERTIES LABELS "host;boundary")

add_executable(JuicerOfxContextControlTests ofx/context_control_test.cpp ofx/context_host.cpp)
target_compile_features(JuicerOfxContextControlTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxContextControlTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxContextControlTests PRIVATE juicer_test_core film_juicer_rust_test_support juicer_ofx_headers)
set_target_properties(JuicerOfxContextControlTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerOfxContextControlTests juicer_test_resources)
juicer_add_host_test(Ofx.Host.ContextControls JuicerOfxContextControlTests)
set_tests_properties(Ofx.Host.ContextControls PROPERTIES LABELS "host;boundary")

add_executable(JuicerOfxInstanceTests ofx/instance_test.cpp ofx/instance_host.cpp ../native/ofx_exports.cpp)
target_compile_features(JuicerOfxInstanceTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxInstanceTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxInstanceTests PRIVATE juicer_test_core film_juicer_rust juicer_ofx_headers)
set_target_properties(JuicerOfxInstanceTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerOfxInstanceTests juicer_test_resources)
juicer_add_host_test(Ofx.Host.Instance JuicerOfxInstanceTests
    --reference "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/s6-d/instance.json")
juicer_add_host_test(Ofx.Host.FilmV1Instance JuicerOfxInstanceTests
    --film-v1 --reference "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/s6-d/instance.json")

add_executable(JuicerOfxActionTests ofx/action_test.cpp ofx/instance_host.cpp ../native/ofx_exports.cpp)
target_compile_features(JuicerOfxActionTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxActionTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxActionTests PRIVATE juicer_test_core film_juicer_rust juicer_ofx_headers)
set_target_properties(JuicerOfxActionTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerOfxActionTests juicer_test_resources)
juicer_add_host_test(Ofx.Host.Actions JuicerOfxActionTests --reference "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/s6-d/actions.json")
set_tests_properties(Ofx.Host.Actions PROPERTIES LABELS "host;boundary")

add_executable(JuicerOfxRenderTests ofx/render_test.cpp ofx/instance_host.cpp ../native/ofx_exports.cpp)
target_compile_features(JuicerOfxRenderTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxRenderTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxRenderTests PRIVATE juicer_test_core film_juicer_rust juicer_ofx_headers)
set_target_properties(JuicerOfxRenderTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerOfxRenderTests juicer_test_resources)
juicer_add_host_test(Ofx.Host.Render JuicerOfxRenderTests
    --reference "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/s6-d/render.json")

# Cargo can reuse an archive with an older timestamp after a feature switch.
# A harness built after the switch still needs to relink for the selected variant.
foreach(ofx_callback_target IN ITEMS JuicerOfxModuleTests JuicerOfxInstanceTests JuicerOfxRenderTests)
    set_property(TARGET ${ofx_callback_target} APPEND PROPERTY LINK_DEPENDS
        "$<TARGET_PROPERTY:film_juicer_rust,JUICER_RUST_FEATURE_INPUT>")
endforeach()

foreach(reference_target IN ITEMS JuicerOfxContextTests JuicerOfxInstanceTests JuicerOfxActionTests JuicerOfxRenderTests)
    target_sources(${reference_target} PRIVATE ofx/host_reference.cpp)
    target_compile_definitions(${reference_target} PRIVATE
        JUICER_TEST_PRESET_NAME="${JUICER_TEST_PRESET_NAME}")
endforeach()

add_executable(JuicerOutputColorValueTests ffi/output-color-values.c ffi/value_probe_test.cpp)
juicer_add_host_test(Ffi.Host.OutputColorValues JuicerOutputColorValueTests)
set_target_properties(JuicerOutputColorValueTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
target_include_directories(JuicerOutputColorValueTests PRIVATE "${JUICER_TEST_ROOT}/native"
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi")
target_link_libraries(JuicerOutputColorValueTests PRIVATE juicer_test_core film_juicer_rust_test_support juicer_dependencies)
if(WIN32)
    target_compile_options(JuicerOutputColorValueTests PRIVATE $<$<COMPILE_LANGUAGE:C>:/W4;/WX;/Od;/Ob0;/UNDEBUG>)
else()
    target_compile_options(JuicerOutputColorValueTests PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Werror;-O0;-UNDEBUG>)
    set_property(TARGET JuicerOutputColorValueTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
set_tests_properties(Ffi.Host.OutputColorValues PROPERTIES LABELS "host;boundary")

add_executable(JuicerOfxRenderControlTests ofx/render_control_test.cpp ofx/instance_host.cpp ../native/ofx_exports.cpp)
target_compile_features(JuicerOfxRenderControlTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxRenderControlTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxRenderControlTests PRIVATE juicer_test_core film_juicer_rust_test_support juicer_ofx_headers)
set_target_properties(JuicerOfxRenderControlTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerOfxRenderControlTests juicer_test_resources)
juicer_add_host_test(Ofx.Host.RenderControls JuicerOfxRenderControlTests)
set_tests_properties(Ofx.Host.RenderControls PROPERTIES LABELS "host;boundary")

add_executable(JuicerOfxCaptureTests ofx/capture_test.cpp ofx/instance_host.cpp ../native/ofx_exports.cpp)
target_compile_features(JuicerOfxCaptureTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxCaptureTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxCaptureTests PRIVATE juicer_test_core film_juicer_rust_test_support juicer_ofx_headers)
set_target_properties(JuicerOfxCaptureTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerOfxCaptureTests juicer_test_resources)
juicer_add_host_test(Ofx.Host.Capture JuicerOfxCaptureTests "${CMAKE_CURRENT_SOURCE_DIR}/film/grain-fixtures/manifest.json" "${JUICER_TEST_PRESET_NAME}"
    --identity-reference "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/s6-d/capture-identity.json")
set_tests_properties(Ofx.Host.Capture PROPERTIES LABELS "host;boundary")

add_executable(JuicerOfxInstanceControlTests ofx/instance_control_test.cpp ofx/instance_host.cpp)
target_compile_features(JuicerOfxInstanceControlTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxInstanceControlTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxInstanceControlTests PRIVATE juicer_test_core film_juicer_rust_test_support juicer_ofx_headers)
set_target_properties(JuicerOfxInstanceControlTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerOfxInstanceControlTests juicer_test_resources)
juicer_add_host_test(Ofx.Host.InstanceControls JuicerOfxInstanceControlTests)
set_tests_properties(Ofx.Host.InstanceControls PROPERTIES LABELS "host;boundary")

add_executable(JuicerOfxModuleControlTests ofx/module_control_test.cpp ofx/module_host.cpp ofx/platform_abi_c.c)
target_compile_features(JuicerOfxModuleControlTests PRIVATE cxx_std_20)
target_include_directories(JuicerOfxModuleControlTests PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
target_link_libraries(JuicerOfxModuleControlTests PRIVATE juicer_test_core film_juicer_rust_test_support juicer_ofx_headers)
set_target_properties(JuicerOfxModuleControlTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}" C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
foreach(mode IN ITEMS Construction Panic Terminal)
    string(TOLOWER "${mode}" mode_argument)
    juicer_add_host_test(Ofx.Host.ModuleControls.${mode} JuicerOfxModuleControlTests "${mode_argument}")
    set_tests_properties(Ofx.Host.ModuleControls.${mode} PROPERTIES LABELS "host;boundary")
endforeach()

add_executable(JuicerCudaOwnerUnloadTests ofx/cuda_owner_unload_test.cpp ofx/instance_host.cpp ../native/ofx_exports.cpp)
target_compile_definitions(JuicerCudaOwnerUnloadTests PRIVATE JUICER_OWNER_UNLOAD_TEST=1)
target_link_libraries(JuicerCudaOwnerUnloadTests PRIVATE juicer_test_core film_juicer_rust_test_support juicer_ofx_headers)
set_target_properties(JuicerCudaOwnerUnloadTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerCudaOwnerUnloadTests juicer_test_resources)
add_test(NAME Ofx.Host.CudaOwnerUnload COMMAND JuicerCudaOwnerUnloadTests)
set_tests_properties(Ofx.Host.CudaOwnerUnload PROPERTIES LABELS "host" TIMEOUT 300)

add_dependencies(JuicerCudaTerminalTests juicer_test_resources)

add_executable(JuicerScatterHalationHostTests
    scatter_halation/recipe_descriptor_test.cpp)
target_sources(JuicerScatterHalationHostTests PRIVATE ffi/spatial_optics_fixture.cpp)
target_include_directories(JuicerScatterHalationHostTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ffi")
target_link_libraries(JuicerScatterHalationHostTests PRIVATE
    juicer_test_core film_juicer_rust_test_support GTest::gtest)
target_compile_definitions(JuicerScatterHalationHostTests PRIVATE
    JUICER_TEST_RESOURCE_DIR="${JUICER_TEST_RESOURCE_DIR}")
set_target_properties(JuicerScatterHalationHostTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerScatterHalationHostTests juicer_test_resources)

add_executable(JuicerScratchRequestTests resource/scratch_request_test.cpp)
target_include_directories(JuicerScratchRequestTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ffi")
target_link_libraries(JuicerScratchRequestTests PRIVATE juicer_core film_juicer_rust GTest::gtest juicer_host_gtest_main)
set_target_properties(JuicerScratchRequestTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Resource.Host.ScratchRequest COMMAND JuicerScratchRequestTests)
set_tests_properties(Resource.Host.ScratchRequest PROPERTIES LABELS "host" TIMEOUT 300)

set(JUICER_PREPARED_TEST_SOURCES
    ffi/prepared_projection_test.cpp ffi/prepared_span_test.cpp
    ffi/prepared_descriptors.cpp)
add_executable(JuicerPreparedProjectionTests ${JUICER_PREPARED_TEST_SOURCES} ffi/spatial_optics_fixture.cpp)
target_link_libraries(JuicerPreparedProjectionTests PRIVATE juicer_test_core film_juicer_rust_test_support GTest::gtest juicer_host_gtest_main)
target_compile_definitions(JuicerPreparedProjectionTests PRIVATE
    FJ_PREPARED_STATE_PRESET="${JUICER_TEST_PRESET_NAME}"
    FJ_SCANNER_BINDING_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/scanner/color-fixtures/scanner-binding-v1.json")
set_target_properties(JuicerPreparedProjectionTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Ffi.Host.PreparedProjection COMMAND JuicerPreparedProjectionTests)
set_tests_properties(Ffi.Host.PreparedProjection PROPERTIES LABELS "host" TIMEOUT 300)
set_tests_properties(Ffi.Host.PreparedProjection PROPERTIES ENVIRONMENT "JUICER_TEST_PRESET_NAME=${JUICER_TEST_PRESET_NAME}")
add_executable(JuicerPreparedStateTests
    ffi/prepared_state_test.cpp ffi/prepared_state_abi_c.c
    ffi/prepared_state_oracle.cpp
    ffi/prepared_descriptors.cpp ffi/spatial_optics_fixture.cpp)
target_link_libraries(JuicerPreparedStateTests PRIVATE juicer_test_core film_juicer_rust_test_support GTest::gtest)
target_compile_definitions(JuicerPreparedStateTests PRIVATE
    FJ_PREPARED_STATE_PRESET="${JUICER_TEST_PRESET_NAME}")
set_target_properties(JuicerPreparedStateTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerPreparedStateTests juicer_test_resources)
add_test(NAME Ffi.Host.PreparedState COMMAND JuicerPreparedStateTests "${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_ARTIFACT_ROOT}/s4-e-integration-replacement" --gtest_fail_if_no_test_selected --gtest_filter=PreparedStateAbi.* --expect-projection-oracle=unused)
add_test(NAME Prepare.Host.StateProducts COMMAND JuicerPreparedStateTests "${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_ARTIFACT_ROOT}/s4-e-integration-replacement" --gtest_fail_if_no_test_selected --gtest_filter=PreparedStateProducts.* --expect-projection-oracle=loaded)
set_tests_properties(Ffi.Host.PreparedState Prepare.Host.StateProducts PROPERTIES LABELS "host" TIMEOUT 300)
set_tests_properties(Ffi.Host.PreparedState Prepare.Host.StateProducts PROPERTIES
    ENVIRONMENT "JUICER_TEST_PRESET_NAME=${JUICER_TEST_PRESET_NAME}")
include(ffi/prepared_sanitizers.cmake)
add_test(NAME Ffi.Host.SanitizerRunner
    COMMAND "${Python3_EXECUTABLE}" -B -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi"
        -p test_prepared_sanitizer_control.py)
set_tests_properties(Ffi.Host.SanitizerRunner PROPERTIES LABELS "host" TIMEOUT 300)
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

# Raw lookup facades and prepared-state publication use the existing test consumer.
add_executable(JuicerAssetLookupTests ffi/asset_lookup_test.cpp)
target_link_libraries(JuicerAssetLookupTests PRIVATE juicer_test_core film_juicer_rust_test_support GTest::gtest juicer_host_gtest_main)
target_compile_definitions(JuicerAssetLookupTests PRIVATE JUICER_LOOKUP_RESOURCE_DIR="${JUICER_TEST_RESOURCE_DIR}")
set_target_properties(JuicerAssetLookupTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Assets.Host.LookupSafety COMMAND JuicerAssetLookupTests)
set_tests_properties(Assets.Host.LookupSafety PROPERTIES LABELS "host" TIMEOUT 300)


gtest_add_tests(TARGET JuicerScatterHalationHostTests
    EXTRA_ARGS --gtest_fail_if_no_test_selected
    TEST_PREFIX "ScatterHalation.Host." TEST_LIST juicer_scatter_host_tests)
set_tests_properties(${juicer_scatter_host_tests} PROPERTIES
    LABELS "host" TIMEOUT 300
    ENVIRONMENT "JUICER_TEST_ARTIFACT_DIR=${JUICER_TEST_ARTIFACT_ROOT}/scatter-halation-host")

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
add_custom_target(juicer_diffusion_reference DEPENDS "${DIFFUSION_GENERATED_HEADER}")
add_dependencies(juicer_analysis_inputs juicer_diffusion_reference)
add_executable(JuicerDiffusionHostReferenceTest
    diffusion/host_reference_test.cpp
    "${DIFFUSION_GENERATED_HEADER}")
target_include_directories(JuicerDiffusionHostReferenceTest PRIVATE
    "${DIFFUSION_GENERATED_DIR}")
target_sources(JuicerDiffusionHostReferenceTest PRIVATE ffi/spatial_optics_fixture.cpp)
target_include_directories(JuicerDiffusionHostReferenceTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ffi")
target_link_libraries(JuicerDiffusionHostReferenceTest PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerDiffusionHostReferenceTest PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Diffusion.HostReference
    COMMAND JuicerDiffusionHostReferenceTest)
set_tests_properties(Diffusion.HostReference PROPERTIES
    LABELS "host;reference"
    TIMEOUT 300)

add_test(NAME Gamma.Comparator
    COMMAND "${Python3_EXECUTABLE}" -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/gamma"
        -p test_compare.py
        -v)
set_tests_properties(Gamma.Comparator PROPERTIES
    LABELS "host"
    TIMEOUT 300
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
        TIMEOUT 300
        ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
endif()

set(juicer_processor_reference "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/spectral-v1/processor-${JUICER_TEST_PRESET_NAME}.json")
add_executable(JuicerProcessorReferenceProbe ofx/processor_reference_test.cpp ofx/render_assertions.cpp
    ofx/instance_host.cpp ../native/ofx_exports.cpp)
target_sources(JuicerProcessorReferenceProbe PRIVATE ffi/spatial_optics_fixture.cpp ffi/prepared_frame_fixture.cpp)
target_sources(JuicerProcessorReferenceProbe PRIVATE ofx/scanner_reference_capture.cpp ../src/Cuda/Scan/JuicerCudaScanPipeline.cu)
target_link_libraries(JuicerProcessorReferenceProbe PRIVATE
    juicer_test_core film_juicer_rust_test_support CUDA::cuda_driver)
target_compile_definitions(JuicerProcessorReferenceProbe PRIVATE
    JUICER_SCAN_DENSITY_CAPTURE_TEST=1
    JUICER_ACCEPTED_CAPTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/spectral-v1/accepted-clean-${JUICER_TEST_PRESET_NAME}.json"
    JUICER_ACCEPTED_CAPTURE_PRESET="${JUICER_TEST_PRESET_NAME}"
    JUICER_ARCHIVED_ACCEPTED_CAPTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/accepted_cuda_clean_samples.json"
    JUICER_PROCESSOR_REFERENCE_PATH="${juicer_processor_reference}"
    JUICER_PROCESSOR_REFERENCE_PRESET="${JUICER_TEST_PRESET_NAME}")
set_target_properties(JuicerProcessorReferenceProbe PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerProcessorReferenceProbe juicer_test_resources)
add_test(NAME Ofx.Gpu.ProcessorReference COMMAND JuicerProcessorReferenceProbe)
add_test(NAME Ofx.Gpu.ProcessorReferenceInputs COMMAND JuicerProcessorReferenceProbe --processor-reference-inputs)
set_tests_properties(Ofx.Gpu.ProcessorReferenceInputs PROPERTIES LABELS "gpu;reference" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
string(REGEX REPLACE "-debug$" "-release" juicer_processor_scanner_preset "${JUICER_TEST_PRESET_NAME}")
add_test(NAME Ofx.Gpu.ProcessorScannerReferenceInputs COMMAND JuicerProcessorReferenceProbe --processor-scanner-reference-inputs
    "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/spectral-v1/processor-scanner-${juicer_processor_scanner_preset}.json")
set_tests_properties(Ofx.Gpu.ProcessorScannerReferenceInputs PROPERTIES LABELS "gpu;reference" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
set_tests_properties(Ofx.Gpu.ProcessorReference PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_test(NAME Ofx.Gpu.SequentialOwners COMMAND JuicerProcessorReferenceProbe --sequential-owners)
set_tests_properties(Ofx.Gpu.SequentialOwners PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_test(NAME Ofx.Gpu.CutoverContract COMMAND JuicerProcessorReferenceProbe --cutover-contract)
set_tests_properties(Ofx.Gpu.CutoverContract PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_library(juicer_render_test_objects OBJECT
    ../native/juicer_cuda_api.cpp ../native/juicer_cuda_call_scope.cpp ../src/Cuda/JuicerCudaExecutor.cpp ../src/ProcessRoot.cpp)
target_link_libraries(juicer_render_test_objects PRIVATE juicer_settings juicer_ofx_headers)
target_compile_definitions(juicer_render_test_objects PRIVATE JUICER_CUDA_RENDER_TEST_HOOK=1)
add_executable(JuicerPreparedBoundaryProbe
    ofx/processor_reference_test.cpp ofx/render_assertions.cpp
    ofx/scanner_reference_capture.cpp ../src/Cuda/Scan/JuicerCudaScanPipeline.cu
    ofx/instance_host.cpp ../native/ofx_exports.cpp
    ffi/spatial_optics_fixture.cpp
    ffi/prepared_frame_fixture.cpp
    ffi/prepared_boundary.cpp
    ffi/production_state_contract.cpp
    ffi/cuda_render_test.cpp
    $<TARGET_OBJECTS:juicer_render_test_objects>
    $<TARGET_OBJECTS:juicer_pinned_upload_test_objects>
    ffi/prepared_boundary.c
    ffi/prepared_descriptors.cpp)
target_link_libraries(JuicerPreparedBoundaryProbe PRIVATE juicer_test_core film_juicer_rust_test_support CUDA::cuda_driver)
add_test(NAME Ffi.Gpu.ProductionPreparedRender
    COMMAND JuicerProductionPreparedRenderProbe
        "${JUICER_STAGE_PREFIX}/${JUICER_BUNDLE}/Contents/${JUICER_ARCH_DIR}/$<TARGET_FILE_NAME:juicer>"
        "${JUICER_STAGE_PREFIX}/${JUICER_BUNDLE}/Contents/Resources"
        "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/spectral-v1/ordinary-${JUICER_TEST_PRESET_NAME}.json"
        "${JUICER_TEST_PRESET_NAME}" --summary)
set_tests_properties(Ffi.Gpu.ProductionPreparedRender PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
add_test(NAME Ffi.Gpu.FilmV1Staged
    COMMAND JuicerProductionPreparedRenderProbe --film-v1-witness
        "${JUICER_STAGE_PREFIX}/${JUICER_BUNDLE}/Contents/${JUICER_ARCH_DIR}/$<TARGET_FILE_NAME:juicer>")
set_tests_properties(Ffi.Gpu.FilmV1Staged PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
if(juicer_grain_aperture_consumer)
    add_test(NAME Grain.Gpu.FilmOfxApertures COMMAND ${juicer_grain_aperture_consumer} --film-v1-apertures)
    set_tests_properties(Grain.Gpu.FilmOfxApertures PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
endif()
if(MSVC)
    target_compile_options(JuicerPreparedBoundaryProbe PRIVATE $<$<COMPILE_LANGUAGE:C>:/W3;/WX>)
else()
    target_compile_options(JuicerPreparedBoundaryProbe PRIVATE
        $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Werror;-pedantic-errors>)
endif()
target_compile_definitions(JuicerPreparedBoundaryProbe PRIVATE
    JUICER_PINNED_UPLOAD_TEST_HOOK=1
    JUICER_CUDA_RENDER_TEST_HOOK=1
    JUICER_SCAN_DENSITY_CAPTURE_TEST=1
    JUICER_PREPARED_BOUNDARY_TEST=1
    JUICER_TEST_RESOURCE_DIR="${JUICER_TEST_RESOURCE_DIR}"
    JUICER_PROCESSOR_REFERENCE_PATH="${juicer_processor_reference}"
    JUICER_PROCESSOR_REFERENCE_PRESET="${JUICER_TEST_PRESET_NAME}")
set_target_properties(JuicerPreparedBoundaryProbe PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerPreparedBoundaryProbe juicer_test_resources)
foreach(prepared_case IN ITEMS negative-direct negative-print positive-direct positive-print
        combined-print glare-plus-zero glare-minus-zero mallett-direct arctic-print)
    add_test(NAME "Ffi.Gpu.PreparedBoundary.${prepared_case}"
        COMMAND JuicerPreparedBoundaryProbe --prepared-case "${prepared_case}")
    set_tests_properties("Ffi.Gpu.PreparedBoundary.${prepared_case}" PROPERTIES
        LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
endforeach()
set_property(TEST Ffi.Gpu.PreparedBoundary.negative-print APPEND PROPERTY
    ENVIRONMENT "JUICER_TEST_ARTIFACT_DIR=${JUICER_TEST_ARTIFACT_ROOT}/print-preflash")

foreach(render_case IN ITEMS negative-direct negative-print positive-direct positive-print)
    add_test(NAME "Ffi.Gpu.CudaRender.${render_case}"
        COMMAND JuicerPreparedBoundaryProbe --state-contract "${render_case}")
    set_tests_properties("Ffi.Gpu.CudaRender.${render_case}" PROPERTIES
        LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
endforeach()

add_test(NAME Ffi.Gpu.PreparedStateLifetime
    COMMAND JuicerPreparedBoundaryProbe --state-lifetime direct-and-print)
set_tests_properties(Ffi.Gpu.PreparedStateLifetime PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

# Isolated copies keep failure injection and observations out of juicer_core.
add_library(juicer_executor_failure_test_objects OBJECT
    ../native/juicer_cuda_api.cpp
    ../native/juicer_cuda_call_scope.cpp
    ../src/Cuda/JuicerCudaExecutor.cpp
    ../src/Cuda/JuicerCudaResources.cpp
    ../src/ProcessRoot.cpp)
target_link_libraries(juicer_executor_failure_test_objects PRIVATE juicer_settings juicer_ofx_headers)
target_compile_definitions(juicer_executor_failure_test_objects PRIVATE
    JUICER_EXECUTOR_FAILURE_TEST_HOOK=1)
add_executable(JuicerExecutorFailureOrderProbe ofx/processor_reference_test.cpp ofx/processor_test_hooks.cpp ofx/render_assertions.cpp
    ofx/instance_host.cpp ../native/ofx_exports.cpp
    ofx/scanner_reference_capture.cpp ../src/Cuda/Scan/JuicerCudaScanPipeline.cu
    $<TARGET_OBJECTS:juicer_executor_failure_test_objects> ffi/spatial_optics_fixture.cpp ffi/prepared_frame_fixture.cpp)
target_link_libraries(JuicerExecutorFailureOrderProbe PRIVATE
    juicer_test_core film_juicer_rust_test_support CUDA::cuda_driver)
target_compile_definitions(JuicerExecutorFailureOrderProbe PRIVATE
    JUICER_EXECUTOR_FAILURE_TEST_HOOK=1
    JUICER_SCAN_DENSITY_CAPTURE_TEST=1
    JUICER_PROCESSOR_REFERENCE_PATH="${juicer_processor_reference}"
    JUICER_PROCESSOR_REFERENCE_PRESET="${JUICER_TEST_PRESET_NAME}"
    JUICER_WORKSPACE_REFERENCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/spectral-v1/workspace-${JUICER_TEST_PRESET_NAME}.json")
set_target_properties(JuicerExecutorFailureOrderProbe PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerExecutorFailureOrderProbe juicer_test_resources)
add_test(NAME Ofx.Gpu.NoiseSourceLifetime COMMAND JuicerExecutorFailureOrderProbe --noise-lifetime)
set_tests_properties(Ofx.Gpu.NoiseSourceLifetime PROPERTIES
    LABELS "gpu"
    TIMEOUT 300
    RESOURCE_LOCK juicer_gpu
    ENVIRONMENT "JUICER_TEST_ARTIFACT_DIR=${JUICER_TEST_ARTIFACT_ROOT}/noise-source-lifetime")
add_test(NAME Ofx.Gpu.WorkspaceTransitions COMMAND JuicerExecutorFailureOrderProbe --workspace-transitions)
add_test(NAME Ofx.Gpu.WorkspaceReferenceInputs COMMAND JuicerExecutorFailureOrderProbe --workspace-reference-inputs)
set_tests_properties(Ofx.Gpu.WorkspaceReferenceInputs PROPERTIES LABELS "gpu;reference" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
set_tests_properties(Ofx.Gpu.WorkspaceTransitions PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
add_test(NAME Ofx.Gpu.GrainFrame COMMAND JuicerExecutorFailureOrderProbe --grain-frame
    "${CMAKE_CURRENT_SOURCE_DIR}/grain/frame-fixtures/manifest.json" "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ofx.Gpu.GrainFrame PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
add_test(NAME Ofx.Gpu.EffectsFrame COMMAND JuicerExecutorFailureOrderProbe --effects-frame
    "${CMAKE_CURRENT_SOURCE_DIR}/film/effects-frame-fixtures/manifest.json" "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ofx.Gpu.EffectsFrame PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
add_test(NAME Ofx.Gpu.DirFrame COMMAND JuicerExecutorFailureOrderProbe --dir-frame
    "${CMAKE_CURRENT_SOURCE_DIR}/dir/frame-fixtures/manifest.json" "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ofx.Gpu.DirFrame PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_test(NAME Ofx.Gpu.ExecutorFailureOrder
    COMMAND JuicerExecutorFailureOrderProbe --failure-order)
set_tests_properties(Ofx.Gpu.ExecutorFailureOrder PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
foreach(resource_failure IN ITEMS DefectFence GrainUpload OutputGamut)
    add_test(NAME "Ofx.Gpu.ResourceFailure.${resource_failure}"
        COMMAND JuicerExecutorFailureOrderProbe --resource-failure "${resource_failure}")
    set_tests_properties("Ofx.Gpu.ResourceFailure.${resource_failure}" PROPERTIES
        LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
endforeach()

add_executable(JuicerOfxMeasurementTests benchmarks/observation_test.cpp
    benchmarks/observation.cpp benchmarks/observation_contract.cpp benchmarks/observation_c11.c)
target_link_libraries(JuicerOfxMeasurementTests PRIVATE
    juicer_settings GTest::gtest juicer_host_gtest_main)
set_target_properties(JuicerOfxMeasurementTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF JUICER_RUST_VARIANT none
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(WIN32)
    target_compile_options(JuicerOfxMeasurementTests PRIVATE $<$<COMPILE_LANGUAGE:C>:/TC;/W4;/WX>)
else()
    target_compile_options(JuicerOfxMeasurementTests PRIVATE $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Werror;-pedantic-errors>)
endif()
juicer_add_host_test(Ofx.Host.Measurement JuicerOfxMeasurementTests)

set(juicer_processor_rust_pixels "${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/spectral-v1/render-gpu-${JUICER_TEST_PRESET_NAME}.json")
foreach(mode IN ITEMS Gpu Trace)
    if(mode STREQUAL "Gpu")
        set(target JuicerOfxRenderGpuTests)
        set(source ofx/render_gpu_test.cpp)
        set(rust_library film_juicer_rust)
        set(registration Ofx.Gpu.Render)
    else()
        set(target JuicerOfxRenderTraceTests)
        set(source ofx/render_trace_test.cpp)
        set(rust_library film_juicer_rust_test_support)
        set(registration Ofx.Gpu.RenderTrace)
    endif()
    add_executable(${target} ${source} ofx/instance_host.cpp ofx/render_assertions.cpp ofx/render_gpu_reference.cpp
        ../native/ofx_exports.cpp)
    if(mode STREQUAL "Trace")
        target_sources(${target} PRIVATE ofx/processor_test_hooks.cpp
            ofx/production_images.cpp ofx/scanner_reference_capture.cpp ../src/Cuda/Scan/JuicerCudaScanPipeline.cu
            $<TARGET_OBJECTS:juicer_executor_failure_test_objects>)
        target_compile_definitions(${target} PRIVATE JUICER_EXECUTOR_FAILURE_TEST_HOOK=1 JUICER_SCAN_DENSITY_CAPTURE_TEST=1
            JUICER_PARAMETER_IDENTITIES_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/exposure/parameter-identities-${JUICER_TEST_PRESET_NAME}.json"
            JUICER_SPECTRAL_TRACE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/spectral-v1/fixed-trace-${JUICER_TEST_PRESET_NAME}.json")
    endif()
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_include_directories(${target} PRIVATE "${JUICER_TEST_ROOT}/native" "${JUICER_TEST_ROOT}/third_party")
    target_compile_definitions(${target} PRIVATE JUICER_PROCESSOR_RUST_PATH="${juicer_processor_rust_pixels}"
        JUICER_SPECTRAL_TRACE_PRESET="${JUICER_TEST_PRESET_NAME}")
    target_link_libraries(${target} PRIVATE juicer_test_core ${rust_library} juicer_ofx_headers CUDA::cudart CUDA::cuda_driver)
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
    add_dependencies(${target} juicer_test_resources)
    if(mode STREQUAL "Gpu")
        add_test(NAME ${registration} COMMAND ${target} --reference "${juicer_processor_rust_pixels}")
    else()
        add_test(NAME ${registration} COMMAND ${target})
        add_test(NAME Ofx.Gpu.ScannerReferenceInputs COMMAND ${target} --scanner-reference-inputs)
        add_test(NAME Ofx.Gpu.RenderReferenceInputs COMMAND ${target} --render-gpu-reference-inputs)
        add_test(NAME Ofx.Gpu.PrintGlareLaunches COMMAND ${target} --print-glare-launches)
        add_test(NAME Ofx.Gpu.RetainedHighGainReference COMMAND ${target} --retained-high-gain-reference)
        set_tests_properties(Ofx.Gpu.ScannerReferenceInputs Ofx.Gpu.RenderReferenceInputs Ofx.Gpu.PrintGlareLaunches Ofx.Gpu.RetainedHighGainReference
            PROPERTIES LABELS "gpu;reference" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
    endif()
    set_tests_properties(${registration} PROPERTIES LABELS "gpu;boundary" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
endforeach()

add_executable(JuicerAdapterTraceProbe
    ofx/frame_geometry_test_hooks.cpp
    ofx/adapter_trace_test.cpp ../native/ofx_exports.cpp)
target_include_directories(JuicerAdapterTraceProbe PRIVATE "${JUICER_TEST_ROOT}/native")
target_link_libraries(JuicerAdapterTraceProbe PRIVATE
    juicer_test_core film_juicer_rust_test_support juicer_ofx_headers CUDA::cuda_driver)
set_target_properties(JuicerAdapterTraceProbe PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerAdapterTraceProbe juicer_test_resources juicer_test_module)
target_sources(JuicerAdapterTraceProbe PRIVATE ffi/spatial_optics_fixture.cpp)
add_test(NAME Ofx.Gpu.FrameGeometry COMMAND JuicerAdapterTraceProbe --frame-geometry
    "${CMAKE_CURRENT_SOURCE_DIR}/frame/geometry-fixtures/manifest.json" "${JUICER_TEST_PRESET_NAME}"
    "${JUICER_TEST_ARTIFACT_ROOT}/ofx/frame-geometry.log" "${CMAKE_CURRENT_SOURCE_DIR}/scatter_halation/frame-fixtures/manifest.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/diffusion/frame-fixtures/manifest.json")
set_tests_properties(Ofx.Gpu.FrameGeometry PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu
    ENVIRONMENT "JUICER_DIAGNOSTICS=3;JUICER_DIAGNOSTICS_PATH=${JUICER_TEST_ARTIFACT_ROOT}/ofx/frame-geometry-diagnostic.log")
add_test(NAME Ofx.Gpu.AdapterTrace
    COMMAND JuicerAdapterTraceProbe
        --trace-output
        "${JUICER_TEST_ARTIFACT_ROOT}/ofx/adapter_trace.txt"
        --grain-fixtures "${CMAKE_CURRENT_SOURCE_DIR}/film/grain-fixtures/manifest.json" "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ofx.Gpu.AdapterTrace PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerScatterHalationGpuTests
    scatter_halation/integration_test.cpp ofx/render_assertions.cpp
    scatter_halation/integration_test_cases.cpp)

add_test(NAME Ofx.Gpu.StreamContract
    COMMAND JuicerAdapterTraceProbe --stream-contract "${JUICER_TEST_BINARY_DIR}/$<TARGET_FILE_NAME:juicer>"
        --trace-output "${JUICER_TEST_ARTIFACT_ROOT}/ofx/stream-contract.log")
set_tests_properties(Ofx.Gpu.StreamContract PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu
    ENVIRONMENT "JUICER_DIAGNOSTICS=3;JUICER_DIAGNOSTICS_PATH=${JUICER_TEST_ARTIFACT_ROOT}/ofx/stream-contract-diagnostic.log")

add_library(juicer_context_drain_test_root OBJECT
    ../src/ProcessRoot.cpp ../src/Cuda/Film/JuicerCudaHighlightBoost.cu)
target_link_libraries(juicer_context_drain_test_root PRIVATE juicer_settings)
target_compile_definitions(juicer_context_drain_test_root PRIVATE
    JUICER_CONTEXT_DRAIN_TEST_HOOK=1)
target_compile_definitions(JuicerScatterHalationGpuTests PRIVATE
    JUICER_CONTEXT_DRAIN_TEST_HOOK=1)
target_sources(JuicerScatterHalationGpuTests PRIVATE
    $<TARGET_OBJECTS:juicer_context_drain_test_root>)
target_include_directories(JuicerScatterHalationGpuTests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/scatter_halation")
target_compile_definitions(JuicerScatterHalationGpuTests PRIVATE
    JUICER_PARAMETER_IDENTITY_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/exposure/parameter-identities-${JUICER_TEST_PRESET_NAME}.json")
target_sources(JuicerScatterHalationGpuTests PRIVATE ffi/spatial_optics_fixture.cpp ffi/prepared_frame_fixture.cpp)
target_include_directories(JuicerScatterHalationGpuTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ffi")
target_link_libraries(JuicerScatterHalationGpuTests PRIVATE
    juicer_test_core film_juicer_rust_test_support CUDA::cuda_driver)
set_target_properties(JuicerScatterHalationGpuTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerScatterHalationGpuTests juicer_test_resources)

function(juicer_add_scatter_gpu_test test_name case_group)
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
        TIMEOUT 300
        RESOURCE_LOCK juicer_gpu)
endfunction()

juicer_add_scatter_gpu_test(HostContracts host-contracts)
juicer_add_scatter_gpu_test(PreparedFrame prepared-frame)
juicer_add_scatter_gpu_test(TerminalRetention terminal-retention)
juicer_add_scatter_gpu_test(FocusedCudaReference focused-cuda-reference)
set_property(TEST ScatterHalation.Gpu.FocusedCudaReference
    PROPERTY LABELS "gpu;reference")
juicer_add_scatter_gpu_test(RouteBoundaries route-boundaries)
juicer_add_scatter_gpu_test(CapturedCarrier captured-carrier)
set_property(TEST ScatterHalation.Gpu.CapturedCarrier
    PROPERTY LABELS "gpu;reference")
juicer_add_scatter_gpu_test(ZeroWork zero-work)
juicer_add_scatter_gpu_test(Lifecycle lifecycle)

add_executable(JuicerGrainScratchTests grain/scratch_reuse_test.cpp ffi/spatial_optics_fixture.cpp
    ffi/prepared_frame_fixture.cpp)
target_link_libraries(JuicerGrainScratchTests PRIVATE
    juicer_test_core film_juicer_rust_test_support GTest::gtest CUDA::cuda_driver)
set_target_properties(JuicerGrainScratchTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerGrainScratchTests juicer_test_resources)
add_test(NAME Grain.Gpu.ScratchReuse COMMAND JuicerGrainScratchTests
    --gtest_fail_if_no_test_selected --gtest_filter=-GrainScratch.ResetRetiresReusedWorkspace)
# Reset retirement exercises process-local CUDA state after scratch reuse.
add_test(NAME Grain.Gpu.ScratchReset COMMAND JuicerGrainScratchTests
    --gtest_fail_if_no_test_selected --gtest_filter=GrainScratch.ResetRetiresReusedWorkspace)
set_tests_properties(Grain.Gpu.ScratchReuse Grain.Gpu.ScratchReset PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerGrainPrototype grain/grain_prototype.cpp grain/film_reference.cu
    grain/film_bank.cpp grain/film_oracle.cpp grain/prototype_benchmark.cpp ffi/prepared_frame_fixture.cpp
    ofx/instance_host.cpp ../native/ofx_exports.cpp)
# Correlate SAT profiling with this prototype CUDA source in Release builds.
set_property(SOURCE grain/film_reference.cu APPEND PROPERTY COMPILE_OPTIONS -lineinfo)
target_link_libraries(JuicerGrainPrototype PRIVATE juicer_test_core film_juicer_rust_test_support juicer_ofx_headers)
target_compile_definitions(JuicerGrainPrototype PRIVATE
    JUICER_GRAIN_TEST_BANK="${JUICER_TEST_RESOURCE_DIR}/grain/portra-400"
    JUICER_GRAIN_TEST_TABLES="${CMAKE_CURRENT_SOURCE_DIR}/grain/portra-table-fixtures")
set_target_properties(JuicerGrainPrototype PROPERTIES LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerGrainPrototype juicer_test_resources)
juicer_add_host_test(Grain.Host.FilmReferenceOracle JuicerGrainPrototype host)
add_test(NAME Grain.Gpu.FilmReference COMMAND JuicerGrainPrototype gpu)
set_tests_properties(Grain.Gpu.FilmReference PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerDirExposureCacheTests dir/exposure_cache_test.cpp ffi/spatial_optics_fixture.cpp
    ffi/prepared_frame_fixture.cpp
    dir/boundary_test_hooks.cpp $<TARGET_OBJECTS:juicer_dir_boundary_test_objects>)
target_compile_definitions(JuicerDirExposureCacheTests PRIVATE
    JUICER_DIR_BOUNDARY_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/dir/boundary-fixtures/${JUICER_TEST_PRESET_NAME}.json")
target_link_libraries(JuicerDirExposureCacheTests PRIVATE
    juicer_test_core film_juicer_rust_test_support GTest::gtest CUDA::cuda_driver)
set_target_properties(JuicerDirExposureCacheTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerDirExposureCacheTests juicer_test_resources)
add_test(NAME Dir.Gpu.ExposureCache COMMAND JuicerDirExposureCacheTests)
set_tests_properties(Dir.Gpu.ExposureCache PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

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

add_executable(JuicerScannerOutputTests scanner/output_encoding_test.cpp
    ffi/prepared_frame_fixture.cpp)
target_include_directories(JuicerScannerOutputTests PRIVATE ffi)
target_link_libraries(JuicerScannerOutputTests PRIVATE juicer_output_encoding_oracle
    juicer_test_core film_juicer_rust_test_support GTest::gtest CUDA::cuda_driver)
set_target_properties(JuicerScannerOutputTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerScannerOutputTests juicer_test_resources)
foreach(gaussian_consumer IN ITEMS JuicerScannerOutputTests JuicerGrainScratchTests JuicerDirExposureCacheTests)
    target_compile_definitions(${gaussian_consumer} PRIVATE
        JUICER_GAUSSIAN_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/optics/gaussian-fixtures/${JUICER_TEST_PRESET_NAME}.json")
endforeach()
add_test(NAME Scanner.Gpu.OutputEncoding COMMAND JuicerScannerOutputTests --gtest_fail_if_no_test_selected --gtest_filter=-ScannerRoutes.CorrectionRoutes)
add_test(NAME Scanner.Gpu.CorrectionRoutes COMMAND JuicerScannerOutputTests --gtest_fail_if_no_test_selected --gtest_filter=ScannerRoutes.CorrectionRoutes)
set_tests_properties(Scanner.Gpu.CorrectionRoutes PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
set_tests_properties(Scanner.Gpu.OutputEncoding PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

add_test(NAME Ofx.Gpu.AcceptedCudaCaptures COMMAND JuicerProcessorReferenceProbe --accepted-clean-captures)
add_test(NAME Ofx.Gpu.AcceptedCleanReferenceInputs COMMAND JuicerProcessorReferenceProbe --accepted-clean-reference-inputs)
set_tests_properties(Ofx.Gpu.AcceptedCleanReferenceInputs PROPERTIES LABELS "gpu;reference" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
set_tests_properties(Ofx.Gpu.AcceptedCudaCaptures PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)


add_executable(JuicerSpectralBootstrapTests ffi/spectral_bootstrap_test.cpp)
target_link_libraries(JuicerSpectralBootstrapTests PRIVATE juicer_test_core film_juicer_rust_test_support GTest::gtest)
set_target_properties(JuicerSpectralBootstrapTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Assets.Host.SpectralBootstrap COMMAND JuicerSpectralBootstrapTests
    --gtest_fail_if_no_test_selected "--gtest_filter=SpectralBootstrap.*"
    "${JUICER_TEST_RESOURCE_DIR}"
    "${JUICER_TEST_ARTIFACT_ROOT}/trusted-resource-decoding/C8/slice-3/bootstrap")
set_tests_properties(Assets.Host.SpectralBootstrap PROPERTIES LABELS "host" TIMEOUT 300)

add_library(juicer_illuminant_test_objects OBJECT ffi/test_asset_bridge.cpp ffi/test_asset_library.cpp)
target_link_libraries(juicer_illuminant_test_objects PRIVATE juicer_settings juicer_ofx_headers)
target_compile_definitions(juicer_illuminant_test_objects PRIVATE JUICER_ILLUMINANT_TEST_HOOK=1)
add_executable(JuicerIlluminantCalibrationTests ffi/illuminant_calibration_abi_c.c ffi/illuminant_calibration_test.cpp ffi/illuminant_construction_test.cpp
    ffi/prepared_descriptors.cpp
    $<TARGET_OBJECTS:juicer_illuminant_test_objects>)
target_link_libraries(JuicerIlluminantCalibrationTests PRIVATE juicer_test_core film_juicer_rust_test_support GTest::gtest)
target_compile_definitions(JuicerIlluminantCalibrationTests PRIVATE JUICER_ILLUMINANT_TEST_HOOK=1)
set_target_properties(JuicerIlluminantCalibrationTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
foreach(illuminant_group IN ITEMS IlluminantAbi IlluminantLibrary IlluminantMath IlluminantConstruction CalibrationAbi CalibrationNative IlluminantNative)
    add_test(NAME Assets.Host.${illuminant_group} COMMAND JuicerIlluminantCalibrationTests
        --gtest_fail_if_no_test_selected "--gtest_filter=${illuminant_group}.*"
        "${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_ROOT}/tests/ffi/fixtures/illuminants"
        "${JUICER_TEST_ARTIFACT_ROOT}/trusted-resource-decoding/C8/slice-4/consumers")
    set_tests_properties(Assets.Host.${illuminant_group} PROPERTIES LABELS "host" TIMEOUT 300)
endforeach()

add_executable(JuicerColorPreparationTests ffi/color_abi_c.c ffi/color_preparation_test.cpp
    ffi/prepared_frame_fixture.cpp)
set_target_properties(JuicerColorPreparationTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_link_libraries(JuicerColorPreparationTests PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerColorPreparationTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")

juicer_add_host_test(Ffi.Host.SharedFixtureLoading JuicerColorPreparationTests
    "${JUICER_TEST_ARTIFACT_ROOT}/fixtures/shared-loader" unused fixture-format)

juicer_add_host_test(Color.Host.Cat16 JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/${JUICER_TEST_PRESET_NAME}.json" leaf)
juicer_add_host_test(Ffi.Host.ColorPreparation JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/${JUICER_TEST_PRESET_NAME}.json" preparation)

juicer_add_host_test(Color.Host.Cat02 JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/cat02-${JUICER_TEST_PRESET_NAME}.json" cat02-leaf)
juicer_add_host_test(Ffi.Host.Cat02Preparation JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/cat02-${JUICER_TEST_PRESET_NAME}.json" cat02-preparation)

juicer_add_host_test(Color.Host.InputColor JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/input-color-${JUICER_TEST_PRESET_NAME}.json" input-leaf)
juicer_add_host_test(Ffi.Host.InputColorPreparation JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/input-color-${JUICER_TEST_PRESET_NAME}.json" input-preparation)
# Admission hooks and asset caches are process-local; bound host contention to two slots.
set_tests_properties(Ffi.Host.InputColorPreparation PROPERTIES PROCESSORS 2)

add_test(NAME Color.Gpu.ScannerReuse COMMAND JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/${JUICER_TEST_PRESET_NAME}.json" reuse-gpu)
set_tests_properties(Color.Gpu.ScannerReuse PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)

# These consumers use identical fixture flags; hook-specific replacements stay local.
add_library(juicer_spectral_fixture_objects OBJECT ffi/spectral_products_test.cpp)
add_library(juicer_spectral_allocation_test_objects OBJECT ffi/spectral_allocation_probe.cpp)
foreach(fixture_target IN ITEMS juicer_spectral_fixture_objects juicer_spectral_allocation_test_objects)
    target_link_libraries(${fixture_target} PRIVATE juicer_settings juicer_ofx_headers)
    set_target_properties(${fixture_target} PROPERTIES POSITION_INDEPENDENT_CODE OFF)
    if(UNIX)
        target_compile_options(${fixture_target} PRIVATE -fPIE)
    endif()
endforeach()

add_executable(JuicerExposurePreparationTests ffi/exposure_abi_c.c ffi/exposure_preparation_test.cpp
    $<TARGET_OBJECTS:juicer_spectral_fixture_objects>
    )
target_link_libraries(JuicerExposurePreparationTests PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerExposurePreparationTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
foreach(exposure_case IN ITEMS reference sensitivity boundary)
    if(exposure_case STREQUAL "boundary")
        set(exposure_name Ffi.Host.ExposurePreparation)
    elseif(exposure_case STREQUAL "reference")
        set(exposure_name Exposure.Host.Reference)
    else()
        set(exposure_name Exposure.Host.Sensitivity)
    endif()
    if(exposure_case STREQUAL "reference")
        add_test(NAME ${exposure_name} COMMAND JuicerExposurePreparationTests ${exposure_case}
            "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/exposure/numerical.json"
            "${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_ARTIFACT_ROOT}/exposure/reference")
    else()
        add_test(NAME ${exposure_name} COMMAND JuicerExposurePreparationTests ${exposure_case}
            "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/exposure/numerical.json")
    endif()
    set_tests_properties(${exposure_name} PROPERTIES LABELS "host" TIMEOUT 300)
endforeach()
add_test(NAME Exposure.Host.Products COMMAND JuicerExposurePreparationTests products
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/exposure/products-${JUICER_TEST_PRESET_NAME}.json"
    "${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_ARTIFACT_ROOT}/exposure/products")
set_tests_properties(Exposure.Host.Products PROPERTIES LABELS "host" TIMEOUT 300)
add_test(NAME Exposure.Host.Admission COMMAND JuicerExposurePreparationTests admission
    "${JUICER_TEST_RESOURCE_DIR}")
set_tests_properties(Exposure.Host.Admission PROPERTIES LABELS "host" TIMEOUT 300)

add_test(NAME Exposure.Host.FixtureFormat COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p test_exposure_fixtures.py -v)
set_tests_properties(Exposure.Host.FixtureFormat PROPERTIES LABELS "host" TIMEOUT 300
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

add_test(NAME Ffi.Host.SharedFixtureFormat COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p test_shared_fixtures.py -v)
set_tests_properties(Ffi.Host.SharedFixtureFormat PROPERTIES LABELS "host" TIMEOUT 300
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")


add_executable(JuicerMallettExposureTests ffi/mallett_exposure_abi_c.c ffi/mallett_exposure_test.cpp ffi/mallett_caller_test.cpp)
target_link_libraries(JuicerMallettExposureTests PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerMallettExposureTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
foreach(mallett_case IN ITEMS numerical boundary)
    add_test(NAME Exposure.Host.Mallett.${mallett_case} COMMAND JuicerMallettExposureTests
        "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/exposure/mallett-manifest.json" "${JUICER_TEST_PRESET_NAME}" ${mallett_case})
    set_tests_properties(Exposure.Host.Mallett.${mallett_case} PROPERTIES LABELS "host" TIMEOUT 300)
endforeach()

add_test(NAME Exposure.Host.Mallett.Spectra COMMAND JuicerMallettExposureTests
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/exposure/mallett-manifest.json" "${JUICER_TEST_PRESET_NAME}" spectra
    "${JUICER_TEST_RESOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/cat02-${JUICER_TEST_PRESET_NAME}.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/input-color-${JUICER_TEST_PRESET_NAME}.json")
set_tests_properties(Exposure.Host.Mallett.Spectra PROPERTIES LABELS "host" TIMEOUT 300)

foreach(mallett_case IN ITEMS products admission)
    add_test(NAME Exposure.Host.Mallett.${mallett_case} COMMAND JuicerMallettExposureTests
        "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/exposure/mallett-manifest.json" "${JUICER_TEST_PRESET_NAME}" ${mallett_case} "${JUICER_TEST_RESOURCE_DIR}"
        "${JUICER_TEST_ARTIFACT_ROOT}/mallett-source/${mallett_case}")
    set_tests_properties(Exposure.Host.Mallett.${mallett_case} PROPERTIES LABELS "host" TIMEOUT 300)
endforeach()

add_test(NAME Exposure.Host.Mallett.FixtureFormat COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p test_mallett_exposure_fixtures.py -v)
set_tests_properties(Exposure.Host.Mallett.FixtureFormat PROPERTIES LABELS "host" TIMEOUT 300
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

add_executable(JuicerSpectralPreparationTests ffi/spectral_preparation_abi_c.c ffi/spectral_preparation_test.cpp $<TARGET_OBJECTS:juicer_spectral_fixture_objects> )
target_link_libraries(JuicerSpectralPreparationTests PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerSpectralPreparationTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerSpectralPreparationTests juicer_test_resources)
foreach(spectral_case IN ITEMS tables white inverse edge)
    if(spectral_case STREQUAL "edge")
        set(spectral_name Ffi.Host.SpectralPreparation)
    elseif(spectral_case STREQUAL "inverse")
        set(spectral_name Spectral.Host.SInverse)
    elseif(spectral_case STREQUAL "white")
        set(spectral_name Spectral.Host.White)
    else()
        set(spectral_name Spectral.Host.Tables)
    endif()
    add_test(NAME ${spectral_name} COMMAND JuicerSpectralPreparationTests ${spectral_case}
        "${JUICER_TEST_ROOT}/tests/ffi/fixtures/spectral/${JUICER_TEST_PRESET_NAME}.json"
        "${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_ARTIFACT_ROOT}/spectral/${spectral_case}")
    set_tests_properties(${spectral_name} PROPERTIES LABELS "host" TIMEOUT 300)
endforeach()

foreach(spectral_case IN ITEMS products admission)
    if(spectral_case STREQUAL "products")
        set(spectral_name Spectral.Host.Products)
    else()
        set(spectral_name Spectral.Host.Admission)
    endif()
    add_test(NAME ${spectral_name} COMMAND JuicerSpectralPreparationTests ${spectral_case}
        "${JUICER_TEST_ROOT}/tests/ffi/fixtures/spectral/${JUICER_TEST_PRESET_NAME}.json"
        "${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_ARTIFACT_ROOT}/spectral/${spectral_case}")
    set_tests_properties(${spectral_name} PROPERTIES LABELS "host" TIMEOUT 300)
endforeach()

add_executable(JuicerTcLutTests ffi/tc_lut_abi_c.c ffi/tc_lut_test.cpp $<TARGET_OBJECTS:juicer_spectral_fixture_objects>)
target_link_libraries(JuicerTcLutTests PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerTcLutTests PROPERTIES LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
foreach(tc_case IN ITEMS boundary numerical products admission)
    add_test(NAME Reconstruction.Host.TcLut.${tc_case} COMMAND JuicerTcLutTests ${tc_case} "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/tc_lut/manifest.json" "${JUICER_TEST_PRESET_NAME}")
    set_tests_properties(Reconstruction.Host.TcLut.${tc_case} PROPERTIES LABELS "host" TIMEOUT 300)
endforeach()

add_executable(JuicerTcLutLifetimeTests ffi/tc_lut_lifetime_test.cpp)
target_link_libraries(JuicerTcLutLifetimeTests PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerTcLutLifetimeTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Reconstruction.Host.TcLut.Lifetime COMMAND JuicerTcLutLifetimeTests "${JUICER_TEST_RESOURCE_DIR}")
set_tests_properties(Reconstruction.Host.TcLut.Lifetime PROPERTIES LABELS "host" TIMEOUT 300)

add_test(NAME Reconstruction.Host.TcLut.FixtureFormat COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p test_tc_lut_fixtures.py -v)
set_tests_properties(Reconstruction.Host.TcLut.FixtureFormat PROPERTIES LABELS "host" TIMEOUT 300 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

add_executable(JuicerInputHullTests ffi/input_hull_abi_c.c ffi/input_hull_test.cpp $<TARGET_OBJECTS:juicer_spectral_fixture_objects> $<TARGET_OBJECTS:juicer_spectral_allocation_test_objects> ffi/prepared_descriptors.cpp)
target_link_libraries(JuicerInputHullTests PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerInputHullTests PROPERTIES LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
foreach(hull_case IN ITEMS boundary binding products cache admission-1 admission-2 admission-3 admission-4 admission-5 admission-6)
    add_test(NAME Gamut.Host.InputHull.${hull_case} COMMAND JuicerInputHullTests ${hull_case}
        "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/input_hull/manifest.json" "${JUICER_TEST_PRESET_NAME}")
    set_tests_properties(Gamut.Host.InputHull.${hull_case} PROPERTIES LABELS "host" TIMEOUT 300)
endforeach()
add_test(NAME Gamut.Host.InputHull.FixtureFormat COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p test_input_hull_fixtures.py -v)
set_tests_properties(Gamut.Host.InputHull.FixtureFormat PROPERTIES LABELS "host" TIMEOUT 300 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

add_executable(JuicerOutputGamutTests ffi/output_gamut_abi_c.c ffi/output_gamut_test.cpp)
target_link_libraries(JuicerOutputGamutTests PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerOutputGamutTests PROPERTIES LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
foreach(gamut_case IN ITEMS boundary binding)
    add_test(NAME Gamut.Host.Output.${gamut_case} COMMAND JuicerOutputGamutTests ${gamut_case}
        "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/output_gamut/manifest.json" "${JUICER_TEST_PRESET_NAME}")
    set_tests_properties(Gamut.Host.Output.${gamut_case} PROPERTIES LABELS "host" TIMEOUT 300)
endforeach()
add_executable(JuicerOutputGamutCacheTests ffi/output_gamut_cache_test.cpp ffi/spectral_allocation_probe.cpp)
target_link_libraries(JuicerOutputGamutCacheTests PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerOutputGamutCacheTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME Gamut.Host.Output.Cache COMMAND JuicerOutputGamutCacheTests "${JUICER_TEST_RESOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/output_gamut/manifest.json" "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Gamut.Host.Output.Cache PROPERTIES LABELS "host" TIMEOUT 300)
add_test(NAME Gamut.Host.Output.FixtureFormat COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p test_output_gamut_fixtures.py -v)
set_tests_properties(Gamut.Host.Output.FixtureFormat PROPERTIES LABELS "host" TIMEOUT 300 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ffi/output-color-development.json")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/ffi/output-color-development.json" output_color_development)
string(JSON output_color_development_target GET "${output_color_development}" target)
string(JSON output_color_development_test GET "${output_color_development}" test)
string(JSON output_color_development_fixture GET "${output_color_development}" fixture)
add_executable(${output_color_development_target} ffi/output_color_abi_c.c ffi/output_color_test.cpp
    $<TARGET_OBJECTS:juicer_scanner_test_objects>)
target_compile_definitions(JuicerOutputColorTests PRIVATE JUICER_SCANNER_TEST_HOOK=1)
target_link_libraries(JuicerOutputColorTests PRIVATE juicer_test_core film_juicer_rust_test_support)
set_target_properties(JuicerOutputColorTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_test(NAME ${output_color_development_test} COMMAND ${output_color_development_target}
    "${JUICER_TEST_ROOT}/${output_color_development_fixture}")
set_tests_properties(Ffi.Host.OutputColor PROPERTIES LABELS "host" TIMEOUT 300)
add_test(NAME Ffi.Host.ScannerColor COMMAND JuicerOutputColorTests scanner-color
    "${CMAKE_CURRENT_SOURCE_DIR}/scanner/color-fixtures/native.json"
    "${JUICER_TEST_ROOT}/${output_color_development_fixture}"
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/cat02-${JUICER_TEST_PRESET_NAME}.json")
set_tests_properties(Ffi.Host.ScannerColor PROPERTIES LABELS "host" TIMEOUT 300)


add_library(juicer_output_encoding_oracle STATIC scanner/output_encoding_oracle.cpp)
target_include_directories(juicer_output_encoding_oracle PUBLIC scanner "${JUICER_TEST_ROOT}/src")
target_compile_features(juicer_output_encoding_oracle PRIVATE cxx_std_20)
target_link_libraries(JuicerOutputColorTests PRIVATE juicer_output_encoding_oracle)


add_dependencies(JuicerProductionOfxProbe juicer_test_module)
add_test(NAME Ffi.Host.ProductionOfx COMMAND JuicerProductionOfxProbe
    "${JUICER_TEST_BINARY_DIR}/$<TARGET_FILE_NAME:juicer>")
add_test(NAME Ffi.Host.ProductionCudaAbi COMMAND JuicerProductionCudaAbiProbe)
set_tests_properties(Ffi.Host.ProductionOfx Ffi.Host.ProductionCudaAbi PROPERTIES
    LABELS "host" TIMEOUT 300 ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1")
