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

include("${CMAKE_CURRENT_SOURCE_DIR}/ToolingTests.cmake")
if(CMAKE_HOST_WIN32)
    set(JUICER_TOOLING_HOST_WINDOWS TRUE)
else()
    set(JUICER_TOOLING_HOST_WINDOWS FALSE)
endif()
juicer_register_tooling_tests(
    ROOT "${PROJECT_SOURCE_DIR}"
    ARTIFACT_ROOT "${JUICER_TEST_ARTIFACT_ROOT}"
    PYTHON "${Python3_EXECUTABLE}"
    CMAKE_COMMAND "${CMAKE_COMMAND}"
    CTEST_COMMAND "${CMAKE_CTEST_COMMAND}"
    CARGO "${JUICER_CARGO_EXECUTABLE}"
    NINJA "${CMAKE_MAKE_PROGRAM}"
    CC "${CMAKE_C_COMPILER}"
    BUILD_TYPE "${CMAKE_BUILD_TYPE}"
    HOST_WINDOWS "${JUICER_TOOLING_HOST_WINDOWS}"
    TESTS_OUT JUICER_TOOLING_TESTS)

set(JUICER_RUST_TEST_COMMAND
    "${JUICER_CARGO_EXECUTABLE}" test
    --locked
    --package film-juicer-core
    --target "${JUICER_RUST_TARGET}"
    --target-dir "${JUICER_RUST_TARGET_DIR}")
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND JUICER_RUST_TEST_COMMAND --release)
endif()
set(JUICER_RUST_CORE_TEST_COMMAND
    "${CMAKE_COMMAND}" -E env
    "RUSTFLAGS="
    "CARGO_BUILD_RUSTFLAGS="
    "CARGO_ENCODED_RUSTFLAGS="
    ${JUICER_RUST_TEST_COMMAND})
# Keep cold test compilation in the build phase, outside the CTest runtime budget.
# Cargo owns test-source and include-file freshness on incremental builds.
add_custom_target(juicer_rust_core_tests ALL
    COMMAND ${JUICER_RUST_CORE_TEST_COMMAND} --no-run
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Preparing core Rust test executables (${JUICER_RUST_CARGO_PROFILE})"
    VERBATIM
    USES_TERMINAL)
add_test(NAME Rust.Core COMMAND ${JUICER_RUST_CORE_TEST_COMMAND} -- --test-threads=2)
set_tests_properties(Rust.Core PROPERTIES
    LABELS "host"
    TIMEOUT 120
    PROCESSORS 2
    RESOURCE_LOCK juicer_cargo
    ENVIRONMENT "JUICER_TEST_ARTIFACT_DIR=${JUICER_TEST_ARTIFACT_ROOT}/rust-core"
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
    LABELS "host" TIMEOUT 120 RESOURCE_LOCK juicer_cargo WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

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

add_executable(JuicerSpectralOwnerTests ffi/spectral_abi_c.c ffi/spectral_owner_test.cpp)
set_target_properties(JuicerSpectralOwnerTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_include_directories(JuicerSpectralOwnerTests PRIVATE "${PROJECT_SOURCE_DIR}/native")
target_link_libraries(JuicerSpectralOwnerTests PRIVATE film_juicer_rust GTest::gtest)
if(WIN32)
    target_compile_options(JuicerSpectralOwnerTests PRIVATE /W3 /WX)
else()
    target_compile_options(JuicerSpectralOwnerTests PRIVATE -Wall -Wextra -Werror -pedantic-errors)
endif()
add_test(NAME Ffi.Host.SpectralOwner COMMAND JuicerSpectralOwnerTests
    "${JUICER_TEST_RESOURCE_DIR}"
    "${JUICER_TEST_ARTIFACT_ROOT}/trusted-resource-decoding/C8/slice-3/spectral-owner")
set_tests_properties(Ffi.Host.SpectralOwner PROPERTIES LABELS "host" TIMEOUT 60)

add_library(juicer_noise_test_objects OBJECT ../src/RustAssetBridge.cpp ../native/juicer_cuda_api.cpp)
target_link_libraries(juicer_noise_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_noise_test_objects PRIVATE JUICER_NOISE_TEST_HOOK=1)
add_executable(JuicerNoiseOwnerTests ffi/noise_owner_test.cpp $<TARGET_OBJECTS:juicer_noise_test_objects>)
target_link_libraries(JuicerNoiseOwnerTests PRIVATE juicer_core GTest::gtest)
target_compile_definitions(JuicerNoiseOwnerTests PRIVATE JUICER_NOISE_TEST_HOOK=1)
set_target_properties(JuicerNoiseOwnerTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerNoiseOwnerTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_dependencies(JuicerNoiseOwnerTests juicer_test_resources)
add_test(NAME Ffi.Host.NoiseOwner COMMAND JuicerNoiseOwnerTests "${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_ARTIFACT_ROOT}/trusted-resource-decoding/C8/slice-5/noise-owner")
set_tests_properties(Ffi.Host.NoiseOwner PROPERTIES LABELS "host" TIMEOUT 120)

add_library(juicer_catalog_test_objects OBJECT ../src/RustAssetBridge.cpp)
target_link_libraries(juicer_catalog_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_catalog_test_objects PRIVATE JUICER_CATALOG_TEST_HOOK=1)
add_executable(JuicerCatalogOwnerTests ffi/catalog_abi_c.c ffi/catalog_owner_test.cpp
    $<TARGET_OBJECTS:juicer_catalog_test_objects>)
target_link_libraries(JuicerCatalogOwnerTests PRIVATE juicer_core)
target_compile_definitions(JuicerCatalogOwnerTests PRIVATE JUICER_CATALOG_TEST_HOOK=1)
set_target_properties(JuicerCatalogOwnerTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerCatalogOwnerTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Ffi.Host.CatalogOwner COMMAND JuicerCatalogOwnerTests
    "${JUICER_TEST_ARTIFACT_ROOT}/trusted-resource-decoding/C8/slice-2/catalog")
set_tests_properties(Ffi.Host.CatalogOwner PROPERTIES LABELS "host" TIMEOUT 60)

add_library(juicer_profile_conversion_test_objects OBJECT ../src/RustAssetBridge.cpp)
target_link_libraries(juicer_profile_conversion_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_profile_conversion_test_objects PRIVATE JUICER_PROFILE_CONVERSION_TEST_HOOK=1)
add_executable(JuicerProductionProfileOwnerTests ffi/production_profile_abi_c.c ffi/production_profile_owner_test.cpp
    $<TARGET_OBJECTS:juicer_profile_conversion_test_objects>)
target_link_libraries(JuicerProductionProfileOwnerTests PRIVATE juicer_core GTest::gtest)

add_library(juicer_film_input_test_objects OBJECT ../src/RustFilmInputBridge.cpp)
target_link_libraries(juicer_film_input_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_film_input_test_objects PRIVATE JUICER_FILM_INPUT_TEST_HOOK=1)
add_executable(JuicerFilmInputTests ffi/film_input_abi_c.c ffi/film_input_test.cpp
    $<TARGET_OBJECTS:juicer_film_input_test_objects>)
target_link_libraries(JuicerFilmInputTests PRIVATE juicer_core)
set_target_properties(JuicerFilmInputTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerFilmInputTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_dependencies(JuicerFilmInputTests juicer_test_resources)
if(WIN32)
    set(JUICER_FILM_INPUT_FIXTURE "windows.json")
else()
    set(JUICER_FILM_INPUT_FIXTURE "linux.json")
endif()
add_test(NAME Ffi.Host.FilmInput COMMAND JuicerFilmInputTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/input-fixtures/${JUICER_FILM_INPUT_FIXTURE}")
set_tests_properties(Ffi.Host.FilmInput PROPERTIES LABELS "host" TIMEOUT 120)
add_test(NAME Film.Host.InputProducts COMMAND JuicerFilmInputTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/input-fixtures/${JUICER_FILM_INPUT_FIXTURE}" products)
set_tests_properties(Film.Host.InputProducts PROPERTIES LABELS "host" TIMEOUT 180)

add_library(juicer_film_development_test_objects OBJECT ../src/RustAssetBridge.cpp)
add_library(juicer_film_recipe_test_objects OBJECT ../src/RustFilmRecipeBridge.cpp)
add_library(juicer_scanner_test_objects OBJECT ../src/RustScannerBridge.cpp)
target_link_libraries(juicer_scanner_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_scanner_test_objects PRIVATE JUICER_SCANNER_TEST_HOOK=1)
add_library(juicer_film_recipe_state_test_objects OBJECT ../src/ResourceAssetLibrary.cpp)
target_link_libraries(juicer_film_recipe_state_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_film_recipe_state_test_objects PRIVATE JUICER_FILM_RECIPE_STATE_TEST_HOOK=1)
target_link_libraries(juicer_film_recipe_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_film_recipe_test_objects PRIVATE JUICER_FILM_RECIPE_TEST_HOOK=1)
add_library(juicer_gaussian_test_objects OBJECT ../src/Cuda/JuicerCudaResources.cpp)
target_link_libraries(juicer_gaussian_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_gaussian_test_objects PRIVATE JUICER_GAUSSIAN_TEST_HOOK=1)
add_library(juicer_dir_boundary_test_objects OBJECT ../src/Cuda/JuicerCudaResources.cpp)
target_link_libraries(juicer_dir_boundary_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_dir_boundary_test_objects PRIVATE JUICER_DIR_BOUNDARY_TEST_HOOK=1)
add_executable(JuicerFilmRecipeTests ffi/film_recipe_abi_c.c ffi/film_recipe_test.cpp ffi/spectral_allocation_probe.cpp ffi/spatial_optics_fixture.cpp dir/boundary_test_hooks.cpp
    $<TARGET_OBJECTS:juicer_dir_boundary_test_objects>
    $<TARGET_OBJECTS:juicer_scanner_test_objects>
    $<TARGET_OBJECTS:juicer_film_recipe_test_objects>
    $<TARGET_OBJECTS:juicer_film_recipe_state_test_objects>)
target_link_libraries(JuicerFilmRecipeTests PRIVATE juicer_core)
target_compile_definitions(JuicerFilmRecipeTests PRIVATE JUICER_SCANNER_TEST_HOOK=1)
set_target_properties(JuicerFilmRecipeTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerFilmRecipeTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_dependencies(JuicerFilmRecipeTests juicer_test_resources)
add_test(NAME Ffi.Host.GrainFrame COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/grain/frame-fixtures/manifest.json" grain-frame "${JUICER_TEST_PRESET_NAME}")
add_test(NAME Film.Host.GrainFrameProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/grain/frame-fixtures/manifest.json" grain-frame-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.GrainFrame Film.Host.GrainFrameProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.DirBoundary COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/dir/boundary-fixtures/manifest.json" dir-boundary "${JUICER_TEST_PRESET_NAME}")
add_test(NAME Film.Host.DirBoundaryProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/dir/boundary-fixtures/manifest.json" dir-boundary-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.DirBoundary Film.Host.DirBoundaryProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.DirFrame COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/dir/frame-fixtures/manifest.json" dir-frame "${JUICER_TEST_PRESET_NAME}")
add_test(NAME Film.Host.DirFrameProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/dir/frame-fixtures/manifest.json" dir-frame-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.DirFrame Film.Host.DirFrameProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.DiffusionFrame COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/diffusion/frame-fixtures/manifest.json" diffusion-frame "${JUICER_TEST_PRESET_NAME}")
add_test(NAME Optics.Host.DiffusionFrameProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/diffusion/frame-fixtures/manifest.json" diffusion-frame-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.DiffusionFrame Optics.Host.DiffusionFrameProducts PROPERTIES LABELS "host" TIMEOUT 180)
if(WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Keep the full numerical fixture separate at the preset's four-worker limit.
    set_tests_properties(Optics.Host.DiffusionFrameProducts PROPERTIES PROCESSORS 4)
endif()
add_test(NAME Ffi.Host.HalationFrame COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scatter_halation/frame-fixtures/manifest.json" halation-frame "${JUICER_TEST_PRESET_NAME}")
add_test(NAME Optics.Host.HalationFrameProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scatter_halation/frame-fixtures/manifest.json" halation-frame-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.HalationFrame Optics.Host.HalationFrameProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.FrameGeometry COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/frame/geometry-fixtures/manifest.json" frame "${JUICER_TEST_PRESET_NAME}")
add_test(NAME Frame.Host.GeometryProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/frame/geometry-fixtures/manifest.json" frame-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.FrameGeometry Frame.Host.GeometryProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.GaussianKernel COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/optics/gaussian-fixtures/manifest.json" gaussian "${JUICER_TEST_PRESET_NAME}")
add_test(NAME Optics.Host.GaussianKernelProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/optics/gaussian-fixtures/manifest.json" gaussian-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.GaussianKernel Optics.Host.GaussianKernelProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.ScannerPostEffects COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scanner/post-effects-fixtures/manifest.json" scanner-post "${JUICER_TEST_PRESET_NAME}")
add_test(NAME Scanner.Host.PostEffectsProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scanner/post-effects-fixtures/manifest.json" scanner-post-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.ScannerPostEffects Scanner.Host.PostEffectsProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.ScannerLutDescriptor COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scanner/descriptor-fixtures/manifest.json" scanner-descriptor "${JUICER_TEST_PRESET_NAME}")
add_test(NAME Scanner.Host.LutDescriptorProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scanner/descriptor-fixtures/manifest.json" scanner-descriptor-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.ScannerLutDescriptor Scanner.Host.LutDescriptorProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.ScannerLut COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scanner/lut-fixtures/manifest.json" scanner-lut "${JUICER_TEST_PRESET_NAME}")
add_test(NAME Scanner.Host.LutProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scanner/lut-fixtures/manifest.json" scanner-lut-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.ScannerLut Scanner.Host.LutProducts PROPERTIES LABELS "host" TIMEOUT 180)
if(WIN32)
    set(JUICER_FILM_DIR_FIXTURE windows)
else()
    set(JUICER_FILM_DIR_FIXTURE linux)
endif()
add_test(NAME Ffi.Host.FilmRecipes COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/dir-fixtures/${JUICER_FILM_DIR_FIXTURE}.json")
set_tests_properties(Ffi.Host.FilmRecipes PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.FilmReference COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/reference-fixtures/manifest.json" film-reference "${JUICER_TEST_PRESET_NAME}")
add_test(NAME Film.Host.ReferenceProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/reference-fixtures/manifest.json" film-reference-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.FilmReference Film.Host.ReferenceProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Film.Host.DirDensityProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/dir-fixtures/${JUICER_FILM_DIR_FIXTURE}.json" products)
set_tests_properties(Film.Host.DirDensityProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.VisualGrain COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/grain-fixtures/manifest.json" grain "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.VisualGrain PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Film.Host.VisualGrainProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/grain-fixtures/manifest.json" grain-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Film.Host.VisualGrainProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.PhysicalEffects COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/effects-fixtures/manifest.json" effects "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.PhysicalEffects PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Film.Host.PhysicalEffectsProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/effects-fixtures/manifest.json" effects-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Film.Host.PhysicalEffectsProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.SpatialOptics COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/optics-fixtures/manifest.json" optics "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.SpatialOptics PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Film.Host.SpatialOpticsProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/optics-fixtures/manifest.json" optics-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Film.Host.SpatialOpticsProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.PrintIlluminant COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/illuminant-fixtures/rust-manifest.json" print-illuminant "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.PrintIlluminant PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Print.Host.IlluminantProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/illuminant-fixtures/rust-manifest.json" print-illuminant-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Print.Host.IlluminantProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.PrintPreflash COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/preflash-fixtures/manifest.json" print-preflash "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.PrintPreflash PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Print.Host.PreflashProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/preflash-fixtures/manifest.json" print-preflash-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Print.Host.PreflashProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.PrintBalance COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/balance-fixtures/manifest.json" print-balance "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.PrintBalance PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Print.Host.BalanceProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/balance-fixtures/manifest.json" print-balance-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Print.Host.BalanceProducts PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Ffi.Host.ScannerCorrection COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scanner/correction-fixtures/manifest.json" scanner-correction "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ffi.Host.ScannerCorrection PROPERTIES LABELS "host" TIMEOUT 180
    ENVIRONMENT "JUICER_DIAGNOSTICS=2;JUICER_DIAGNOSTICS_PATH=${JUICER_TEST_ARTIFACT_ROOT}/scanner/correction-boundary-trace.log")
add_test(NAME Scanner.Host.CorrectionProducts COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/scanner/correction-fixtures/manifest.json" scanner-correction-products "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Scanner.Host.CorrectionProducts PROPERTIES LABELS "host" TIMEOUT 180
    ENVIRONMENT "JUICER_DIAGNOSTICS=2;JUICER_DIAGNOSTICS_PATH=${JUICER_TEST_ARTIFACT_ROOT}/scanner/correction-product-trace.log")
add_test(NAME Print.Gpu.BalanceCache COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/balance-fixtures/manifest.json" print-balance-cache "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Print.Gpu.BalanceCache PROPERTIES LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)
add_test(NAME Print.Gpu.PreflashCache COMMAND JuicerFilmRecipeTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/print/preflash-fixtures/manifest.json" print-preflash-cache "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Print.Gpu.PreflashCache PROPERTIES LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)
target_link_libraries(juicer_film_development_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_film_development_test_objects PRIVATE JUICER_FILM_DEVELOPMENT_TEST_HOOK=1)
add_executable(JuicerFilmDevelopmentTests ffi/film_development_abi_c.c ffi/film_development_test.cpp ffi/spectral_allocation_probe.cpp
    $<TARGET_OBJECTS:juicer_film_development_test_objects>)
target_link_libraries(JuicerFilmDevelopmentTests PRIVATE juicer_core)
target_compile_definitions(JuicerFilmDevelopmentTests PRIVATE JUICER_FILM_DEVELOPMENT_TEST_HOOK=1)
set_target_properties(JuicerFilmDevelopmentTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerFilmDevelopmentTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_dependencies(JuicerFilmDevelopmentTests juicer_test_resources)
add_test(NAME Ffi.Host.FilmDevelopment COMMAND JuicerFilmDevelopmentTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/fixtures/${JUICER_TEST_PRESET_NAME}.json")
set_tests_properties(Ffi.Host.FilmDevelopment PROPERTIES LABELS "host" TIMEOUT 120 PROCESSORS 2)
add_test(NAME Film.Host.DevelopmentProducts COMMAND JuicerFilmDevelopmentTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/film/fixtures/${JUICER_TEST_PRESET_NAME}.json" products)
set_tests_properties(Film.Host.DevelopmentProducts PROPERTIES LABELS "host" TIMEOUT 120 PROCESSORS 2)
target_compile_definitions(JuicerProductionProfileOwnerTests PRIVATE JUICER_PROFILE_CONVERSION_TEST_HOOK=1)
set_target_properties(JuicerProductionProfileOwnerTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerProductionProfileOwnerTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Ffi.Host.ProductionProfileOwner COMMAND JuicerProductionProfileOwnerTests
    "${JUICER_TEST_RESOURCE_DIR}" "${PROJECT_SOURCE_DIR}/tests/profile/fixtures"
    "${JUICER_TEST_ARTIFACT_ROOT}/trusted-resource-decoding/C8/slice-2/profile-owner")
set_tests_properties(Ffi.Host.ProductionProfileOwner PROPERTIES LABELS "host" TIMEOUT 60)

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
set_target_properties(JuicerCudaTerminalTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerCudaTerminalTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
foreach(terminal_case IN ITEMS host-cache-failure host-cache-native-precedence partial-library construction catalog-retention registered-reentry success serialization allocation standard unknown typed wording runtime context-loss diagnostic-allocation accepting-failure
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
target_compile_definitions(JuicerPinnedUploadTests PRIVATE JUICER_PINNED_UPLOAD_TEST_HOOK=1
    JUICER_TEST_RESOURCE_DIR="${JUICER_TEST_RESOURCE_DIR}")
add_dependencies(JuicerPinnedUploadTests juicer_test_resources)
set_target_properties(JuicerPinnedUploadTests PROPERTIES LINKER_LANGUAGE CXX)
if(UNIX)
    set_property(TARGET JuicerPinnedUploadTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Resource.Gpu.PinnedUpload COMMAND JuicerPinnedUploadTests)
set_tests_properties(Resource.Gpu.PinnedUpload PROPERTIES LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)

# Gaussian cache failure and collision cases use the real context-owned miss path.
add_executable(JuicerGaussianKernelTests resource/gaussian_kernel_test.cpp
    resource/gaussian_kernel_test_hooks.cpp ffi/spectral_allocation_probe.cpp
    $<TARGET_OBJECTS:juicer_gaussian_test_objects>)
target_link_libraries(JuicerGaussianKernelTests PRIVATE juicer_core CUDA::cuda_driver)
target_compile_definitions(JuicerGaussianKernelTests PRIVATE JUICER_GAUSSIAN_TEST_HOOK=1)
set_target_properties(JuicerGaussianKernelTests PROPERTIES LINKER_LANGUAGE CXX)
if(UNIX)
    set_property(TARGET JuicerGaussianKernelTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Resource.Gpu.GaussianKernel COMMAND JuicerGaussianKernelTests
    "${CMAKE_CURRENT_SOURCE_DIR}/optics/gaussian-fixtures/${JUICER_TEST_PRESET_NAME}-payload.json")
set_tests_properties(Resource.Gpu.GaussianKernel PROPERTIES LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)

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

# One explicit development owner; linkage determines dependencies, not test relevance.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/route/development.json")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/route/development.json" route_development)
string(JSON route_development_target GET "${route_development}" target)
string(JSON route_development_test GET "${route_development}" test)
string(JSON route_development_source GET "${route_development}" source)
add_executable(${route_development_target}
    route/route_bridge_test.cpp "${PROJECT_SOURCE_DIR}/${route_development_source}")
target_include_directories(JuicerRouteBridgeTests PRIVATE
    "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/native")
target_link_libraries(JuicerRouteBridgeTests PRIVATE film_juicer_rust)
if(WIN32)
    target_compile_options(JuicerRouteBridgeTests PRIVATE /W3 /WX)
else()
    target_compile_options(JuicerRouteBridgeTests PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME ${route_development_test} COMMAND ${route_development_target})
set_tests_properties(Route.Host.Bridge PROPERTIES LABELS "host" TIMEOUT 30)

add_test(NAME Quality.RustNaming
    COMMAND "${CMAKE_COMMAND}" -E env
        "JUICER_CARGO=${JUICER_CARGO_EXECUTABLE}"
        "JUICER_RUST_TARGET=${JUICER_RUST_TARGET}"
        "JUICER_TEST_ARTIFACT_DIR=${JUICER_TEST_ARTIFACT_ROOT}/quality/rust-naming"
        "JUICER_RUST_PROBE_DIR=${JUICER_RUST_TARGET_DIR}/quality-probes/naming"
        "JUICER_RUST_PROBE_PROFILE=${JUICER_TEST_CONFIGURATION}"
        "PYTHONDONTWRITEBYTECODE=1"
        "${Python3_EXECUTABLE}" -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/quality"
            -p test_rust_naming.py -v)
set_tests_properties(Quality.RustNaming PROPERTIES
    LABELS "host" TIMEOUT 180)

add_test(NAME Quality.RustBoundaries
    COMMAND "${CMAKE_COMMAND}" -E env
        "JUICER_CARGO=${JUICER_CARGO_EXECUTABLE}"
        "JUICER_RUST_TARGET=${JUICER_RUST_TARGET}"
        "JUICER_TEST_ARTIFACT_DIR=${JUICER_TEST_ARTIFACT_ROOT}/quality/rust-boundaries"
        "JUICER_RUST_PROBE_DIR=${JUICER_RUST_TARGET_DIR}/quality-probes/boundaries"
        "JUICER_RUST_PROBE_PROFILE=${JUICER_TEST_CONFIGURATION}"
        "PYTHONDONTWRITEBYTECODE=1"
        "${Python3_EXECUTABLE}" -m unittest discover
            -s "${CMAKE_CURRENT_SOURCE_DIR}/quality"
            -p test_rust_boundaries.py -v)
set_tests_properties(Quality.RustBoundaries PROPERTIES
    LABELS "host" TIMEOUT 1200)

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
add_dependencies(JuicerProductionProfileOwnerTests juicer_test_resources)
add_dependencies(JuicerCudaTerminalTests juicer_test_resources)

add_executable(JuicerScatterHalationHostTests
    scatter_halation/recipe_descriptor_test.cpp)
target_sources(JuicerScatterHalationHostTests PRIVATE ffi/spatial_optics_fixture.cpp)
target_include_directories(JuicerScatterHalationHostTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ffi")
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

set(JUICER_PREPARED_TEST_SOURCES
    ffi/prepared_projection_test.cpp ffi/prepared_span_test.cpp
    ffi/prepared_descriptors.cpp)
add_executable(JuicerPreparedProjectionTests ${JUICER_PREPARED_TEST_SOURCES} ffi/spatial_optics_fixture.cpp)
target_link_libraries(JuicerPreparedProjectionTests PRIVATE juicer_core GTest::gtest_main)
set_target_properties(JuicerPreparedProjectionTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerPreparedProjectionTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Ffi.Host.PreparedProjection COMMAND JuicerPreparedProjectionTests)
set_tests_properties(Ffi.Host.PreparedProjection PROPERTIES LABELS "host" TIMEOUT 30)
set_tests_properties(Ffi.Host.PreparedProjection PROPERTIES ENVIRONMENT "JUICER_TEST_PRESET_NAME=${JUICER_TEST_PRESET_NAME}")
include(ffi/prepared_sanitizers.cmake)
add_test(NAME Ffi.Host.SanitizerRunner
    COMMAND "${Python3_EXECUTABLE}" -B -m unittest discover
        -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p test_prepared_sanitizer_control.py -v)
set_tests_properties(Ffi.Host.SanitizerRunner PROPERTIES LABELS "host" TIMEOUT 30)

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
add_library(juicer_asset_lookup_test_objects OBJECT ../src/RenderRecipe.cpp)
target_link_libraries(juicer_asset_lookup_test_objects PRIVATE juicer_settings)
target_include_directories(juicer_asset_lookup_test_objects PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ffi")
target_compile_definitions(juicer_asset_lookup_test_objects PRIVATE JUICER_ASSET_LOOKUP_TEST_HOOK=1)
add_executable(JuicerAssetLookupTests
    ffi/asset_lookup_test.cpp $<TARGET_OBJECTS:juicer_asset_lookup_test_objects>)
target_link_libraries(JuicerAssetLookupTests PRIVATE juicer_core GTest::gtest_main)
target_compile_definitions(JuicerAssetLookupTests PRIVATE JUICER_ASSET_LOOKUP_TEST_HOOK=1)
target_compile_definitions(JuicerAssetLookupTests PRIVATE JUICER_LOOKUP_RESOURCE_DIR="${JUICER_TEST_RESOURCE_DIR}")
add_dependencies(JuicerAssetLookupTests juicer_test_resources)
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
    PROPERTIES LABELS "host\\;product" TIMEOUT 120)

gtest_discover_tests(JuicerScatterHalationHostTests
    TEST_PREFIX "ScatterHalation.Host."
    DISCOVERY_MODE POST_BUILD
    DISCOVERY_TIMEOUT 30
    PROPERTIES
        LABELS "host\\;product"
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
target_sources(JuicerDiffusionHostReferenceTest PRIVATE ffi/spatial_optics_fixture.cpp)
target_include_directories(JuicerDiffusionHostReferenceTest PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ffi")
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

add_executable(JuicerProcessorReferenceProbe ofx/processor_reference_test.cpp ofx/render_assertions.cpp)
target_sources(JuicerProcessorReferenceProbe PRIVATE ffi/spatial_optics_fixture.cpp)
target_link_libraries(JuicerProcessorReferenceProbe PRIVATE
    juicer_core juicer_ofxs CUDA::cuda_driver)
target_compile_definitions(JuicerProcessorReferenceProbe PRIVATE
    JUICER_ACCEPTED_CAPTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/accepted_cuda_clean_samples.json"
    JUICER_PROCESSOR_REFERENCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/exposure/processor-${JUICER_TEST_PRESET_NAME}.dat")
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
    ofx/processor_reference_test.cpp ofx/render_assertions.cpp
    ffi/spatial_optics_fixture.cpp
    ffi/prepared_boundary.cpp
    ffi/cuda_render_test.cpp
    $<TARGET_OBJECTS:juicer_render_test_objects>
    $<TARGET_OBJECTS:juicer_pinned_upload_test_objects>
    ffi/prepared_boundary.c
    ffi/prepared_descriptors.cpp)
add_library(juicer_illuminant_preflash_root OBJECT ../src/ProcessRoot.cpp)
target_link_libraries(juicer_illuminant_preflash_root PRIVATE juicer_settings)
target_compile_definitions(juicer_illuminant_preflash_root PRIVATE JUICER_ILLUMINANT_TEST_HOOK=1)
target_sources(JuicerPreparedBoundaryProbe PRIVATE $<TARGET_OBJECTS:juicer_illuminant_preflash_root>)
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
    JUICER_TEST_RESOURCE_DIR="${JUICER_TEST_RESOURCE_DIR}"
    JUICER_PROCESSOR_REFERENCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/exposure/processor-${JUICER_TEST_PRESET_NAME}.dat")
set_target_properties(JuicerPreparedBoundaryProbe PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerPreparedBoundaryProbe juicer_test_resources)
if(UNIX)
    set_property(TARGET JuicerPreparedBoundaryProbe PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
foreach(prepared_case IN ITEMS negative-direct negative-print positive-direct positive-print
        combined-print glare-plus-zero glare-minus-zero mallett-direct arctic-print)
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
    ../src/RustAssetBridge.cpp
    ../src/CudaRenderProjection.cpp
    ../native/juicer_cuda_api.cpp
    ../src/Cuda/JuicerCudaExecutor.cpp
    ../src/Cuda/JuicerCudaResources.cpp
    ../src/ProcessRoot.cpp)
target_link_libraries(juicer_executor_failure_test_objects PRIVATE juicer_settings juicer_ofx_headers)
target_compile_definitions(juicer_executor_failure_test_objects PRIVATE
    JUICER_EXECUTOR_FAILURE_TEST_HOOK=1 JUICER_NOISE_TEST_HOOK=1)
add_executable(JuicerExecutorFailureOrderProbe ofx/processor_reference_test.cpp ofx/render_assertions.cpp
    $<TARGET_OBJECTS:juicer_executor_failure_test_objects> ffi/spatial_optics_fixture.cpp)
target_link_libraries(JuicerExecutorFailureOrderProbe PRIVATE
    juicer_core juicer_ofxs CUDA::cuda_driver)
target_compile_definitions(JuicerExecutorFailureOrderProbe PRIVATE
    JUICER_EXECUTOR_FAILURE_TEST_HOOK=1 JUICER_NOISE_TEST_HOOK=1
    JUICER_PROCESSOR_REFERENCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/exposure/processor-${JUICER_TEST_PRESET_NAME}.dat")
set_target_properties(JuicerExecutorFailureOrderProbe PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerExecutorFailureOrderProbe juicer_test_resources)
if(UNIX)
    set_property(TARGET JuicerExecutorFailureOrderProbe
        PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME Ofx.Gpu.NoiseSourceLifetime COMMAND JuicerExecutorFailureOrderProbe --noise-lifetime)
set_tests_properties(Ofx.Gpu.NoiseSourceLifetime PROPERTIES LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
add_test(NAME Ofx.Gpu.GrainFrame COMMAND JuicerExecutorFailureOrderProbe --grain-frame
    "${CMAKE_CURRENT_SOURCE_DIR}/grain/frame-fixtures/manifest.json" "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ofx.Gpu.GrainFrame PROPERTIES LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)
add_test(NAME Ofx.Gpu.DirFrame COMMAND JuicerExecutorFailureOrderProbe --dir-frame
    "${CMAKE_CURRENT_SOURCE_DIR}/dir/frame-fixtures/manifest.json" "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ofx.Gpu.DirFrame PROPERTIES LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)

add_test(NAME Ofx.Gpu.ExecutorFailureOrder
    COMMAND JuicerExecutorFailureOrderProbe --failure-order)
set_tests_properties(Ofx.Gpu.ExecutorFailureOrder PROPERTIES
    LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)
foreach(resource_failure IN ITEMS DefectFence GrainUpload ScannerLut)
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
    ofx/frame_geometry_test_hooks.cpp
    ofx/adapter_trace_test.cpp route/route_failure_adapter.cpp ../src/JuicerEffect.cpp)
# Isolate the fixture replacement while linking the real route metadata/resolver.
set_property(SOURCE ../src/JuicerEffect.cpp ofx/adapter_trace_test.cpp ofx/frame_geometry_test_hooks.cpp
    APPEND PROPERTY COMPILE_DEFINITIONS JUICER_FRAME_GEOMETRY_TEST_HOOK=1)
target_compile_definitions(JuicerAdapterTraceProbe PRIVATE
    resolve_scan_route=resolve_scan_route_with_fault)
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
target_sources(JuicerAdapterTraceProbe PRIVATE ffi/spatial_optics_fixture.cpp)
add_test(NAME Ofx.Gpu.FrameGeometry COMMAND JuicerAdapterTraceProbe --frame-geometry
    "${CMAKE_CURRENT_SOURCE_DIR}/frame/geometry-fixtures/manifest.json" "${JUICER_TEST_PRESET_NAME}"
    "${JUICER_TEST_ARTIFACT_ROOT}/ofx/frame-geometry.log" "${CMAKE_CURRENT_SOURCE_DIR}/scatter_halation/frame-fixtures/manifest.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/diffusion/frame-fixtures/manifest.json")
set_tests_properties(Ofx.Gpu.FrameGeometry PROPERTIES LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu
    ENVIRONMENT "JUICER_DIAGNOSTICS=3;JUICER_DIAGNOSTICS_PATH=${JUICER_TEST_ARTIFACT_ROOT}/ofx/frame-geometry-diagnostic.log")
add_test(NAME Ofx.Gpu.AdapterTrace
    COMMAND JuicerAdapterTraceProbe
        --trace-output
        "${JUICER_TEST_ARTIFACT_ROOT}/ofx/adapter_trace.txt"
        --grain-fixtures "${CMAKE_CURRENT_SOURCE_DIR}/film/grain-fixtures/manifest.json" "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Ofx.Gpu.AdapterTrace PROPERTIES
    LABELS "gpu" TIMEOUT 1800 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerScatterHalationGpuTests
    scatter_halation/integration_test.cpp ofx/render_assertions.cpp
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
target_compile_definitions(JuicerScatterHalationGpuTests PRIVATE
    JUICER_PARAMETER_IDENTITY_PATH="${CMAKE_CURRENT_SOURCE_DIR}/ofx/fixtures/exposure/parameter-identities-${JUICER_TEST_PRESET_NAME}.json")
target_sources(JuicerScatterHalationGpuTests PRIVATE ffi/spatial_optics_fixture.cpp)
target_include_directories(JuicerScatterHalationGpuTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ffi")
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

add_executable(JuicerGrainScratchTests grain/scratch_reuse_test.cpp ffi/spatial_optics_fixture.cpp)
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

add_executable(JuicerDirExposureCacheTests dir/exposure_cache_test.cpp ffi/spatial_optics_fixture.cpp
    dir/boundary_test_hooks.cpp $<TARGET_OBJECTS:juicer_dir_boundary_test_objects>)
target_compile_definitions(JuicerDirExposureCacheTests PRIVATE
    JUICER_DIR_BOUNDARY_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/dir/boundary-fixtures/${JUICER_TEST_PRESET_NAME}.json")
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
target_include_directories(JuicerScannerOutputTests PRIVATE ffi)
target_link_libraries(JuicerScannerOutputTests PRIVATE juicer_output_encoding_oracle
    juicer_core GTest::gtest CUDA::cuda_driver)
set_target_properties(JuicerScannerOutputTests PROPERTIES
    LINKER_LANGUAGE CXX
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
add_dependencies(JuicerScannerOutputTests juicer_test_resources)
if(UNIX)
    set_property(TARGET JuicerScannerOutputTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
foreach(gaussian_consumer IN ITEMS JuicerScannerOutputTests JuicerGrainScratchTests JuicerDirExposureCacheTests)
    target_compile_definitions(${gaussian_consumer} PRIVATE
        JUICER_GAUSSIAN_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/optics/gaussian-fixtures/${JUICER_TEST_PRESET_NAME}.json")
endforeach()
add_test(NAME Scanner.Gpu.OutputEncoding COMMAND JuicerScannerOutputTests --gtest_filter=-ScannerRoutes.CorrectionRoutes)
add_test(NAME Scanner.Gpu.CorrectionRoutes COMMAND JuicerScannerOutputTests --gtest_filter=ScannerRoutes.CorrectionRoutes)
set_tests_properties(Scanner.Gpu.CorrectionRoutes PROPERTIES
    LABELS "gpu" TIMEOUT 300 RESOURCE_LOCK juicer_gpu)
set_tests_properties(Scanner.Gpu.OutputEncoding PROPERTIES
    LABELS "gpu" TIMEOUT 1200 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerRenderAssertionTests ofx/render_assertions_test.cpp ofx/render_assertions.cpp)
target_compile_features(JuicerRenderAssertionTests PRIVATE cxx_std_20)
target_link_libraries(JuicerRenderAssertionTests PRIVATE GTest::gtest_main)
add_test(NAME Ofx.Host.RenderAssertions COMMAND JuicerRenderAssertionTests)
set_tests_properties(Ofx.Host.RenderAssertions PROPERTIES LABELS "host" TIMEOUT 30)

add_test(NAME Ofx.Gpu.AcceptedCudaCaptures COMMAND JuicerProcessorReferenceProbe --accepted-clean-captures)
set_tests_properties(Ofx.Gpu.AcceptedCudaCaptures PROPERTIES LABELS "gpu" TIMEOUT 180 RESOURCE_LOCK juicer_gpu)

add_dependencies(JuicerSpectralOwnerTests juicer_test_resources)

add_library(juicer_spectral_test_objects OBJECT ../src/RustAssetBridge.cpp ../src/SpectralProcessing.cpp)
target_link_libraries(juicer_spectral_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_spectral_test_objects PRIVATE JUICER_SPECTRAL_TEST_HOOK=1)
add_executable(JuicerSpectralBootstrapTests ffi/spectral_bootstrap_test.cpp
    $<TARGET_OBJECTS:juicer_spectral_test_objects>)
target_link_libraries(JuicerSpectralBootstrapTests PRIVATE juicer_core GTest::gtest)
target_compile_definitions(JuicerSpectralBootstrapTests PRIVATE JUICER_SPECTRAL_TEST_HOOK=1)
set_target_properties(JuicerSpectralBootstrapTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerSpectralBootstrapTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_dependencies(JuicerSpectralBootstrapTests juicer_test_resources)
foreach(spectral_group IN ITEMS Bootstrap Copy)
    add_test(NAME Assets.Host.Spectral${spectral_group} COMMAND JuicerSpectralBootstrapTests
        "--gtest_filter=Spectral${spectral_group}.*"
        "${JUICER_TEST_RESOURCE_DIR}"
        "${JUICER_TEST_ARTIFACT_ROOT}/trusted-resource-decoding/C8/slice-3/bootstrap")
    set_tests_properties(Assets.Host.Spectral${spectral_group} PROPERTIES LABELS "host" TIMEOUT 120)
endforeach()

add_library(juicer_illuminant_test_objects OBJECT ../src/RustAssetBridge.cpp ../src/ResourceAssetLibrary.cpp ../src/JuicerState.cpp ../src/CudaRenderProjection.cpp)
target_link_libraries(juicer_illuminant_test_objects PRIVATE juicer_settings juicer_ofx_headers)
target_compile_definitions(juicer_illuminant_test_objects PRIVATE JUICER_ILLUMINANT_TEST_HOOK=1)
add_executable(JuicerIlluminantCalibrationTests ffi/illuminant_calibration_abi_c.c ffi/illuminant_calibration_test.cpp ffi/illuminant_construction_test.cpp
    $<TARGET_OBJECTS:juicer_illuminant_test_objects>)
target_link_libraries(JuicerIlluminantCalibrationTests PRIVATE juicer_core GTest::gtest)
target_compile_definitions(JuicerIlluminantCalibrationTests PRIVATE JUICER_ILLUMINANT_TEST_HOOK=1)
set_target_properties(JuicerIlluminantCalibrationTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerIlluminantCalibrationTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_dependencies(JuicerIlluminantCalibrationTests juicer_test_resources)
foreach(illuminant_group IN ITEMS IlluminantAbi IlluminantCopy IlluminantLibrary IlluminantMath IlluminantConstruction CalibrationAbi CalibrationNative IlluminantNative)
    add_test(NAME Assets.Host.${illuminant_group} COMMAND JuicerIlluminantCalibrationTests
        "--gtest_filter=${illuminant_group}.*"
        "${JUICER_TEST_RESOURCE_DIR}" "${PROJECT_SOURCE_DIR}/tests/ffi/fixtures/illuminants"
        "${JUICER_TEST_ARTIFACT_ROOT}/trusted-resource-decoding/C8/slice-4/consumers")
    set_tests_properties(Assets.Host.${illuminant_group} PROPERTIES LABELS "host" TIMEOUT 120)
endforeach()

add_executable(JuicerColorPreparationTests ffi/color_abi_c.c ffi/color_preparation_test.cpp
    $<TARGET_OBJECTS:juicer_admission_test_state>
    $<TARGET_OBJECTS:juicer_profile_conversion_test_objects>)
target_compile_definitions(JuicerColorPreparationTests PRIVATE
    JUICER_ADMISSION_TEST_HOOK=1 JUICER_PROFILE_CONVERSION_TEST_HOOK=1)
set_target_properties(JuicerColorPreparationTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF)
target_link_libraries(JuicerColorPreparationTests PRIVATE juicer_core)
set_target_properties(JuicerColorPreparationTests PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerColorPreparationTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_dependencies(JuicerColorPreparationTests juicer_test_resources)

add_test(NAME Ffi.Host.SharedFixtureLoading COMMAND JuicerColorPreparationTests
    "${JUICER_TEST_ARTIFACT_ROOT}/fixtures/shared-loader" unused fixture-format)
set_tests_properties(Ffi.Host.SharedFixtureLoading PROPERTIES LABELS "host" TIMEOUT 30)

add_test(NAME Color.Host.Cat16 COMMAND JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/${JUICER_TEST_PRESET_NAME}.json" leaf)
set_tests_properties(Color.Host.Cat16 PROPERTIES LABELS "host" TIMEOUT 60)
add_test(NAME Ffi.Host.ColorPreparation COMMAND JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/${JUICER_TEST_PRESET_NAME}.json" preparation)
set_tests_properties(Ffi.Host.ColorPreparation PROPERTIES LABELS "host" TIMEOUT 180)

add_test(NAME Color.Host.Cat02 COMMAND JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/cat02-${JUICER_TEST_PRESET_NAME}.json" cat02-leaf)
set_tests_properties(Color.Host.Cat02 PROPERTIES LABELS "host" TIMEOUT 60)
add_test(NAME Ffi.Host.Cat02Preparation COMMAND JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/cat02-${JUICER_TEST_PRESET_NAME}.json" cat02-preparation)
set_tests_properties(Ffi.Host.Cat02Preparation PROPERTIES LABELS "host" TIMEOUT 180)

add_test(NAME Color.Host.InputColor COMMAND JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/input-color-${JUICER_TEST_PRESET_NAME}.json" input-leaf)
set_tests_properties(Color.Host.InputColor PROPERTIES LABELS "host" TIMEOUT 60)
add_test(NAME Ffi.Host.InputColorPreparation COMMAND JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/input-color-${JUICER_TEST_PRESET_NAME}.json" input-preparation)
# Admission hooks and asset caches are process-local; bound host contention to two slots.
set_tests_properties(Ffi.Host.InputColorPreparation PROPERTIES LABELS "host" TIMEOUT 600 PROCESSORS 2)

add_test(NAME Color.Gpu.ScannerReuse COMMAND JuicerColorPreparationTests
    "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/${JUICER_TEST_PRESET_NAME}.json" reuse-gpu)
set_tests_properties(Color.Gpu.ScannerReuse PROPERTIES LABELS "gpu" TIMEOUT 120 RESOURCE_LOCK juicer_gpu)

add_executable(JuicerExposurePreparationTests ffi/exposure_abi_c.c ffi/exposure_preparation_test.cpp
    ffi/spectral_products_test.cpp ffi/spectral_allocation_probe.cpp)
target_link_libraries(JuicerExposurePreparationTests PRIVATE juicer_core)
set_target_properties(JuicerExposurePreparationTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerExposurePreparationTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
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
    set_tests_properties(${exposure_name} PROPERTIES LABELS "host" TIMEOUT 90)
endforeach()
add_dependencies(JuicerExposurePreparationTests juicer_test_resources)
add_test(NAME Exposure.Host.Products COMMAND JuicerExposurePreparationTests products
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/exposure/products-${JUICER_TEST_PRESET_NAME}.json"
    "${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_ARTIFACT_ROOT}/exposure/products")
set_tests_properties(Exposure.Host.Products PROPERTIES LABELS "host" TIMEOUT 300)
add_test(NAME Exposure.Host.Admission COMMAND JuicerExposurePreparationTests admission
    "${JUICER_TEST_RESOURCE_DIR}")
set_tests_properties(Exposure.Host.Admission PROPERTIES LABELS "host" TIMEOUT 600)

add_test(NAME Exposure.Host.FixtureFormat COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p test_exposure_fixtures.py -v)
set_tests_properties(Exposure.Host.FixtureFormat PROPERTIES LABELS "host" TIMEOUT 30
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

add_test(NAME Ffi.Host.SharedFixtureFormat COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p test_shared_fixtures.py -v)
set_tests_properties(Ffi.Host.SharedFixtureFormat PROPERTIES LABELS "host" TIMEOUT 30
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")


add_executable(JuicerMallettExposureTests ffi/mallett_exposure_abi_c.c ffi/mallett_exposure_test.cpp ffi/mallett_caller_test.cpp
    $<TARGET_OBJECTS:juicer_admission_test_state>)
target_compile_definitions(JuicerMallettExposureTests PRIVATE JUICER_ADMISSION_TEST_HOOK=1)
target_link_libraries(JuicerMallettExposureTests PRIVATE juicer_core)
set_target_properties(JuicerMallettExposureTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerMallettExposureTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
foreach(mallett_case IN ITEMS numerical boundary)
    add_test(NAME Exposure.Host.Mallett.${mallett_case} COMMAND JuicerMallettExposureTests
        "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/exposure/mallett-manifest.json" "${JUICER_TEST_PRESET_NAME}" ${mallett_case})
    set_tests_properties(Exposure.Host.Mallett.${mallett_case} PROPERTIES LABELS "host" TIMEOUT 120)
endforeach()

add_dependencies(JuicerMallettExposureTests juicer_test_resources)
add_test(NAME Exposure.Host.Mallett.Spectra COMMAND JuicerMallettExposureTests
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/exposure/mallett-manifest.json" "${JUICER_TEST_PRESET_NAME}" spectra
    "${JUICER_TEST_RESOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/cat02-${JUICER_TEST_PRESET_NAME}.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/input-color-${JUICER_TEST_PRESET_NAME}.json")
set_tests_properties(Exposure.Host.Mallett.Spectra PROPERTIES LABELS "host" TIMEOUT 120)

foreach(mallett_case IN ITEMS products admission)
    add_test(NAME Exposure.Host.Mallett.${mallett_case} COMMAND JuicerMallettExposureTests
        "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/exposure/mallett-manifest.json" "${JUICER_TEST_PRESET_NAME}" ${mallett_case} "${JUICER_TEST_RESOURCE_DIR}")
    set_tests_properties(Exposure.Host.Mallett.${mallett_case} PROPERTIES LABELS "host" TIMEOUT 600)
endforeach()

add_test(NAME Exposure.Host.Mallett.FixtureFormat COMMAND "${Python3_EXECUTABLE}" -m unittest discover
    -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p test_mallett_exposure_fixtures.py -v)
set_tests_properties(Exposure.Host.Mallett.FixtureFormat PROPERTIES LABELS "host" TIMEOUT 30
    ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

add_executable(JuicerSpectralPreparationTests ffi/spectral_preparation_abi_c.c ffi/spectral_preparation_test.cpp ffi/spectral_products_test.cpp ffi/spectral_allocation_probe.cpp)
target_link_libraries(JuicerSpectralPreparationTests PRIVATE juicer_core)
set_target_properties(JuicerSpectralPreparationTests PROPERTIES
    LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerSpectralPreparationTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
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
        "${PROJECT_SOURCE_DIR}/tests/ffi/fixtures/spectral/${JUICER_TEST_PRESET_NAME}.json"
        "${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_ARTIFACT_ROOT}/spectral/${spectral_case}")
    set_tests_properties(${spectral_name} PROPERTIES LABELS "host" TIMEOUT 120)
endforeach()

foreach(spectral_case IN ITEMS products admission)
    if(spectral_case STREQUAL "products")
        set(spectral_name Spectral.Host.Products)
    else()
        set(spectral_name Spectral.Host.Admission)
    endif()
    add_test(NAME ${spectral_name} COMMAND JuicerSpectralPreparationTests ${spectral_case}
        "${PROJECT_SOURCE_DIR}/tests/ffi/fixtures/spectral/${JUICER_TEST_PRESET_NAME}.json"
        "${JUICER_TEST_RESOURCE_DIR}" "${JUICER_TEST_ARTIFACT_ROOT}/spectral/${spectral_case}")
    set_tests_properties(${spectral_name} PROPERTIES LABELS "host" TIMEOUT 300)
endforeach()

add_executable(JuicerTcLutTests ffi/tc_lut_abi_c.c ffi/tc_lut_test.cpp ffi/spectral_products_test.cpp)
target_link_libraries(JuicerTcLutTests PRIVATE juicer_core)
set_target_properties(JuicerTcLutTests PROPERTIES LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerTcLutTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_dependencies(JuicerTcLutTests juicer_test_resources)
foreach(tc_case IN ITEMS boundary numerical products admission)
    add_test(NAME Reconstruction.Host.TcLut.${tc_case} COMMAND JuicerTcLutTests ${tc_case} "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/tc_lut/manifest.json" "${JUICER_TEST_PRESET_NAME}")
    set_tests_properties(Reconstruction.Host.TcLut.${tc_case} PROPERTIES LABELS "host" TIMEOUT 600)
endforeach()

add_library(juicer_tc_lut_test_objects OBJECT ../src/RustReconstructionBridge.cpp)
target_link_libraries(juicer_tc_lut_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_tc_lut_test_objects PRIVATE JUICER_TC_LUT_TEST_HOOK=1)
add_executable(JuicerTcLutLifetimeTests ffi/tc_lut_lifetime_test.cpp ffi/spectral_allocation_probe.cpp $<TARGET_OBJECTS:juicer_tc_lut_test_objects>)
target_link_libraries(JuicerTcLutLifetimeTests PRIVATE juicer_core)
target_compile_definitions(JuicerTcLutLifetimeTests PRIVATE JUICER_TC_LUT_TEST_HOOK=1)
set_target_properties(JuicerTcLutLifetimeTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerTcLutLifetimeTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_dependencies(JuicerTcLutLifetimeTests juicer_test_resources)
add_test(NAME Reconstruction.Host.TcLut.Lifetime COMMAND JuicerTcLutLifetimeTests "${JUICER_TEST_RESOURCE_DIR}")
set_tests_properties(Reconstruction.Host.TcLut.Lifetime PROPERTIES LABELS "host" TIMEOUT 180)

add_test(NAME Reconstruction.Host.TcLut.FixtureFormat COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p test_tc_lut_fixtures.py -v)
set_tests_properties(Reconstruction.Host.TcLut.FixtureFormat PROPERTIES LABELS "host" TIMEOUT 30 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

add_library(juicer_input_hull_test_objects OBJECT ../src/RustGamutBridge.cpp)
target_link_libraries(juicer_input_hull_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_input_hull_test_objects PRIVATE JUICER_INPUT_HULL_TEST_HOOK=1)
add_executable(JuicerInputHullTests ffi/input_hull_abi_c.c ffi/input_hull_test.cpp ffi/spectral_products_test.cpp ffi/spectral_allocation_probe.cpp $<TARGET_OBJECTS:juicer_input_hull_test_objects>)
target_link_libraries(JuicerInputHullTests PRIVATE juicer_core)
set_target_properties(JuicerInputHullTests PROPERTIES LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerInputHullTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_dependencies(JuicerInputHullTests juicer_test_resources)
foreach(hull_case IN ITEMS boundary binding products cache admission-1 admission-2 admission-3 admission-4 admission-5 admission-6)
    add_test(NAME Gamut.Host.InputHull.${hull_case} COMMAND JuicerInputHullTests ${hull_case}
        "${JUICER_TEST_RESOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/input_hull/manifest.json" "${JUICER_TEST_PRESET_NAME}")
    set_tests_properties(Gamut.Host.InputHull.${hull_case} PROPERTIES LABELS "host" TIMEOUT 180)
endforeach()
add_test(NAME Gamut.Host.InputHull.Math
    COMMAND "${CMAKE_COMMAND}" -E env "RUSTFLAGS=" "CARGO_BUILD_RUSTFLAGS=" "CARGO_ENCODED_RUSTFLAGS="
        ${JUICER_RUST_TEST_COMMAND} --features test-support --test input_hull)
set_tests_properties(Gamut.Host.InputHull.Math PROPERTIES LABELS "host" TIMEOUT 120 RESOURCE_LOCK juicer_cargo WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
add_test(NAME Gamut.Host.InputHull.FixtureFormat COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p test_input_hull_fixtures.py -v)
set_tests_properties(Gamut.Host.InputHull.FixtureFormat PROPERTIES LABELS "host" TIMEOUT 30 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

add_executable(JuicerOutputGamutTests ffi/output_gamut_abi_c.c ffi/output_gamut_test.cpp)
target_link_libraries(JuicerOutputGamutTests PRIVATE juicer_core)
set_target_properties(JuicerOutputGamutTests PROPERTIES LINKER_LANGUAGE CXX C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerOutputGamutTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
foreach(gamut_case IN ITEMS boundary binding)
    add_test(NAME Gamut.Host.Output.${gamut_case} COMMAND JuicerOutputGamutTests ${gamut_case}
        "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/output_gamut/manifest.json" "${JUICER_TEST_PRESET_NAME}")
    set_tests_properties(Gamut.Host.Output.${gamut_case} PROPERTIES LABELS "host" TIMEOUT 180)
endforeach()
add_test(NAME Gamut.Host.Output.Math
    COMMAND "${CMAKE_COMMAND}" -E env "RUSTFLAGS=" "CARGO_BUILD_RUSTFLAGS=" "CARGO_ENCODED_RUSTFLAGS="
        ${JUICER_RUST_TEST_COMMAND} --features test-support --test output_gamut)
set_tests_properties(Gamut.Host.Output.Math PROPERTIES LABELS "host" TIMEOUT 120 RESOURCE_LOCK juicer_cargo WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")

add_library(juicer_output_gamut_test_objects OBJECT ../src/ResourceAssetLibrary.cpp ../src/RustGamutBridge.cpp)
target_link_libraries(juicer_output_gamut_test_objects PRIVATE juicer_settings)
target_compile_definitions(juicer_output_gamut_test_objects PRIVATE JUICER_OUTPUT_GAMUT_TEST_HOOK=1)
add_executable(JuicerOutputGamutCacheTests ffi/output_gamut_cache_test.cpp $<TARGET_OBJECTS:juicer_output_gamut_test_objects>)
target_link_libraries(JuicerOutputGamutCacheTests PRIVATE juicer_core)
set_target_properties(JuicerOutputGamutCacheTests PROPERTIES LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerOutputGamutCacheTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_dependencies(JuicerOutputGamutCacheTests juicer_test_resources)
add_test(NAME Gamut.Host.Output.Cache COMMAND JuicerOutputGamutCacheTests "${JUICER_TEST_RESOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/output_gamut/manifest.json" "${JUICER_TEST_PRESET_NAME}")
set_tests_properties(Gamut.Host.Output.Cache PROPERTIES LABELS "host" TIMEOUT 180)
add_test(NAME Gamut.Host.Output.FixtureFormat COMMAND "${Python3_EXECUTABLE}" -m unittest discover -s "${CMAKE_CURRENT_SOURCE_DIR}/ffi" -p test_output_gamut_fixtures.py -v)
set_tests_properties(Gamut.Host.Output.FixtureFormat PROPERTIES LABELS "host" TIMEOUT 30 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ffi/output-color-development.json")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/ffi/output-color-development.json" output_color_development)
string(JSON output_color_development_target GET "${output_color_development}" target)
string(JSON output_color_development_test GET "${output_color_development}" test)
string(JSON output_color_development_fixture GET "${output_color_development}" fixture)
add_executable(${output_color_development_target} ffi/output_color_abi_c.c ffi/output_color_test.cpp
    $<TARGET_OBJECTS:juicer_scanner_test_objects>)
target_compile_definitions(JuicerOutputColorTests PRIVATE JUICER_SCANNER_TEST_HOOK=1)
target_link_libraries(JuicerOutputColorTests PRIVATE juicer_core)
set_target_properties(JuicerOutputColorTests PROPERTIES
    C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
if(UNIX)
    set_property(TARGET JuicerOutputColorTests PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
add_test(NAME ${output_color_development_test} COMMAND ${output_color_development_target}
    "${PROJECT_SOURCE_DIR}/${output_color_development_fixture}")
set_tests_properties(Ffi.Host.OutputColor PROPERTIES LABELS "host" TIMEOUT 60)
add_test(NAME Ffi.Host.ScannerColor COMMAND JuicerOutputColorTests scanner-color
    "${CMAKE_CURRENT_SOURCE_DIR}/scanner/color-fixtures/native.json"
    "${PROJECT_SOURCE_DIR}/${output_color_development_fixture}"
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/fixtures/color/cat02-${JUICER_TEST_PRESET_NAME}.json")
set_tests_properties(Ffi.Host.ScannerColor PROPERTIES LABELS "host" TIMEOUT 60)


add_library(juicer_output_encoding_oracle STATIC scanner/output_encoding_oracle.cpp)
target_include_directories(juicer_output_encoding_oracle PUBLIC scanner "${PROJECT_SOURCE_DIR}/src")
target_compile_features(juicer_output_encoding_oracle PRIVATE cxx_std_20)
target_link_libraries(JuicerOutputColorTests PRIVATE juicer_output_encoding_oracle)


# Relevance and runtime prerequisites are independent. Every group has one owner.
set(juicer_boundary_tests Ffi.Host.GrainFrame Ffi.Host.DirBoundary Ffi.Host.DirFrame Ffi.Host.DiffusionFrame Ffi.Host.HalationFrame Ffi.Host.FrameGeometry Ffi.Host.GaussianKernel Ffi.Host.ScannerPostEffects Ffi.Host.FilmReference
    Ffi.Host.VisualGrain Ffi.Host.PrintIlluminant Ffi.Host.PrintPreflash Ffi.Host.PrintBalance Ffi.Host.ScannerCorrection Ffi.Host.ScannerColor Ffi.Host.ScannerLut Scanner.Host.LutProducts Ffi.Host.ScannerLutDescriptor Quality.RustNaming Quality.RustBoundaries)
set(juicer_corpus_tests Ffi.Host.SharedFixtureFormat Exposure.Host.FixtureFormat Exposure.Host.Mallett.FixtureFormat
    Reconstruction.Host.TcLut.FixtureFormat Gamut.Host.InputHull.FixtureFormat
    Gamut.Host.Output.FixtureFormat)
get_property(juicer_registered_tests DIRECTORY PROPERTY TESTS)
foreach(juicer_test IN LISTS juicer_registered_tests)
    if(juicer_test IN_LIST JUICER_TOOLING_TESTS)
        set(juicer_relevance tooling)
    elseif(juicer_test IN_LIST juicer_boundary_tests)
        set(juicer_relevance boundary)
    elseif(juicer_test IN_LIST juicer_corpus_tests)
        set(juicer_relevance corpus)
    else()
        set(juicer_relevance product)
    endif()
    set_property(TEST "${juicer_test}" APPEND PROPERTY LABELS "${juicer_relevance}")
endforeach()
