cmake_minimum_required(VERSION 4.3)

# CUDA 13.2 on Windows uses hybrid cudart. CMake 4.3 has no Hybrid runtime
# enumerator; leave its automatic runtime selection empty and link the imported
# cudart target below. Native CUDA compiler detection and compilation still apply.
if(CMAKE_HOST_WIN32)
    set(CMAKE_CUDA_RUNTIME_LIBRARY "")
    set(CMAKE_CUDA_FLAGS_INIT "--cudart=hybrid")
    if(NOT CMAKE_CUDA_HOST_COMPILER)
        find_program(JUICER_NVCC_MSVC cl.exe REQUIRED)
        set(CMAKE_CUDA_HOST_COMPILER "${JUICER_NVCC_MSVC}" CACHE FILEPATH "NVCC MSVC host")
    endif()
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT ProgramDatabase)
    # MSVC's /RTC stack/uninitialised checks cannot be combined with /O2.
    set(CMAKE_MSVC_RUNTIME_CHECKS "")
endif()

project(FilmJuicer LANGUAGES CXX CUDA)
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$" OR
   NOT (WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    message(FATAL_ERROR "Film-Juicer supports Windows and Linux x86-64 only")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Both configurations use the same optimisation policy and retain assertions.
# Debug adds symbols and invariant diagnostics to the optimised product.
foreach(language IN ITEMS CXX CUDA)
    string(REGEX REPLACE "(^| )[/-]DNDEBUG( |$)" " "
        CMAKE_${language}_FLAGS_RELEASE "${CMAKE_${language}_FLAGS_RELEASE}")
    set(CMAKE_${language}_FLAGS_DEBUG "${CMAKE_${language}_FLAGS_RELEASE}")
    if(NOT WIN32)
        string(APPEND CMAKE_${language}_FLAGS_DEBUG " -g")
    endif()
endforeach()
# Bound memory use independently of Ninja's overall worker count.
set_property(GLOBAL PROPERTY JOB_POOLS juicer_compile=8 juicer_link=2)
set(CMAKE_JOB_POOL_COMPILE juicer_compile)
set(CMAKE_JOB_POOL_LINK juicer_link)
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "75-real;75-virtual")
    message(FATAL_ERROR "The supported GPU policy is SM 7.5 plus compute_75 PTX; use the shared presets")
endif()
if(NOT CMAKE_BUILD_TYPE MATCHES "^(Debug|Release)$")
    message(FATAL_ERROR "Use a Debug or Release Ninja preset")
endif()

include(cmake/JuicerDependencies.cmake)
include(cmake/JuicerSources.cmake)
include(cmake/JuicerRust.cmake)

include(cmake/JuicerSettings.cmake)

add_library(juicer_native OBJECT ${JUICER_NATIVE_SOURCES})
target_link_libraries(juicer_native PRIVATE juicer_settings)
add_library(juicer_host OBJECT ${JUICER_HOST_SOURCES})
target_link_libraries(juicer_host PRIVATE juicer_settings juicer_ofx_headers)
add_library(juicer_adapter OBJECT ${JUICER_ADAPTER_SOURCES})
target_link_libraries(juicer_adapter PRIVATE juicer_settings juicer_ofx_headers)

# Compiler analysis consumes the ordinary objects for isolation checks and the
# Rust archive. Testing adds generated headers consumed by compiler analysis;
# the configured database retains every test-hook variant.
add_custom_target(juicer_analysis_inputs)
add_dependencies(juicer_analysis_inputs
    juicer_native juicer_host juicer_adapter juicer_rust_production_build)

# Preserve one aggregate archive for current product/test links and isolated
# test-hook object replacement, without compiling a second CUDA implementation.
add_library(juicer_core STATIC
    $<TARGET_OBJECTS:juicer_native>
    $<TARGET_OBJECTS:juicer_host>
    $<TARGET_OBJECTS:juicer_adapter>)
target_link_libraries(juicer_core PUBLIC juicer_settings juicer_ofx_headers)
set_target_properties(juicer_core PROPERTIES LINKER_LANGUAGE CXX)

add_library(juicer MODULE src/main.cpp)
target_link_libraries(juicer PRIVATE juicer_core film_juicer_rust juicer_ofxs juicer_dependencies)
set_target_properties(juicer PROPERTIES PREFIX "" SUFFIX ".ofx" LINKER_LANGUAGE CXX)
if(UNIX)
    # Static cuFFT contains relocatable device code even though our kernels do
    # not need separable compilation. Resolve it once at the final module.
    set_property(TARGET juicer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_options(juicer PRIVATE
        -static-libstdc++ -static-libgcc
        "LINKER:--version-script=${PROJECT_SOURCE_DIR}/cmake/juicer.exports"
        "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
    set_property(TARGET juicer APPEND PROPERTY LINK_DEPENDS "${PROJECT_SOURCE_DIR}/cmake/juicer.exports")
endif()

add_executable(JuicerProductionPreparedRenderProbe EXCLUDE_FROM_ALL
    tests/ffi/production-prepared-render.cpp src/main.cpp)
target_link_libraries(JuicerProductionPreparedRenderProbe PRIVATE
    juicer_core film_juicer_rust juicer_ofxs juicer_dependencies CUDA::cuda_driver)
set_target_properties(JuicerProductionPreparedRenderProbe PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/bin")
if(UNIX)
    set_property(TARGET JuicerProductionPreparedRenderProbe PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# These existing C11 value/ABI probes need the real native dependency of the
# production Rust archive. CMake owns their final link; normal builds exclude them.
set(JUICER_PRODUCTION_C_PROBES)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    enable_language(C)
    find_package(Python3 3.13 EXACT REQUIRED COMPONENTS Interpreter)
    if(WIN32)
        set(probe_preset windows-clang-release)
    else()
        set(probe_preset linux-release)
    endif()
    set(probe_fixture_directories
        tests/film/fixtures tests/film/input-fixtures tests/film/reference-fixtures
        tests/film/dir-fixtures tests/film/grain-fixtures tests/film/effects-fixtures
        tests/print/balance-fixtures tests/scanner/correction-fixtures
        tests/scanner/color-fixtures tests/dir/boundary-fixtures
        tests/film/effects-frame-fixtures tests/grain/frame-fixtures
        tests/dir/frame-fixtures tests/diffusion/frame-fixtures
        tests/scatter_halation/frame-fixtures tests/frame/geometry-fixtures
        tests/scanner/post-effects-fixtures tests/scanner/descriptor-fixtures)
    set(probe_fixture_inputs)
    foreach(directory IN LISTS probe_fixture_directories)
        file(GLOB_RECURSE fixture_inputs CONFIGURE_DEPENDS
            "${PROJECT_SOURCE_DIR}/${directory}/*.json")
        list(APPEND probe_fixture_inputs ${fixture_inputs})
    endforeach()
    set(probe_expected_header "${CMAKE_BINARY_DIR}/production-probes/production-film-expected.h")
    add_custom_command(OUTPUT "${probe_expected_header}"
        COMMAND "${Python3_EXECUTABLE}" -B "${PROJECT_SOURCE_DIR}/tests/ffi/production_contract.py"
            --preset "${probe_preset}" --artifacts "${CMAKE_BINARY_DIR}/production-probes"
            --generate-expected-header
        DEPENDS ${probe_fixture_inputs}
            tests/ffi/production_contract.py tests/ffi/shared_fixture.py
            tests/ffi/compact_exposure_fixtures.py tests/ffi/grain_probe_contract.py
            tests/ffi/effects_frame_probe_contract.py
        COMMENT "Generating C11 production fixture expectations"
        VERBATIM)
    foreach(probe IN ITEMS Film OutputColor)
        if(probe STREQUAL "Film")
            set(probe_source tests/ffi/production-film.c.in)
        else()
            set(probe_source tests/ffi/production-output-color.c.in)
        endif()
        set(probe_target JuicerProduction${probe}Probe)
        set_source_files_properties(${probe_source} PROPERTIES LANGUAGE C)
        add_executable(${probe_target} EXCLUDE_FROM_ALL ${probe_source})
        if(probe STREQUAL "Film")
            target_sources(${probe_target} PRIVATE "${probe_expected_header}")
        endif()
        set_target_properties(${probe_target} PROPERTIES
            C_STANDARD 11 C_STANDARD_REQUIRED ON C_EXTENSIONS OFF
            LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests/bin")
        target_include_directories(${probe_target} PRIVATE
            "${PROJECT_SOURCE_DIR}/tests/ffi" "${CMAKE_BINARY_DIR}/production-probes")
        target_link_libraries(${probe_target} PRIVATE
            juicer_core film_juicer_rust juicer_dependencies)
        # Preserve the probes' original C compilation and active assertions.
        if(WIN32)
            target_compile_options(${probe_target} PRIVATE /W4 /WX /Od /Ob0 /UNDEBUG)
        else()
            target_compile_options(${probe_target} PRIVATE -Wall -Wextra -Werror -O0 -UNDEBUG)
            set_property(TARGET ${probe_target} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        endif()
        list(APPEND JUICER_PRODUCTION_C_PROBES ${probe_target})
    endforeach()
endif()

include(cmake/JuicerBundle.cmake)

option(BUILD_TESTING "Build the public Film-Juicer test suite" OFF)
include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests EXCLUDE_FROM_ALL)
endif()

juicer_write_rust_artifacts(juicer JuicerProductionPreparedRenderProbe
    ${JUICER_PRODUCTION_C_PROBES} ${JUICER_RUST_TEST_CONSUMERS})
