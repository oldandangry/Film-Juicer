cmake_minimum_required(VERSION 3.24)

project(FilmJuicerToolingTests LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Tooling test configuration" FORCE)
endif()
if(NOT CMAKE_BUILD_TYPE MATCHES "^(Debug|Release)$")
    message(FATAL_ERROR "Tooling tests support Debug or Release configurations")
endif()

find_package(Python3 3.13 EXACT REQUIRED COMPONENTS Interpreter)
find_program(JUICER_TOOLING_CARGO NAMES cargo cargo.exe REQUIRED)

get_filename_component(JUICER_TOOLING_DEFAULT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(JUICER_TOOLING_ROOT "${JUICER_TOOLING_DEFAULT_ROOT}" CACHE PATH
    "Film-Juicer checkout root containing tests and Rust sources")
string(TOLOWER "${CMAKE_BUILD_TYPE}" tooling_configuration)
if(WIN32)
    set(tooling_preset "windows-clang-${tooling_configuration}")
else()
    set(tooling_preset "linux-${tooling_configuration}")
endif()
set(JUICER_TOOLING_ARTIFACT_ROOT
    "${JUICER_TOOLING_ROOT}/out/validation/${tooling_preset}" CACHE PATH
    "Tooling test artifact root")

include(CTest)
include("${JUICER_TOOLING_ROOT}/tests/ToolingTests.cmake")
if(CMAKE_HOST_WIN32)
    set(JUICER_TOOLING_HOST_WINDOWS TRUE)
else()
    set(JUICER_TOOLING_HOST_WINDOWS FALSE)
endif()
juicer_register_tooling_tests(
    ROOT "${JUICER_TOOLING_ROOT}"
    ARTIFACT_ROOT "${JUICER_TOOLING_ARTIFACT_ROOT}"
    PYTHON "${Python3_EXECUTABLE}"
    CMAKE_COMMAND "${CMAKE_COMMAND}"
    CTEST_COMMAND "${CMAKE_CTEST_COMMAND}"
    CARGO "${JUICER_TOOLING_CARGO}"
    NINJA "${CMAKE_MAKE_PROGRAM}"
    CC "${CMAKE_C_COMPILER}"
    BUILD_TYPE "${CMAKE_BUILD_TYPE}"
    HOST_WINDOWS "${JUICER_TOOLING_HOST_WINDOWS}"
    TESTS_OUT JUICER_TOOLING_TESTS)

# A standalone tooling entry contains no product or GPU tests. All registered
# cases belong to the tooling selection, including the product suite's existing
# Ffi.Host.RustBuild consumer contract.
foreach(tooling_test IN LISTS JUICER_TOOLING_TESTS)
    set_property(TEST "${tooling_test}" APPEND PROPERTY LABELS tooling)
endforeach()
