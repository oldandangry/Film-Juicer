if(CMAKE_SCRIPT_MODE_FILE AND DEFINED JUICER_RUST_TESTS_METADATA_MODE)
    foreach(required_variable IN ITEMS
        JUICER_RUST_TESTS_CARGO
        JUICER_RUST_TESTS_NEXTEST
        JUICER_RUST_TESTS_ROOT
        JUICER_RUST_TESTS_TARGET
        JUICER_RUST_TESTS_TARGET_DIR
        JUICER_RUST_TESTS_BINARIES_METADATA
        JUICER_RUST_TESTS_CARGO_METADATA
        JUICER_RUST_TESTS_PROFILE)
        if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
            message(FATAL_ERROR "Missing ${required_variable} for Rust test metadata preparation")
        endif()
    endforeach()

    string(TIMESTAMP JUICER_RUST_TESTS_CARGO_START "%s%f" UTC)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "CARGO_TARGET_DIR=${JUICER_RUST_TESTS_TARGET_DIR}"
            "${JUICER_RUST_TESTS_CARGO}" metadata
            --format-version 1
            --locked
            --no-default-features
            --features film-juicer-plugin/test-support
            --filter-platform "${JUICER_RUST_TESTS_TARGET}"
            --manifest-path "${JUICER_RUST_TESTS_ROOT}/Cargo.toml"
        WORKING_DIRECTORY "${JUICER_RUST_TESTS_ROOT}"
        RESULT_VARIABLE JUICER_RUST_TESTS_CARGO_METADATA_RESULT
        OUTPUT_FILE "${JUICER_RUST_TESTS_CARGO_METADATA}"
        ERROR_VARIABLE JUICER_RUST_TESTS_CARGO_METADATA_ERROR)
    string(TIMESTAMP JUICER_RUST_TESTS_CARGO_END "%s%f" UTC)
    math(EXPR JUICER_RUST_TESTS_CARGO_US
        "${JUICER_RUST_TESTS_CARGO_END} - ${JUICER_RUST_TESTS_CARGO_START}")
    if(NOT JUICER_RUST_TESTS_CARGO_METADATA_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Could not save Cargo metadata for Rust tests: "
            "${JUICER_RUST_TESTS_CARGO_METADATA_ERROR}")
    endif()

    set(JUICER_RUST_TESTS_PROFILE_ARGUMENT)
    if(JUICER_RUST_TESTS_PROFILE STREQUAL "release")
        set(JUICER_RUST_TESTS_PROFILE_ARGUMENT --release)
    elseif(NOT JUICER_RUST_TESTS_PROFILE STREQUAL "debug")
        message(FATAL_ERROR "Unsupported Rust test Cargo profile: ${JUICER_RUST_TESTS_PROFILE}")
    endif()
    string(TIMESTAMP JUICER_RUST_TESTS_BINARIES_START "%s%f" UTC)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            "CARGO=${JUICER_RUST_TESTS_CARGO}"
            "RUSTFLAGS="
            "CARGO_BUILD_RUSTFLAGS="
            "CARGO_ENCODED_RUSTFLAGS="
            "${JUICER_RUST_TESTS_NEXTEST}" nextest list
            --locked
            --workspace
            --tests
            --no-default-features
            --features film-juicer-plugin/test-support
            --target "${JUICER_RUST_TESTS_TARGET}"
            --target-dir "${JUICER_RUST_TESTS_TARGET_DIR}"
            ${JUICER_RUST_TESTS_PROFILE_ARGUMENT}
            --list-type binaries-only
            --message-format json
            --config-file "${JUICER_RUST_TESTS_ROOT}/.config/nextest.toml"
        WORKING_DIRECTORY "${JUICER_RUST_TESTS_ROOT}"
        RESULT_VARIABLE JUICER_RUST_TESTS_BINARIES_RESULT
        OUTPUT_FILE "${JUICER_RUST_TESTS_BINARIES_METADATA}"
        ERROR_VARIABLE JUICER_RUST_TESTS_BINARIES_ERROR)
    string(TIMESTAMP JUICER_RUST_TESTS_BINARIES_END "%s%f" UTC)
    math(EXPR JUICER_RUST_TESTS_BINARIES_US
        "${JUICER_RUST_TESTS_BINARIES_END} - ${JUICER_RUST_TESTS_BINARIES_START}")
    if(NOT JUICER_RUST_TESTS_BINARIES_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Could not save nextest binaries metadata for Rust tests: "
            "${JUICER_RUST_TESTS_BINARIES_ERROR}")
    endif()
    message(STATUS "${JUICER_RUST_TESTS_BINARIES_ERROR}")
    file(WRITE "${JUICER_RUST_TESTS_TARGET_DIR}/metadata/preparation-timings.json"
        "{\"cargo_metadata_microseconds\":${JUICER_RUST_TESTS_CARGO_US},\"binaries_metadata_microseconds\":${JUICER_RUST_TESTS_BINARIES_US}}\n")
    return()
endif()

if(NOT DEFINED JUICER_RUST_SOURCE_DIR)
    set(JUICER_RUST_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
endif()

set(JUICER_RUST_VERSION "1.98.1")
set(JUICER_CARGO_EXECUTABLE "" CACHE FILEPATH "Cargo executable for the pinned Film-Juicer toolchain")
set(JUICER_NEXTEST_EXECUTABLE "" CACHE FILEPATH "cargo-nextest executable for focused Rust tests")

if(BUILD_TESTING)
    if(NOT JUICER_NEXTEST_EXECUTABLE)
        find_program(JUICER_NEXTEST_FOUND
            NAMES cargo-nextest cargo-nextest.exe
            HINTS
                "$ENV{USERPROFILE}/.cargo/bin"
                "$ENV{HOME}/.cargo/bin")
        if(JUICER_NEXTEST_FOUND)
            set(JUICER_NEXTEST_EXECUTABLE "${JUICER_NEXTEST_FOUND}"
                CACHE FILEPATH "cargo-nextest executable for focused Rust tests" FORCE)
        endif()
    endif()
    if(NOT JUICER_NEXTEST_EXECUTABLE)
        message(FATAL_ERROR "BUILD_TESTING requires cargo-nextest 0.9.143 on the native host")
    endif()
    if(JUICER_NEXTEST_EXECUTABLE)
        execute_process(
            COMMAND "${JUICER_NEXTEST_EXECUTABLE}" nextest --version
            WORKING_DIRECTORY "${JUICER_RUST_SOURCE_DIR}"
            RESULT_VARIABLE JUICER_NEXTEST_VERSION_RESULT
            OUTPUT_VARIABLE JUICER_NEXTEST_VERSION_OUTPUT
            ERROR_VARIABLE JUICER_NEXTEST_VERSION_ERROR
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT JUICER_NEXTEST_VERSION_RESULT EQUAL 0 OR
           NOT JUICER_NEXTEST_VERSION_OUTPUT MATCHES "^cargo-nextest 0\\.9\\.143($| )")
            message(FATAL_ERROR
                "Rust.Core and Rust.Bridge require cargo-nextest 0.9.143: "
                "${JUICER_NEXTEST_VERSION_OUTPUT}${JUICER_NEXTEST_VERSION_ERROR}")
        endif()
    else()
        message(FATAL_ERROR "BUILD_TESTING requires cargo-nextest 0.9.143 for Rust.Core and Rust.Bridge")
    endif()
endif()

if(NOT JUICER_CARGO_EXECUTABLE)
    find_program(JUICER_CARGO_FOUND
        NAMES cargo cargo.exe
        HINTS
            "$ENV{USERPROFILE}/.cargo/bin"
            "$ENV{HOME}/.cargo/bin"
        REQUIRED)
    set(JUICER_CARGO_EXECUTABLE "${JUICER_CARGO_FOUND}"
        CACHE FILEPATH "Cargo executable for the pinned Film-Juicer toolchain" FORCE)
endif()

get_filename_component(JUICER_RUST_BIN_DIR "${JUICER_CARGO_EXECUTABLE}" DIRECTORY)
if(WIN32)
    set(JUICER_RUSTC_EXECUTABLE "${JUICER_RUST_BIN_DIR}/rustc.exe")
    set(JUICER_RUST_TARGET "x86_64-pc-windows-msvc")
    set(JUICER_RUST_ARCHIVE_NAME "film_juicer_plugin.lib")
else()
    set(JUICER_RUSTC_EXECUTABLE "${JUICER_RUST_BIN_DIR}/rustc")
    set(JUICER_RUST_TARGET "x86_64-unknown-linux-gnu")
    set(JUICER_RUST_ARCHIVE_NAME "libfilm_juicer_plugin.a")
endif()
if(NOT EXISTS "${JUICER_RUSTC_EXECUTABLE}")
    message(FATAL_ERROR "rustc was not found beside Cargo: ${JUICER_RUSTC_EXECUTABLE}")
endif()

execute_process(
    COMMAND "${JUICER_CARGO_EXECUTABLE}" --version
    WORKING_DIRECTORY "${JUICER_RUST_SOURCE_DIR}"
    RESULT_VARIABLE JUICER_CARGO_VERSION_RESULT
    OUTPUT_VARIABLE JUICER_CARGO_VERSION_OUTPUT
    ERROR_VARIABLE JUICER_CARGO_VERSION_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT JUICER_CARGO_VERSION_RESULT EQUAL 0 OR
   NOT JUICER_CARGO_VERSION_OUTPUT MATCHES "^cargo ${JUICER_RUST_VERSION} ")
    message(FATAL_ERROR
        "Film-Juicer requires Cargo ${JUICER_RUST_VERSION}: "
        "${JUICER_CARGO_VERSION_OUTPUT}${JUICER_CARGO_VERSION_ERROR}")
endif()

execute_process(
    COMMAND "${JUICER_RUSTC_EXECUTABLE}" --version --verbose
    WORKING_DIRECTORY "${JUICER_RUST_SOURCE_DIR}"
    RESULT_VARIABLE JUICER_RUSTC_VERSION_RESULT
    OUTPUT_VARIABLE JUICER_RUSTC_VERSION_OUTPUT
    ERROR_VARIABLE JUICER_RUSTC_VERSION_ERROR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT JUICER_RUSTC_VERSION_RESULT EQUAL 0 OR
   NOT JUICER_RUSTC_VERSION_OUTPUT MATCHES "^rustc ${JUICER_RUST_VERSION} " OR
   NOT JUICER_RUSTC_VERSION_OUTPUT MATCHES "host: ${JUICER_RUST_TARGET}($|[\r\n])")
    message(FATAL_ERROR
        "Film-Juicer requires rustc ${JUICER_RUST_VERSION} for ${JUICER_RUST_TARGET}: "
        "${JUICER_RUSTC_VERSION_OUTPUT}${JUICER_RUSTC_VERSION_ERROR}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(JUICER_RUST_CARGO_PROFILE "release")
    set(JUICER_RUST_PROFILE_ARGUMENT --release)
else()
    set(JUICER_RUST_CARGO_PROFILE "debug")
    set(JUICER_RUST_PROFILE_ARGUMENT)
endif()
if(JUICER_CUDA_FREE_TESTS)
    set(JUICER_RUST_TARGET_DIR "${JUICER_PRESET_BUILD_DIR}/cargo")
else()
    set(JUICER_RUST_TARGET_DIR "${CMAKE_BINARY_DIR}/cargo")
endif()
set(JUICER_RUST_PRODUCTION_TARGET_DIR "${JUICER_RUST_TARGET_DIR}/production")
set(JUICER_RUST_TEST_SUPPORT_TARGET_DIR "${JUICER_RUST_TARGET_DIR}/test-support")

if(BUILD_TESTING AND JUICER_NEXTEST_EXECUTABLE)
    set(JUICER_RUST_TESTS_TARGET_DIR "${JUICER_RUST_TEST_SUPPORT_TARGET_DIR}")
    set(JUICER_RUST_TESTS_METADATA_DIR "${JUICER_RUST_TESTS_TARGET_DIR}/metadata")
    set(JUICER_RUST_TESTS_BINARIES_METADATA
        "${JUICER_RUST_TESTS_METADATA_DIR}/binaries.json")
    set(JUICER_RUST_TESTS_CARGO_METADATA
        "${JUICER_RUST_TESTS_METADATA_DIR}/cargo.json")
    set(JUICER_RUST_TESTS_PROFILE "${JUICER_RUST_CARGO_PROFILE}")
    add_custom_target(juicer_rust_tests
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${JUICER_RUST_TESTS_METADATA_DIR}"
        COMMAND "${CMAKE_COMMAND}"
            "-DJUICER_RUST_TESTS_METADATA_MODE=ON"
            "-DJUICER_RUST_TESTS_CARGO=${JUICER_CARGO_EXECUTABLE}"
            "-DJUICER_RUST_TESTS_NEXTEST=${JUICER_NEXTEST_EXECUTABLE}"
            "-DJUICER_RUST_TESTS_ROOT=${JUICER_RUST_SOURCE_DIR}"
            "-DJUICER_RUST_TESTS_TARGET=${JUICER_RUST_TARGET}"
            "-DJUICER_RUST_TESTS_TARGET_DIR=${JUICER_RUST_TESTS_TARGET_DIR}"
            "-DJUICER_RUST_TESTS_BINARIES_METADATA=${JUICER_RUST_TESTS_BINARIES_METADATA}"
            "-DJUICER_RUST_TESTS_CARGO_METADATA=${JUICER_RUST_TESTS_CARGO_METADATA}"
            "-DJUICER_RUST_TESTS_PROFILE=${JUICER_RUST_TESTS_PROFILE}"
            -P "${JUICER_RUST_SOURCE_DIR}/cmake/JuicerRust.cmake"
        WORKING_DIRECTORY "${JUICER_RUST_SOURCE_DIR}"
        COMMENT "Preparing Rust core and bridge test binaries (${JUICER_RUST_TESTS_PROFILE})"
        VERBATIM
        USES_TERMINAL)
endif()

file(GLOB_RECURSE JUICER_RUST_SOURCES CONFIGURE_DEPENDS
    LIST_DIRECTORIES FALSE "${JUICER_RUST_SOURCE_DIR}/rust/*.rs")
set(JUICER_RUST_BUILD_INPUTS
    "${JUICER_RUST_SOURCE_DIR}/Cargo.toml"
    "${JUICER_RUST_SOURCE_DIR}/Cargo.lock"
    "${JUICER_RUST_SOURCE_DIR}/clippy.toml"
    "${JUICER_RUST_SOURCE_DIR}/rust-toolchain.toml"
    "${JUICER_RUST_SOURCE_DIR}/rustfmt.toml"
    "${JUICER_RUST_SOURCE_DIR}/rust/film-juicer-core/Cargo.toml"
    "${JUICER_RUST_SOURCE_DIR}/rust/film-juicer-plugin/Cargo.toml"
    ${JUICER_RUST_SOURCES})

function(juicer_add_rust_artifact variant rust_target)
    set(directory "${JUICER_RUST_TARGET_DIR}/${variant}")
    set(features none)
    set(feature_argument)
    if(variant STREQUAL "test-support")
        set(features test-support)
        set(feature_argument --features test-support)
    elseif(NOT variant STREQUAL "production")
        message(FATAL_ERROR "Unsupported Rust artifact: ${variant}")
    endif()
    string(REPLACE "-" "_" build_variant "${variant}")
    set(build_target "juicer_rust_${build_variant}_build")
    set(archive "${directory}/${JUICER_RUST_TARGET}/${JUICER_RUST_CARGO_PROFILE}/${JUICER_RUST_ARCHIVE_NAME}")
    set(stamp "${directory}/build.stamp")
    set(feature_input "${directory}/features.txt")
    file(CONFIGURE OUTPUT "${directory}/CACHEDIR.TAG"
        CONTENT "Signature: 8a477f597d28d172789f06886806bc55\n" @ONLY)
    file(CONFIGURE OUTPUT "${feature_input}"
        CONTENT "features=@features@\ntarget=@JUICER_RUST_TARGET@\nprofile=@JUICER_RUST_CARGO_PROFILE@\ncargo=@JUICER_CARGO_EXECUTABLE@\n" @ONLY)
    add_custom_command(
        OUTPUT "${stamp}"
        BYPRODUCTS "${archive}"
        COMMAND "${CMAKE_COMMAND}" -E env
            "RUSTFLAGS=" "CARGO_BUILD_RUSTFLAGS=" "CARGO_ENCODED_RUSTFLAGS="
            "${JUICER_CARGO_EXECUTABLE}" build
                --locked --package film-juicer-plugin --no-default-features
                ${feature_argument}
                --target "${JUICER_RUST_TARGET}" --target-dir "${directory}"
                ${JUICER_RUST_PROFILE_ARGUMENT}
        # A fresh Cargo result can keep the archive unchanged. Only successful
        # preparation advances this variant's stamp; consumers track the archive.
        COMMAND "${CMAKE_COMMAND}" -E touch "${stamp}"
        WORKING_DIRECTORY "${JUICER_RUST_SOURCE_DIR}"
        DEPENDS ${JUICER_RUST_BUILD_INPUTS} "${feature_input}"
        COMMENT "Building Film-Juicer Rust ${variant} (${JUICER_RUST_TARGET}, ${JUICER_RUST_CARGO_PROFILE})"
        VERBATIM USES_TERMINAL)
    add_custom_target(${build_target} DEPENDS "${stamp}")
    add_library(${rust_target} STATIC IMPORTED GLOBAL)
    set_target_properties(${rust_target} PROPERTIES
        IMPORTED_LOCATION "${archive}"
        JUICER_RUST_VARIANT "${variant}"
        INTERFACE_JUICER_RUST_VARIANT "${variant}"
        JUICER_RUST_FEATURE_INPUT "${feature_input}"
        JUICER_RUST_BUILD_STAMP "${stamp}"
        JUICER_RUST_TARGET_DIR "${directory}")
    set_property(TARGET ${rust_target} APPEND PROPERTY COMPATIBLE_INTERFACE_STRING JUICER_RUST_VARIANT)
    add_dependencies(${rust_target} ${build_target})
endfunction()

juicer_add_rust_artifact(production film_juicer_rust)
if(BUILD_TESTING)
    juicer_add_rust_artifact(test-support film_juicer_rust_test_support)
endif()

set(JUICER_RUST_PROBE_SOURCE "${CMAKE_BINARY_DIR}/cargo-native-link-probe.rs")
set(JUICER_RUST_PROBE_TEXT "pub extern \"C\" fn cargo_native_link_probe() {}\n")
file(WRITE "${JUICER_RUST_PROBE_SOURCE}" "${JUICER_RUST_PROBE_TEXT}")
string(SHA256 JUICER_RUST_NATIVE_CURRENT_KEY
    "${JUICER_RUSTC_EXECUTABLE}|${JUICER_RUSTC_VERSION_OUTPUT}|${JUICER_RUST_TARGET}|${JUICER_RUST_PROBE_TEXT}")
if(JUICER_RUST_NATIVE_CACHE_KEY STREQUAL JUICER_RUST_NATIVE_CURRENT_KEY AND
   NOT "${JUICER_RUST_NATIVE_CACHE_LINE}" STREQUAL "")
    set(JUICER_RUST_NATIVE_LINE "${JUICER_RUST_NATIVE_CACHE_LINE}")
else()
    if(DEFINED ENV{TEMP} AND NOT "$ENV{TEMP}" STREQUAL "")
        set(JUICER_RUST_PROBE_DIR "$ENV{TEMP}")
    else()
        set(JUICER_RUST_PROBE_DIR "/tmp")
    endif()
    string(SHA256 JUICER_RUST_PROBE_ID "${CMAKE_BINARY_DIR}")
    string(SUBSTRING "${JUICER_RUST_PROBE_ID}" 0 12 JUICER_RUST_PROBE_ID)
    set(JUICER_RUST_NATIVE_PROBE
        "${JUICER_RUST_PROBE_DIR}/juicer-rust-native-libs-${JUICER_RUST_PROBE_ID}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    execute_process(
        COMMAND "${JUICER_RUSTC_EXECUTABLE}"
            --crate-name film_juicer_link_probe
            --edition=2024
            --target "${JUICER_RUST_TARGET}"
            --crate-type staticlib
            --print native-static-libs
            "${JUICER_RUST_PROBE_SOURCE}"
            -o "${JUICER_RUST_NATIVE_PROBE}"
        WORKING_DIRECTORY "${JUICER_RUST_SOURCE_DIR}"
        RESULT_VARIABLE JUICER_RUST_NATIVE_RESULT
        OUTPUT_VARIABLE JUICER_RUST_NATIVE_OUTPUT
        ERROR_VARIABLE JUICER_RUST_NATIVE_ERROR)
    file(REMOVE "${JUICER_RUST_NATIVE_PROBE}")
    set(JUICER_RUST_NATIVE_REPORT
        "${JUICER_RUST_NATIVE_OUTPUT}\n${JUICER_RUST_NATIVE_ERROR}")
    if(NOT JUICER_RUST_NATIVE_RESULT EQUAL 0 OR
       NOT JUICER_RUST_NATIVE_REPORT MATCHES "native-static-libs:[ \t]*([^\r\n]+)")
        message(FATAL_ERROR
            "rustc did not report native static-library requirements: ${JUICER_RUST_NATIVE_REPORT}")
    endif()
    set(JUICER_RUST_NATIVE_LINE "${CMAKE_MATCH_1}")
    set(JUICER_RUST_NATIVE_CACHE_KEY "${JUICER_RUST_NATIVE_CURRENT_KEY}"
        CACHE INTERNAL "Qualified toolchain-only rustc native-link requirement key" FORCE)
    set(JUICER_RUST_NATIVE_CACHE_LINE "${JUICER_RUST_NATIVE_LINE}"
        CACHE INTERNAL "Qualified toolchain-only rustc native static-library requirements" FORCE)
endif()
separate_arguments(JUICER_RUST_NATIVE_TOKENS NATIVE_COMMAND "${JUICER_RUST_NATIVE_LINE}")
set(JUICER_RUST_NATIVE_LIBRARIES)
set(JUICER_RUST_NATIVE_LINK_OPTIONS)
foreach(token IN LISTS JUICER_RUST_NATIVE_TOKENS)
    if(token MATCHES "^-l(.+)$")
        list(APPEND JUICER_RUST_NATIVE_LIBRARIES "${CMAKE_MATCH_1}")
    elseif(token MATCHES "\\.lib$")
        list(APPEND JUICER_RUST_NATIVE_LIBRARIES "${token}")
    elseif(token MATCHES "^/defaultlib:")
        list(APPEND JUICER_RUST_NATIVE_LINK_OPTIONS "${token}")
    else()
        message(FATAL_ERROR "Unsupported rustc native-static-libs token: ${token}")
    endif()
endforeach()
if(NOT JUICER_RUST_NATIVE_LIBRARIES)
    message(FATAL_ERROR "rustc reported no native static-library requirements")
endif()
foreach(rust_target IN ITEMS film_juicer_rust film_juicer_rust_test_support)
    if(TARGET ${rust_target})
        target_link_libraries(${rust_target} INTERFACE ${JUICER_RUST_NATIVE_LIBRARIES})
        if(JUICER_RUST_NATIVE_LINK_OPTIONS)
            target_link_options(${rust_target} INTERFACE ${JUICER_RUST_NATIVE_LINK_OPTIONS})
        endif()
    endif()
endforeach()

# This is final-link metadata, not a second artifact selector. Final consumers
# must name their archive directly; neutral native cores carry neither variant.
function(juicer_write_rust_artifacts)
    foreach(core IN ITEMS juicer_core juicer_test_core)
        if(TARGET ${core})
            get_target_property(interface ${core} INTERFACE_LINK_LIBRARIES)
            foreach(rust_target IN ITEMS film_juicer_rust film_juicer_rust_test_support)
                if(rust_target IN_LIST interface)
                    message(FATAL_ERROR "${core} must remain Rust-neutral")
                endif()
            endforeach()
        endif()
    endforeach()
    set(json "{\n\"schema\":\"rust-artifacts-v1\",\n\"build\":\"${CMAKE_BINARY_DIR}\",\n\"variants\":{")
    set(separator "")
    foreach(rust_target IN ITEMS film_juicer_rust film_juicer_rust_test_support)
        if(NOT TARGET ${rust_target})
            continue()
        endif()
        get_target_property(variant ${rust_target} JUICER_RUST_VARIANT)
        set(features "")
        if(variant STREQUAL "test-support")
            set(features "\"test-support\"")
        endif()
        string(APPEND json "${separator}\"${variant}\":{\"archive\":\"$<TARGET_FILE:${rust_target}>\",\"features\":[${features}],"
            "\"feature_input\":\"$<TARGET_PROPERTY:${rust_target},JUICER_RUST_FEATURE_INPUT>\","
            "\"stamp\":\"$<TARGET_PROPERTY:${rust_target},JUICER_RUST_BUILD_STAMP>\","
            "\"target_dir\":\"$<TARGET_PROPERTY:${rust_target},JUICER_RUST_TARGET_DIR>\"}")
        set(separator ",")
    endforeach()
    string(APPEND json "},\n\"native_archive\":\"$<TARGET_FILE:juicer_core>\",\n\"consumers\":{")
    set(separator "")
    foreach(target IN LISTS ARGN)
        get_target_property(libraries ${target} LINK_LIBRARIES)
        set(selected)
        foreach(rust_target IN ITEMS film_juicer_rust film_juicer_rust_test_support)
            if(rust_target IN_LIST libraries)
                list(APPEND selected ${rust_target})
            endif()
        endforeach()
        list(LENGTH selected count)
        if(count EQUAL 1)
            list(GET selected 0 rust_target)
            get_target_property(variant ${rust_target} JUICER_RUST_VARIANT)
        elseif(count EQUAL 0)
            get_target_property(variant ${target} JUICER_RUST_VARIANT)
            if(NOT variant STREQUAL "none")
                message(FATAL_ERROR "${target} must explicitly select one Rust artifact or declare none")
            endif()
        else()
            message(FATAL_ERROR "${target} mixes production and test-support Rust artifacts")
        endif()
        string(APPEND json "${separator}\"${target}\":{\"binary\":\"$<TARGET_FILE:${target}>\",\"variant\":\"${variant}\"}")
        set(separator ",")
    endforeach()
    string(APPEND json "}\n}\n")
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/tests/rust-artifacts.json" CONTENT "${json}")
endfunction()

message(STATUS "Rust ${JUICER_RUST_VERSION}: target=${JUICER_RUST_TARGET}, profile=${JUICER_RUST_CARGO_PROFILE}, separate production/test-support artifacts")
message(STATUS "Rust toolchain-only native static libraries: ${JUICER_RUST_NATIVE_LINE}")
