set(JUICER_RUST_VERSION "1.98.1")
set(JUICER_CARGO_EXECUTABLE "" CACHE FILEPATH "Cargo executable for the pinned Film-Juicer toolchain")

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
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
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
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
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
set(JUICER_RUST_TARGET_DIR "${CMAKE_BINARY_DIR}/cargo")
set(JUICER_RUST_ARCHIVE
    "${JUICER_RUST_TARGET_DIR}/${JUICER_RUST_TARGET}/${JUICER_RUST_CARGO_PROFILE}/${JUICER_RUST_ARCHIVE_NAME}")

file(GLOB_RECURSE JUICER_RUST_SOURCES CONFIGURE_DEPENDS
    LIST_DIRECTORIES FALSE
    "${PROJECT_SOURCE_DIR}/rust/*.rs")
set(JUICER_RUST_BUILD_INPUTS
    "${PROJECT_SOURCE_DIR}/Cargo.toml"
    "${PROJECT_SOURCE_DIR}/Cargo.lock"
    "${PROJECT_SOURCE_DIR}/clippy.toml"
    "${PROJECT_SOURCE_DIR}/rust-toolchain.toml"
    "${PROJECT_SOURCE_DIR}/rustfmt.toml"
    "${PROJECT_SOURCE_DIR}/rust/film-juicer-core/Cargo.toml"
    "${PROJECT_SOURCE_DIR}/rust/film-juicer-plugin/Cargo.toml"
    ${JUICER_RUST_SOURCES})

add_custom_command(
    OUTPUT "${JUICER_RUST_ARCHIVE}"
    COMMAND "${CMAKE_COMMAND}" -E env
        "RUSTFLAGS="
        "CARGO_BUILD_RUSTFLAGS="
        "CARGO_ENCODED_RUSTFLAGS="
        "${JUICER_CARGO_EXECUTABLE}" build
            --locked
            --package film-juicer-plugin
            --target "${JUICER_RUST_TARGET}"
            --target-dir "${JUICER_RUST_TARGET_DIR}"
            ${JUICER_RUST_PROFILE_ARGUMENT}
    COMMAND "${CMAKE_COMMAND}" -E touch "${JUICER_RUST_ARCHIVE}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    DEPENDS ${JUICER_RUST_BUILD_INPUTS}
    COMMENT
        "Building Film-Juicer Rust static library (${JUICER_RUST_TARGET}, ${JUICER_RUST_CARGO_PROFILE})"
    VERBATIM
    USES_TERMINAL)
add_custom_target(juicer_rust_build DEPENDS "${JUICER_RUST_ARCHIVE}")

add_library(film_juicer_rust STATIC IMPORTED GLOBAL)
set_target_properties(film_juicer_rust PROPERTIES
    IMPORTED_LOCATION "${JUICER_RUST_ARCHIVE}")
add_dependencies(film_juicer_rust juicer_rust_build)

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
        WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
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
target_link_libraries(film_juicer_rust INTERFACE ${JUICER_RUST_NATIVE_LIBRARIES})
if(JUICER_RUST_NATIVE_LINK_OPTIONS)
    target_link_options(film_juicer_rust INTERFACE ${JUICER_RUST_NATIVE_LINK_OPTIONS})
endif()

message(STATUS
    "Rust ${JUICER_RUST_VERSION}: target=${JUICER_RUST_TARGET}, "
    "profile=${JUICER_RUST_CARGO_PROFILE}, archive=${JUICER_RUST_ARCHIVE}")
message(STATUS "Rust toolchain-only native static libraries: ${JUICER_RUST_NATIVE_LINE}")
