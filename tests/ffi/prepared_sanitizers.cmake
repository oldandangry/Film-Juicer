option(JUICER_PREPARED_FFI_SANITIZERS "Build the bounded Debug prepared-host ASan/UBSan pilot" OFF)
if(NOT JUICER_PREPARED_FFI_SANITIZERS)
    return()
endif()
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(FATAL_ERROR "The prepared-host sanitizer pilot requires Debug")
endif()

# Replace these two archive members only in the pilot executable. Other core,
# Rust, GoogleTest and CUDA device objects remain uninstrumented.
add_library(juicer_prepared_sanitizer_objects OBJECT
    "${PROJECT_SOURCE_DIR}/native/juicer_cuda_prepared.cpp"
    "${PROJECT_SOURCE_DIR}/native/juicer_cuda_descriptors.cpp")
target_link_libraries(juicer_prepared_sanitizer_objects PRIVATE juicer_settings)
add_executable(JuicerPreparedSanitizerTests ${JUICER_PREPARED_TEST_SOURCES}
    $<TARGET_OBJECTS:juicer_prepared_sanitizer_objects>)
target_link_libraries(JuicerPreparedSanitizerTests PRIVATE juicer_core GTest::gtest_main)
add_executable(JuicerPreparedSanitizerControl ffi/fixtures/sanitizer/control.cpp
    $<TARGET_OBJECTS:juicer_prepared_sanitizer_objects>)
target_link_libraries(JuicerPreparedSanitizerControl PRIVATE juicer_core)
foreach(target IN ITEMS juicer_prepared_sanitizer_objects JuicerPreparedSanitizerTests JuicerPreparedSanitizerControl)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-sanitize-recover=undefined)
    if(WIN32)
        # Match the uninstrumented archive's STL definitions without weakening
        # access instrumentation or bypassing the linker's ODR checks.
        target_compile_definitions(${target} PRIVATE
            _DISABLE_VECTOR_ANNOTATION _DISABLE_STRING_ANNOTATION _DISABLE_OPTIONAL_ANNOTATION)
    else()
        target_compile_options(${target} PRIVATE -fno-omit-frame-pointer)
    endif()
endforeach()
foreach(target IN ITEMS JuicerPreparedSanitizerTests JuicerPreparedSanitizerControl)
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX
        RUNTIME_OUTPUT_DIRECTORY "${JUICER_TEST_BINARY_DIR}")
    if(UNIX)
        target_link_options(${target} PRIVATE "$<HOST_LINK:-fsanitize=address>" "$<HOST_LINK:-fsanitize=undefined>")
        set_property(TARGET ${target} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()
endforeach()

add_test(NAME Ffi.Host.PreparedSanitizers COMMAND JuicerPreparedSanitizerTests)
add_test(NAME Ffi.Host.SanitizerCleanControl COMMAND JuicerPreparedSanitizerControl clean)
foreach(mode IN ITEMS memory undefined)
    add_test(NAME Ffi.Host.SanitizerControl.${mode}
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/ffi/run_prepared_sanitizer_control.py"
            "$<TARGET_FILE:JuicerPreparedSanitizerControl>" ${mode}
            "${JUICER_TEST_ARTIFACT_ROOT}/prepared-sanitizers")
endforeach()
set(sanitizer_tests Ffi.Host.PreparedSanitizers Ffi.Host.SanitizerCleanControl
    Ffi.Host.SanitizerControl.memory Ffi.Host.SanitizerControl.undefined)
set_tests_properties(${sanitizer_tests} PROPERTIES LABELS "host;sanitizer" TIMEOUT 60
    ENVIRONMENT "ASAN_OPTIONS=detect_leaks=0:halt_on_error=1;UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1;PYTHONDONTWRITEBYTECODE=1")
if(WIN32)
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-resource-dir
        OUTPUT_VARIABLE sanitizer_resource_dir OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY TIMEOUT 10)
    set(sanitizer_runtime_dir "${sanitizer_resource_dir}/lib/windows")
    if(NOT EXISTS "${sanitizer_runtime_dir}/clang_rt.asan_dynamic-x86_64.dll")
        message(FATAL_ERROR "The prepared-host pilot requires the compiler's x64 ASan runtime")
    endif()
    # CMake invokes lld-link directly. Match clang-cl's ASan/UBSan driver link;
    # the dynamic ASan runtime also supplies the UBSan handlers on Windows.
    foreach(target IN ITEMS JuicerPreparedSanitizerTests JuicerPreparedSanitizerControl)
        target_link_libraries(${target} PRIVATE "${sanitizer_runtime_dir}/clang_rt.asan_dynamic-x86_64.lib")
        target_link_options(${target} PRIVATE /INCREMENTAL:NO /INCLUDE:__asan_seh_interceptor
            "/WHOLEARCHIVE:${sanitizer_runtime_dir}/clang_rt.asan_dynamic_runtime_thunk-x86_64.lib")
    endforeach()
    get_filename_component(sanitizer_compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    set_tests_properties(${sanitizer_tests} PROPERTIES ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:${sanitizer_runtime_dir};PATH=path_list_prepend:${sanitizer_compiler_dir}")
endif()
