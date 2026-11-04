# Shared tooling registrations for the product test suite and the lightweight
# native tooling entry point. Keep checkout paths and artifact routing explicit
# because PROJECT_SOURCE_DIR differs between those two callers.
function(juicer_add_python_test)
    cmake_parse_arguments(ARG "" "NAME" "COMMAND;TARGETS" ${ARGN})
    if(NOT ARG_NAME OR NOT ARG_COMMAND)
        message(FATAL_ERROR "juicer_add_python_test requires NAME and COMMAND")
    endif()
    if(JUICER_PYTHON_ROOT)
        set(root "${JUICER_PYTHON_ROOT}")
        set(preset "${JUICER_PYTHON_PRESET}")
    else()
        set(root "${PROJECT_SOURCE_DIR}")
        set(preset "${JUICER_TEST_PRESET_NAME}")
    endif()
    list(FIND ARG_COMMAND "${Python3_EXECUTABLE}" interpreter)
    if(interpreter LESS 0)
        message(FATAL_ERROR "Python test needs the configured interpreter: ${ARG_NAME}")
    endif()
    list(SUBLIST ARG_COMMAND 0 ${interpreter} prefix)
    list(SUBLIST ARG_COMMAND ${interpreter} -1 command)
    add_test(NAME "${ARG_NAME}" COMMAND ${prefix} "${Python3_EXECUTABLE}" -B
        "${root}/scripts/verify.py" python-tests --preset "${preset}"
        --build-dir "${CMAKE_BINARY_DIR}" -- ${command})
    set_property(GLOBAL APPEND PROPERTY JUICER_PYTHON_TEST_NAMES "${ARG_NAME}")
    set_property(GLOBAL PROPERTY "JUICER_PYTHON_CLOSURE_${ARG_NAME}" "${ARG_TARGETS}")
endfunction()

function(juicer_write_python_targets)
    get_property(names GLOBAL PROPERTY JUICER_PYTHON_TEST_NAMES)
    set(json "{\n")
    set(separator "")
    foreach(name IN LISTS names)
        get_property(targets GLOBAL PROPERTY "JUICER_PYTHON_CLOSURE_${name}")
        set(values "")
        set(comma "")
        foreach(target IN LISTS targets)
            if(NOT "${target}" STREQUAL "")
                string(APPEND values "${comma}\"${target}\"")
                set(comma ",")
            endif()
        endforeach()
        string(APPEND json "${separator}\"${name}\":[${values}]")
        set(separator ",\n")
    endforeach()
    string(APPEND json "\n}\n")
    file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/tests/python-targets.json" CONTENT "${json}")
endfunction()

function(juicer_register_tooling_tests)
    cmake_parse_arguments(ARG "" "ROOT;ARTIFACT_ROOT;PYTHON;CMAKE_COMMAND;CTEST_COMMAND;CARGO;NINJA;CC;BUILD_TYPE;HOST_WINDOWS;TESTS_OUT" "" ${ARGN})
    foreach(required IN ITEMS ROOT ARTIFACT_ROOT PYTHON CMAKE_COMMAND CTEST_COMMAND CARGO NINJA CC BUILD_TYPE HOST_WINDOWS TESTS_OUT)
        if(NOT DEFINED ARG_${required} OR "${ARG_${required}}" STREQUAL "")
            message(FATAL_ERROR "juicer_register_tooling_tests requires ${required}")
        endif()
    endforeach()

    set(test_source "${ARG_ROOT}/tests")
    set(JUICER_PYTHON_ROOT "${ARG_ROOT}")
    if(ARG_HOST_WINDOWS)
        set(platform windows-clang)
    else()
        set(platform linux)
    endif()
    string(TOLOWER "${ARG_BUILD_TYPE}" configuration)
    set(JUICER_PYTHON_PRESET "${platform}-${configuration}")
    set(tooling_tests)
    foreach(group IN ITEMS NativeResults HostResults GpuResults)
        if(group STREQUAL "NativeResults")
            set(pattern test_native_results.py)
        elseif(group STREQUAL "HostResults")
            set(pattern test_host_results.py)
        else()
            set(pattern test_gpu_results.py)
        endif()
        juicer_add_python_test(NAME "Quality.${group}" COMMAND "${ARG_PYTHON}" -B -m unittest discover
            -s "${test_source}/quality" -p "${pattern}" -v)
        set_tests_properties("Quality.${group}" PROPERTIES LABELS "host" TIMEOUT 120)
        list(APPEND tooling_tests "Quality.${group}")
    endforeach()
    juicer_add_python_test(NAME Quality.Dispatcher
        COMMAND "${ARG_PYTHON}" -m unittest discover
            -s "${test_source}/quality"
            -p test_check_quality.py
            -v)
    set_tests_properties(Quality.Dispatcher PROPERTIES
        LABELS "host"
        TIMEOUT 600
        ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    list(APPEND tooling_tests Quality.Dispatcher)

    juicer_add_python_test(NAME Quality.ToolingLint
        COMMAND "${ARG_PYTHON}" -m unittest discover
            -s "${test_source}/quality" -p test_tooling_lint.py -v)
    set_tests_properties(Quality.ToolingLint PROPERTIES
        LABELS "host" TIMEOUT 600 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    list(APPEND tooling_tests Quality.ToolingLint)

    # These native verification controls are currently required only by the
    # Debug tooling lane. Their test selection stays identical in both entries.
    if(ARG_BUILD_TYPE STREQUAL "Debug")
        foreach(verification_group IN ITEMS Execution BuildEvidence BuildRegistration Verify Checkpoint)
            if(verification_group STREQUAL "BuildEvidence")
                set(verification_pattern "test_build_evidence.py")
            elseif(verification_group STREQUAL "BuildRegistration")
                set(verification_pattern "test_build_registration.py")
            elseif(verification_group STREQUAL "Checkpoint")
                set(verification_pattern "test_checkpoint.py")
            elseif(verification_group STREQUAL "Execution")
                set(verification_pattern "test_execution.py")
            else()
                set(verification_pattern "test_verify.py")
            endif()
            juicer_add_python_test(NAME "Quality.${verification_group}"
                COMMAND "${ARG_PYTHON}" -B -m unittest discover
                    -s "${test_source}/quality" -p "${verification_pattern}" -v --durations 10)
            set_tests_properties("Quality.${verification_group}" PROPERTIES LABELS "host" TIMEOUT 600
                ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1;JUICER_TEST_ARTIFACT_DIR=${ARG_ARTIFACT_ROOT}/verification-controls")
            if(ARG_HOST_WINDOWS AND verification_group STREQUAL "Checkpoint")
                set_tests_properties("Quality.${verification_group}" PROPERTIES PROCESSORS 4)
            endif()
            list(APPEND tooling_tests "Quality.${verification_group}")
        endforeach()
    endif()

    juicer_add_python_test(NAME Ctest.Host.Report
        COMMAND "${ARG_PYTHON}" -m unittest discover
            -s "${test_source}/ctest" -p test_report.py -v)
    set_tests_properties(Ctest.Host.Report PROPERTIES
        LABELS "host" TIMEOUT 600 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    list(APPEND tooling_tests Ctest.Host.Report)

    juicer_add_python_test(NAME Ctest.Host.Repetition
        COMMAND "${ARG_PYTHON}" -B -m unittest discover
            -s "${test_source}/ctest" -p test_repeat.py -v)
    set_tests_properties(Ctest.Host.Repetition PROPERTIES LABELS "host" TIMEOUT 600
        ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1;JUICER_CTEST_EXECUTABLE=${ARG_CTEST_COMMAND};JUICER_TEST_ARTIFACT_DIR=${ARG_ARTIFACT_ROOT}/ctest-repetition-controls")
    list(APPEND tooling_tests Ctest.Host.Repetition)

    juicer_add_python_test(NAME Ffi.Host.RustBuild
        COMMAND "${ARG_PYTHON}" "${test_source}/ffi/run_rust_build_contract.py"
            --cargo "${ARG_CARGO}" --cmake "${ARG_CMAKE_COMMAND}"
            --ninja "${ARG_NINJA}" --cc "${ARG_CC}"
            --artifacts "${ARG_ARTIFACT_ROOT}/ffi/rust-build")
    set_tests_properties(Ffi.Host.RustBuild PROPERTIES LABELS "host" TIMEOUT 600)
    list(APPEND tooling_tests Ffi.Host.RustBuild)

    set(${ARG_TESTS_OUT} "${tooling_tests}" PARENT_SCOPE)
endfunction()
