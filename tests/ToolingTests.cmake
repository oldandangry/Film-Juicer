# Shared tooling registrations for the product test suite and the lightweight
# native tooling entry point. Keep checkout paths and artifact routing explicit
# because PROJECT_SOURCE_DIR differs between those two callers.
function(juicer_register_tooling_tests)
    cmake_parse_arguments(ARG "" "ROOT;ARTIFACT_ROOT;PYTHON;CMAKE_COMMAND;CTEST_COMMAND;CARGO;NINJA;CC;BUILD_TYPE;HOST_WINDOWS;TESTS_OUT" "" ${ARGN})
    foreach(required IN ITEMS ROOT ARTIFACT_ROOT PYTHON CMAKE_COMMAND CTEST_COMMAND CARGO NINJA CC BUILD_TYPE HOST_WINDOWS TESTS_OUT)
        if(NOT DEFINED ARG_${required} OR "${ARG_${required}}" STREQUAL "")
            message(FATAL_ERROR "juicer_register_tooling_tests requires ${required}")
        endif()
    endforeach()

    set(test_source "${ARG_ROOT}/tests")
    set(tooling_tests)
    add_test(NAME Quality.Dispatcher
        COMMAND "${ARG_PYTHON}" -m unittest discover
            -s "${test_source}/quality"
            -p test_check_quality.py
            -v)
    set_tests_properties(Quality.Dispatcher PROPERTIES
        LABELS "host"
        TIMEOUT 180
        ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    list(APPEND tooling_tests Quality.Dispatcher)

    add_test(NAME Quality.ToolingLint
        COMMAND "${ARG_PYTHON}" -m unittest discover
            -s "${test_source}/quality" -p test_tooling_lint.py -v)
    set_tests_properties(Quality.ToolingLint PROPERTIES
        LABELS "host" TIMEOUT 60 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    list(APPEND tooling_tests Quality.ToolingLint)

    # These native verification controls are currently required only by the
    # Debug tooling lane. Their test selection stays identical in both entries.
    if(ARG_BUILD_TYPE STREQUAL "Debug")
        foreach(verification_group IN ITEMS Execution BuildEvidence BuildRegistration Verify Checkpoint PreparationRenewal)
            if(verification_group STREQUAL "BuildEvidence")
                set(verification_pattern "test_build_evidence.py")
            elseif(verification_group STREQUAL "BuildRegistration")
                set(verification_pattern "test_build_registration.py")
            elseif(verification_group STREQUAL "Checkpoint")
                set(verification_pattern "test_checkpoint.py")
            elseif(verification_group STREQUAL "PreparationRenewal")
                set(verification_pattern "test_preparation_renewal.py")
            elseif(verification_group STREQUAL "Execution")
                set(verification_pattern "test_execution.py")
            else()
                set(verification_pattern "test_verify.py")
            endif()
            add_test(NAME "Quality.${verification_group}"
                COMMAND "${ARG_PYTHON}" -B -m unittest discover
                    -s "${test_source}/quality" -p "${verification_pattern}" -f -v --durations 10)
            if(verification_group STREQUAL "BuildEvidence")
                set(verification_timeout 1800)
            elseif(verification_group STREQUAL "PreparationRenewal" AND ARG_HOST_WINDOWS)
                set(verification_timeout 300)
            else()
                set(verification_timeout 600)
            endif()
            set_tests_properties("Quality.${verification_group}" PROPERTIES LABELS "host" TIMEOUT ${verification_timeout}
                ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1;JUICER_TEST_ARTIFACT_DIR=${ARG_ARTIFACT_ROOT}/verification-controls")
            if(ARG_HOST_WINDOWS AND (verification_group STREQUAL "Checkpoint" OR verification_group STREQUAL "PreparationRenewal"))
                # Full tool-byte validation is I/O heavy; avoid overlap at the four-worker preset limit.
                set_tests_properties("Quality.${verification_group}" PROPERTIES PROCESSORS 4)
            endif()
            list(APPEND tooling_tests "Quality.${verification_group}")
        endforeach()
    endif()

    add_test(NAME Ctest.Host.Report
        COMMAND "${ARG_PYTHON}" -m unittest discover
            -s "${test_source}/ctest" -p test_report.py -v)
    set_tests_properties(Ctest.Host.Report PROPERTIES
        LABELS "host" TIMEOUT 30 ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
    list(APPEND tooling_tests Ctest.Host.Report)

    add_test(NAME Ctest.Host.Repetition
        COMMAND "${ARG_PYTHON}" -B -m unittest discover
            -s "${test_source}/ctest" -p test_repeat.py -v)
    set_tests_properties(Ctest.Host.Repetition PROPERTIES LABELS "host" TIMEOUT 60
        ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1;JUICER_CTEST_EXECUTABLE=${ARG_CTEST_COMMAND};JUICER_TEST_ARTIFACT_DIR=${ARG_ARTIFACT_ROOT}/ctest-repetition-controls")
    list(APPEND tooling_tests Ctest.Host.Repetition)

    add_test(NAME Ffi.Host.RustBuild
        COMMAND "${ARG_PYTHON}" "${test_source}/ffi/run_rust_build_contract.py"
            --cargo "${ARG_CARGO}" --cmake "${ARG_CMAKE_COMMAND}"
            --ninja "${ARG_NINJA}" --cc "${ARG_CC}"
            --artifacts "${ARG_ARTIFACT_ROOT}/ffi/rust-build")
    set_tests_properties(Ffi.Host.RustBuild PROPERTIES LABELS "host" TIMEOUT 240)
    list(APPEND tooling_tests Ffi.Host.RustBuild)

    set(${ARG_TESTS_OUT} "${tooling_tests}" PARENT_SCOPE)
endfunction()
