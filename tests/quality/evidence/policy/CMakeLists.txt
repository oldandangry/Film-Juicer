cmake_minimum_required(VERSION 3.24)

project(BoundedPolicyEvidenceFixture LANGUAGES C)

if(NOT DEFINED Python3_EXECUTABLE OR NOT EXISTS "${Python3_EXECUTABLE}")
    message(FATAL_ERROR "The policy miniature requires its explicit qualified Python interpreter")
endif()
add_executable(BoundedEvidenceChild src/evidence_child.c)
set_target_properties(BoundedEvidenceChild PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
foreach(configuration IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${configuration}" configuration_upper)
    set_property(TARGET BoundedEvidenceChild PROPERTY
        "RUNTIME_OUTPUT_DIRECTORY_${configuration_upper}" "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()
if(MSVC)
    target_compile_definitions(BoundedEvidenceChild PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
include(CTest)
target_compile_definitions(BoundedEvidenceChild PRIVATE POLICY_TESTING_ON=$<BOOL:${BUILD_TESTING}>)
if(BUILD_TESTING)
    foreach(group IN ITEMS product boundary corpus)
        add_test(NAME "Policy.${group}"
            COMMAND BoundedEvidenceChild "${CMAKE_CURRENT_SOURCE_DIR}/input.txt"
                "${CMAKE_CURRENT_BINARY_DIR}/counter-${group}.txt")
        set_tests_properties("Policy.${group}" PROPERTIES LABELS "host;${group}"
            TIMEOUT 0 WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    endforeach()
    add_test(NAME Tooling.Miniature COMMAND "${Python3_EXECUTABLE}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/quality/test_miniature.py")
    set_tests_properties(Tooling.Miniature PROPERTIES LABELS "host;tooling"
        TIMEOUT 0 WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/Policy[1]_include.cmake"
        CONTENT "include(\"policy-leaf.cmake\")\nsubdirs(\"policy-subdir\")\n")
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/policy-leaf.cmake"
        CONTENT "# Fixed policy registration leaf.\n")
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/policy-subdir/CTestTestfile.cmake"
        CONTENT "include(\"../policy-leaf.cmake\")\n")
    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
        "${CMAKE_CURRENT_BINARY_DIR}/Policy[1]_include.cmake")
endif()
