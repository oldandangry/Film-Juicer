cmake_minimum_required(VERSION 3.24)

project(BoundedEvidenceFixture LANGUAGES C)

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
    # The fixture deliberately exercises the portable C11 stdio interface.
    target_compile_definitions(BoundedEvidenceChild PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(NAME Bounded.Read
        COMMAND BoundedEvidenceChild
            "${CMAKE_CURRENT_SOURCE_DIR}/input.txt"
            "${CMAKE_CURRENT_BINARY_DIR}/assertion-count.txt")
    set_tests_properties(Bounded.Read PROPERTIES
        LABELS "host;product"
        TIMEOUT 0)
endif()
