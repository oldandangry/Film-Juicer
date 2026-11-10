cmake_minimum_required(VERSION 4.3)

if(CMAKE_SCRIPT_MODE_FILE)
    if(NOT IS_ABSOLUTE "${JUICER_RESOURCE_STAGE}" OR IS_SYMLINK "${JUICER_RESOURCE_STAGE}")
        message(FATAL_ERROR "Runtime test resource staging needs its own absolute directory")
    endif()
    file(MAKE_DIRECTORY "${JUICER_RESOURCE_STAGE}")
    file(STRINGS "${JUICER_RESOURCE_MEMBERS}" members ENCODING UTF-8)
    file(GLOB_RECURSE staged LIST_DIRECTORIES FALSE RELATIVE "${JUICER_RESOURCE_STAGE}"
        "${JUICER_RESOURCE_STAGE}/*")
    foreach(resource IN LISTS staged)
        if(NOT resource IN_LIST members)
            file(REMOVE "${JUICER_RESOURCE_STAGE}/${resource}")
        endif()
    endforeach()
    return()
endif()

function(juicer_stage_test_resources source stage stamp)
    set(membership "${CMAKE_CURRENT_BINARY_DIR}/resource-members.txt")
    list(JOIN JUICER_RUNTIME_RESOURCES "\n" members)
    file(CONFIGURE OUTPUT "${membership}" CONTENT "${members}\n" @ONLY)
    # Membership changes prune obsolete payloads before any newly selected copies.
    add_custom_command(OUTPUT "${stamp}"
        COMMAND "${CMAKE_COMMAND}"
            "-DJUICER_RESOURCE_STAGE=${stage}"
            "-DJUICER_RESOURCE_MEMBERS=${membership}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
        COMMAND "${CMAKE_COMMAND}" -E touch "${stamp}"
        DEPENDS "${membership}" "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
        COMMENT "Removing obsolete runtime test resources"
        VERBATIM)
    add_custom_target(juicer_test_resource_membership DEPENDS "${stamp}")

    set(outputs)
    foreach(resource IN LISTS JUICER_RUNTIME_RESOURCES)
        get_filename_component(directory "${stage}/${resource}" DIRECTORY)
        add_custom_command(OUTPUT "${stage}/${resource}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${directory}"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${source}/${resource}" "${stage}/${resource}"
            DEPENDS "${source}/${resource}"
            COMMENT "Copying runtime test resource ${resource}"
            VERBATIM)
        list(APPEND outputs "${stage}/${resource}")
    endforeach()
    add_custom_target(juicer_test_resources DEPENDS ${outputs})
    add_dependencies(juicer_test_resources juicer_test_resource_membership)
    if(TARGET juicer_host_owner)
        add_dependencies(juicer_test_resource_membership juicer_host_owner)
        add_dependencies(juicer_test_resources juicer_host_owner)
    endif()
endfunction()
