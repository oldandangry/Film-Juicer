foreach(flags IN ITEMS CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE)
    if("${${flags}}" MATCHES "(^|[ \t\r\n])--?use_fast_math([= \t\r\n]|$)")
        message(FATAL_ERROR "ScatterHalationValidation forbids --use_fast_math")
    endif()
endforeach()
add_executable(ScatterHalationValidation
    ScatterHalationValidation.cpp ScatterHalationValidation.cu)
target_link_libraries(ScatterHalationValidation PRIVATE juicer_core CUDA::cuda_driver)
set_target_properties(ScatterHalationValidation PROPERTIES
    LINKER_LANGUAGE CXX RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/validation")
if(UNIX)
    set_property(TARGET ScatterHalationValidation PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
# Root bootstraps relative to the executable, independently of --resource-root.
add_custom_target(scatter-halation-resources
    COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${PROJECT_SOURCE_DIR}/Resources" "${CMAKE_BINARY_DIR}/Resources"
    VERBATIM)
add_dependencies(ScatterHalationValidation scatter-halation-resources)
