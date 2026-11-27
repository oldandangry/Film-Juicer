# Native compile policy shared by shipping and test targets.
if(NOT DEFINED JUICER_SETTINGS_ROOT)
    set(JUICER_SETTINGS_ROOT "${PROJECT_SOURCE_DIR}")
endif()

add_library(juicer_settings INTERFACE)
target_include_directories(juicer_settings INTERFACE
    "${JUICER_SETTINGS_ROOT}/src" "${JUICER_SETTINGS_ROOT}/native")
target_include_directories(juicer_settings SYSTEM INTERFACE
    "${JUICER_SETTINGS_ROOT}/third_party")
target_include_directories(juicer_settings SYSTEM INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(juicer_settings INTERFACE
    JUICER_DIAGNOSTICS_COMPILED=$<IF:$<CONFIG:Debug>,1,0>)
if(WIN32)
    message(STATUS "CUDA 13.2 Windows: explicit hybrid runtime integration (CMake 4.3)")
    target_compile_definitions(juicer_settings INTERFACE WIN32 NOMINMAX _CRT_SECURE_NO_WARNINGS UNICODE _UNICODE)
    target_compile_options(juicer_settings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:/W3;/WX;/sdl;/permissive-;/EHsc>
        $<$<COMPILE_LANGUAGE:CXX>:/O2;/Oi;/Ob2;/Gy>)
    target_compile_options(juicer_settings INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xptxas=-O3;-Xcompiler=/EHsc,/O2,/Oi,/Ob2,/Gy,/Gw,/GF>)
    target_link_options(juicer_settings INTERFACE "$<HOST_LINK:/DEBUG,/INCREMENTAL:NO,/OPT:REF,/OPT:ICF>")
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "Windows requires ClangCl with NVCC/MSVC")
    endif()
    target_compile_options(juicer_settings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:/Gw;/GF;-flto=thin>)
else()
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
       CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10 OR
       CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16)
        message(FATAL_ERROR "Linux requires CUDA 13.2-supported GCC 10 through 15")
    endif()
    target_compile_options(juicer_settings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>)
    target_compile_options(juicer_settings INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xptxas=-O3>)
endif()
