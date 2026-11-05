# One native compile policy; CUDA-independent tests omit only SDK/device options.
if(NOT DEFINED JUICER_SETTINGS_ROOT)
    set(JUICER_SETTINGS_ROOT "${PROJECT_SOURCE_DIR}")
endif()

add_library(juicer_settings INTERFACE)
target_include_directories(juicer_settings INTERFACE
    "${JUICER_SETTINGS_ROOT}/src" "${JUICER_SETTINGS_ROOT}/native")
target_include_directories(juicer_settings SYSTEM INTERFACE
    "${JUICER_SETTINGS_ROOT}/third_party")
if(NOT JUICER_CUDA_FREE_TESTS)
    target_include_directories(juicer_settings SYSTEM INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
endif()
target_compile_definitions(juicer_settings INTERFACE
    JUICER_DIAGNOSTICS_COMPILED=$<IF:$<CONFIG:Debug>,1,0>)
if(WIN32)
    if(NOT JUICER_CUDA_FREE_TESTS)
        message(STATUS "CUDA 13.2 Windows: explicit hybrid runtime integration (CMake 4.3)")
    endif()
    target_compile_definitions(juicer_settings INTERFACE WIN32 NOMINMAX _CRT_SECURE_NO_WARNINGS
        $<$<COMPILE_LANGUAGE:CXX>:_WINDLL>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:UNICODE;_UNICODE>)
    target_compile_options(juicer_settings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:/W3;/WX;/sdl;/permissive-;/EHsc>
        $<$<COMPILE_LANGUAGE:CXX>:/O2;/Oi;/Ob2;/Gy>)
    if(NOT JUICER_CUDA_FREE_TESTS)
        target_compile_options(juicer_settings INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xptxas=-O3;-Xcompiler=/EHsc,/O2,/Oi,/Ob2,/Gy,/Gw,/GF>)
    endif()
    target_link_options(juicer_settings INTERFACE /DEBUG /INCREMENTAL:NO /OPT:REF /OPT:ICF)
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
    if(NOT JUICER_CUDA_FREE_TESTS)
        target_compile_options(juicer_settings INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>
            $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xptxas=-O3>)
    endif()
endif()
