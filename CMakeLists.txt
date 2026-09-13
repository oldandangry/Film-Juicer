cmake_minimum_required(VERSION 4.3)

# CUDA 13.2 on Windows uses hybrid cudart. CMake 4.3 has no Hybrid runtime
# enumerator; leave its automatic runtime selection empty and link the imported
# cudart target below. Native CUDA compiler detection and compilation still apply.
if(CMAKE_HOST_WIN32)
    set(CMAKE_CUDA_RUNTIME_LIBRARY "")
    set(CMAKE_CUDA_FLAGS_INIT "--cudart=hybrid")
    if(NOT CMAKE_CUDA_HOST_COMPILER)
        find_program(JUICER_NVCC_MSVC cl.exe REQUIRED)
        set(CMAKE_CUDA_HOST_COMPILER "${JUICER_NVCC_MSVC}" CACHE FILEPATH "NVCC MSVC host")
    endif()
    set(CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT ProgramDatabase)
endif()

project(FilmJuicer LANGUAGES CXX CUDA)
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT Embedded)
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR
   NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64)$" OR
   NOT (WIN32 OR CMAKE_SYSTEM_NAME STREQUAL "Linux"))
    message(FATAL_ERROR "Film-Juicer supports Windows and Linux x86-64 only")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# The existing Windows Release configurations retain standard assertions.
# Keep that behavior on both platforms instead of CMake's default NDEBUG.
foreach(language IN ITEMS CXX CUDA)
    string(REGEX REPLACE "(^| )[/-]DNDEBUG( |$)" " "
        CMAKE_${language}_FLAGS_RELEASE "${CMAKE_${language}_FLAGS_RELEASE}")
endforeach()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "75-real;75-virtual")
    message(FATAL_ERROR "The supported GPU policy is SM 7.5 plus compute_75 PTX; use the shared presets")
endif()
if(NOT CMAKE_BUILD_TYPE MATCHES "^(Debug|Release)$")
    message(FATAL_ERROR "Use a Debug or Release Ninja preset")
endif()

include(cmake/JuicerDependencies.cmake)
include(cmake/JuicerSources.cmake)

add_library(juicer_settings INTERFACE)
target_include_directories(juicer_settings INTERFACE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(juicer_settings INTERFACE juicer_dependencies)
target_compile_definitions(juicer_settings INTERFACE
    JUICER_DIAGNOSTICS_COMPILED=$<IF:$<CONFIG:Debug>,1,0>)
if(WIN32)
    message(STATUS "CUDA 13.2 Windows: explicit hybrid runtime integration (CMake 4.3)")
    target_compile_definitions(juicer_settings INTERFACE WIN32 NOMINMAX _CRT_SECURE_NO_WARNINGS
        $<$<COMPILE_LANGUAGE:CXX>:_WINDLL>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:UNICODE;_UNICODE>)
    target_compile_options(juicer_settings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:/W3;/WX;/sdl;/permissive-;/EHsc>
        $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/O2;/Oi;/Ob2;/Gy>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3;-Xptxas=-O3;-Xcompiler=/EHsc,/O2,/Oi,/Ob2,/Gy,/Gw,/GF>)
    target_link_options(juicer_settings INTERFACE /DEBUG /INCREMENTAL:NO
        $<$<CONFIG:Release>:/OPT:REF;/OPT:ICF>)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        target_compile_options(juicer_settings INTERFACE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/Gw;/GF;-flto=thin>)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(juicer_settings INTERFACE
            $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:/GL>)
        target_link_options(juicer_settings INTERFACE $<$<CONFIG:Release>:/LTCG>)
    else()
        message(FATAL_ERROR "Windows requires MSVC or ClangCl with NVCC/MSVC")
    endif()
else()
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
       CMAKE_CXX_COMPILER_VERSION VERSION_LESS 10 OR
       CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 16)
        message(FATAL_ERROR "Linux requires CUDA 13.2-supported GCC 10 through 15")
    endif()
    target_compile_options(juicer_settings INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Werror>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3;-Xptxas=-O3>)
endif()

add_library(juicer_core STATIC ${JUICER_SOURCES})
target_link_libraries(juicer_core PUBLIC juicer_settings)
set_target_properties(juicer_core PROPERTIES LINKER_LANGUAGE CXX)

add_library(juicer MODULE src/main.cpp)
target_link_libraries(juicer PRIVATE juicer_core juicer_ofxs)
set_target_properties(juicer PROPERTIES PREFIX "" SUFFIX ".ofx" LINKER_LANGUAGE CXX)
if(UNIX)
    # Static cuFFT contains relocatable device code even though our kernels do
    # not need separable compilation. Resolve it once at the final module.
    set_property(TARGET juicer PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_options(juicer PRIVATE
        -static-libstdc++ -static-libgcc
        "LINKER:--version-script=${PROJECT_SOURCE_DIR}/cmake/juicer.exports"
        "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
    set_property(TARGET juicer APPEND PROPERTY LINK_DEPENDS "${PROJECT_SOURCE_DIR}/cmake/juicer.exports")
endif()

option(JUICER_BUILD_VALIDATION "Build the tracked scatter/halation acceptance executable" OFF)
if(JUICER_BUILD_VALIDATION)
    add_subdirectory(validation/scatter_halation)
endif()
include(cmake/JuicerBundle.cmake)
