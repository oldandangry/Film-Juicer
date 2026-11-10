if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" OR
   NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "^13\\.2\\.")
    message(FATAL_ERROR "Film-Juicer requires NVIDIA CUDA 13.2; no toolkit or dialect fallback")
endif()
# The CUDA language already selected NVCC's toolkit. Searching every PATH entry
# again for optional CUDA libraries is particularly expensive across WSL mounts.
block(PROPAGATE CUDAToolkit_VERSION CUDAToolkit_LIBRARY_ROOT CUDAToolkit_BIN_DIR CUDAToolkit_INCLUDE_DIRS)
    set(CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH FALSE)
    find_package(CUDAToolkit 13.2 REQUIRED)
    if(NOT CUDAToolkit_VERSION MATCHES "^13\\.2\\.")
        message(FATAL_ERROR "FindCUDAToolkit must resolve the same CUDA 13.2 installation as NVCC")
    endif()
endblock()
find_package(Threads REQUIRED)

add_library(juicer_dependencies INTERFACE)
add_library(juicer_ofx_headers INTERFACE)
target_include_directories(juicer_ofx_headers SYSTEM INTERFACE
    "${PROJECT_SOURCE_DIR}/third_party/ofxs/OpenFX-1.4/include")
target_link_libraries(juicer_dependencies INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
if(WIN32)
    target_link_libraries(juicer_dependencies INTERFACE CUDA::cudart CUDA::cufft)
else()
    target_link_libraries(juicer_dependencies INTERFACE CUDA::cudart_static CUDA::cufft_static)
endif()
