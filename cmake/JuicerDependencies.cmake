if(NOT CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" OR
   NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "^13\\.2\\.")
    message(FATAL_ERROR "Film-Juicer requires NVIDIA CUDA 13.2; no toolkit or dialect fallback")
endif()
find_package(CUDAToolkit 13.2 REQUIRED)
if(NOT CUDAToolkit_VERSION MATCHES "^13\\.2\\.")
    message(FATAL_ERROR "FindCUDAToolkit must resolve the same CUDA 13.2 installation as NVCC")
endif()
find_package(Threads REQUIRED)

add_library(juicer_dependencies INTERFACE)
target_include_directories(juicer_dependencies SYSTEM INTERFACE
    "${PROJECT_SOURCE_DIR}/third_party"
    "${PROJECT_SOURCE_DIR}/third_party/ofxs/OpenFX-1.4/include"
    "${PROJECT_SOURCE_DIR}/third_party/ofxs/Support/include")
target_link_libraries(juicer_dependencies INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
if(WIN32)
    target_link_libraries(juicer_dependencies INTERFACE CUDA::cudart CUDA::cufft)
else()
    target_link_libraries(juicer_dependencies INTERFACE CUDA::cudart_static CUDA::cufft_static)
endif()

add_library(juicer_ofxs OBJECT
    "${PROJECT_SOURCE_DIR}/third_party/ofxs/Support/Library/ofxsCore.cpp"
    "${PROJECT_SOURCE_DIR}/third_party/ofxs/Support/Library/ofxsImageEffect.cpp"
    "${PROJECT_SOURCE_DIR}/third_party/ofxs/Support/Library/ofxsInteract.cpp"
    "${PROJECT_SOURCE_DIR}/third_party/ofxs/Support/Library/ofxsLog.cpp"
    "${PROJECT_SOURCE_DIR}/third_party/ofxs/Support/Library/ofxsMultiThread.cpp"
    "${PROJECT_SOURCE_DIR}/third_party/ofxs/Support/Library/ofxsParams.cpp"
    "${PROJECT_SOURCE_DIR}/third_party/ofxs/Support/Library/ofxsProperty.cpp"
    "${PROJECT_SOURCE_DIR}/third_party/ofxs/Support/Library/ofxsPropertyValidation.cpp")
set_source_files_properties(third_party/ofxs/Support/Library/ofxsImageEffect.cpp
    PROPERTIES COMPILE_DEFINITIONS DEBUG_BUILD)
target_link_libraries(juicer_ofxs PUBLIC juicer_dependencies)
if(WIN32)
    target_link_libraries(juicer_ofxs PRIVATE juicer_settings)
else()
    target_compile_options(juicer_ofxs PRIVATE -Werror)
endif()
