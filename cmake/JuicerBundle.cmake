set(JUICER_BUNDLE "juicer.ofx.bundle")
if(WIN32)
    set(JUICER_ARCH_DIR "Win64")
else()
    set(JUICER_ARCH_DIR "Linux-x86-64")
endif()
set(JUICER_CONTENTS "${JUICER_BUNDLE}/Contents")
install(TARGETS juicer LIBRARY DESTINATION "${JUICER_CONTENTS}/${JUICER_ARCH_DIR}"
    RUNTIME DESTINATION "${JUICER_CONTENTS}/${JUICER_ARCH_DIR}")
install(DIRECTORY Resources/ DESTINATION "${JUICER_CONTENTS}/Resources")
install(FILES LICENSE DESTINATION "${JUICER_CONTENTS}/Legal")
install(FILES third_party/spektrafilm/SPEKTRAFILM_LICENSE.txt third_party/spektrafilm/README.md
    DESTINATION "${JUICER_CONTENTS}/Legal/spektrafilm")
install(DIRECTORY third_party/ofxs/Legal/ DESTINATION "${JUICER_CONTENTS}/Legal/ofxs")
install(FILES third_party/ofxs/PROVENANCE.json DESTINATION "${JUICER_CONTENTS}/Legal/ofxs")
install(FILES third_party/nlohmann/LICENSE.MIT third_party/nlohmann/PROVENANCE.md
    DESTINATION "${JUICER_CONTENTS}/Legal/nlohmann")
# Toolkit package layouts differ. Require the redistributable's license rather
# than silently producing a bundle without CUDA's redistribution terms.
find_file(JUICER_CUDA_LICENSE NAMES LICENSE EULA.txt copyright
    PATHS "${CUDAToolkit_LIBRARY_ROOT}" "/usr/share/doc/cuda-cudart-13-2"
    NO_DEFAULT_PATH REQUIRED)
find_file(JUICER_CUFFT_LICENSE NAMES LICENSE EULA.txt copyright
    PATHS "${CUDAToolkit_LIBRARY_ROOT}" "/usr/share/doc/libcufft-13-2"
    NO_DEFAULT_PATH REQUIRED)
install(FILES "${JUICER_CUDA_LICENSE}" DESTINATION "${JUICER_CONTENTS}/Legal/CUDA"
    RENAME CUDA-LICENSE.txt)
install(FILES "${JUICER_CUFFT_LICENSE}" DESTINATION "${JUICER_CONTENTS}/Legal/CUDA"
    RENAME cuFFT-LICENSE.txt)
if(WIN32)
    # CUDA 13.2 moves DLLs into bin/x64. The current FFT plans do not opt into
    # LTO kernels or callbacks, so they do not consume NVRTC/nvJitLink.
    foreach(runtime IN ITEMS cudart64_13.dll cufft64_12.dll)
        find_file(JUICER_${runtime} NAMES "${runtime}"
            PATHS "${CUDAToolkit_BIN_DIR}/x64" "${CUDAToolkit_BIN_DIR}" NO_DEFAULT_PATH REQUIRED)
        install(FILES "${JUICER_${runtime}}" DESTINATION "${JUICER_CONTENTS}/${JUICER_ARCH_DIR}")
    endforeach()
    set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION "${JUICER_CONTENTS}/${JUICER_ARCH_DIR}")
    set(CMAKE_INSTALL_UCRT_LIBRARIES FALSE)
    include(InstallRequiredSystemLibraries)
else()
    install(FILES README.md DESTINATION "${JUICER_BUNDLE}" RENAME INSTALL.md)
    install(FILES
        third_party/gcc-runtime/GPL-3.0-only.txt
        third_party/gcc-runtime/GCC-RUNTIME-LIBRARY-EXCEPTION-3.1.txt
        third_party/gcc-runtime/PROVENANCE.md
        DESTINATION "${JUICER_CONTENTS}/Legal/gcc-runtime")
endif()

# Stage from an empty tree before each archive so removed resources cannot linger.
add_custom_target(bundle-archive
    COMMAND "${CMAKE_COMMAND}" -E rm -rf "${CMAKE_INSTALL_PREFIX}/${JUICER_BUNDLE}"
    COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --config "$<CONFIG>"
    COMMAND "${CMAKE_COMMAND}" -E chdir "${CMAKE_INSTALL_PREFIX}"
        "${CMAKE_COMMAND}" -E tar czf "${CMAKE_BINARY_DIR}/juicer-${JUICER_ARCH_DIR}.tar.gz"
        "${JUICER_BUNDLE}"
    DEPENDS juicer VERBATIM USES_TERMINAL)
