set(JUICER_BUNDLE "juicer.ofx.bundle")
if(WIN32)
    set(JUICER_ARCH_DIR "Win64")
else()
    set(JUICER_ARCH_DIR "Linux-x86-64")
endif()
set(JUICER_CONTENTS "${JUICER_BUNDLE}/Contents")
install(TARGETS juicer LIBRARY DESTINATION "${JUICER_CONTENTS}/${JUICER_ARCH_DIR}"
    RUNTIME DESTINATION "${JUICER_CONTENTS}/${JUICER_ARCH_DIR}")

# Keep the bundle resource surface explicit. Repository artwork, generators,
# and documentation remain available to the project without becoming runtime
# plug-in payload.
set(JUICER_RUNTIME_RESOURCES
    Noise/Wang/tiles.json
    Noise/Wang/wang_tiles_256x256x16_u8.bin
    Noise/stbn_scalar_512x512x256_u8.bin
    cie1931_2deg.csv
    com.juicer.Juicer.png
    filters/heat_absorbing/schott/KG3.csv
    filters/lens_transmission/canon/canon_24_f28_is.csv
    filters/neutral_print_filters.json
    illuminants/D50.csv
    illuminants/D55.csv
    illuminants/D65.csv
    illuminants/K75P.csv
    illuminants/T.csv
    luts/spectral_upsampling/irradiance_xy_tc.npy
    luts/spectral_upsampling/arctic2026beta04_reflectance_xy_tc.npy
    luts/spectral_upsampling/mallett2019_basis.npy
    profiles/fujifilm_c200.json
    profiles/fujifilm_crystal_archive_typeii.json
    profiles/fujifilm_pro_400h.json
    profiles/fujifilm_provia_100f.json
    profiles/fujifilm_velvia_100.json
    profiles/fujifilm_xtra_400.json
    profiles/kodak_2383.json
    profiles/kodak_2393.json
    profiles/kodak_ektachrome_100.json
    profiles/kodak_ektacolor_edge.json
    profiles/kodak_ektar_100.json
    profiles/kodak_endura_premier.json
    profiles/kodak_gold_200.json
    profiles/kodak_kodachrome_64.json
    profiles/kodak_portra_160.json
    profiles/kodak_portra_400.json
    profiles/kodak_portra_800.json
    profiles/kodak_portra_800_push1.json
    profiles/kodak_portra_800_push2.json
    profiles/kodak_portra_endura.json
    profiles/kodak_supra_endura.json
    profiles/kodak_ultra_endura.json
    profiles/kodak_ultramax_400.json
    profiles/kodak_verita_200d.json
    profiles/kodak_vision3_200t.json
    profiles/kodak_vision3_250d.json
    profiles/kodak_vision3_500t.json
    profiles/kodak_vision3_50d.json)
foreach(resource IN LISTS JUICER_RUNTIME_RESOURCES)
    get_filename_component(resource_directory "${resource}" DIRECTORY)
    if(resource_directory)
        set(resource_destination "${JUICER_CONTENTS}/Resources/${resource_directory}")
    else()
        set(resource_destination "${JUICER_CONTENTS}/Resources")
    endif()
    install(FILES "${PROJECT_SOURCE_DIR}/Resources/${resource}"
        DESTINATION "${resource_destination}")
endforeach()

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
