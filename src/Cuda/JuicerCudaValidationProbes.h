#pragma once

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__) && defined(JUICER_CUDA_VALIDATE_PRIMITIVES) && (JUICER_CUDA_VALIDATE_PRIMITIVES != 0)

#include <cuda_runtime.h>

namespace JuicerCuda {
    struct PipelineRunParams;
}

extern "C" cudaError_t juicer_cuda_probe_density_curve(
    const float* dX,
    const float* dY,
    int n,
    float gammaFactor,
    const float* hLogE,
    int m,
    float* hOut,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_density_curve_sanitize_inf(
    const float* dX,
    const float* dY,
    int n,
    float gammaFactor,
    const float* hLogE,
    int m,
    float* hOut,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_film_log_raw(
    const float filmRaw3[3],
    float outLogRaw3[3],
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_hanatos_layer_exposures(
    const float rgbDWG[3],
    const float* dHanatosLut,
    int hanatosN,
    const float refIllumWhiteXYZ[3],
    const float* dSensB,
    const float* dSensG,
    const float* dSensR,
    int K,
    float exposureScale,
    int applyDeltaLambda,
    float* outE3,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_convert_input_to_DWG(
    const float rgbIn[3],
    int inputColorSpaceIndex,
    int applyCctfDecoding,
    int applyInputChromaticAdapt,
    const float inputRGBToXYZ9[9],
    const float inputXYZAdapt9[9],
    float outRgbDWG[3],
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_tables_layer_exposures(
    const float rgbDWG[3],
    const float S_inv9[9],
    const float refIllumWhiteXYZ[3],
    const float* dAx,
    const float* dAy,
    const float* dAz,
    int K,
    const float* dSensB,
    const float* dSensG,
    const float* dSensR,
    float exposureScale,
    int applyDeltaLambda,
    float* outE3,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_scan_spectral_to_log_xyz(
    const double D_norm3[3],
    int mediumIsNegative,
    const float min_cmy[3],
    const float inv_max_cmy[3],
    const float* dEpsC,
    const float* dEpsM,
    const float* dEpsY,
    const float* dAx,
    const float* dAy,
    const float* dAz,
    const float* dBaseMin,
    int K,
    int hasBaseline,
    float invYn,
    double outLogXYZ3[3],
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_clamp_logE_to_curve_domain(
    const float* dX,
    int n,
    float logE,
    float* outLogE,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_probe_print_pipeline(
    const float* hNegCmy,
    int count,
    const JuicerCuda::PipelineRunParams* hParams,
    float* hOutPrintCmy,
    void* cudaStreamOpaque);

#endif
