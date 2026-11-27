#pragma once

#include <cuda_runtime_api.h>

#include "Cuda/JuicerCudaPayloads.h"

extern "C" cudaError_t juicer_cuda_direct_focused_scanner_post_output(
    const JuicerCuda::FilmScanRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    const JuicerCuda::FilmDefectsPayload* gateDefects,
    const JuicerCuda::GateWeavePayload* weave,
    const float* gateTransmittance,
    int gateTransmittanceWidth,
    int gateTransmittanceHeight,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_scanner_post_output(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    const float* dLensBlurKernel,
    int lensBlurRadius,
    const float* dUnsharpKernel,
    int unsharpRadius,
    float unsharpAmount,
    const JuicerCuda::FilmDefectsPayload* gateDefects,
    const JuicerCuda::GateWeavePayload* weave,
    const float* gateTransmittance,
    int gateTransmittanceWidth,
    int gateTransmittanceHeight,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_apply_film_defects(
    const JuicerCuda::FilmDefectsPayload* defects,
    int width,
    int height,
    float* densityC,
    float* densityM,
    float* densityY,
    float* filmDustTransmittance,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_build_gate_defect_transmittance_focused(
    const JuicerCuda::FilmDefectsPayload* defects,
    float* gateTransmittance,
    int gateTransmittanceWidth,
    int gateTransmittanceHeight,
    void* cudaStreamOpaque);
