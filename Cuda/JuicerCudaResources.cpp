// Cuda/JuicerCudaResources.cpp
//
// Phase 2: WorkingState uploads + primitive validation hooks.
//
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaPayloads.h"

#include "FilmProcessing.h"
#include "ColorTransforms.h"
#include "SpectralProcessing.h"
#include "WorkingState.h"
#include "ScanStage.h"
#include "Print.h"
#include "ExposePrintStage.h"
#include "DevelopPrintStage.h"

#include "GaussianSciPy.h"

#include "Logging.h"
#include "Hash.h"
#include "SpectralContext.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// Implemented in Cuda/JuicerCudaPrimitives.cu
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
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

// Implemented in Cuda/JuicerCudaPrimitives.cu
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

// Implemented in Cuda/JuicerCudaPrimitives.cu
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
    const JuicerCuda::Phase3RunParams* hParams,
    float* hOutPrintCmy,
    void* cudaStreamOpaque);
#endif

namespace JuicerCuda {

    static void free_curve(DeviceCurve& c) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (c.x) {
            cudaFree(c.x);
            c.x = nullptr;
        }
        if (c.y) {
            cudaFree(c.y);
            c.y = nullptr;
        }
#endif
        c.n = 0;
    }

    static void free_hanatos(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.hanatosLut) {
            cudaFree(resources.hanatosLut);
            resources.hanatosLut = nullptr;
        }
#endif
        resources.hanatosN = 0;
    }

    static void free_hanatos_integrated(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.hanatosLutIntegrated) {
            cudaFree(resources.hanatosLutIntegrated);
            resources.hanatosLutIntegrated = nullptr;
        }
#endif
        resources.hanatosNIntegrated = 0;
        resources.hanatosIntegratedBuildCounter = 0;
    }

    static void free_scan_error_flag(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.scanErrorFlag) {
            cudaFree(resources.scanErrorFlag);
            resources.scanErrorFlag = nullptr;
        }
        if (resources.scanErrorHost) {
            cudaFreeHost(resources.scanErrorHost);
            resources.scanErrorHost = nullptr;
        }
        if (resources.scanErrorEventOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(resources.scanErrorEventOpaque);
            cudaEventDestroy(ev);
            resources.scanErrorEventOpaque = nullptr;
        }
#endif
        resources.scanErrorPending = 0;
    }

    static void free_tables(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.tablesAx) {
            cudaFree(resources.tablesAx);
            resources.tablesAx = nullptr;
        }
        if (resources.tablesAy) {
            cudaFree(resources.tablesAy);
            resources.tablesAy = nullptr;
        }
        if (resources.tablesAz) {
            cudaFree(resources.tablesAz);
            resources.tablesAz = nullptr;
        }
#endif
        resources.tablesK = 0;
    }

    static void free_spectral_tables(Resources::DeviceSpectralTables& t) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (t.epsC) { cudaFree(t.epsC); t.epsC = nullptr; }
        if (t.epsM) { cudaFree(t.epsM); t.epsM = nullptr; }
        if (t.epsY) { cudaFree(t.epsY); t.epsY = nullptr; }
        if (t.Ax) { cudaFree(t.Ax); t.Ax = nullptr; }
        if (t.Ay) { cudaFree(t.Ay); t.Ay = nullptr; }
        if (t.Az) { cudaFree(t.Az); t.Az = nullptr; }
        if (t.baseMin) { cudaFree(t.baseMin); t.baseMin = nullptr; }
#endif
        t.K = 0;
        t.hasBaseline = 0;
        t.invYn = 1.0f;
    }

    static void free_scan_medium(Resources::DeviceScanMedium& m) noexcept {
        free_spectral_tables(m.tables);
        m.mediumIsNegative = 1;
        m.min_cmy[0] = m.min_cmy[1] = m.min_cmy[2] = 0.0f;
        m.inv_max_cmy[0] = m.inv_max_cmy[1] = m.inv_max_cmy[2] = 1.0f;
    }

    static void free_scan_lut(Resources::DeviceSpectralLut& lut) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (lut.logXYZ) {
            cudaFree(lut.logXYZ);
            lut.logXYZ = nullptr;
        }
#endif
        lut.res = 0;
        lut.hash = 0;
    }

    static void free_gaussian_kernel(Resources::DeviceGaussianKernel& k) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (k.weights) {
            cudaFree(k.weights);
            k.weights = nullptr;
        }
#endif
        k.radius = 0;
        k.sigma = 0.0f;
    }

    static void free_optics_scratch(Resources::DeviceOpticsScratch& s) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (s.rgbR) { cudaFree(s.rgbR); s.rgbR = nullptr; }
        if (s.rgbG) { cudaFree(s.rgbG); s.rgbG = nullptr; }
        if (s.rgbB) { cudaFree(s.rgbB); s.rgbB = nullptr; }
        if (s.tmp) { cudaFree(s.tmp); s.tmp = nullptr; }
        if (s.blurred) { cudaFree(s.blurred); s.blurred = nullptr; }
#endif
        s.width = 0;
        s.height = 0;
    }

    static void free_spatial_dir_scratch(Resources::DeviceSpatialDirScratch& s) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (s.corrY) { cudaFree(s.corrY); s.corrY = nullptr; }
        if (s.corrM) { cudaFree(s.corrM); s.corrM = nullptr; }
        if (s.corrC) { cudaFree(s.corrC); s.corrC = nullptr; }
        if (s.tmp) { cudaFree(s.tmp); s.tmp = nullptr; }
#endif
        s.width = 0;
        s.height = 0;
    }

    static void free_print_payloads(Resources& resources) noexcept {
        free_curve(resources.printDcC);
        free_curve(resources.printDcM);
        free_curve(resources.printDcY);
        free_curve(resources.printSensC);
        free_curve(resources.printSensM);
        free_curve(resources.printSensY);

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.printIllumFiltered) {
            cudaFree(resources.printIllumFiltered);
            resources.printIllumFiltered = nullptr;
        }
#endif
        resources.printIllumK = 0;
        resources.printIllumYShiftSteps = 0.0f;
        resources.printIllumMShiftSteps = 0.0f;
        resources.printIllumCShiftSteps = 0.0f;
        resources.printIllumShapeK = 0;
        resources.printIllumBuildCounter = 0;
        resources.printIllumRuntimePtr = nullptr;

        resources.printGammaC = 1.0f;
        resources.printGammaM = 1.0f;
        resources.printGammaY = 1.0f;

        resources.printPreflashRaw[0] = resources.printPreflashRaw[1] = resources.printPreflashRaw[2] = 0.0f;
        resources.printPreflashValid = false;
        resources.printPreflashBuildCounter = 0;
        resources.printPreflashRuntimePtr = nullptr;
        resources.printPreflashShapeK = 0;
    }

    static bool alloc_and_upload_array(float*& dst, const float* src, int n, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)dst;
        (void)src;
        (void)n;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!src || n <= 0) {
            outError = std::string(label) + " array is empty";
            return false;
        }
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dst), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(") + label + ") failed: " + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }
        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(") + label + ") failed: " + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            cudaFree(dst);
            dst = nullptr;
            return false;
        }
        return true;
#endif
    }

    static bool alloc_and_upload_curve(DeviceCurve& dst, const Spectral::Curve& src, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)dst;
        (void)src;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        if (src.lambda_nm.empty() || src.linear.empty() || src.lambda_nm.size() != src.linear.size()) {
            outError = "curve has no samples or mismatched arrays";
            return false;
        }

        const int n = static_cast<int>(src.lambda_nm.size());
        if (n <= 0) {
            outError = "curve sample count invalid";
            return false;
        }

        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dst.x), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(curve.x) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }
        err = cudaMalloc(reinterpret_cast<void**>(&dst.y), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(curve.y) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }

        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        err = cudaMemcpyAsync(dst.x, src.lambda_nm.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(curve.x) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }
        err = cudaMemcpyAsync(dst.y, src.linear.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(curve.y) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }

        dst.n = n;
        return true;
#endif
    }

    static bool alloc_and_upload_spectral_samples(DeviceCurve& dst, const std::vector<float>& src, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)dst;
        (void)src;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (src.empty()) {
            outError = std::string(label) + " array is empty";
            return false;
        }
        const int n = static_cast<int>(src.size());
        if (n <= 0) {
            outError = std::string(label) + " sample count invalid";
            return false;
        }
        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dst.y), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(") + label + ") failed: " + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }
        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        err = cudaMemcpyAsync(dst.y, src.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(") + label + ") failed: " + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_curve(dst);
            return false;
        }
        dst.n = n;
        return true;
#endif
    }

    Resources::~Resources() {
        free_curve(densB);
        free_curve(densG);
        free_curve(densR);
        free_curve(dirDensB);
        free_curve(dirDensG);
        free_curve(dirDensR);
        free_curve(sensB);
        free_curve(sensG);
        free_curve(sensR);
        free_tables(*this);
        free_scan_medium(scanNegative);
        free_scan_medium(scanPrint);
        free_scan_lut(scanNegativeLut);
        free_scan_lut(scanPrintLut);
        free_gaussian_kernel(scannerLensBlurKernel);
        free_gaussian_kernel(scannerUnsharpKernel);
        free_gaussian_kernel(scannerGlareKernel);
        free_optics_scratch(scannerScratch);
        free_gaussian_kernel(spatialDirKernel);
        free_spatial_dir_scratch(spatialDirScratch);
        free_print_payloads(*this);
        free_hanatos(*this);
        free_hanatos_integrated(*this);
        free_scan_error_flag(*this);
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (lastUseEventOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(lastUseEventOpaque);
            cudaEventDestroy(ev);
            lastUseEventOpaque = nullptr;
        }
#endif
    }

    Resources* create() noexcept {
        try {
            Resources* r = new Resources();
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
            int dev = -1;
            if (cudaGetDevice(&dev) == cudaSuccess) {
                r->deviceId = dev;
            }
#endif
            return r;
        }
        catch (...) {
            return nullptr;
        }
    }

    void destroy(Resources* resources) noexcept {
        delete resources;
    }

    static bool sync_before_rebuild(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError);

    bool ensure_uploaded(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        (void)cudaStreamOpaque;
        std::lock_guard<std::mutex> lock(resources.m);
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        if (resources.uploadedBuildCounter == ws.buildCounter && ws.buildCounter != 0) {
            return true;
        }

        if (ws.buildCounter == 0) {
            outError = "WorkingState buildCounter is 0";
            return false;
        }

        // Important: OFX CUDA renders are async (we enqueue work on the host stream). When rebuilding
        // GPU resources, ensure no in-flight work can still reference the previous device pointers.
        // We do NOT assume the current render stream matches the stream used by the previous render.
        if (resources.uploadedBuildCounter != ws.buildCounter) {
            if (resources.uploadedBuildCounter != 0) {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
                if (resources.lastUseEventOpaque) {
                    cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(resources.lastUseEventOpaque);
                    const cudaError_t evErr = cudaEventSynchronize(ev);
                    if (evErr != cudaSuccess) {
                        outError = std::string("cudaEventSynchronize before rebuild failed: ") + (cudaGetErrorString(evErr) ? cudaGetErrorString(evErr) : "(unknown)");
                        return false;
                    }
                } else {
                    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
                    const cudaError_t syncErr = cudaStreamSynchronize(stream);
                    if (syncErr != cudaSuccess) {
                        outError = std::string("cudaStreamSynchronize before rebuild failed: ") + (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
                        return false;
                    }
                }
#endif
            }

            // Clear any previously uploaded (or partially uploaded) curves before re-uploading.
            free_curve(resources.densB);
            free_curve(resources.densG);
            free_curve(resources.densR);
            free_curve(resources.dirDensB);
            free_curve(resources.dirDensG);
            free_curve(resources.dirDensR);
            free_curve(resources.sensB);
            free_curve(resources.sensG);
            free_curve(resources.sensR);
            free_tables(resources);
            free_scan_medium(resources.scanNegative);
            free_scan_medium(resources.scanPrint);
            free_print_payloads(resources);

            resources.validatedBuildCounter = 0;
        }

        if (!alloc_and_upload_curve(resources.densB, ws.densB, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.densG, ws.densG, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.densR, ws.densR, cudaStreamOpaque, outError)) return false;

        if (!alloc_and_upload_curve(resources.dirDensB, ws.dirDensB, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.dirDensG, ws.dirDensG, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.dirDensR, ws.dirDensR, cudaStreamOpaque, outError)) return false;

        if (!alloc_and_upload_curve(resources.sensB, ws.sensB, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.sensG, ws.sensG, cudaStreamOpaque, outError)) return false;
        if (!alloc_and_upload_curve(resources.sensR, ws.sensR, cudaStreamOpaque, outError)) return false;

        // Upload per-instance Mallett basis tables (Ax/Ay/Az) and keep a host-side copy of S_inv + ref white.
        {
            const int K = ws.tablesRef.K;
            const bool want =
                ws.spdReady &&
                K == Spectral::gShape.K &&
                static_cast<int>(ws.tablesRef.Ax.size()) == K &&
                static_cast<int>(ws.tablesRef.Ay.size()) == K &&
                static_cast<int>(ws.tablesRef.Az.size()) == K;

            if (!want) {
                free_tables(resources);
            } else if (resources.tablesK != K || !resources.tablesAx || !resources.tablesAy || !resources.tablesAz) {
                free_tables(resources);
                if (!alloc_and_upload_array(resources.tablesAx, ws.tablesRef.Ax.data(), K, cudaStreamOpaque, "tablesAx", outError)) { free_tables(resources); return false; }
                if (!alloc_and_upload_array(resources.tablesAy, ws.tablesRef.Ay.data(), K, cudaStreamOpaque, "tablesAy", outError)) { free_tables(resources); return false; }
                if (!alloc_and_upload_array(resources.tablesAz, ws.tablesRef.Az.data(), K, cudaStreamOpaque, "tablesAz", outError)) { free_tables(resources); return false; }
                resources.tablesK = K;
            }

            for (int i = 0; i < 9; ++i) resources.spdSInv[i] = ws.spdSInv[i];
            for (int i = 0; i < 3; ++i) resources.refIllumWhiteXYZ[i] = ws.filmRaw.refIllumWhiteXYZ[i];
        }

        auto upload_scan_medium = [&](Resources::DeviceScanMedium& dst, const Scanner::ScannerMediumRuntime& medium, std::string& outErrorLocal) -> bool {
            const Spectral::SpectralTables* t = medium.tables;
            if (!t || t->K != Spectral::gShape.K) {
                free_scan_medium(dst);
                return true;
            }
            const int K = t->K;
            const bool arraysOk =
                static_cast<int>(t->epsC.size()) == K &&
                static_cast<int>(t->epsM.size()) == K &&
                static_cast<int>(t->epsY.size()) == K &&
                static_cast<int>(t->Ax.size()) == K &&
                static_cast<int>(t->Ay.size()) == K &&
                static_cast<int>(t->Az.size()) == K &&
                (!t->hasBaseline || static_cast<int>(t->baseMin.size()) == K);
            if (!arraysOk) {
                outErrorLocal = "scan spectral tables missing required arrays";
                return false;
            }

            // Rebuild if size mismatches or not allocated yet.
            if (dst.tables.K != K || !dst.tables.epsC || !dst.tables.Ax) {
                free_scan_medium(dst);
                if (!alloc_and_upload_array(dst.tables.epsC, t->epsC.data(), K, cudaStreamOpaque, "scan.epsC", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.epsM, t->epsM.data(), K, cudaStreamOpaque, "scan.epsM", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.epsY, t->epsY.data(), K, cudaStreamOpaque, "scan.epsY", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.Ax, t->Ax.data(), K, cudaStreamOpaque, "scan.Ax", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.Ay, t->Ay.data(), K, cudaStreamOpaque, "scan.Ay", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (!alloc_and_upload_array(dst.tables.Az, t->Az.data(), K, cudaStreamOpaque, "scan.Az", outErrorLocal)) { free_scan_medium(dst); return false; }
                if (t->hasBaseline) {
                    if (!alloc_and_upload_array(dst.tables.baseMin, t->baseMin.data(), K, cudaStreamOpaque, "scan.baseMin", outErrorLocal)) { free_scan_medium(dst); return false; }
                }

                dst.tables.K = K;
            }

            // Baseline can toggle without changing K; keep device pointer in sync.
            if (t->hasBaseline) {
                if (!dst.tables.baseMin) {
                    if (!alloc_and_upload_array(dst.tables.baseMin, t->baseMin.data(), K, cudaStreamOpaque, "scan.baseMin", outErrorLocal)) { free_scan_medium(dst); return false; }
                }
            }
            else {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
                if (dst.tables.baseMin) {
                    cudaFree(dst.tables.baseMin);
                    dst.tables.baseMin = nullptr;
                }
#endif
            }

            dst.tables.hasBaseline = t->hasBaseline ? 1 : 0;
            dst.tables.invYn = (std::isfinite(t->invYn) && t->invYn > 0.0f) ? t->invYn : 1.0f;

            dst.mediumIsNegative = (medium.medium == Scanner::ScannerMedium::Negative) ? 1 : 0;
            for (int i = 0; i < 3; ++i) {
                dst.min_cmy[i] = medium.range.min_cmy[i];
                dst.inv_max_cmy[i] = (std::isfinite(medium.range.inv_max_cmy[i]) && medium.range.inv_max_cmy[i] > 0.0f)
                    ? medium.range.inv_max_cmy[i]
                    : 1.0f;
            }
            return true;
        };

        {
            std::string scanError;
            if (!upload_scan_medium(resources.scanNegative, ws.negativeMediumRuntime, scanError)) {
                outError = std::string("upload scan negative failed: ") + scanError;
                return false;
            }
            if (!upload_scan_medium(resources.scanPrint, ws.printMediumRuntime, scanError)) {
                outError = std::string("upload scan print failed: ") + scanError;
                return false;
            }
        }

        // Phase 5: upload print pipeline payloads when a valid print runtime is present.
        {
            const Print::Runtime* prt = ws.printRT.get();
            if (!prt || !Print::profile_is_valid(prt->profile)) {
                free_print_payloads(resources);
            }
            else {
                const Print::Profile& p = prt->profile;

                // Upload print density curves (logE->D) for C/M/Y.
                auto ensure_print_curve = [&](DeviceCurve& dst, const Spectral::Curve& src, const char* label, std::string& err) -> bool {
                    const int n = static_cast<int>(src.lambda_nm.size());
                    const bool want = n > 1 && src.linear.size() == src.lambda_nm.size();
                    if (!want) {
                        free_curve(dst);
                        return true;
                    }
                    if (dst.n != n || !dst.x || !dst.y) {
                        free_curve(dst);
                        if (!alloc_and_upload_curve(dst, src, cudaStreamOpaque, err)) {
                            err = std::string(label) + ": " + err;
                            return false;
                        }
                    }
                    return true;
                };

                std::string printErr;
                if (!ensure_print_curve(resources.printDcC, p.dcC, "print dcC", printErr)) { outError = printErr; return false; }
                if (!ensure_print_curve(resources.printDcM, p.dcM, "print dcM", printErr)) { outError = printErr; return false; }
                if (!ensure_print_curve(resources.printDcY, p.dcY, "print dcY", printErr)) { outError = printErr; return false; }

                // Upload print paper sensitivities (linear domain, pinned to shape).
                const int K = Spectral::gShape.K;
                const bool sensOk =
                    K > 0 &&
                    static_cast<int>(p.sensC_log.linear.size()) == K &&
                    static_cast<int>(p.sensM_log.linear.size()) == K &&
                    static_cast<int>(p.sensY_log.linear.size()) == K;
                if (!sensOk) {
                    free_curve(resources.printSensC);
                    free_curve(resources.printSensM);
                    free_curve(resources.printSensY);
                }
                else {
                    if (!resources.printSensC.y || resources.printSensC.n != K) {
                        free_curve(resources.printSensC);
                        if (!alloc_and_upload_spectral_samples(resources.printSensC, p.sensC_log.linear, cudaStreamOpaque, "print sensC", outError)) { return false; }
                    }
                    if (!resources.printSensM.y || resources.printSensM.n != K) {
                        free_curve(resources.printSensM);
                        if (!alloc_and_upload_spectral_samples(resources.printSensM, p.sensM_log.linear, cudaStreamOpaque, "print sensM", outError)) { return false; }
                    }
                    if (!resources.printSensY.y || resources.printSensY.n != K) {
                        free_curve(resources.printSensY);
                        if (!alloc_and_upload_spectral_samples(resources.printSensY, p.sensY_log.linear, cudaStreamOpaque, "print sensY", outError)) { return false; }
                    }
                }

                auto gamma_safe = [](float v) -> float {
                    return (std::isfinite(v) && v > 0.0f) ? v : 1.0f;
                };
                resources.printGammaC = gamma_safe(p.gammaFactor[0]);
                resources.printGammaM = gamma_safe(p.gammaFactor[1]);
                resources.printGammaY = gamma_safe(p.gammaFactor[2]);

                // Preflash raw is computed for (y=m=c=0, Dneg=0) and cached per WorkingState/runtime.
                if (!resources.printPreflashValid ||
                    resources.printPreflashBuildCounter != ws.buildCounter ||
                    resources.printPreflashRuntimePtr != prt ||
                    resources.printPreflashShapeK != Spectral::gShape.K) {
                    const int shapeK = Spectral::gShape.K;
                    const bool haveShape = shapeK > 0;
                    if (!haveShape) {
                        resources.printPreflashRaw[0] = resources.printPreflashRaw[1] = resources.printPreflashRaw[2] = 0.0f;
                        resources.printPreflashValid = false;
                        resources.printPreflashBuildCounter = ws.buildCounter;
                        resources.printPreflashRuntimePtr = nullptr;
                        resources.printPreflashShapeK = 0;
                    }
                    else {
                        auto blend = [](float curveVal, float normalizedAmount) -> float {
                            const float a = std::isfinite(normalizedAmount) ? normalizedAmount : 0.0f;
                            return 1.0f - (1.0f - curveVal) * a;
                        };
                        auto compose_amount = [](float neutralAmount, float deltaSteps) -> float {
                            const float neutral = std::isfinite(neutralAmount)
                                ? std::clamp(neutralAmount, 0.0f, 1.0f)
                                : 0.0f;
                            float ds = std::isfinite(deltaSteps) ? deltaSteps : 0.0f;
                            ds = std::clamp(ds, -Print::kEnlargerSteps, Print::kEnlargerSteps);
                            const float totalSteps = neutral * Print::kEnlargerSteps + ds;
                            return totalSteps / Print::kEnlargerSteps;
                        };

                        const float yAmount = compose_amount(prt->neutralY, 0.0f);
                        const float mAmount = compose_amount(prt->neutralM, 0.0f);
                        const float cAmount = compose_amount(prt->neutralC, 0.0f);

                        const bool hasBL = ws.hasBaseline &&
                            static_cast<int>(ws.baseMin.linear.size()) == shapeK &&
                            static_cast<int>(ws.tablesView.baseMin.size()) == shapeK;

                        double accumC = 0.0;
                        double accumM = 0.0;
                        double accumY = 0.0;
                        for (int i = 0; i < shapeK; ++i) {
                            const float Ee = (prt->illumEnlarger.linear.size() > static_cast<size_t>(i))
                                ? prt->illumEnlarger.linear[static_cast<size_t>(i)]
                                : 1.0f;
                            const float fY = blend(
                                (prt->filterY.linear.size() > static_cast<size_t>(i)) ? prt->filterY.linear[static_cast<size_t>(i)] : 1.0f,
                                yAmount);
                            const float fM = blend(
                                (prt->filterM.linear.size() > static_cast<size_t>(i)) ? prt->filterM.linear[static_cast<size_t>(i)] : 1.0f,
                                mAmount);
                            const float fC = blend(
                                (prt->filterC.linear.size() > static_cast<size_t>(i)) ? prt->filterC.linear[static_cast<size_t>(i)] : 1.0f,
                                cAmount);
                            const float illumFiltered = Ee * (fY * fM * fC);

                            const float baseDensity = hasBL ? ws.tablesView.baseMin[static_cast<size_t>(i)] : 0.0f;
                            const double transmitted = std::pow(10.0, -static_cast<double>(baseDensity)) * static_cast<double>(illumFiltered);
                            const float out = static_cast<float>(transmitted);
                            const float light = std::isnan(out) ? 0.0f : out;
                            const float sC = p.sensC_log.linear[static_cast<size_t>(i)];
                            const float sM = p.sensM_log.linear[static_cast<size_t>(i)];
                            const float sY = p.sensY_log.linear[static_cast<size_t>(i)];
                            const double e64 = static_cast<double>(light);
                            if (!std::isnan(sC)) accumC += e64 * static_cast<double>(sC);
                            if (!std::isnan(sM)) accumM += e64 * static_cast<double>(sM);
                            if (!std::isnan(sY)) accumY += e64 * static_cast<double>(sY);
                        }

                        resources.printPreflashRaw[0] = static_cast<float>(accumC);
                        resources.printPreflashRaw[1] = static_cast<float>(accumM);
                        resources.printPreflashRaw[2] = static_cast<float>(accumY);
                        resources.printPreflashValid = true;
                        resources.printPreflashBuildCounter = ws.buildCounter;
                        resources.printPreflashRuntimePtr = prt;
                        resources.printPreflashShapeK = shapeK;
                    }
                }
            }
        }

        // Upload Hanatos LUT if available (uploaded once and reused across WorkingState rebuilds).
        {
            Spectral::SpectralContext& ctx = Spectral::context();
            const bool hanatosAvailable = ctx.hanatosAvailable.load(std::memory_order_acquire);
            const int N = ctx.hanSpectra.size;
            const int K = ctx.hanSpectra.numSamples;
            const bool want = hanatosAvailable && N > 0 && K == Spectral::kNumSamples && !ctx.hanSpectra.data.empty();
            if (!want) {
                // If Hanatos becomes unavailable (asset missing/mismatch), drop the device copy.
                if (resources.hanatosLut) {
                    if (!sync_before_rebuild(resources, cudaStreamOpaque, "Hanatos LUT", outError)) {
                        return false;
                    }
                    free_hanatos(resources);
                }
                if (resources.hanatosLutIntegrated) {
                    if (!sync_before_rebuild(resources, cudaStreamOpaque, "Hanatos integrated LUT", outError)) {
                        return false;
                    }
                    free_hanatos_integrated(resources);
                }
            } else {
                if (!resources.hanatosLut || resources.hanatosN != N) {
                    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
                    if (resources.hanatosLut) {
                        if (!sync_before_rebuild(resources, cudaStreamOpaque, "Hanatos LUT", outError)) {
                            return false;
                        }
                        free_hanatos(resources);
                    }

                    const size_t count = static_cast<size_t>(N) * static_cast<size_t>(N) * static_cast<size_t>(K);
                    const size_t bytes = count * sizeof(float);
                    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.hanatosLut), bytes);
                    if (err != cudaSuccess) {
                        outError = std::string("cudaMalloc(Hanatos LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                        free_hanatos(resources);
                        return false;
                    }
                    err = cudaMemcpyAsync(resources.hanatosLut, ctx.hanSpectra.data.data(), bytes, cudaMemcpyHostToDevice, stream);
                    if (err != cudaSuccess) {
                        outError = std::string("cudaMemcpyAsync(Hanatos LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                        free_hanatos(resources);
                        return false;
                    }
                    resources.hanatosN = N;
                }
            }

            // Build + upload the Hanatos LUT preintegrated with per-instance sensitivities.
            const bool sensOk =
                static_cast<int>(ws.sensB.linear.size()) == K &&
                static_cast<int>(ws.sensG.linear.size()) == K &&
                static_cast<int>(ws.sensR.linear.size()) == K;
            const bool wantIntegrated = want && sensOk;
            if (!wantIntegrated) {
                if (resources.hanatosLutIntegrated) {
                    if (!sync_before_rebuild(resources, cudaStreamOpaque, "Hanatos integrated LUT", outError)) {
                        return false;
                    }
                    free_hanatos_integrated(resources);
                }
            }
            else {
                const bool needAlloc = (!resources.hanatosLutIntegrated || resources.hanatosNIntegrated != N);
                const bool needUpload = needAlloc || resources.hanatosIntegratedBuildCounter != ws.buildCounter;
                if (needUpload) {
                    std::vector<float> cpu;
                    cpu.resize(static_cast<size_t>(N) * static_cast<size_t>(N) * 4u, 0.0f);

                    const float* lut = ctx.hanSpectra.data.data();
                    const float* sB = ws.sensB.linear.data();
                    const float* sG = ws.sensG.linear.data();
                    const float* sR = ws.sensR.linear.data();
                    const size_t stride = static_cast<size_t>(K);
                    for (int x = 0; x < N; ++x) {
                        for (int y = 0; y < N; ++y) {
                            double accB = 0.0;
                            double accG = 0.0;
                            double accR = 0.0;
                            const size_t base = (static_cast<size_t>(x) * static_cast<size_t>(N) + static_cast<size_t>(y)) * stride;
                            for (int k = 0; k < K; ++k) {
                                const float raw = lut[base + static_cast<size_t>(k)];
                                if (!std::isfinite(raw)) {
                                    continue;
                                }
                                const float e = (raw > 0.0f) ? raw : 0.0f;
                                if (!std::isfinite(e)) {
                                    continue;
                                }
                                const double e64 = static_cast<double>(e);
                                const float sb = sB[k];
                                const float sg = sG[k];
                                const float sr = sR[k];
                                if (std::isfinite(sb)) accB += e64 * static_cast<double>(sb);
                                if (std::isfinite(sg)) accG += e64 * static_cast<double>(sg);
                                if (std::isfinite(sr)) accR += e64 * static_cast<double>(sr);
                            }

                            const size_t outBase = (static_cast<size_t>(x) * static_cast<size_t>(N) + static_cast<size_t>(y)) * 4u;
                            cpu[outBase + 0] = static_cast<float>(accR);
                            cpu[outBase + 1] = static_cast<float>(accG);
                            cpu[outBase + 2] = static_cast<float>(accB);
                            cpu[outBase + 3] = 0.0f;
                        }
                    }

                    const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
                    if (needAlloc) {
                        if (resources.hanatosLutIntegrated) {
                            if (!sync_before_rebuild(resources, cudaStreamOpaque, "Hanatos integrated LUT", outError)) {
                                return false;
                            }
                            free_hanatos_integrated(resources);
                        }
                        const size_t bytes = cpu.size() * sizeof(float);
                        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.hanatosLutIntegrated), bytes);
                        if (err != cudaSuccess) {
                            outError = std::string("cudaMalloc(Hanatos integrated LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                            free_hanatos_integrated(resources);
                            return false;
                        }
                    }

                    const size_t bytes = cpu.size() * sizeof(float);
                    cudaError_t err = cudaMemcpyAsync(resources.hanatosLutIntegrated, cpu.data(), bytes, cudaMemcpyHostToDevice, stream);
                    if (err != cudaSuccess) {
                        outError = std::string("cudaMemcpyAsync(Hanatos integrated LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                        free_hanatos_integrated(resources);
                        return false;
                    }

                    resources.hanatosNIntegrated = N;
                    resources.hanatosIntegratedBuildCounter = ws.buildCounter;
                }
            }
        }

        resources.uploadedBuildCounter = ws.buildCounter;
        return true;
#endif
    }

    static bool sync_before_rebuild(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (resources.lastUseEventOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(resources.lastUseEventOpaque);
            const cudaError_t evErr = cudaEventSynchronize(ev);
            if (evErr != cudaSuccess) {
                outError = std::string("cudaEventSynchronize before ") + label + " rebuild failed: " + (cudaGetErrorString(evErr) ? cudaGetErrorString(evErr) : "(unknown)");
                return false;
            }
        }
        else {
            const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
            const cudaError_t syncErr = cudaStreamSynchronize(stream);
            if (syncErr != cudaSuccess) {
                outError = std::string("cudaStreamSynchronize before ") + label + " rebuild failed: " + (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
                return false;
            }
        }
        return true;
#endif
    }

    bool ensure_scan_lut(Resources& resources, const WorkingState& ws, bool negativeMedium, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)negativeMedium;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        const Scanner::ScannerMediumRuntime& medium = negativeMedium ? ws.negativeMediumRuntime : ws.printMediumRuntime;
        const Scanner::ScannerStaticKey& staticKey = negativeMedium ? ws.negativeStaticKey : ws.printStaticKey;

        if (!medium.tables || medium.tables->K <= 0) {
            outError = "scan LUT build requested but medium tables are unavailable";
            return false;
        }

        const std::uint32_t res = std::clamp(staticKey.lutResolution, 17u, 128u);
        const std::uint64_t expectedHash = Hash::hash_bytes(&staticKey.hash, sizeof(staticKey.hash));
        if (expectedHash == 0) {
            outError = "scan LUT staticKey hash invalid";
            return false;
        }

        Resources::DeviceSpectralLut* dst = negativeMedium ? &resources.scanNegativeLut : &resources.scanPrintLut;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
            if (dst->logXYZ && dst->res == res && dst->hash == expectedHash) {
                return true;
            }
        }

        // Build CPU LUT first (can overlap with any in-flight GPU work) before we synchronize to
        // safely retire the previous device buffer.
        const size_t sRes = static_cast<size_t>(res);
        const size_t voxels = sRes * sRes * sRes;
        const size_t count = voxels * 3u;
        std::vector<double> cpu;
        cpu.resize(count);

        for (std::uint32_t z = 0; z < res; ++z) {
            const double nz = (res > 1u) ? static_cast<double>(z) / static_cast<double>(res - 1u) : 0.0;
            for (std::uint32_t y = 0; y < res; ++y) {
                const double ny = (res > 1u) ? static_cast<double>(y) / static_cast<double>(res - 1u) : 0.0;
                for (std::uint32_t x = 0; x < res; ++x) {
                    const double nx = (res > 1u) ? static_cast<double>(x) / static_cast<double>(res - 1u) : 0.0;
                    const size_t idx = (static_cast<size_t>(z) * sRes + static_cast<size_t>(y)) * sRes + static_cast<size_t>(x);
                    const size_t base = idx * 3u;
                    const double D_norm[3] = { nx, ny, nz };
                    double logXYZ[3] = { 0.0, 0.0, 0.0 };
                    Pipeline::ScanStage::spectral_to_log_xyz(medium, D_norm, logXYZ);
                    if (!std::isfinite(logXYZ[0]) || !std::isfinite(logXYZ[1]) || !std::isfinite(logXYZ[2])) {
                        outError = "scan LUT build produced non-finite logXYZ";
                        return false;
                    }
                    cpu[base + 0] = logXYZ[0];
                    cpu[base + 1] = logXYZ[1];
                    cpu[base + 2] = logXYZ[2];
                }
            }
        }

        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
            if (dst->logXYZ && dst->res == res && dst->hash == expectedHash) {
                return true;
            }

            if (dst->logXYZ) {
                // Ensure no in-flight work can still reference the previous device LUT.
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "scan LUT", outError)) {
                    return false;
                }
                free_scan_lut(*dst);
            }

            double* dLut = nullptr;
            const size_t bytes = count * sizeof(double);
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dLut), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(scan LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_scan_lut(*dst);
                return false;
            }
            err = cudaMemcpyAsync(dLut, cpu.data(), bytes, cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) {
                outError = std::string("cudaMemcpyAsync(scan LUT) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                cudaFree(dLut);
                free_scan_lut(*dst);
                return false;
            }

            dst->logXYZ = dLut;
            dst->res = res;
            dst->hash = expectedHash;
            return true;
        }
#endif
    }

    bool ensure_scan_error_flag(Resources& resources, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        if (!resources.scanErrorFlag) {
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.scanErrorFlag), sizeof(int));
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(scan error flag) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                resources.scanErrorFlag = nullptr;
                return false;
            }
        }
        if (!resources.scanErrorHost) {
            cudaError_t err = cudaMallocHost(reinterpret_cast<void**>(&resources.scanErrorHost), sizeof(int));
            if (err != cudaSuccess) {
                resources.scanErrorHost = nullptr;
            }
        }
        if (!resources.scanErrorEventOpaque) {
            cudaEvent_t ev = nullptr;
            cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            if (err == cudaSuccess && ev) {
                resources.scanErrorEventOpaque = reinterpret_cast<void*>(ev);
            }
        }
        return true;
#endif
    }

    bool ensure_optics_scratch(Resources& resources, int width, int height, bool needUnsharpScratch, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)width;
        (void)height;
        (void)needUnsharpScratch;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        if (width <= 0 || height <= 0) {
            outError = "optics scratch dimensions invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        const bool dimsMatch = (resources.scannerScratch.width == width && resources.scannerScratch.height == height);
        const bool haveBase = resources.scannerScratch.rgbR && resources.scannerScratch.rgbG && resources.scannerScratch.rgbB && resources.scannerScratch.tmp;

        if (!dimsMatch || !haveBase) {
            if (resources.scannerScratch.rgbR || resources.scannerScratch.tmp || resources.scannerScratch.blurred) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "optics scratch", outError)) {
                    return false;
                }
            }
            free_optics_scratch(resources.scannerScratch);

            const size_t n = static_cast<size_t>(width) * static_cast<size_t>(height);
            const size_t bytes = n * sizeof(float);
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.scannerScratch.rgbR), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(scannerScratch.rgbR) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_optics_scratch(resources.scannerScratch);
                return false;
            }
            err = cudaMalloc(reinterpret_cast<void**>(&resources.scannerScratch.rgbG), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(scannerScratch.rgbG) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_optics_scratch(resources.scannerScratch);
                return false;
            }
            err = cudaMalloc(reinterpret_cast<void**>(&resources.scannerScratch.rgbB), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(scannerScratch.rgbB) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_optics_scratch(resources.scannerScratch);
                return false;
            }
            err = cudaMalloc(reinterpret_cast<void**>(&resources.scannerScratch.tmp), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(scannerScratch.tmp) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                free_optics_scratch(resources.scannerScratch);
                return false;
            }

            resources.scannerScratch.width = width;
            resources.scannerScratch.height = height;
        }

        if (needUnsharpScratch) {
            if (!resources.scannerScratch.blurred) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "unsharp scratch", outError)) {
                    return false;
                }
                const size_t n = static_cast<size_t>(resources.scannerScratch.width) * static_cast<size_t>(resources.scannerScratch.height);
                const size_t bytes = n * sizeof(float);
                const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&resources.scannerScratch.blurred), bytes);
                if (err != cudaSuccess) {
                    outError = std::string("cudaMalloc(scannerScratch.blurred) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }
            }
        }
        else {
            if (resources.scannerScratch.blurred) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "unsharp scratch free", outError)) {
                    return false;
                }
                cudaFree(resources.scannerScratch.blurred);
                resources.scannerScratch.blurred = nullptr;
            }
        }

        return true;
#endif
    }

    bool ensure_spatial_dir_scratch(Resources& resources, int width, int height, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)width;
        (void)height;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        if (width <= 0 || height <= 0) {
            outError = "spatial DIR scratch dimensions invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        Resources::DeviceSpatialDirScratch& scratch = resources.spatialDirScratch;
        if (scratch.width == width && scratch.height == height && scratch.corrY && scratch.corrM && scratch.corrC && scratch.tmp) {
            return true;
        }

        if (scratch.corrY || scratch.corrM || scratch.corrC || scratch.tmp) {
            if (!sync_before_rebuild(resources, cudaStreamOpaque, "spatial DIR scratch", outError)) {
                return false;
            }
            free_spatial_dir_scratch(scratch);
        }

        const size_t total = static_cast<size_t>(width) * static_cast<size_t>(height);
        const size_t bytes = total * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&scratch.corrY), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(spatial DIR corrY) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_spatial_dir_scratch(scratch);
            return false;
        }
        err = cudaMalloc(reinterpret_cast<void**>(&scratch.corrM), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(spatial DIR corrM) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_spatial_dir_scratch(scratch);
            return false;
        }
        err = cudaMalloc(reinterpret_cast<void**>(&scratch.corrC), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(spatial DIR corrC) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_spatial_dir_scratch(scratch);
            return false;
        }
        err = cudaMalloc(reinterpret_cast<void**>(&scratch.tmp), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(spatial DIR tmp) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_spatial_dir_scratch(scratch);
            return false;
        }

        scratch.width = width;
        scratch.height = height;
        return true;
#endif
    }

    bool ensure_spatial_dir_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)kernel;
        (void)sigma;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        if (!(std::isfinite(sigma)) || sigma <= 0.0f) {
            if (kernel.weights) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "spatial DIR kernel free", outError)) {
                    return false;
                }
            }
            free_gaussian_kernel(kernel);
            return true;
        }

        const int radiusRaw = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
        const int radius = std::min(radiusRaw, 75);
        if (radius <= 0) {
            if (kernel.weights) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "spatial DIR kernel free", outError)) {
                    return false;
                }
            }
            free_gaussian_kernel(kernel);
            return true;
        }

        const bool same = (kernel.weights && kernel.radius == radius && std::fabs(kernel.sigma - sigma) <= 1e-6f);
        if (same) {
            return true;
        }

        std::vector<float> cpu;
        cpu.resize(static_cast<size_t>(2 * radius + 1));
        const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma) * 2.0;
        double wsum = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            const double w = std::exp(-(static_cast<double>(i * i)) / s2);
            cpu[static_cast<size_t>(i + radius)] = static_cast<float>(w);
            wsum += w;
        }
        const double invW = (wsum != 0.0) ? (1.0 / wsum) : 0.0;
        for (float& w : cpu) {
            w = static_cast<float>(static_cast<double>(w) * invW);
        }

        if (kernel.weights) {
            if (!sync_before_rebuild(resources, cudaStreamOpaque, "spatial DIR kernel", outError)) {
                return false;
            }
            free_gaussian_kernel(kernel);
        }

        const size_t bytes = cpu.size() * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&kernel.weights), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(spatial DIR kernel) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_gaussian_kernel(kernel);
            return false;
        }
        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        err = cudaMemcpyAsync(kernel.weights, cpu.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(spatial DIR kernel) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            cudaFree(kernel.weights);
            free_gaussian_kernel(kernel);
            return false;
        }

        kernel.radius = radius;
        kernel.sigma = sigma;
        return true;
#endif
    }

    bool ensure_gaussian_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)kernel;
        (void)sigma;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        if (!(std::isfinite(sigma)) || sigma <= 0.0f) {
            if (kernel.weights) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "gaussian kernel free", outError)) {
                    return false;
                }
            }
            free_gaussian_kernel(kernel);
            return true;
        }

        const int radiusRaw = JuicerGaussian::scipy_gaussian_radius(sigma, 4.0f);
        const int radius = std::min(radiusRaw, 75);
        if (radius <= 0) {
            if (kernel.weights) {
                if (!sync_before_rebuild(resources, cudaStreamOpaque, "gaussian kernel free", outError)) {
                    return false;
                }
            }
            free_gaussian_kernel(kernel);
            return true;
        }

        const bool same = (kernel.weights && kernel.radius == radius && std::fabs(kernel.sigma - sigma) <= 1e-6f);
        if (same) {
            return true;
        }

        std::vector<float> cpu;
        cpu.resize(static_cast<size_t>(2 * radius + 1));
        const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma) * 2.0;
        double wsum = 0.0;
        for (int i = -radius; i <= radius; ++i) {
            const double w = std::exp(-(static_cast<double>(i * i)) / s2);
            cpu[static_cast<size_t>(i + radius)] = static_cast<float>(w);
            wsum += w;
        }
        const double invW = (wsum != 0.0) ? (1.0 / wsum) : 0.0;
        for (float& w : cpu) {
            w = static_cast<float>(static_cast<double>(w) * invW);
        }

        if (kernel.weights) {
            if (!sync_before_rebuild(resources, cudaStreamOpaque, "gaussian kernel", outError)) {
                return false;
            }
            free_gaussian_kernel(kernel);
        }

        const size_t bytes = cpu.size() * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&kernel.weights), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(gaussian kernel) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            free_gaussian_kernel(kernel);
            return false;
        }
        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        err = cudaMemcpyAsync(kernel.weights, cpu.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(gaussian kernel) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            cudaFree(kernel.weights);
            free_gaussian_kernel(kernel);
            return false;
        }

        kernel.radius = radius;
        kernel.sigma = sigma;
        return true;
#endif
    }

    bool ensure_print_illuminant_filtered(
        Resources& resources,
        const WorkingState& ws,
        const Print::Runtime& prt,
        const Print::Params& prm,
        void* cudaStreamOpaque,
        std::string& outError)
    {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)prt;
        (void)prm;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        {
            int cur = -1;
            const cudaError_t devErr = cudaGetDevice(&cur);
            if (devErr != cudaSuccess || cur < 0) {
                outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
                return false;
            }
            if (resources.deviceId < 0) {
                resources.deviceId = cur;
            }
            if (resources.deviceId != cur) {
                outError = "CUDA device mismatch for cached resources";
                return false;
            }
        }

        const int K = Spectral::gShape.K;
        if (K <= 0) {
            outError = "spectral shape invalid";
            return false;
        }

        // Normalize filter step keys for cache parity with ExposePrintStage.
        const float yKey = std::isfinite(prm.yFilter) ? prm.yFilter : 0.0f;
        const float mKey = std::isfinite(prm.mFilter) ? prm.mFilter : 0.0f;
        const float cKey = 0.0f;

        const bool cached =
            resources.printIllumFiltered &&
            resources.printIllumK == K &&
            resources.printIllumShapeK == K &&
            resources.printIllumBuildCounter == ws.buildCounter &&
            resources.printIllumRuntimePtr == &prt &&
            resources.printIllumYShiftSteps == yKey &&
            resources.printIllumMShiftSteps == mKey &&
            resources.printIllumCShiftSteps == cKey;
        if (cached) {
            return true;
        }

        auto blend = [](float curveVal, float normalizedAmount) -> float {
            const float a = std::isfinite(normalizedAmount) ? normalizedAmount : 0.0f;
            return 1.0f - (1.0f - curveVal) * a;
        };
        auto compose_amount = [](float neutralAmount, float deltaSteps) -> float {
            const float neutral = std::isfinite(neutralAmount)
                ? std::clamp(neutralAmount, 0.0f, 1.0f)
                : 0.0f;
            float ds = std::isfinite(deltaSteps) ? deltaSteps : 0.0f;
            ds = std::clamp(ds, -Print::kEnlargerSteps, Print::kEnlargerSteps);
            const float totalSteps = neutral * Print::kEnlargerSteps + ds;
            return totalSteps / Print::kEnlargerSteps;
        };

        const float yAmount = compose_amount(prt.neutralY, yKey);
        const float mAmount = compose_amount(prt.neutralM, mKey);
        const float cAmount = compose_amount(prt.neutralC, cKey);

        std::vector<float> cpu;
        cpu.resize(static_cast<size_t>(K));
        for (int i = 0; i < K; ++i) {
            const float Ee = (prt.illumEnlarger.linear.size() > static_cast<size_t>(i))
                ? prt.illumEnlarger.linear[static_cast<size_t>(i)]
                : 1.0f;
            const float fY = blend(
                (prt.filterY.linear.size() > static_cast<size_t>(i)) ? prt.filterY.linear[static_cast<size_t>(i)] : 1.0f,
                yAmount);
            const float fM = blend(
                (prt.filterM.linear.size() > static_cast<size_t>(i)) ? prt.filterM.linear[static_cast<size_t>(i)] : 1.0f,
                mAmount);
            const float fC = blend(
                (prt.filterC.linear.size() > static_cast<size_t>(i)) ? prt.filterC.linear[static_cast<size_t>(i)] : 1.0f,
                cAmount);
            cpu[static_cast<size_t>(i)] = Ee * (fY * fM * fC);
        }

        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;

        if (resources.printIllumFiltered) {
            if (!sync_before_rebuild(resources, cudaStreamOpaque, "print illuminant filtered", outError)) {
                return false;
            }
            cudaFree(resources.printIllumFiltered);
            resources.printIllumFiltered = nullptr;
            resources.printIllumK = 0;
        }

        float* dIllum = nullptr;
        const size_t bytes = static_cast<size_t>(K) * sizeof(float);
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&dIllum), bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(print illuminant filtered) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }
        err = cudaMemcpyAsync(dIllum, cpu.data(), bytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            outError = std::string("cudaMemcpyAsync(print illuminant filtered) failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            cudaFree(dIllum);
            return false;
        }

        resources.printIllumFiltered = dIllum;
        resources.printIllumK = K;
        resources.printIllumYShiftSteps = yKey;
        resources.printIllumMShiftSteps = mKey;
        resources.printIllumCShiftSteps = cKey;
        resources.printIllumShapeK = K;
        resources.printIllumBuildCounter = ws.buildCounter;
        resources.printIllumRuntimePtr = &prt;
        return true;
#endif
    }

    bool validate_density_primitives(Resources& resources, const WorkingState& ws, void* cudaStreamOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        if (resources.validatedBuildCounter == ws.buildCounter && ws.buildCounter != 0) {
            return true;
        }

        // Lightweight parity probe: sample density curves at a few canonical logE values and compare
        // against the CPU implementation (FilmProcessing.h). This is a development-only check and is
        // expected to be compiled out in shipping builds.
        static constexpr float kLogE[5] = { -3.0f, -1.0f, 0.0f, 2.0f, 4.0f };
        float outB[5] = {};
        float outG[5] = {};
        float outR[5] = {};

        cudaError_t err = ::juicer_cuda_probe_density_curve(resources.densB.x, resources.densB.y, resources.densB.n,
            ws.gammaFactorB, kLogE, 5, outB, cudaStreamOpaque);
        if (err != cudaSuccess) {
            outError = std::string("probe densB failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }
        err = ::juicer_cuda_probe_density_curve(resources.densG.x, resources.densG.y, resources.densG.n,
            ws.gammaFactorG, kLogE, 5, outG, cudaStreamOpaque);
        if (err != cudaSuccess) {
            outError = std::string("probe densG failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }
        err = ::juicer_cuda_probe_density_curve(resources.densR.x, resources.densR.y, resources.densR.n,
            ws.gammaFactorR, kLogE, 5, outR, cudaStreamOpaque);
        if (err != cudaSuccess) {
            outError = std::string("probe densR failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }

        float maxAbs = 0.0f;
        for (int i = 0; i < 5; ++i) {
            const float cpuB = Spectral::sample_density_at_logE(ws.densB, kLogE[i], ws.gammaFactorB);
            const float cpuG = Spectral::sample_density_at_logE(ws.densG, kLogE[i], ws.gammaFactorG);
            const float cpuR = Spectral::sample_density_at_logE(ws.densR, kLogE[i], ws.gammaFactorR);

            if (!std::isfinite(cpuB) || !std::isfinite(cpuG) || !std::isfinite(cpuR) ||
                !std::isfinite(outB[i]) || !std::isfinite(outG[i]) || !std::isfinite(outR[i])) {
                outError = "density probe produced non-finite values";
                return false;
            }

            maxAbs = std::max(maxAbs, std::fabs(outB[i] - cpuB));
            maxAbs = std::max(maxAbs, std::fabs(outG[i] - cpuG));
            maxAbs = std::max(maxAbs, std::fabs(outR[i] - cpuR));
        }

        // This primitive is expected to match exactly (same math, float-only) within a tiny epsilon.
        if (!(maxAbs <= 1e-6f)) {
            outError = "density probe mismatch: maxAbs=" + std::to_string(maxAbs);
            return false;
        }

        // Validate +/-inf sanitization before sampling (DevelopFilmStage parity).
        {
            const float kLogEInf[3] = {
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                0.0f
            };
            float outBInf[3] = {};
            float outGInf[3] = {};
            float outRInf[3] = {};

            auto cpu_sanitize_inf = [](float logE, const Spectral::Curve& c) -> float {
                if (std::isfinite(logE) || std::isnan(logE)) {
                    return logE;
                }
                float xmin = 0.0f, xmax = 0.0f;
                const size_t n = c.lambda_nm.size();
                if (n == 0 || c.linear.size() != n) {
                    return logE;
                }
                size_t begin = 0;
                while (begin < n && !std::isfinite(c.lambda_nm[begin])) {
                    ++begin;
                }
                if (begin == n) {
                    return logE;
                }
                size_t end = n - 1;
                while (end > begin && !std::isfinite(c.lambda_nm[end])) {
                    --end;
                }
                xmin = c.lambda_nm[begin];
                xmax = c.lambda_nm[end];
                if (!std::isfinite(xmin) || !std::isfinite(xmax) || !(xmax >= xmin)) {
                    return logE;
                }
                return (logE > 0.0f) ? xmax : xmin;
            };

            cudaError_t err = ::juicer_cuda_probe_density_curve_sanitize_inf(resources.densB.x, resources.densB.y, resources.densB.n,
                ws.gammaFactorB, kLogEInf, 3, outBInf, cudaStreamOpaque);
            if (err != cudaSuccess) {
                outError = std::string("probe densB sanitize_inf failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                return false;
            }
            err = ::juicer_cuda_probe_density_curve_sanitize_inf(resources.densG.x, resources.densG.y, resources.densG.n,
                ws.gammaFactorG, kLogEInf, 3, outGInf, cudaStreamOpaque);
            if (err != cudaSuccess) {
                outError = std::string("probe densG sanitize_inf failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                return false;
            }
            err = ::juicer_cuda_probe_density_curve_sanitize_inf(resources.densR.x, resources.densR.y, resources.densR.n,
                ws.gammaFactorR, kLogEInf, 3, outRInf, cudaStreamOpaque);
            if (err != cudaSuccess) {
                outError = std::string("probe densR sanitize_inf failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                return false;
            }

            float maxDiff = 0.0f;
            for (int i = 0; i < 3; ++i) {
                const float cpuB = Spectral::sample_density_at_logE(ws.densB, cpu_sanitize_inf(kLogEInf[i], ws.densB), ws.gammaFactorB);
                const float cpuG = Spectral::sample_density_at_logE(ws.densG, cpu_sanitize_inf(kLogEInf[i], ws.densG), ws.gammaFactorG);
                const float cpuR = Spectral::sample_density_at_logE(ws.densR, cpu_sanitize_inf(kLogEInf[i], ws.densR), ws.gammaFactorR);
                maxDiff = std::max(maxDiff, std::fabs(outBInf[i] - cpuB));
                maxDiff = std::max(maxDiff, std::fabs(outGInf[i] - cpuG));
                maxDiff = std::max(maxDiff, std::fabs(outRInf[i] - cpuR));
            }
            if (!(maxDiff <= 1e-6f)) {
                outError = "sanitize_inf_logE mismatch: maxAbs=" + std::to_string(maxDiff);
                return false;
            }
        }

        // Validate film log-raw computation parity (DevelopFilmStage::compute_log_raw).
        {
            const float samples[][3] = {
                { 0.184f, 0.184f, 0.184f },
                { -0.5f, 0.0f, 2.0f },
                { std::numeric_limits<float>::quiet_NaN(), 1.0f, std::numeric_limits<float>::infinity() }
            };

            for (const auto& filmRaw : samples) {
                float gpuLogRaw[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_film_log_raw(filmRaw, gpuLogRaw, cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("film log-raw probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                constexpr float kEps = 1e-10f;
                float cpuLogRaw[3] = {
                    std::log10(std::fmax(filmRaw[0], 0.0f) + kEps),
                    std::log10(std::fmax(filmRaw[1], 0.0f) + kEps),
                    std::log10(std::fmax(filmRaw[2], 0.0f) + kEps)
                };

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    if (std::isfinite(cpuLogRaw[c]) && std::isfinite(gpuLogRaw[c])) {
                        maxDiff = std::max(maxDiff, std::fabs(cpuLogRaw[c] - gpuLogRaw[c]));
                    }
                    else if (std::isnan(cpuLogRaw[c]) != std::isnan(gpuLogRaw[c])) {
                        outError = "film log-raw NaN mismatch";
                        return false;
                    }
                    else if (std::isinf(cpuLogRaw[c]) != std::isinf(gpuLogRaw[c])) {
                        outError = "film log-raw inf mismatch";
                        return false;
                    }
                }
                if (!(maxDiff <= 1e-6f)) {
                    outError = "film log-raw mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // Validate input RGB -> DWG conversion against the CPU implementation.
        {
            const int inputColorSpaceIndex = Spectral::inputColorSpaceToIndex(ws.filmRaw.inputColorSpace);
            const int applyDecode = ws.filmRaw.applyCctfDecoding ? 1 : 0;
            const int applyAdapt = ws.filmRaw.applyInputChromaticAdapt ? 1 : 0;

            const float samplesIn[][3] = {
                { 0.184f, 0.184f, 0.184f },
                { 1.2f, -0.1f, 0.5f },   // includes negative input channel (must match CPU sanitize-only behavior)
                { 0.0f, 0.5f, 2.0f }
            };

            for (const auto& rgbIn : samplesIn) {
                float gpuDWG[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_convert_input_to_DWG(
                    rgbIn,
                    inputColorSpaceIndex,
                    applyDecode,
                    applyAdapt,
                    ws.filmRaw.inputRGBToXYZ.m,
                    ws.filmRaw.inputXYZAdapt.m,
                    gpuDWG,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("convert_input_rgb_to_DWG probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                float cpuDWG[3] = { 0.0f, 0.0f, 0.0f };
                Spectral::convert_input_rgb_to_DWG(ws.filmRaw, rgbIn, cpuDWG);

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    maxDiff = std::max(maxDiff, std::fabs(gpuDWG[c] - cpuDWG[c]));
                }
                if (!(maxDiff <= 2e-5f)) {
                    outError = "convert_input_rgb_to_DWG mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // Validate DIR "clamp corrected logE to curve domain" behavior (Couplers::apply_runtime_logE_with_curves clamp_to).
        {
            auto cpu_clamp_to = [](float le, const Spectral::Curve& c) -> float {
                if (c.lambda_nm.empty()) return le;
                const float xmin = c.lambda_nm.front();
                const float xmax = c.lambda_nm.back();
                if (!std::isfinite(le)) return xmin;
                return std::min(std::max(le, xmin), xmax);
            };

            const float samples[] = {
                -1000.0f,
                1000.0f,
                -std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::infinity(),
                std::numeric_limits<float>::quiet_NaN(),
                0.0f
            };

            for (float le : samples) {
                float outB = 0.0f, outG = 0.0f, outR = 0.0f;
                cudaError_t err = ::juicer_cuda_probe_clamp_logE_to_curve_domain(resources.densB.x, resources.densB.n, le, &outB, cudaStreamOpaque);
                if (err != cudaSuccess) { outError = std::string("clamp_to densB probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)"); return false; }
                err = ::juicer_cuda_probe_clamp_logE_to_curve_domain(resources.densG.x, resources.densG.n, le, &outG, cudaStreamOpaque);
                if (err != cudaSuccess) { outError = std::string("clamp_to densG probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)"); return false; }
                err = ::juicer_cuda_probe_clamp_logE_to_curve_domain(resources.densR.x, resources.densR.n, le, &outR, cudaStreamOpaque);
                if (err != cudaSuccess) { outError = std::string("clamp_to densR probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)"); return false; }

                const float cpuB = cpu_clamp_to(le, ws.densB);
                const float cpuG = cpu_clamp_to(le, ws.densG);
                const float cpuR = cpu_clamp_to(le, ws.densR);

                auto eq = [](float a, float b) -> bool {
                    if (std::isnan(a) && std::isnan(b)) return true;
                    return a == b;
                };

                if (!eq(outB, cpuB) || !eq(outG, cpuG) || !eq(outR, cpuR)) {
                    outError = "DIR clamp_to mismatch";
                    return false;
                }
            }
        }

        // Validate scan-stage spectral_to_log_xyz primitive (negative medium) against CPU.
        if (resources.scanNegative.tables.K == Spectral::gShape.K &&
            resources.scanNegative.tables.epsC &&
            resources.scanNegative.tables.Ax)
        {
            const double samples[][3] = {
                { 0.0, 0.0, 0.0 },
                { 0.5, 0.5, 0.5 },
                { 1.0, 1.0, 1.0 },
                { 0.1, 0.7, 0.3 }
            };

            for (const auto& D_norm : samples) {
                double gpuLogXYZ[3] = { 0.0, 0.0, 0.0 };
                const cudaError_t err = ::juicer_cuda_probe_scan_spectral_to_log_xyz(
                    D_norm,
                    resources.scanNegative.mediumIsNegative,
                    resources.scanNegative.min_cmy,
                    resources.scanNegative.inv_max_cmy,
                    resources.scanNegative.tables.epsC,
                    resources.scanNegative.tables.epsM,
                    resources.scanNegative.tables.epsY,
                    resources.scanNegative.tables.Ax,
                    resources.scanNegative.tables.Ay,
                    resources.scanNegative.tables.Az,
                    resources.scanNegative.tables.baseMin,
                    resources.scanNegative.tables.K,
                    resources.scanNegative.tables.hasBaseline,
                    resources.scanNegative.tables.invYn,
                    gpuLogXYZ,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("scan logXYZ probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                double cpuLogXYZ[3] = { 0.0, 0.0, 0.0 };
                Pipeline::ScanStage::spectral_to_log_xyz(ws.negativeMediumRuntime, D_norm, cpuLogXYZ);

                double maxDiff = 0.0;
                for (int c = 0; c < 3; ++c) {
                    if (!std::isfinite(cpuLogXYZ[c]) || !std::isfinite(gpuLogXYZ[c])) {
                        outError = "scan logXYZ produced non-finite values";
                        return false;
                    }
                    maxDiff = std::max(maxDiff, std::fabs(cpuLogXYZ[c] - gpuLogXYZ[c]));
                }
                if (!(maxDiff <= 1e-4)) {
                    outError = "scan logXYZ mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // Validate scan-stage spectral_to_log_xyz primitive (print medium) against CPU when available.
        if (ws.printMediumRuntime.tables &&
            resources.scanPrint.tables.K == Spectral::gShape.K &&
            resources.scanPrint.tables.epsC &&
            resources.scanPrint.tables.Ax)
        {
            const double samples[][3] = {
                { 0.0, 0.0, 0.0 },
                { 0.5, 0.5, 0.5 },
                { 1.0, 1.0, 1.0 },
                { 0.1, 0.7, 0.3 }
            };

            for (const auto& D_norm : samples) {
                double gpuLogXYZ[3] = { 0.0, 0.0, 0.0 };
                const cudaError_t err = ::juicer_cuda_probe_scan_spectral_to_log_xyz(
                    D_norm,
                    resources.scanPrint.mediumIsNegative,
                    resources.scanPrint.min_cmy,
                    resources.scanPrint.inv_max_cmy,
                    resources.scanPrint.tables.epsC,
                    resources.scanPrint.tables.epsM,
                    resources.scanPrint.tables.epsY,
                    resources.scanPrint.tables.Ax,
                    resources.scanPrint.tables.Ay,
                    resources.scanPrint.tables.Az,
                    resources.scanPrint.tables.baseMin,
                    resources.scanPrint.tables.K,
                    resources.scanPrint.tables.hasBaseline,
                    resources.scanPrint.tables.invYn,
                    gpuLogXYZ,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("scan logXYZ (print) probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                double cpuLogXYZ[3] = { 0.0, 0.0, 0.0 };
                Pipeline::ScanStage::spectral_to_log_xyz(ws.printMediumRuntime, D_norm, cpuLogXYZ);

                double maxDiff = 0.0;
                for (int c = 0; c < 3; ++c) {
                    if (!std::isfinite(cpuLogXYZ[c]) || !std::isfinite(gpuLogXYZ[c])) {
                        outError = "scan logXYZ (print) produced non-finite values";
                        return false;
                    }
                    maxDiff = std::max(maxDiff, std::fabs(cpuLogXYZ[c] - gpuLogXYZ[c]));
                }
                if (!(maxDiff <= 1e-4)) {
                    outError = "scan logXYZ (print) mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // Validate Mallett (tables + S_inv) exposure primitive against CPU (forced non-Hanatos path).
        if (ws.spdReady && resources.tablesAx && resources.tablesAy && resources.tablesAz && resources.tablesK == Spectral::gShape.K) {
            const float samples[][3] = {
                { 0.184f, 0.184f, 0.184f }, // mid-gray
                { 0.9f, 0.1f, 0.1f },       // red-ish
                { 0.05f, 0.2f, 0.9f }       // blue-ish
            };

            for (const auto& rgbDWG : samples) {
                float gpuE[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_tables_layer_exposures(
                    rgbDWG,
                    resources.spdSInv,
                    resources.refIllumWhiteXYZ,
                    resources.tablesAx,
                    resources.tablesAy,
                    resources.tablesAz,
                    resources.tablesK,
                    resources.sensB.y,
                    resources.sensG.y,
                    resources.sensR.y,
                    1.0f,
                    1,
                    gpuE,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("tables exposure probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                std::vector<float> Ee;
                Spectral::reconstruct_Ee_from_DWG_RGB_with_tables(rgbDWG, ws.tablesRef, ws.spdSInv, Ee);
                float cpuE[3] = { 0.0f, 0.0f, 0.0f };
                Spectral::layerExposures_from_sceneSPD_with_curves(Ee, ws.sensB, ws.sensG, ws.sensR, cpuE, 1.0f, true);

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    maxDiff = std::max(maxDiff, std::fabs(gpuE[c] - cpuE[c]));
                }
                if (!(maxDiff <= 2e-4f)) {
                    outError = "tables exposure mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        // If Hanatos is available and uploaded, validate the SPD→exposure integration primitive on a
        // few representative DWG RGB samples.
        if (resources.hanatosLut && resources.hanatosN > 0) {
            const float refWhite[3] = {
                ws.tablesRef.refIllumWhiteXYZ[0],
                ws.tablesRef.refIllumWhiteXYZ[1],
                ws.tablesRef.refIllumWhiteXYZ[2]
            };
            const float samples[][3] = {
                { 0.184f, 0.184f, 0.184f }, // mid-gray
                { 0.9f, 0.1f, 0.1f },       // red-ish
                { 0.05f, 0.2f, 0.9f }       // blue-ish
            };

            for (const auto& rgbDWG : samples) {
                float gpuE[3] = { 0.0f, 0.0f, 0.0f };
                const cudaError_t err = ::juicer_cuda_probe_hanatos_layer_exposures(
                    rgbDWG,
                    resources.hanatosLut,
                    resources.hanatosN,
                    refWhite,
                    resources.sensB.y,
                    resources.sensG.y,
                    resources.sensR.y,
                    Spectral::kNumSamples,
                    1.0f,
                    0,
                    gpuE,
                    cudaStreamOpaque);
                if (err != cudaSuccess) {
                    outError = std::string("Hanatos exposure probe failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                    return false;
                }

                // CPU reference: match Spectral::rgbDWG_to_layerExposures_from_tables_with_curves
                // semantics for Hanatos path (applyDeltaLambda = false).
                std::vector<float> Ee;
                Spectral::reconstruct_Ee_from_DWG_RGB_hanatos(rgbDWG, Ee, refWhite);
                float cpuE[3] = { 0.0f, 0.0f, 0.0f };
                Spectral::layerExposures_from_sceneSPD_with_curves(Ee, ws.sensB, ws.sensG, ws.sensR, cpuE, 1.0f, false);

                float maxDiff = 0.0f;
                for (int c = 0; c < 3; ++c) {
                    maxDiff = std::max(maxDiff, std::fabs(gpuE[c] - cpuE[c]));
                }
                if (!(maxDiff <= 2e-4f)) {
                    outError = "Hanatos exposure mismatch: maxAbs=" + std::to_string(maxDiff);
                    return false;
                }
            }
        }

        resources.validatedBuildCounter = ws.buildCounter;
        return true;
#endif
    }

    bool validate_print_primitives(
        Resources& resources,
        const WorkingState& ws,
        const Print::Runtime& prt,
        const Print::Params& prm,
        float midgrayFactor,
        void* cudaStreamOpaque,
        std::string& outError)
    {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ws;
        (void)prt;
        (void)prm;
        (void)midgrayFactor;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        // Cache: avoid re-running the (expensive) CPU-vs-GPU probe every frame. This is for debug-only
        // validation; correctness is still enforced when the key changes.
        std::uint64_t paramsHash = Hash::kFnvOffset;
        auto hash_u32 = [&](std::uint32_t v) {
            Hash::hash_bytes_update(paramsHash, &v, sizeof(v));
        };
        auto hash_u64 = [&](std::uint64_t v) {
            Hash::hash_bytes_update(paramsHash, &v, sizeof(v));
        };
        auto hash_f32 = [&](float v) {
            float vv = v;
            if (vv == 0.0f) {
                vv = 0.0f; // canonicalize -0.0f to +0.0f
            }
            std::uint32_t bits = 0;
            std::memcpy(&bits, &vv, sizeof(bits));
            hash_u32(bits);
        };
        auto hash_bool = [&](bool v) {
            const std::uint32_t b = v ? 1U : 0U;
            hash_u32(b);
        };

        hash_u64(ws.buildCounter);
        hash_f32(prm.exposure);
        hash_f32(prm.preflashExposure);
        hash_f32(prm.yFilter);
        hash_f32(prm.mFilter);
        hash_bool(prm.exposureCompensationEnabled);
        hash_f32(prm.exposureCompensationScale);
        hash_f32(midgrayFactor);

        {
            std::lock_guard<std::mutex> lock(resources.m);
            if (resources.validatedPrintBuildCounter == ws.buildCounter &&
                resources.validatedPrintParamsHash == paramsHash) {
                outError.clear();
                return true;
            }
        }

        // Ensure current print illuminant (filtered by print params) is available on device.
        {
            std::string illumError;
            if (!ensure_print_illuminant_filtered(resources, ws, prt, prm, cudaStreamOpaque, illumError)) {
                outError = std::string("ensure_print_illuminant_filtered failed: ") + illumError;
                return false;
            }
        }

        static constexpr float kNegCmySamples[][3] = {
            { 0.0f, 0.0f, 0.0f },
            { 0.2f, 0.3f, 0.4f },
            { 0.5f, 0.5f, 0.5f },
            { 1.0f, 0.8f, 0.6f },
            { 2.0f, 1.5f, 1.0f },
            { 3.0f, 3.0f, 3.0f }
        };
        constexpr int kCount = static_cast<int>(sizeof(kNegCmySamples) / sizeof(kNegCmySamples[0]));

        const float kMid = (std::isfinite(midgrayFactor) && midgrayFactor > 0.0f) ? midgrayFactor : 1.0f;

        float cpuOut[kCount * 3] = {};
        Pipeline::PrintPipelineScratch scratch{};
        for (int i = 0; i < kCount; ++i) {
            Pipeline::ExposePrintInputs in{};
            in.printRuntime = &prt;
            in.printParams = &prm;
            in.negativeDensity.v[0] = kNegCmySamples[i][0];
            in.negativeDensity.v[1] = kNegCmySamples[i][1];
            in.negativeDensity.v[2] = kNegCmySamples[i][2];
            in.midgrayFactor = kMid;

            Pipeline::ExposePrintOutputs ex{};
            if (!Pipeline::ExposePrintStage::run(ws, in, ex, scratch)) {
                outError = "CPU ExposePrintStage::run failed";
                return false;
            }

            Pipeline::DevelopPrintInputs din{};
            din.printRuntime = &prt;
            din.printLogRaw = ex.printLogRaw;
            Pipeline::DevelopPrintOutputs dout{};
            if (!Pipeline::DevelopPrintStage::run(din, dout)) {
                outError = "CPU DevelopPrintStage::run failed";
                return false;
            }

            cpuOut[i * 3 + 0] = dout.printDensity.v[0];
            cpuOut[i * 3 + 1] = dout.printDensity.v[1];
            cpuOut[i * 3 + 2] = dout.printDensity.v[2];
        }

        JuicerCuda::Phase3RunParams run{};
        run.printActive = 1;
        run.printExposure = prm.exposure;
        run.printPreflashExposure = prm.preflashExposure;
        run.printMidgrayFactor = kMid;

        float inNeg[kCount * 3] = {};
        for (int i = 0; i < kCount; ++i) {
            inNeg[i * 3 + 0] = kNegCmySamples[i][0];
            inNeg[i * 3 + 1] = kNegCmySamples[i][1];
            inNeg[i * 3 + 2] = kNegCmySamples[i][2];
        }
        float gpuOut[kCount * 3] = {};

        {
            std::lock_guard<std::mutex> lock(resources.m);
            run.negTables.epsC = resources.scanNegative.tables.epsC;
            run.negTables.epsM = resources.scanNegative.tables.epsM;
            run.negTables.epsY = resources.scanNegative.tables.epsY;
            run.negTables.baseMin = resources.scanNegative.tables.baseMin;
            run.negTables.K = resources.scanNegative.tables.K;
            run.negTables.hasBaseline = resources.scanNegative.tables.hasBaseline;
            run.negTables.invYn = resources.scanNegative.tables.invYn;
            run.negTables.mediumIsNegative = resources.scanNegative.mediumIsNegative;
            for (int j = 0; j < 3; ++j) {
                run.negTables.min_cmy[j] = resources.scanNegative.min_cmy[j];
                run.negTables.inv_max_cmy[j] = resources.scanNegative.inv_max_cmy[j];
            }

            run.printIllumFiltered = resources.printIllumFiltered;
            run.printIllumK = resources.printIllumK;
            run.printSensC = { resources.printSensC.x, resources.printSensC.y, resources.printSensC.n };
            run.printSensM = { resources.printSensM.x, resources.printSensM.y, resources.printSensM.n };
            run.printSensY = { resources.printSensY.x, resources.printSensY.y, resources.printSensY.n };
            run.printDcC = { resources.printDcC.x, resources.printDcC.y, resources.printDcC.n };
            run.printDcM = { resources.printDcM.x, resources.printDcM.y, resources.printDcM.n };
            run.printDcY = { resources.printDcY.x, resources.printDcY.y, resources.printDcY.n };
            run.printGammaC = resources.printGammaC;
            run.printGammaM = resources.printGammaM;
            run.printGammaY = resources.printGammaY;
            for (int j = 0; j < 3; ++j) {
                run.printPreflashRaw[j] = resources.printPreflashRaw[j];
            }
        }

        const cudaError_t probeErr = ::juicer_cuda_probe_print_pipeline(inNeg, kCount, &run, gpuOut, cudaStreamOpaque);
        if (probeErr != cudaSuccess) {
            outError = std::string("GPU print pipeline probe failed: ") + (cudaGetErrorString(probeErr) ? cudaGetErrorString(probeErr) : "(unknown)");
            return false;
        }

        float maxAbs = 0.0f;
        for (int i = 0; i < kCount; ++i) {
            for (int c = 0; c < 3; ++c) {
                const float a = cpuOut[i * 3 + c];
                const float b = gpuOut[i * 3 + c];
                if (!std::isfinite(a) || !std::isfinite(b)) {
                    outError = "print pipeline probe produced non-finite values";
                    return false;
                }
                maxAbs = std::max(maxAbs, std::fabs(a - b));
            }
        }

        if (!(maxAbs <= 1e-3f)) {
            outError = "print pipeline mismatch: maxAbs=" + std::to_string(maxAbs);
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(resources.m);
            resources.validatedPrintBuildCounter = ws.buildCounter;
            resources.validatedPrintParamsHash = paramsHash;
        }

        return true;
#endif
    }

    void record_use(Resources& resources, void* cudaStreamOpaque) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
        std::lock_guard<std::mutex> lock(resources.m);
        if (!resources.lastUseEventOpaque) {
            cudaEvent_t ev = nullptr;
            const cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            if (err != cudaSuccess || !ev) {
                return;
            }
            resources.lastUseEventOpaque = reinterpret_cast<void*>(ev);
        }
        cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(resources.lastUseEventOpaque);
        (void)cudaEventRecord(ev, stream);
#else
        (void)resources;
        (void)cudaStreamOpaque;
#endif
    }

} // namespace JuicerCuda
