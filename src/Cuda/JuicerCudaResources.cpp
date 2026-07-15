// Cuda/JuicerCudaResources.cpp
//
// WorkingState uploads + primitive validation hooks.
//
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"

#include "FilmProcessing.h"
#include "ColorTransforms.h"
#include "PipelineTypes.h"
#include "SpectralProcessing.h"
#include "Print.h"
#include "JuicerState.h"
#include "ProcessRoot.h"

#include "GaussianSciPy.h"

#include "Logging.h"
#include "Hash.h"
#include "nlohmann/json.hpp"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include <cuda_runtime.h>
#include <cuda.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <atomic>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace JuicerCuda {
    namespace Precompute {

        struct CanonicalScanLutCpu {
            std::vector<float> log2XYZ;
            std::vector<float> slopeC;
            std::vector<float> slopeM;
            std::vector<float> slopeY;
            std::vector<float> cellMin;
            std::vector<float> cellMax;
        };

        namespace {

            size_t scan_lut_index(std::uint32_t res, std::uint32_t c, std::uint32_t m, std::uint32_t y, std::uint32_t out) {
                return (((static_cast<size_t>(c) * res + m) * res + y) * 3u) + out;
            }

            size_t scan_lut_cell_index(std::uint32_t cellRes, std::uint32_t c, std::uint32_t m, std::uint32_t y, std::uint32_t out) {
                return (((static_cast<size_t>(c) * cellRes + m) * cellRes + y) * 3u) + out;
            }

            void fill_monotone_slopes_1d(const std::vector<double>& values, std::vector<double>& slopes) {
                const size_t size = values.size();
                slopes.assign(size, 0.0);
                if (size <= 1u) {
                    return;
                }

                std::vector<double> deltas(size - 1u);
                for (size_t index = 0; index + 1u < size; ++index) {
                    deltas[index] = values[index + 1u] - values[index];
                }
                if (size == 2u) {
                    slopes[0] = deltas[0];
                    slopes[1] = deltas[0];
                    return;
                }

                double left = 0.5 * (3.0 * deltas[0] - deltas[1]);
                if (left * deltas[0] <= 0.0) {
                    left = 0.0;
                } else if (deltas[0] * deltas[1] < 0.0 && std::abs(left) > std::abs(3.0 * deltas[0])) {
                    left = 3.0 * deltas[0];
                }
                slopes[0] = left;

                for (size_t index = 1u; index + 1u < size; ++index) {
                    const double previous = deltas[index - 1u];
                    const double next = deltas[index];
                    slopes[index] =
                        previous == 0.0 || next == 0.0 || previous * next <= 0.0
                            ? 0.0
                            : 2.0 * previous * next / (previous + next);
                }

                double right = 0.5 * (3.0 * deltas[size - 2u] - deltas[size - 3u]);
                if (right * deltas[size - 2u] <= 0.0) {
                    right = 0.0;
                } else if (deltas[size - 2u] * deltas[size - 3u] < 0.0 &&
                           std::abs(right) > std::abs(3.0 * deltas[size - 2u])) {
                    right = 3.0 * deltas[size - 2u];
                }
                slopes[size - 1u] = right;
            }

            int scipy_reflect_index(int index, int size) {
                if (size <= 1) {
                    return 0;
                }
                while (index < 0 || index >= size) {
                    index = index < 0 ? -index - 1 : 2 * size - index - 1;
                }
                return index;
            }

            bool build_blurred_hanatos_spectra(
                const NpySpectraLUT& source,
                float sigma,
                std::vector<float>& out,
                std::string& outError) {
                if (!(sigma > 0.0f)) {
                    out = source.data;
                    return true;
                }
                const int n = source.size;
                const int k = source.numSamples;
                const int radius = JuicerGaussian::scipy_gaussian_radius(sigma);
                if (n <= 0 || k <= 0 || radius <= 0 || source.data.size() != static_cast<size_t>(n) * n * k) {
                    outError = "direct Hanatos spectral blur source is invalid";
                    return false;
                }

                std::vector<double> kernel(static_cast<size_t>(radius * 2 + 1));
                double kernelSum = 0.0;
                const double sigmaSquared = static_cast<double>(sigma) * static_cast<double>(sigma);
                for (int offset = -radius; offset <= radius; ++offset) {
                    const double weight = std::exp(-0.5 * static_cast<double>(offset * offset) / sigmaSquared);
                    const int kernelOffset = offset + radius;
                    const size_t kernelIndex = static_cast<size_t>(kernelOffset);
                    kernel[kernelIndex] = weight;
                    kernelSum += weight;
                }
                if (!(std::isfinite(kernelSum) && kernelSum > 0.0)) {
                    outError = "direct Hanatos spectral blur kernel is invalid";
                    return false;
                }
                for (double& weight : kernel) {
                    weight /= kernelSum;
                }

                out.assign(source.data.size(), 0.0f);
                for (int c = 0; c < n; ++c) {
                    for (int m = 0; m < n; ++m) {
                        const size_t base =
                            (static_cast<size_t>(c) * static_cast<size_t>(n) + static_cast<size_t>(m)) *
                            static_cast<size_t>(k);
                        for (int sample = 0; sample < k; ++sample) {
                            double value = 0.0;
                            for (int offset = -radius; offset <= radius; ++offset) {
                                const int reflected = scipy_reflect_index(sample + offset, k);
                                const int kernelOffset = offset + radius;
                                const size_t kernelIndex = static_cast<size_t>(kernelOffset);
                                value += kernel[kernelIndex] *
                                         static_cast<double>(source.data[base + static_cast<size_t>(reflected)]);
                            }
                            if (!std::isfinite(value)) {
                                outError = "direct Hanatos spectral blur produced non-finite data";
                                return false;
                            }
                            out[base + static_cast<size_t>(sample)] = static_cast<float>(value);
                        }
                    }
                }
                return true;
            }

            // NOLINTBEGIN(bugprone-easily-swappable-parameters)
            double eval_hanatos_surface(
                const std::array<float, 15>& params,
                double tcC,
                double tcM,
                double centerC,
                double centerM) {
                const double x = tcC - centerC;
                const double y = tcM - centerM;
                const double x2 = x * x;
                const double y2 = y * y;
                const double x3 = x2 * x;
                const double y3 = y2 * y;
                const double raw =
                    params[1] * x + params[2] * y + params[3] * x2 + params[4] * y2 + params[5] * x * y +
                    params[6] * x3 + params[7] * y3 + params[8] * x2 * y + params[9] * x * y2 +
                    params[10] * x2 * x2 + params[11] * y2 * y2 + params[12] * x3 * y +
                    params[13] * x2 * y2 + params[14] * x * y3;
                constexpr double kMaxCorrectionStops = 2.0;
                return raw / std::sqrt(1.0 + (raw / kMaxCorrectionStops) * (raw / kMaxCorrectionStops));
            }
            // NOLINTEND(bugprone-easily-swappable-parameters)

        } // namespace

        bool build_canonical_scan_lut_cpu(
            const Scanner::ScannerMediumRuntime& medium,
            std::uint32_t res,
            CanonicalScanLutCpu& out,
            std::string& outError) {
            if (res < 2u) {
                outError = "canonical scan LUT resolution must be at least two";
                return false;
            }

            const size_t voxelValues = static_cast<size_t>(res) * res * res * 3u;
            const std::uint32_t cellRes = res - 1u;
            const size_t cellValues = static_cast<size_t>(cellRes) * cellRes * cellRes * 3u;
            std::vector<double> log2XYZ(voxelValues, 0.0);
            out.log2XYZ.assign(voxelValues, 0.0f);
            out.slopeC.assign(voxelValues, 0.0f);
            out.slopeM.assign(voxelValues, 0.0f);
            out.slopeY.assign(voxelValues, 0.0f);
            out.cellMin.assign(cellValues, 0.0f);
            out.cellMax.assign(cellValues, 0.0f);

            constexpr double kLog2_10 = 3.32192809488736234787;

            for (std::uint32_t c = 0; c < res; ++c) {
                const double nc = static_cast<double>(c) / static_cast<double>(res - 1u);
                for (std::uint32_t m = 0; m < res; ++m) {
                    const double nm = static_cast<double>(m) / static_cast<double>(res - 1u);
                    for (std::uint32_t y = 0; y < res; ++y) {
                        const double ny = static_cast<double>(y) / static_cast<double>(res - 1u);
                        const double normalizedCmy[3] = {nc, nm, ny};
                        double logXYZ[3] = {0.0, 0.0, 0.0};
                        Scanner::spectral_to_log_xyz(medium, normalizedCmy, logXYZ);
                        for (std::uint32_t output = 0; output < 3u; ++output) {
                            if (!std::isfinite(logXYZ[output])) {
                                outError = "canonical scan LUT build produced non-finite log2 XYZ";
                                return false;
                            }
                            const size_t index = scan_lut_index(res, c, m, y, output);
                            log2XYZ[index] = logXYZ[output] * kLog2_10;
                            out.log2XYZ[index] = static_cast<float>(log2XYZ[index]);
                        }
                    }
                }
            }

            std::vector<double> line(res);
            std::vector<double> slopes;
            for (std::uint32_t m = 0; m < res; ++m) {
                for (std::uint32_t y = 0; y < res; ++y) {
                    for (std::uint32_t output = 0; output < 3u; ++output) {
                        for (std::uint32_t c = 0; c < res; ++c) {
                            line[c] = log2XYZ[scan_lut_index(res, c, m, y, output)];
                        }
                        fill_monotone_slopes_1d(line, slopes);
                        for (std::uint32_t c = 0; c < res; ++c) {
                            out.slopeC[scan_lut_index(res, c, m, y, output)] = static_cast<float>(slopes[c]);
                        }
                    }
                }
            }
            for (std::uint32_t c = 0; c < res; ++c) {
                for (std::uint32_t y = 0; y < res; ++y) {
                    for (std::uint32_t output = 0; output < 3u; ++output) {
                        for (std::uint32_t m = 0; m < res; ++m) {
                            line[m] = log2XYZ[scan_lut_index(res, c, m, y, output)];
                        }
                        fill_monotone_slopes_1d(line, slopes);
                        for (std::uint32_t m = 0; m < res; ++m) {
                            out.slopeM[scan_lut_index(res, c, m, y, output)] = static_cast<float>(slopes[m]);
                        }
                    }
                }
            }
            for (std::uint32_t c = 0; c < res; ++c) {
                for (std::uint32_t m = 0; m < res; ++m) {
                    for (std::uint32_t output = 0; output < 3u; ++output) {
                        for (std::uint32_t y = 0; y < res; ++y) {
                            line[y] = log2XYZ[scan_lut_index(res, c, m, y, output)];
                        }
                        fill_monotone_slopes_1d(line, slopes);
                        for (std::uint32_t y = 0; y < res; ++y) {
                            out.slopeY[scan_lut_index(res, c, m, y, output)] = static_cast<float>(slopes[y]);
                        }
                    }
                }
            }

            for (std::uint32_t c = 0; c < cellRes; ++c) {
                for (std::uint32_t m = 0; m < cellRes; ++m) {
                    for (std::uint32_t y = 0; y < cellRes; ++y) {
                        for (std::uint32_t output = 0; output < 3u; ++output) {
                            double minimum = log2XYZ[scan_lut_index(res, c, m, y, output)];
                            double maximum = minimum;
                            for (std::uint32_t dc = 0; dc < 2u; ++dc) {
                                for (std::uint32_t dm = 0; dm < 2u; ++dm) {
                                    for (std::uint32_t dy = 0; dy < 2u; ++dy) {
                                        const double sample =
                                            log2XYZ[scan_lut_index(res, c + dc, m + dm, y + dy, output)];
                                        minimum = std::min(minimum, sample);
                                        maximum = std::max(maximum, sample);
                                    }
                                }
                            }
                            const size_t cellIndex = scan_lut_cell_index(cellRes, c, m, y, output);
                            out.cellMin[cellIndex] = static_cast<float>(minimum);
                            out.cellMax[cellIndex] = static_cast<float>(maximum);
                        }
                    }
                }
            }
            return true;
        }

        bool build_direct_hanatos_integrated_lut_cpu(
            const Spectral::SpectralContext& context,
            const FilmRawRecipe& filmRaw,
            const float referenceWhiteXYZ[3],
            std::vector<float>& out,
            std::string& outError) {
            const int n = context.hanSpectra.size;
            const int k = context.hanSpectra.numSamples;
            if (n <= 0 || k != Spectral::kNumSamples || filmRaw.hanatosLutHash == 0) {
                outError = "direct Hanatos integrated LUT input is invalid";
                return false;
            }

            std::vector<float> spectra;
            if (!build_blurred_hanatos_spectra(
                    context.hanSpectra,
                    filmRaw.hanatos.spectralGaussianBlur,
                    spectra,
                    outError)) {
                return false;
            }

            double centerC = 0.0;
            double centerM = 0.0;
            if (filmRaw.hanatos.applySurface) {
                const double sum =
                    static_cast<double>(referenceWhiteXYZ[0]) +
                    static_cast<double>(referenceWhiteXYZ[1]) +
                    static_cast<double>(referenceWhiteXYZ[2]);
                if (!(std::isfinite(sum) && sum > 0.0)) {
                    outError = "direct Hanatos adaptation surface reference illuminant is invalid";
                    return false;
                }
                const double x = std::clamp(static_cast<double>(referenceWhiteXYZ[0]) / sum, 0.0, 1.0);
                const double y = std::clamp(static_cast<double>(referenceWhiteXYZ[1]) / sum, 0.0, 1.0);
                centerC = (1.0 - x) * (1.0 - x);
                centerM = std::clamp(y / std::max(1.0 - x, 1e-10), 0.0, 1.0);
            }

            out.assign(static_cast<size_t>(n) * static_cast<size_t>(n) * 4u, 0.0f);
            for (int c = 0; c < n; ++c) {
                const double tcC = static_cast<double>(c) / static_cast<double>(n - 1);
                for (int m = 0; m < n; ++m) {
                    const double tcM = static_cast<double>(m) / static_cast<double>(n - 1);
                    double raw[3] = {0.0, 0.0, 0.0};
                    const size_t base =
                        (static_cast<size_t>(c) * static_cast<size_t>(n) + static_cast<size_t>(m)) *
                        static_cast<size_t>(k);
                    for (int sample = 0; sample < k; ++sample) {
                        const double energy = static_cast<double>(spectra[base + static_cast<size_t>(sample)]);
                        const auto& sensitivity = filmRaw.finalSensitivity[static_cast<size_t>(sample)];
                        raw[0] += energy * static_cast<double>(sensitivity[0]);
                        raw[1] += energy * static_cast<double>(sensitivity[1]);
                        raw[2] += energy * static_cast<double>(sensitivity[2]);
                    }
                    if (filmRaw.hanatos.applySurface) {
                        for (size_t channel = 0; channel < 3u; ++channel) {
                            const double correction = eval_hanatos_surface(
                                filmRaw.hanatos.surfaceParams[channel],
                                tcC,
                                tcM,
                                centerC,
                                centerM);
                            raw[channel] *= std::exp2(correction);
                        }
                    }
                    const size_t outBase =
                        (static_cast<size_t>(c) * static_cast<size_t>(n) + static_cast<size_t>(m)) * 4u;
                    for (size_t channel = 0; channel < 3u; ++channel) {
                        if (!std::isfinite(raw[channel])) {
                            outError = "direct Hanatos integrated LUT produced non-finite data";
                            return false;
                        }
                        out[outBase + channel] = static_cast<float>(raw[channel]);
                    }
                }
            }
            return true;
        }

        void build_hanatos_integrated_lut_cpu(const Spectral::SpectralContext& ctx, const WorkingState& ws, std::vector<float>& out) {
            const int N = ctx.hanSpectra.size;
            const int K = ctx.hanSpectra.numSamples;
            out.clear();
            out.resize(static_cast<size_t>(N) * static_cast<size_t>(N) * 4u, 0.0f);

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
                        if (std::isfinite(sb))
                            accB += e64 * static_cast<double>(sb);
                        if (std::isfinite(sg))
                            accG += e64 * static_cast<double>(sg);
                        if (std::isfinite(sr))
                            accR += e64 * static_cast<double>(sr);
                    }

                    const size_t outBase = (static_cast<size_t>(x) * static_cast<size_t>(N) + static_cast<size_t>(y)) * 4u;
                    out[outBase + 0] = static_cast<float>(accR);
                    out[outBase + 1] = static_cast<float>(accG);
                    out[outBase + 2] = static_cast<float>(accB);
                    out[outBase + 3] = 0.0f;
                }
            }
        }

        bool build_print_preflash_raw(const WorkingState& ws, const Print::Runtime& prt, float outRaw[3], int& outShapeK) {
            return Pipeline::compute_preflash_raw(ws, prt, outRaw, outShapeK);
        }
    } // namespace Precompute
} // namespace JuicerCuda

namespace JuicerCuda {

    bool validate_resource_owner_locked(Resources& resources, std::string& outError, bool bindIfUnset) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)bindIfUnset;
        outError = "CUDA is not enabled";
        return false;
#else
        int cur = -1;
        const cudaError_t devErr = cudaGetDevice(&cur);
        if (devErr != cudaSuccess || cur < 0) {
            outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(devErr) ? cudaGetErrorString(devErr) : "(unknown)");
            return false;
        }
        if (bindIfUnset && resources.deviceId < 0) {
            resources.deviceId = cur;
        }
        if (resources.deviceId != cur) {
            outError = "CUDA device mismatch for cached resources";
            return false;
        }
        return true;
#endif
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    struct HostToDeviceCopyRequest {
        const char* stage = nullptr;
        const char* label = nullptr;
    };

    static bool enqueue_host_to_device_copy(
        const HostToDeviceCopyRequest& request,
        void* dst,
        const void* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        std::string& outError);
#endif

    static void free_stbn(Resources& resources) noexcept;
    static void free_wang(Resources& resources) noexcept;
    static void free_scan_error_readbacks(Resources& resources) noexcept;
    static void free_tables(Resources& resources) noexcept;
    static void free_scan_medium(Resources::DeviceScanMedium& m) noexcept;
    static void free_scan_lut(Resources::DeviceSpectralLut& lut) noexcept;
    static void free_gaussian_kernel(Resources::DeviceGaussianKernel& k) noexcept;
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool retire_tables_locked(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError);
    static bool retire_scan_medium_locked(Resources& resources, Resources::DeviceScanMedium& m, void* cudaStreamOpaque, const char* label, std::string& outError);
    static bool is_async_device_ptr_tracked_locked(const Resources& resources, const void* ptr) noexcept;
    static void untrack_async_device_ptr_locked(Resources& resources, void* ptr) noexcept;
    static cudaError_t device_free_async_compat(void* ptr, cudaStream_t stream) noexcept;
#endif
    static void free_optics_scratch(Resources& resources, Resources::DeviceOpticsScratch& s, void* cudaStreamOpaque) noexcept;
    static void free_spatial_dir_scratch(Resources& resources, Resources::DeviceSpatialDirScratch& s, void* cudaStreamOpaque) noexcept;
    static void free_shared_tmp_plane(Resources& resources) noexcept;

} // namespace JuicerCuda

namespace JuicerCuda {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    struct DeferredDestroyEntry {
        Resources* resources = nullptr;
        int ownerDeviceId = -1;
        void* ownerContextOpaque = nullptr;
    };

    static std::mutex& deferred_destroy_mutex() {
        static std::mutex mutex;
        return mutex;
    }

    static std::vector<DeferredDestroyEntry>& deferred_destroy_queue() {
        static std::vector<DeferredDestroyEntry> queue;
        return queue;
    }

#if defined(_WIN32)
    using CuCtxGetCurrentFn = CUresult(CUDAAPI*)(CUcontext*);

    struct CudaDriverDispatch {
        CuCtxGetCurrentFn cuCtxGetCurrent = nullptr;
        const char* loadError = nullptr;
    };

    static const CudaDriverDispatch& cuda_driver_dispatch_for_teardown() {
        static CudaDriverDispatch dispatch{};
        static std::once_flag once;
        std::call_once(once, []() {
            HMODULE module = GetModuleHandleA("nvcuda.dll");
            if (!module) {
                module = LoadLibraryA("nvcuda.dll");
            }
            if (!module) {
                dispatch.loadError = "nvcuda.dll not available";
                return;
            }
            dispatch.cuCtxGetCurrent =
                reinterpret_cast<CuCtxGetCurrentFn>(GetProcAddress(module, "cuCtxGetCurrent"));
            if (!dispatch.cuCtxGetCurrent) {
                dispatch.loadError = "cuCtxGetCurrent symbol not found";
            }
        });
        return dispatch;
    }
#endif

    static bool query_current_cuda_context(void*& outContextOpaque, std::string& outError) {
        outContextOpaque = nullptr;
        outError.clear();
#if defined(_WIN32)
        const CudaDriverDispatch& dispatch = cuda_driver_dispatch_for_teardown();
        if (!dispatch.cuCtxGetCurrent) {
            outError = dispatch.loadError ? dispatch.loadError : "driver dispatch unavailable";
            return false;
        }
        CUcontext currentContext = nullptr;
        const CUresult result = dispatch.cuCtxGetCurrent(&currentContext);
        if (result != CUDA_SUCCESS) {
            outError = std::string("cuCtxGetCurrent failed (code=") + std::to_string(static_cast<int>(result)) + ")";
            return false;
        }
        if (!currentContext) {
            outError = "current CUDA context is null";
            return false;
        }
        outContextOpaque = reinterpret_cast<void*>(currentContext);
        return true;
#else
        outError = "cuCtxGetCurrent loader unsupported on this platform";
        return false;
#endif
    }

    static bool query_current_cuda_device(int& outDeviceId, std::string& outError) {
        outDeviceId = -1;
        outError.clear();
        const cudaError_t err = cudaGetDevice(&outDeviceId);
        if (err != cudaSuccess || outDeviceId < 0) {
            outError = std::string("cudaGetDevice failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            outDeviceId = -1;
            return false;
        }
        return true;
    }

    static bool owner_matches_current(
        int ownerDeviceId,
        void* ownerContextOpaque,
        int currentDeviceId,
        bool currentDeviceValid,
        void* currentContextOpaque,
        bool currentContextValid,
        bool& outDeviceMatch,
        bool& outContextMatch) {
        outDeviceMatch = (ownerDeviceId < 0) ||
                         (currentDeviceValid && currentDeviceId == ownerDeviceId);
        outContextMatch = (ownerContextOpaque == nullptr) ||
                          (currentContextValid && currentContextOpaque == ownerContextOpaque);
        return outDeviceMatch && outContextMatch;
    }

    static void trace_teardown_event(
        const char* stage,
        const char* outcome,
        int ownerDeviceId,
        void* ownerContextOpaque,
        int currentDeviceId,
        void* currentContextOpaque,
        bool deviceMatch,
        bool contextMatch,
        bool switchAttempted,
        bool switchSucceeded,
        bool managerRetireAttempted,
        bool managerRetireAccepted,
        std::size_t deferredQueueDepth,
        const std::string& detail) noexcept {
        try {
            if (!JTRACE_ENABLED(2)) {
                return;
            }
            const std::uintptr_t ownerContextBits =
                reinterpret_cast<std::uintptr_t>(ownerContextOpaque);
            const std::uintptr_t currentContextBits =
                reinterpret_cast<std::uintptr_t>(currentContextOpaque);
            std::ostringstream oss;
            oss << "stage=" << (stage ? stage : "unknown")
                << " outcome=" << (outcome ? outcome : "unknown")
                << " owner_device=" << ownerDeviceId
                << " current_device=" << currentDeviceId
                << " owner_context=" << ownerContextBits
                << " current_context=" << currentContextBits
                << " device_match=" << (deviceMatch ? 1 : 0)
                << " context_match=" << (contextMatch ? 1 : 0)
                << " switch_attempted=" << (switchAttempted ? 1 : 0)
                << " switch_succeeded=" << (switchSucceeded ? 1 : 0)
                << " retire_attempted=" << (managerRetireAttempted ? 1 : 0)
                << " retire_accepted=" << (managerRetireAccepted ? 1 : 0)
                << " deferred_queue_depth=" << static_cast<unsigned long long>(deferredQueueDepth);
            if (!detail.empty()) {
                oss << " detail=" << detail;
            }
            JTRACE("MSTDN", oss.str());
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    static void reap_deferred_destroy_queue(const char* stage) {
        std::vector<DeferredDestroyEntry> readyEntries;
        std::size_t remainingDepth = 0;
        int currentDeviceId = -1;
        std::string deviceError;
        const bool currentDeviceValid = query_current_cuda_device(currentDeviceId, deviceError);
        void* currentContextOpaque = nullptr;
        std::string contextError;
        const bool currentContextValid = query_current_cuda_context(currentContextOpaque, contextError);

        {
            std::lock_guard<std::mutex> lock(deferred_destroy_mutex());
            auto& queue = deferred_destroy_queue();
            if (queue.empty()) {
                return;
            }
            for (std::size_t i = 0; i < queue.size();) {
                bool deviceMatch = false;
                bool contextMatch = false;
                if (owner_matches_current(
                        queue[i].ownerDeviceId,
                        queue[i].ownerContextOpaque,
                        currentDeviceId,
                        currentDeviceValid,
                        currentContextOpaque,
                        currentContextValid,
                        deviceMatch,
                        contextMatch)) {
                    readyEntries.push_back(queue[i]);
                    queue[i] = queue.back();
                    queue.pop_back();
                    continue;
                }
                ++i;
            }
            remainingDepth = queue.size();
        }

        for (const DeferredDestroyEntry& entry : readyEntries) {
            bool deviceMatch = false;
            bool contextMatch = false;
            (void)owner_matches_current(
                entry.ownerDeviceId,
                entry.ownerContextOpaque,
                currentDeviceId,
                currentDeviceValid,
                currentContextOpaque,
                currentContextValid,
                deviceMatch,
                contextMatch);
            std::string detail;
            if (!deviceError.empty()) {
                detail = deviceError;
            }
            if (!contextError.empty()) {
                if (!detail.empty()) {
                    detail += "; ";
                }
                detail += contextError;
            }
            trace_teardown_event(
                stage,
                "deferred_reap",
                entry.ownerDeviceId,
                entry.ownerContextOpaque,
                currentDeviceId,
                currentContextOpaque,
                deviceMatch,
                contextMatch,
                false,
                false,
                false,
                false,
                remainingDepth,
                detail);
            delete entry.resources;
        }
    }
#endif


#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool add_u64_saturating(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t& out) noexcept {
        if (lhs > (std::numeric_limits<std::uint64_t>::max() - rhs)) {
            out = std::numeric_limits<std::uint64_t>::max();
            return false;
        }
        out = lhs + rhs;
        return true;
    }

    static std::uint64_t bytes_for_count_u64(
        std::size_t count,
        std::size_t elementBytes,
        bool& overflow) noexcept {
        if (overflow) {
            return std::numeric_limits<std::uint64_t>::max();
        }
        if (count == 0 || elementBytes == 0) {
            return 0;
        }

        const std::uint64_t count64 = static_cast<std::uint64_t>(count);
        const std::uint64_t elementBytes64 = static_cast<std::uint64_t>(elementBytes);
        if (count64 > (std::numeric_limits<std::uint64_t>::max() / elementBytes64)) {
            overflow = true;
            return std::numeric_limits<std::uint64_t>::max();
        }
        return count64 * elementBytes64;
    }

    struct ResidencyByteTarget {
        std::uint64_t& total;
        bool& overflow;
    };

    static void add_residency_bytes(
        std::uint64_t bytes,
        ResidencyByteTarget target) noexcept {
        std::uint64_t next = 0;
        if (!add_u64_saturating(target.total, bytes, next)) {
            target.overflow = true;
        }
        target.total = next;
    }

    static void refresh_scratch_residency_state_locked(Resources& resources) noexcept {
        Resources::ScratchResidencyState next{};
        next.retainedGeneration = std::max<std::uint64_t>(1ull, resources.scratchResidency.retainedGeneration);

        bool overflow = false;
        const std::uint64_t opticsPlaneBytes =
            bytes_for_count_u64(resources.scannerScratch.capacityElements, sizeof(float), overflow);
        const std::uint64_t spatialDirPlaneBytes =
            bytes_for_count_u64(resources.spatialDirScratch.capacityElements, sizeof(float), overflow);
        const std::uint64_t gateMaskBytes =
            bytes_for_count_u64(resources.scannerScratch.gateMaskCapacityElements, sizeof(float), overflow);
        const std::uint64_t sharedTmpBytes =
            bytes_for_count_u64(resources.sharedTmpCapacityElements, sizeof(float), overflow);
        if (overflow) {
            next.overflow = true;
        }

        auto add_candidate_plane = [&](ResourceManager::ScratchPolicyCandidate candidate, bool live, std::uint64_t bytes) {
            if (!live || bytes == 0) {
                return;
            }
            const std::size_t index = ResourceManager::scratch_policy_candidate_index(candidate);
            add_residency_bytes(
                bytes,
                ResidencyByteTarget{next.candidateLiveBytes[index], next.overflow});
        };

        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsBase, resources.scannerScratch.rgbR != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsBase, resources.scannerScratch.rgbG != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsBase, resources.scannerScratch.rgbB != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsBlurred, resources.scannerScratch.blurred != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsAux, resources.scannerScratch.aux != nullptr, opticsPlaneBytes);
        add_candidate_plane(
            ResourceManager::ScratchPolicyCandidate::OpticsBase,
            resources.scannerScratch.grainFrameUniforms != nullptr,
            sizeof(GrainFrameUniforms));
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsGrainLayerWork, resources.scannerScratch.grainTmp != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsGrainShared, resources.scannerScratch.grainTmpShared != nullptr, opticsPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::OpticsGateMask, resources.scannerScratch.gateMask != nullptr, gateMaskBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.rawCorrectionY != nullptr, spatialDirPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.rawCorrectionM != nullptr, spatialDirPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.rawCorrectionC != nullptr, spatialDirPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.filteredCorrectionY != nullptr, spatialDirPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.filteredCorrectionM != nullptr, spatialDirPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.filteredCorrectionC != nullptr, spatialDirPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.filterTempM != nullptr, spatialDirPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.filterTempC != nullptr, spatialDirPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.logRawB != nullptr, spatialDirPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.logRawG != nullptr, spatialDirPlaneBytes);
        add_candidate_plane(ResourceManager::ScratchPolicyCandidate::SpatialDirBase, resources.spatialDirScratch.logRawR != nullptr, spatialDirPlaneBytes);

        if (resources.sharedTmpPlane) {
            next.helperSharedBytes = sharedTmpBytes;
        }

        auto assign_non_policy_bytes = [&](ResourceManager::ScratchHelperNonPolicyAllocation allocation, bool live, std::uint64_t bytes) {
            if (!live || bytes == 0) {
                return;
            }
            const std::size_t index = ResourceManager::scratch_helper_non_policy_index(allocation);
            next.helperNonPolicyBytes[index] = bytes;
            add_residency_bytes(
                bytes,
                ResidencyByteTarget{next.helperNonPolicyTotalBytes, next.overflow});
        };

        assign_non_policy_bytes(
            ResourceManager::ScratchHelperNonPolicyAllocation::ScanErrorHost,
            !resources.pendingScanErrorReadbacks.empty(),
            static_cast<std::uint64_t>(resources.pendingScanErrorReadbacks.size()) * sizeof(int));

        for (std::uint64_t bytes : next.candidateLiveBytes) {
            add_residency_bytes(
                bytes,
                ResidencyByteTarget{next.policyLiveRetainedBytes, next.overflow});
        }
        add_residency_bytes(
            next.helperSharedBytes,
            ResidencyByteTarget{next.policyLiveRetainedBytes, next.overflow});
        next.totalLiveRetainedBytes = next.policyLiveRetainedBytes;
        add_residency_bytes(
            next.helperNonPolicyTotalBytes,
            ResidencyByteTarget{next.totalLiveRetainedBytes, next.overflow});

        const bool changed =
            next.candidateLiveBytes != resources.scratchResidency.candidateLiveBytes ||
            next.helperNonPolicyBytes != resources.scratchResidency.helperNonPolicyBytes ||
            next.helperSharedBytes != resources.scratchResidency.helperSharedBytes ||
            next.helperNonPolicyTotalBytes != resources.scratchResidency.helperNonPolicyTotalBytes ||
            next.policyLiveRetainedBytes != resources.scratchResidency.policyLiveRetainedBytes ||
            next.totalLiveRetainedBytes != resources.scratchResidency.totalLiveRetainedBytes ||
            next.overflow != resources.scratchResidency.overflow;
        if (changed) {
            next.retainedGeneration =
                (resources.scratchResidency.retainedGeneration == std::numeric_limits<std::uint64_t>::max())
                    ? std::numeric_limits<std::uint64_t>::max()
                    : std::max<std::uint64_t>(1ull, resources.scratchResidency.retainedGeneration + 1ull);
        }

        resources.scratchResidency = next;
    }
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static void reap_retire_queue_locked(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        for (size_t i = 0; i < resources.retireQueue.size();) {
            Resources::RetireEntry& e = resources.retireQueue[i];
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(e.doneEventOpaque);
            if (!ev) {
                // Retain allocations whose completion fence is missing; freeing early could
                // race an in-flight launch. Blocking teardown owns final cleanup.
                ++i;
                continue;
            }

            const cudaError_t q = cudaEventQuery(ev);
            if (q == cudaSuccess) {
                if (e.kind == Resources::RetireKind::DeviceFree && e.ptr) {
                    cudaFree(e.ptr);
                } else if (e.kind == Resources::RetireKind::DeviceFreeAsync && e.ptr) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
                    const cudaError_t asyncErr = cudaFreeAsync(e.ptr, nullptr);
                    if (asyncErr != cudaSuccess) {
                        cudaFree(e.ptr);
                    }
#else
                    cudaFree(e.ptr);
#endif
                } else if (e.kind == Resources::RetireKind::HostPinnedFree && e.ptr) {
                    cudaFreeHost(e.ptr);
                } else if (e.kind == Resources::RetireKind::EventDestroy && e.ptr) {
                    cudaEventDestroy(reinterpret_cast<cudaEvent_t>(e.ptr));
                }

                // Return the fence to the pool; destroy it if the pool cannot grow.
                try {
                    resources.retireEventPoolOpaque.push_back(e.doneEventOpaque);
                } catch (...) {
                    JuicerLogging::discard_current_exception();
                    cudaEventDestroy(ev);
                    e.doneEventOpaque = nullptr;
                }
                if (resources.retireBytes >= e.bytes) {
                    resources.retireBytes -= e.bytes;
                }
                if (e.scratchTier && resources.retireScratchBytes >= e.bytes) {
                    resources.retireScratchBytes -= e.bytes;
                }
                resources.retireQueue[i] = resources.retireQueue.back();
                resources.retireQueue.pop_back();
                continue;
            }
            if (q == cudaErrorNotReady) {
                ++i;
                continue;
            }

            // On unexpected CUDA errors, keep the entry so we don't free too early.
            ++i;
        }
#else
        (void)resources;
#endif
    }
#endif

    static void drain_retire_queue_blocking(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        for (Resources::RetireEntry& e : resources.retireQueue) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(e.doneEventOpaque);
            if (ev) {
                (void)cudaEventSynchronize(ev);
            }
            if (e.kind == Resources::RetireKind::DeviceFree && e.ptr) {
                cudaFree(e.ptr);
            } else if (e.kind == Resources::RetireKind::DeviceFreeAsync && e.ptr) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
                const cudaError_t asyncErr = cudaFreeAsync(e.ptr, nullptr);
                if (asyncErr == cudaSuccess) {
                    (void)cudaStreamSynchronize(nullptr);
                } else {
                    cudaFree(e.ptr);
                }
#else
                cudaFree(e.ptr);
#endif
            } else if (e.kind == Resources::RetireKind::HostPinnedFree && e.ptr) {
                cudaFreeHost(e.ptr);
            } else if (e.kind == Resources::RetireKind::EventDestroy && e.ptr) {
                cudaEventDestroy(reinterpret_cast<cudaEvent_t>(e.ptr));
            }
            if (ev) {
                cudaEventDestroy(ev);
                e.doneEventOpaque = nullptr;
            }
        }
        resources.retireQueue.clear();
        resources.retireBytes = 0;
        resources.retireScratchBytes = 0;

        // Destroy pooled events.
        for (void* p : resources.retireEventPoolOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(p);
            if (ev) {
                cudaEventDestroy(ev);
            }
        }
        resources.retireEventPoolOpaque.clear();
#else
        (void)resources;
#endif
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool acquire_retire_event_locked(Resources& resources, void*& outEventOpaque, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)outEventOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!resources.retireEventPoolOpaque.empty()) {
            outEventOpaque = resources.retireEventPoolOpaque.back();
            resources.retireEventPoolOpaque.pop_back();
            return true;
        }
        cudaEvent_t ev = nullptr;
        const cudaError_t err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (err != cudaSuccess || !ev) {
            outError = std::string("cudaEventCreateWithFlags failed: ") + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            outEventOpaque = nullptr;
            return false;
        }
        outEventOpaque = reinterpret_cast<void*>(ev);
        return true;
#endif
    }
#endif

    static void release_frame_use_event_entry(Resources::PendingFrameUseEvent& entry) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (entry.eventOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(entry.eventOpaque);
            cudaEventDestroy(ev);
        }
#endif
        entry = Resources::PendingFrameUseEvent{};
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static void reap_frame_use_events_locked(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        std::vector<Resources::PendingFrameUseEvent>& pending = resources.pendingFrameUseEvents;
        for (std::size_t i = 0; i < pending.size();) {
            Resources::PendingFrameUseEvent& entry = pending[i];
            cudaEvent_t ev = entry.eventOpaque
                                 ? reinterpret_cast<cudaEvent_t>(entry.eventOpaque)
                                 : nullptr;
            if (!ev) {
                release_frame_use_event_entry(entry);
                pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(i));
                continue;
            }
            const cudaError_t queryErr = cudaEventQuery(ev);
            if (queryErr == cudaSuccess) {
                release_frame_use_event_entry(entry);
                pending.erase(pending.begin() + static_cast<std::ptrdiff_t>(i));
                continue;
            }
            ++i;
        }
#else
        (void)resources;
#endif
    }

    static bool wait_for_frame_use_events_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        reap_frame_use_events_locked(resources);
        if (resources.pendingFrameUseEvents.empty()) {
            return true;
        }

        const cudaStream_t stream = cudaStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                                        : nullptr;
        for (const Resources::PendingFrameUseEvent& entry : resources.pendingFrameUseEvents) {
            if (!entry.eventOpaque) {
                continue;
            }
            const cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(entry.eventOpaque);
            const cudaError_t waitErr = cudaStreamWaitEvent(stream, ev, 0);
            if (waitErr != cudaSuccess) {
                outError = std::string("cudaStreamWaitEvent before ") + (label ? label : "resource") + " update failed: " + (cudaGetErrorString(waitErr) ? cudaGetErrorString(waitErr) : "(unknown)");
                return false;
            }
        }
        return true;
#endif
    }

    struct RetireFenceRecord {
        void* retireEventOpaque = nullptr;
        void* cudaStreamOpaque = nullptr;
        const char* label = nullptr;
    };

    static bool record_retire_fence_locked(
        Resources& resources,
        const RetireFenceRecord& record,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)record;
        outError = "CUDA is not enabled";
        return false;
#else
        cudaEvent_t retireEv = reinterpret_cast<cudaEvent_t>(record.retireEventOpaque);
        if (!retireEv) {
            outError = "retire fence event missing";
            return false;
        }
        const cudaStream_t stream = record.cudaStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(record.cudaStreamOpaque)
                                        : nullptr;
        if (!wait_for_frame_use_events_locked(resources, record.cudaStreamOpaque, record.label, outError)) {
            return false;
        }
        const cudaError_t recErr = cudaEventRecord(retireEv, stream);
        if (recErr != cudaSuccess) {
            outError = std::string("cudaEventRecord for ") + (record.label ? record.label : "resource") + " retire failed: " + (cudaGetErrorString(recErr) ? cudaGetErrorString(recErr) : "(unknown)");
            return false;
        }
        return true;
#endif
    }

    static bool retire_ptr_locked(Resources& resources, void* ptr, std::size_t bytes, Resources::RetireKind kind, void* cudaStreamOpaque, const char* label, std::string& outError, bool scratchTier = false) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ptr;
        (void)bytes;
        (void)kind;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!ptr) {
            return true;
        }

        bool asyncTracked = false;
        Resources::RetireKind effectiveKind = kind;
        if (kind == Resources::RetireKind::DeviceFree) {
            asyncTracked = is_async_device_ptr_tracked_locked(resources, ptr);
            if (asyncTracked) {
                effectiveKind = Resources::RetireKind::DeviceFreeAsync;
            }
        }

        reap_retire_queue_locked(resources);

        void* retireEventOpaque = nullptr;
        if (!acquire_retire_event_locked(resources, retireEventOpaque, outError)) {
            if (outError.empty()) {
                outError = std::string("retire fence acquisition failed for ") + (label ? label : "resource");
            }
            return false;
        }

        RetireFenceRecord fenceRecord{};
        fenceRecord.retireEventOpaque = retireEventOpaque;
        fenceRecord.cudaStreamOpaque = cudaStreamOpaque;
        fenceRecord.label = label;
        if (!record_retire_fence_locked(resources, fenceRecord, outError)) {
            // If we can't record the retire fence, fail closed instead of falling back to a blocking sync.
            cudaEventDestroy(reinterpret_cast<cudaEvent_t>(retireEventOpaque));
            if (outError.empty()) {
                outError = std::string("retire fence record failed for ") + (label ? label : "resource");
            }
            return false;
        }

        Resources::RetireEntry e{};
        e.ptr = ptr;
        e.bytes = bytes;
        e.kind = effectiveKind;
        e.scratchTier = scratchTier;
        e.doneEventOpaque = retireEventOpaque;
        resources.retireQueue.push_back(e);
        resources.retireBytes += bytes;
        if (scratchTier) {
            resources.retireScratchBytes += bytes;
        }
        if (effectiveKind == Resources::RetireKind::DeviceFreeAsync && asyncTracked) {
            untrack_async_device_ptr_locked(resources, ptr);
        }
        return true;
#endif
    }

#endif

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
        c.domainBegin = 0;
        c.domainEnd = 0;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool retire_curve_locked(Resources& resources, DeviceCurve& c, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)c;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t bytes = static_cast<size_t>(std::max(0, c.n)) * sizeof(float);
        if (c.x) {
            if (!retire_ptr_locked(resources, c.x, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            c.x = nullptr;
        }
        if (c.y) {
            if (!retire_ptr_locked(resources, c.y, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            c.y = nullptr;
        }
        c.n = 0;
        c.domainBegin = 0;
        c.domainEnd = 0;
        return true;
#endif
    }
#endif

    static void free_density_layers(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                if (resources.densityCurvesLayers[layer][ch]) {
                    cudaFree(resources.densityCurvesLayers[layer][ch]);
                    resources.densityCurvesLayers[layer][ch] = nullptr;
                }
            }
        }
#endif
        for (int ch = 0; ch < 3; ++ch) {
            resources.densityCurvesLayersChannelN[ch] = 0;
        }
        resources.hasDensityCurvesLayers = 0;
        resources.directDensityLayersHash = 0;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool retire_density_layers_locked(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        for (int ch = 0; ch < 3; ++ch) {
            const size_t bytes =
                static_cast<size_t>(std::max(0, resources.densityCurvesLayersChannelN[ch])) *
                sizeof(float);
            for (int layer = 0; layer < 3; ++layer) {
                if (resources.densityCurvesLayers[layer][ch]) {
                    if (!retire_ptr_locked(resources, resources.densityCurvesLayers[layer][ch], bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                        return false;
                    }
                    resources.densityCurvesLayers[layer][ch] = nullptr;
                }
            }
            resources.densityCurvesLayersChannelN[ch] = 0;
        }
        resources.hasDensityCurvesLayers = 0;
        resources.directDensityLayersHash = 0;
        return true;
#endif
    }
#endif

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
        resources.hanatosIntegratedKeyHash = 0;
    }

    static void free_mallett_basis(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.mallettBasis) {
            cudaFree(resources.mallettBasis);
            resources.mallettBasis = nullptr;
        }
#endif
        resources.mallettBasisK = 0;
    }


    static void free_spectral_tables(Resources::DeviceSpectralTables& t) noexcept;
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool retire_spectral_tables_locked(
        Resources& resources,
        Resources::DeviceSpectralTables& t,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError);
#endif

    static void free_print_payloads(Resources& resources) noexcept {
        free_curve(resources.printDcC);
        free_curve(resources.printDcM);
        free_curve(resources.printDcY);
        free_curve(resources.printSensC);
        free_curve(resources.printSensM);
        free_curve(resources.printSensY);
        free_spectral_tables(resources.printFilmDensityTables);

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (resources.printIllumFiltered) {
            cudaFree(resources.printIllumFiltered);
            resources.printIllumFiltered = nullptr;
        }
        if (resources.printPreflashIllumFiltered) {
            cudaFree(resources.printPreflashIllumFiltered);
            resources.printPreflashIllumFiltered = nullptr;
        }
#endif
        resources.printIllumK = 0;
        resources.printIllumYShiftSteps = 0.0f;
        resources.printIllumMShiftSteps = 0.0f;
        resources.printIllumCShiftSteps = 0.0f;
        resources.printIllumNeutralFilterHash = 0;
        resources.printIllumShapeK = 0;
        resources.printIllumBuildCounter = 0;
        resources.printIllumCoreHash = 0;
        resources.printPreflashIllumK = 0;
        resources.printIllumFilteredHostValid = false;
        resources.printPreflashIllumFilteredHostValid = false;

        resources.printGammaC = 1.0f;
        resources.printGammaM = 1.0f;
        resources.printGammaY = 1.0f;

        resources.printPreflashRaw[0] = resources.printPreflashRaw[1] = resources.printPreflashRaw[2] = 0.0f;
        resources.printPreflashValid = false;
        resources.printPreflashKeyHash = 0;
        resources.printPreflashShapeK = 0;
        resources.printBalanceFactorMidgray = 1.0f;
        resources.printBalanceFactorMidgrayComp = 1.0f;
        resources.printBalanceNormalizer = 1.0f;
        resources.printFilmDensityTablesDescriptorHash = 0;
        resources.printProfileTablesDescriptorHash = 0;
        resources.printMainIlluminantDescriptorHash = 0;
        resources.printPreflashIlluminantDescriptorHash = 0;
        resources.printPreflashRawDescriptorHash = 0;
        resources.printBalanceDescriptorHash = 0;
        resources.printPreparationDescriptorHash = 0;
        resources.printPreparationCounter = 0;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool retire_print_payloads_locked(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!retire_curve_locked(resources, resources.printDcC, cudaStreamOpaque, label, outError))
            return false;
        if (!retire_curve_locked(resources, resources.printDcM, cudaStreamOpaque, label, outError))
            return false;
        if (!retire_curve_locked(resources, resources.printDcY, cudaStreamOpaque, label, outError))
            return false;
        if (!retire_curve_locked(resources, resources.printSensC, cudaStreamOpaque, label, outError))
            return false;
        if (!retire_curve_locked(resources, resources.printSensM, cudaStreamOpaque, label, outError))
            return false;
        if (!retire_curve_locked(resources, resources.printSensY, cudaStreamOpaque, label, outError))
            return false;
        if (!retire_spectral_tables_locked(
                resources,
                resources.printFilmDensityTables,
                cudaStreamOpaque,
                label,
                outError))
            return false;

        if (resources.printIllumFiltered) {
            const size_t bytes = static_cast<size_t>(std::max(0, resources.printIllumK)) * sizeof(float);
            if (!retire_ptr_locked(resources, resources.printIllumFiltered, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.printIllumFiltered = nullptr;
        }
        if (resources.printPreflashIllumFiltered) {
            const size_t bytes = static_cast<size_t>(std::max(0, resources.printPreflashIllumK)) * sizeof(float);
            if (!retire_ptr_locked(resources, resources.printPreflashIllumFiltered, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.printPreflashIllumFiltered = nullptr;
        }

        resources.printIllumK = 0;
        resources.printIllumYShiftSteps = 0.0f;
        resources.printIllumMShiftSteps = 0.0f;
        resources.printIllumCShiftSteps = 0.0f;
        resources.printIllumNeutralFilterHash = 0;
        resources.printIllumShapeK = 0;
        resources.printIllumBuildCounter = 0;
        resources.printIllumCoreHash = 0;
        resources.printPreflashIllumK = 0;
        resources.printIllumFilteredHostValid = false;
        resources.printPreflashIllumFilteredHostValid = false;

        resources.printGammaC = 1.0f;
        resources.printGammaM = 1.0f;
        resources.printGammaY = 1.0f;

        resources.printPreflashRaw[0] = resources.printPreflashRaw[1] = resources.printPreflashRaw[2] = 0.0f;
        resources.printPreflashValid = false;
        resources.printPreflashKeyHash = 0;
        resources.printPreflashShapeK = 0;
        resources.printBalanceFactorMidgray = 1.0f;
        resources.printBalanceFactorMidgrayComp = 1.0f;
        resources.printBalanceNormalizer = 1.0f;
        resources.printFilmDensityTablesDescriptorHash = 0;
        resources.printProfileTablesDescriptorHash = 0;
        resources.printMainIlluminantDescriptorHash = 0;
        resources.printPreflashIlluminantDescriptorHash = 0;
        resources.printPreflashRawDescriptorHash = 0;
        resources.printBalanceDescriptorHash = 0;
        resources.printPreparationDescriptorHash = 0;
        resources.printPreparationCounter = 0;
        return true;
#endif
    }
#endif

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
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
        if (!enqueue_host_to_device_copy(
                HostToDeviceCopyRequest{"alloc_and_upload_array", label},
                dst,
                src,
                bytes,
                cudaStreamOpaque,
                outError)) {
            cudaFree(dst);
            dst = nullptr;
            return false;
        }
        return true;
#endif
    }

    static bool relock_resources_after_offlock_upload(
        Resources& resources,
        std::unique_lock<std::mutex>* resourcesLock,
        std::string& outError) {
        if (!resourcesLock) {
            return true;
        }
        resourcesLock->lock();
        reap_retire_queue_locked(resources);
        return validate_resource_owner_locked(resources, outError, false);
    }

    static bool alloc_and_upload_bytes(
        void*& dst,
        const void* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)dst;
        (void)src;
        (void)bytes;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!src || bytes == 0) {
            outError = std::string(label ? label : "buffer") + " buffer is empty";
            return false;
        }

        const cudaError_t err = cudaMalloc(&dst, bytes);
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(") + (label ? label : "buffer") + ") failed: " + (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            return false;
        }

        if (!enqueue_host_to_device_copy(
                HostToDeviceCopyRequest{"alloc_and_upload_bytes", label},
                dst,
                src,
                bytes,
                cudaStreamOpaque,
                outError)) {
            cudaFree(dst);
            dst = nullptr;
            return false;
        }

        return true;
#endif
    }

    static bool alloc_and_upload_bytes_locked(
        Resources& resources,
        void*& dst,
        const void* src,
        std::size_t bytes,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>* resourcesLock,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)dst;
        (void)src;
        (void)bytes;
        (void)cudaStreamOpaque;
        (void)resourcesLock;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (resourcesLock) {
            resourcesLock->unlock();
        }
        const bool allocOk =
            alloc_and_upload_bytes(dst, src, bytes, cudaStreamOpaque, label, outError);
        if (!relock_resources_after_offlock_upload(resources, resourcesLock, outError)) {
            if (dst) {
                cudaFree(dst);
                dst = nullptr;
            }
            return false;
        }
        return allocOk;
#endif
    }

    static bool upload_array_locked(
        Resources& resources,
        float*& dst,
        int currentN,
        const float* src,
        int n,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>* resourcesLock,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)dst;
        (void)currentN;
        (void)src;
        (void)n;
        (void)cudaStreamOpaque;
        (void)resourcesLock;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!src || n <= 0) {
            outError = std::string(label ? label : "array") + " array is empty";
            return false;
        }

        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        if (dst && currentN == n) {
            const bool waitOk = wait_for_frame_use_events_locked(
                resources,
                cudaStreamOpaque,
                label,
                outError);
            const bool copyOk = waitOk && enqueue_host_to_device_copy(
                                              HostToDeviceCopyRequest{"upload_array_locked", label},
                                              dst,
                                              src,
                                              bytes,
                                              cudaStreamOpaque,
                                              outError);
            return copyOk;
        }

        float* tmp = nullptr;
        if (resourcesLock) {
            resourcesLock->unlock();
        }
        const bool allocOk =
            alloc_and_upload_array(tmp, src, n, cudaStreamOpaque, label, outError);
        if (!relock_resources_after_offlock_upload(resources, resourcesLock, outError)) {
            if (tmp) {
                cudaFree(tmp);
            }
            return false;
        }
        if (!allocOk) {
            return false;
        }

        const size_t oldBytes = static_cast<size_t>(std::max(0, currentN)) * sizeof(float);
        if (dst) {
            if (!retire_ptr_locked(resources, dst, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                cudaFree(tmp);
                return false;
            }
        }

        dst = tmp;
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

        int domainBegin = 0;
        while (domainBegin < n && !std::isfinite(src.lambda_nm[static_cast<size_t>(domainBegin)])) {
            ++domainBegin;
        }
        int domainEnd = n - 1;
        while (domainEnd > domainBegin && !std::isfinite(src.lambda_nm[static_cast<size_t>(domainEnd)])) {
            --domainEnd;
        }
        dst.domainBegin = domainBegin;
        dst.domainEnd = domainEnd;

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

        if (!enqueue_host_to_device_copy(
                HostToDeviceCopyRequest{"alloc_and_upload_curve", "curve.x"},
                dst.x,
                src.lambda_nm.data(),
                bytes,
                cudaStreamOpaque,
                outError)) {
            outError = std::string("curve.x upload failed: ") + outError;
            free_curve(dst);
            return false;
        }
        if (!enqueue_host_to_device_copy(
                HostToDeviceCopyRequest{"alloc_and_upload_curve", "curve.y"},
                dst.y,
                src.linear.data(),
                bytes,
                cudaStreamOpaque,
                outError)) {
            outError = std::string("curve.y upload failed: ") + outError;
            free_curve(dst);
            return false;
        }

        dst.n = n;
        return true;
#endif
    }

    static bool upload_curve_locked(
        Resources& resources,
        DeviceCurve& dst,
        const Spectral::Curve& src,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>* resourcesLock,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)dst;
        (void)src;
        (void)cudaStreamOpaque;
        (void)resourcesLock;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (src.lambda_nm.empty() || src.linear.empty() || src.lambda_nm.size() != src.linear.size()) {
            outError = std::string(label) + ": curve has no samples or mismatched arrays";
            return false;
        }
        const int n = static_cast<int>(src.lambda_nm.size());
        if (n <= 0) {
            outError = std::string(label) + ": curve sample count invalid";
            return false;
        }

        const size_t bytes = static_cast<size_t>(n) * sizeof(float);

        // If the allocation matches, update in place to avoid alloc/free churn (common during slider scrubs).
        if (dst.x && dst.y && dst.n == n) {
            const char* baseLabel = label ? label : "curve";
            const std::string labelX = std::string(baseLabel) + ".x";
            int domainBegin = 0;
            while (domainBegin < n && !std::isfinite(src.lambda_nm[static_cast<size_t>(domainBegin)])) {
                ++domainBegin;
            }
            int domainEnd = n - 1;
            while (domainEnd > domainBegin && !std::isfinite(src.lambda_nm[static_cast<size_t>(domainEnd)])) {
                --domainEnd;
            }

            const bool waitOk = wait_for_frame_use_events_locked(
                resources,
                cudaStreamOpaque,
                baseLabel,
                outError);
            const bool copyXOk = waitOk && enqueue_host_to_device_copy(
                                               HostToDeviceCopyRequest{"upload_curve_locked", labelX.c_str()},
                                               dst.x,
                                               src.lambda_nm.data(),
                                               bytes,
                                               cudaStreamOpaque,
                                               outError);
            const std::string labelY = std::string(baseLabel) + ".y";
            const bool copyYOk = copyXOk && enqueue_host_to_device_copy(
                                                HostToDeviceCopyRequest{"upload_curve_locked", labelY.c_str()},
                                                dst.y,
                                                src.linear.data(),
                                                bytes,
                                                cudaStreamOpaque,
                                                outError);
            if (!copyXOk) {
                outError = std::string(baseLabel) + ".x upload failed: " + outError;
                return false;
            }
            if (!copyYOk) {
                outError = std::string(baseLabel) + ".y upload failed: " + outError;
                return false;
            }

            dst.domainBegin = domainBegin;
            dst.domainEnd = domainEnd;
            dst.n = n;
            return true;
        }

        DeviceCurve tmp{};
        if (resourcesLock) {
            resourcesLock->unlock();
        }
        const bool allocOk =
            alloc_and_upload_curve(tmp, src, cudaStreamOpaque, outError);
        if (!relock_resources_after_offlock_upload(resources, resourcesLock, outError)) {
            free_curve(tmp);
            return false;
        }
        if (!allocOk) {
            outError = std::string(label) + ": " + outError;
            return false;
        }

        const size_t oldBytes = static_cast<size_t>(std::max(0, dst.n)) * sizeof(float);
        if (dst.x) {
            if (!retire_ptr_locked(resources, dst.x, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                free_curve(tmp);
                return false;
            }
            dst.x = nullptr;
        }
        if (dst.y) {
            if (!retire_ptr_locked(resources, dst.y, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                free_curve(tmp);
                return false;
            }
            dst.y = nullptr;
        }

        dst = tmp;
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
        if (!enqueue_host_to_device_copy(
                HostToDeviceCopyRequest{"alloc_and_upload_spectral_samples", label},
                dst.y,
                src.data(),
                bytes,
                cudaStreamOpaque,
                outError)) {
            free_curve(dst);
            return false;
        }
        dst.n = n;
        dst.domainBegin = 0;
        dst.domainEnd = n - 1;
        return true;
#endif
    }

    static bool upload_spectral_samples_locked(
        Resources& resources,
        DeviceCurve& dst,
        const std::vector<float>& src,
        void* cudaStreamOpaque,
        std::unique_lock<std::mutex>* resourcesLock,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)dst;
        (void)src;
        (void)cudaStreamOpaque;
        (void)resourcesLock;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (src.empty()) {
            outError = std::string(label ? label : "spectral samples") + " array is empty";
            return false;
        }

        const int n = static_cast<int>(src.size());
        if (n <= 0) {
            outError = std::string(label ? label : "spectral samples") + " sample count invalid";
            return false;
        }

        const size_t bytes = static_cast<size_t>(n) * sizeof(float);
        if (dst.y && dst.n == n) {
            const bool waitOk = wait_for_frame_use_events_locked(
                resources,
                cudaStreamOpaque,
                label,
                outError);
            const bool copyOk = waitOk && enqueue_host_to_device_copy(
                                              HostToDeviceCopyRequest{"upload_spectral_samples_locked", label},
                                              dst.y,
                                              src.data(),
                                              bytes,
                                              cudaStreamOpaque,
                                              outError);
            if (!copyOk) {
                return false;
            }
            dst.n = n;
            dst.domainBegin = 0;
            dst.domainEnd = n - 1;
            return true;
        }

        DeviceCurve tmp{};
        if (resourcesLock) {
            resourcesLock->unlock();
        }
        const bool allocOk =
            alloc_and_upload_spectral_samples(tmp, src, cudaStreamOpaque, label, outError);
        if (!relock_resources_after_offlock_upload(resources, resourcesLock, outError)) {
            free_curve(tmp);
            return false;
        }
        if (!allocOk) {
            return false;
        }

        if (!retire_curve_locked(resources, dst, cudaStreamOpaque, label, outError)) {
            free_curve(tmp);
            return false;
        }

        dst = tmp;
        return true;
#endif
    }

#endif

    Resources::~Resources() noexcept {
        try {
            drain_retire_queue_blocking(*this);
            free_curve(densB);
            free_curve(densG);
            free_curve(densR);
            free_density_layers(*this);
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
            free_gaussian_kernel(grainBlurKernel);
            free_gaussian_kernel(grainBlurKernelMid);
            free_gaussian_kernel(grainBlurKernelCoarse);
            for (int layer = 0; layer < 3; ++layer) {
                for (int ch = 0; ch < 3; ++ch) {
                    free_gaussian_kernel(grainDyeKernel[layer][ch]);
                }
            }
            for (int i = 0; i < 3; ++i) {
                free_gaussian_kernel(halationKernel[i]);
                free_gaussian_kernel(halationScatterKernel[i]);
            }
            free_optics_scratch(*this, scannerScratch, nullptr);
            for (auto& kernel : spatialDirKernels) {
                free_gaussian_kernel(kernel);
            }
            free_spatial_dir_scratch(*this, spatialDirScratch, nullptr);
            free_shared_tmp_plane(*this);
            free_stbn(*this);
            free_wang(*this);
            free_print_payloads(*this);
            free_hanatos(*this);
            free_hanatos_integrated(*this);
            free_mallett_basis(*this);
            free_scan_error_readbacks(*this);
            for (PendingFrameUseEvent& entry : pendingFrameUseEvents) {
                release_frame_use_event_entry(entry);
            }
            pendingFrameUseEvents.clear();
            asyncDeviceAllocPointers.clear();
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    Resources* create() noexcept {
        try {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
            reap_deferred_destroy_queue("create");
#endif
            Resources* r = new Resources();
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
            int dev = -1;
            if (cudaGetDevice(&dev) == cudaSuccess) {
                r->deviceId = dev;
            }
            void* contextOpaque = nullptr;
            std::string contextError;
            if (query_current_cuda_context(contextOpaque, contextError)) {
                r->ownerContextOpaque = contextOpaque;
            }
#endif
            return r;
        } catch (...) {
            return nullptr;
        }
    }

    void destroy(Resources* resources) noexcept {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        try {
            delete resources;
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
#else
        try {
            if (!resources) {
                return;
            }

            reap_deferred_destroy_queue("destroy_pre");

            const int ownerDeviceId = resources->deviceId;
            void* ownerContextOpaque = resources->ownerContextOpaque;
            int currentDeviceId = -1;
            std::string currentDeviceError;
            const bool currentDeviceValid = query_current_cuda_device(currentDeviceId, currentDeviceError);
            void* currentContextOpaque = nullptr;
            std::string currentContextError;
            const bool currentContextValid = query_current_cuda_context(currentContextOpaque, currentContextError);

            bool deviceMatch = false;
            bool contextMatch = false;
            bool ownerMatch = owner_matches_current(
                ownerDeviceId,
                ownerContextOpaque,
                currentDeviceId,
                currentDeviceValid,
                currentContextOpaque,
                currentContextValid,
                deviceMatch,
                contextMatch);

            bool switchAttempted = false;
            bool switchSucceeded = false;
            int restoreDeviceId = currentDeviceId;
            bool restoreDevice = false;

            if (!ownerMatch && ownerDeviceId >= 0 &&
                (!currentDeviceValid || currentDeviceId != ownerDeviceId)) {
                switchAttempted = true;
                const cudaError_t setErr = cudaSetDevice(ownerDeviceId);
                if (setErr == cudaSuccess) {
                    switchSucceeded = true;
                    restoreDevice = currentDeviceValid && currentDeviceId != ownerDeviceId;

                    int switchedDeviceId = -1;
                    std::string switchedDeviceError;
                    const bool switchedDeviceValid = query_current_cuda_device(switchedDeviceId, switchedDeviceError);

                    void* switchedContextOpaque = nullptr;
                    std::string switchedContextError;
                    const bool switchedContextValid =
                        query_current_cuda_context(switchedContextOpaque, switchedContextError);

                    ownerMatch = owner_matches_current(
                        ownerDeviceId,
                        ownerContextOpaque,
                        switchedDeviceId,
                        switchedDeviceValid,
                        switchedContextOpaque,
                        switchedContextValid,
                        deviceMatch,
                        contextMatch);

                    if (switchedDeviceValid) {
                        currentDeviceId = switchedDeviceId;
                    }
                    if (switchedContextValid) {
                        currentContextOpaque = switchedContextOpaque;
                    }
                    if (!switchedDeviceError.empty()) {
                        currentDeviceError = switchedDeviceError;
                    }
                    if (!switchedContextError.empty()) {
                        currentContextError = switchedContextError;
                    }
                } else {
                    currentContextError = std::string("cudaSetDevice failed: ") + (cudaGetErrorString(setErr) ? cudaGetErrorString(setErr) : "(unknown)");
                }
            }

            if (ownerMatch) {
                std::size_t deferredQueueDepth = 0;
                {
                    std::lock_guard<std::mutex> lock(deferred_destroy_mutex());
                    deferredQueueDepth = deferred_destroy_queue().size();
                }
                std::string detail;
                if (!currentDeviceError.empty()) {
                    detail = currentDeviceError;
                }
                if (!currentContextError.empty()) {
                    if (!detail.empty()) {
                        detail += "; ";
                    }
                    detail += currentContextError;
                }
                trace_teardown_event(
                    "destroy",
                    "direct_free",
                    ownerDeviceId,
                    ownerContextOpaque,
                    currentDeviceId,
                    currentContextOpaque,
                    deviceMatch,
                    contextMatch,
                    switchAttempted,
                    switchSucceeded,
                    false,
                    false,
                    deferredQueueDepth,
                    detail);
                delete resources;

                if (restoreDevice && restoreDeviceId >= 0 && restoreDeviceId != ownerDeviceId) {
                    (void)cudaSetDevice(restoreDeviceId);
                }
                reap_deferred_destroy_queue("destroy_post");
                return;
            }

            if (restoreDevice && restoreDeviceId >= 0 && restoreDeviceId != ownerDeviceId) {
                (void)cudaSetDevice(restoreDeviceId);
            }

            bool managerRetireAttempted = false;
            bool managerRetireAccepted = false;
            std::string managerRetireError;
            if (ownerDeviceId >= 0 && ownerContextOpaque) {
                managerRetireAttempted = true;
                managerRetireAccepted =
                    JuicerProcess::root().retire_idle_context(ownerDeviceId, ownerContextOpaque, managerRetireError);
            }

            std::size_t deferredQueueDepth = 0;
            {
                std::lock_guard<std::mutex> lock(deferred_destroy_mutex());
                auto& queue = deferred_destroy_queue();
                DeferredDestroyEntry entry{};
                entry.resources = resources;
                entry.ownerDeviceId = ownerDeviceId;
                entry.ownerContextOpaque = ownerContextOpaque;
                queue.push_back(entry);
                deferredQueueDepth = queue.size();
            }

            std::string detail;
            if (!currentDeviceError.empty()) {
                detail = currentDeviceError;
            }
            if (!currentContextError.empty()) {
                if (!detail.empty()) {
                    detail += "; ";
                }
                detail += currentContextError;
            }
            if (!managerRetireError.empty()) {
                if (!detail.empty()) {
                    detail += "; ";
                }
                detail += managerRetireError;
            }
            trace_teardown_event(
                "destroy",
                "deferred_enqueue",
                ownerDeviceId,
                ownerContextOpaque,
                currentDeviceId,
                currentContextOpaque,
                deviceMatch,
                contextMatch,
                switchAttempted,
                switchSucceeded,
                managerRetireAttempted,
                managerRetireAccepted,
                deferredQueueDepth,
                detail);
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
#endif
    }

    bool reap_retired_allocations(Resources& resources, std::size_t& reclaimedBytes, std::string& outError) {
        reclaimedBytes = 0;
        outError.clear();
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }
        const std::size_t beforeBytes = resources.retireBytes;
        reap_retire_queue_locked(resources);
        const std::size_t afterBytes = resources.retireBytes;
        reclaimedBytes = (beforeBytes >= afterBytes) ? (beforeBytes - afterBytes) : 0;
        return true;
#endif
    }

    void snapshot_scratch_stage1_state(
        Resources& resources,
        ResourceManager::ScratchStage1DecisionState& outState) noexcept {
        outState = ResourceManager::ScratchStage1DecisionState{};
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        std::lock_guard<std::mutex> lock(resources.m);
        refresh_scratch_residency_state_locked(resources);
        outState.policyLiveRetainedBytes = resources.scratchResidency.policyLiveRetainedBytes;
        outState.retainedGeneration = resources.scratchResidency.retainedGeneration;
#else
        (void)resources;
#endif
    }

    void snapshot_scratch_residency_view(
        Resources& resources,
        ResourceManager::ScratchResidencyView& outView) noexcept {
        outView = ResourceManager::ScratchResidencyView{};
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        std::lock_guard<std::mutex> lock(resources.m);
        refresh_scratch_residency_state_locked(resources);
        for (std::size_t i = 0; i < ResourceManager::kScratchPolicyCandidateCount; ++i) {
            outView.candidates[i].candidate = ResourceManager::kScratchPolicyCandidateOrder[i];
            outView.candidates[i].liveRetainedBytes = resources.scratchResidency.candidateLiveBytes[i];
        }
        outView.helperNonPolicyBytes = resources.scratchResidency.helperNonPolicyBytes;
        outView.helperSharedBytes = resources.scratchResidency.helperSharedBytes;
        outView.helperNonPolicyTotalBytes = resources.scratchResidency.helperNonPolicyTotalBytes;
        outView.policyLiveRetainedBytes = resources.scratchResidency.policyLiveRetainedBytes;
        outView.totalLiveRetainedBytes = resources.scratchResidency.totalLiveRetainedBytes;
        outView.retirePendingScratchBytes = static_cast<std::uint64_t>(
            std::min<std::size_t>(
                resources.retireScratchBytes,
                static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
        outView.retainedGeneration = resources.scratchResidency.retainedGeneration;
        outView.overflow = resources.scratchResidency.overflow;
#else
        (void)resources;
#endif
    }

    bool retain_frame_use_event(
        Resources& resources,
        void*& eventOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)eventOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        outError.clear();
        if (!eventOpaque) {
            return true;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        reap_frame_use_events_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        try {
            Resources::PendingFrameUseEvent entry{};
            entry.eventOpaque = eventOpaque;
            resources.pendingFrameUseEvents.push_back(entry);
        } catch (...) {
            outError = "frame use event retention failed";
            return false;
        }

        eventOpaque = nullptr;
        return true;
#endif
    }


// Split implementation sections (single-TU include model to preserve exact behavior while
// reducing monolithic file size and keeping ownership boundaries explicit).
#include "Cuda/JuicerCudaResourcesServing.inc"
    static bool prepare_focused_route_resources(
        Resources& resources,
        const DirectResourcePreparation& request,
        bool printRoute,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)request;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        outError.clear();
        if (!request.recipe || !request.exposureTables || !request.spdSInv ||
            !request.filmRawConfig || !request.scannerTables || !request.scannerColor ||
            !request.scannerLutDescriptor) {
            outError = "direct resource preparation request is incomplete";
            return false;
        }

        const RenderRecipe& recipe = *request.recipe;
        const FilmRawRecipe& filmRaw = recipe.filmRaw;
        const FilmDevelopRecipe& filmDevelop = recipe.filmDevelop;
        const DirCouplersRecipe& dirCouplers = recipe.dirCouplers;
        const DensityBoundsRecipe& densityBounds = recipe.densityBounds;
        const bool wantDensityLayers =
            recipe.visualGrain.active && recipe.visualGrain.sublayersActive;
        const Scanner::ScannerSpectralLutDescriptor& scannerDescriptor =
            *request.scannerLutDescriptor;
        if ((printRoute ? !recipe.printStructuralReady : !recipe.directStructuralReady) ||
            Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute) != printRoute ||
            recipe.hash == 0 ||
            filmRaw.finalSensitivityHash == 0 ||
            filmDevelop.normalizedDensityCurvesHash == 0 ||
            densityBounds.hash == 0 ||
            scannerDescriptor.hash == 0 ||
            scannerDescriptor.densityBoundsHash != densityBounds.hash) {
            outError = "direct resource descriptor mismatch";
            return false;
        }
        if (request.exposureTables->K != Spectral::kNumSamples ||
            request.scannerTables->K != Spectral::kNumSamples ||
            filmDevelop.logExposure.empty() ||
            filmDevelop.logExposure.size() != filmDevelop.normalizedDensityCurves.size()) {
            outError = "direct resource host derivation shape mismatch";
            return false;
        }

        std::lock_guard<std::mutex> servingUpdateLock(resources.servingUpdateMutex);
        std::unique_lock<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        const bool alreadyPrepared =
            resources.directFinalSensitivityHash == filmRaw.finalSensitivityHash &&
            resources.directDensityCurvesHash == filmDevelop.normalizedDensityCurvesHash &&
            resources.directDensityLayersHash == (wantDensityLayers ? filmDevelop.densityCurvesLayersHash : 0) &&
            resources.directDirHash == (dirCouplers.active ? dirCouplers.hash : 0) &&
            resources.directDensityBoundsHash == densityBounds.hash &&
            resources.directScannerDescriptorHash == scannerDescriptor.hash &&
            resources.directSelectedMethod == filmRaw.rgbToRawMethod &&
            resources.sensB.x && resources.sensG.x && resources.sensR.x &&
            resources.densB.x && resources.densG.x && resources.densR.x &&
            (!wantDensityLayers ||
             (resources.hasDensityCurvesLayers &&
              resources.densityCurvesLayers[0][0] && resources.densityCurvesLayers[0][1] &&
              resources.densityCurvesLayers[0][2] && resources.densityCurvesLayers[1][0] &&
              resources.densityCurvesLayers[1][1] && resources.densityCurvesLayers[1][2] &&
              resources.densityCurvesLayers[2][0] && resources.densityCurvesLayers[2][1] &&
              resources.densityCurvesLayers[2][2])) &&
            (wantDensityLayers || !resources.hasDensityCurvesLayers) &&
            (!dirCouplers.active ||
             (resources.dirDensB.x && resources.dirDensG.x && resources.dirDensR.x)) &&
            resources.tablesAx && resources.tablesAy && resources.tablesAz && resources.tablesIllum &&
            (printRoute ? resources.scanPrint.tables.epsC : resources.scanNegative.tables.epsC) &&
            (printRoute ? resources.scanPrintLut.canonical_ready() : resources.scanNegativeLut.canonical_ready()) &&
            (printRoute ? resources.scanPrintLut.hash : resources.scanNegativeLut.hash) == scannerDescriptor.hash &&
            ((filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025 &&
              resources.hanatosLut && resources.hanatosLutIntegrated &&
              resources.hanatosIntegratedKeyHash == filmRaw.hanatosLutHash &&
              !resources.mallettBasis) ||
             (filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Mallett2019 &&
              resources.mallettBasis && !resources.hanatosLut && !resources.hanatosLutIntegrated));
        if (alreadyPrepared) {
            return true;
        }

        Spectral::Curve sensB;
        Spectral::Curve sensG;
        Spectral::Curve sensR;
        sensB.lambda_nm.assign(Spectral::gShape.wavelengths.begin(), Spectral::gShape.wavelengths.end());
        sensG.lambda_nm = sensB.lambda_nm;
        sensR.lambda_nm = sensB.lambda_nm;
        sensB.linear.resize(Spectral::kNumSamples);
        sensG.linear.resize(Spectral::kNumSamples);
        sensR.linear.resize(Spectral::kNumSamples);
        for (int sample = 0; sample < Spectral::kNumSamples; ++sample) {
            const auto& rgb = filmRaw.finalSensitivity[static_cast<std::size_t>(sample)];
            sensB.linear[static_cast<std::size_t>(sample)] = rgb[2];
            sensG.linear[static_cast<std::size_t>(sample)] = rgb[1];
            sensR.linear[static_cast<std::size_t>(sample)] = rgb[0];
        }

        Spectral::Curve densB;
        Spectral::Curve densG;
        Spectral::Curve densR;
        densB.lambda_nm = filmDevelop.logExposure;
        densG.lambda_nm = filmDevelop.logExposure;
        densR.lambda_nm = filmDevelop.logExposure;
        densB.linear.resize(filmDevelop.normalizedDensityCurves.size());
        densG.linear.resize(filmDevelop.normalizedDensityCurves.size());
        densR.linear.resize(filmDevelop.normalizedDensityCurves.size());
        for (std::size_t sample = 0; sample < filmDevelop.normalizedDensityCurves.size(); ++sample) {
            const auto& rgb = filmDevelop.normalizedDensityCurves[sample];
            densB.linear[sample] = rgb[2];
            densG.linear[sample] = rgb[1];
            densR.linear[sample] = rgb[0];
        }

        if (!upload_curve_locked(resources, resources.sensB, sensB, cudaStreamOpaque, &lock, "direct finalSensB", outError) ||
            !upload_curve_locked(resources, resources.sensG, sensG, cudaStreamOpaque, &lock, "direct finalSensG", outError) ||
            !upload_curve_locked(resources, resources.sensR, sensR, cudaStreamOpaque, &lock, "direct finalSensR", outError) ||
            !upload_curve_locked(resources, resources.densB, densB, cudaStreamOpaque, &lock, "direct normalizedDensB", outError) ||
            !upload_curve_locked(resources, resources.densG, densG, cudaStreamOpaque, &lock, "direct normalizedDensG", outError) ||
            !upload_curve_locked(resources, resources.densR, densR, cudaStreamOpaque, &lock, "direct normalizedDensR", outError)) {
            return false;
        }

        if (wantDensityLayers) {
            const int densitySamples = static_cast<int>(filmDevelop.logExposure.size());
            if (!filmDevelop.densityCurvesLayersRequired ||
                filmDevelop.densityCurvesLayersHash == 0 ||
                densitySamples <= 0) {
                outError = "MissingRequiredResource phase=9B field=density_curves_layers";
                return false;
            }
            for (int layer = 0; layer < 3; ++layer) {
                for (int ch = 0; ch < 3; ++ch) {
                    if (static_cast<int>(filmDevelop.densityCurvesLayers[layer][ch].size()) !=
                        densitySamples) {
                        outError = "MalformedRequiredProfileData phase=9B field=density_curves_layers shape";
                        return false;
                    }
                }
            }
            bool canReuse =
                resources.hasDensityCurvesLayers &&
                resources.directDensityLayersHash == filmDevelop.densityCurvesLayersHash &&
                resources.densityCurvesLayersChannelN[0] == densitySamples &&
                resources.densityCurvesLayersChannelN[1] == densitySamples &&
                resources.densityCurvesLayersChannelN[2] == densitySamples;
            for (int layer = 0; layer < 3 && canReuse; ++layer) {
                for (int ch = 0; ch < 3; ++ch) {
                    canReuse = canReuse && resources.densityCurvesLayers[layer][ch] != nullptr;
                }
            }
            if (!canReuse) {
                if (!retire_density_layers_locked(
                        resources,
                        cudaStreamOpaque,
                        "focused density_curves_layers",
                        outError)) {
                    return false;
                }
            }
            for (int layer = 0; layer < 3; ++layer) {
                for (int ch = 0; ch < 3; ++ch) {
                    if (!upload_array_locked(
                            resources,
                            resources.densityCurvesLayers[layer][ch],
                            canReuse ? resources.densityCurvesLayersChannelN[ch] : 0,
                            filmDevelop.densityCurvesLayers[layer][ch].data(),
                            densitySamples,
                            cudaStreamOpaque,
                            &lock,
                            "focused density_curves_layers",
                            outError)) {
                        return false;
                    }
                }
            }
            for (int ch = 0; ch < 3; ++ch) {
                resources.densityCurvesLayersChannelN[ch] = densitySamples;
            }
            resources.hasDensityCurvesLayers = 1;
            resources.directDensityLayersHash = filmDevelop.densityCurvesLayersHash;
        } else if (resources.hasDensityCurvesLayers || resources.directDensityLayersHash != 0) {
            if (!retire_density_layers_locked(
                    resources,
                    cudaStreamOpaque,
                    "inactive focused density_curves_layers",
                    outError)) {
                return false;
            }
        }

        if (dirCouplers.active) {
            if (dirCouplers.hash == 0 ||
                dirCouplers.precorrectedDensityCurvesHash == 0 ||
                dirCouplers.precorrectedDensityCurves.size() != filmDevelop.logExposure.size()) {
                outError = "direct DIR resource descriptor mismatch";
                return false;
            }
            Spectral::Curve dirB = densB;
            Spectral::Curve dirG = densG;
            Spectral::Curve dirR = densR;
            for (std::size_t sample = 0; sample < dirCouplers.precorrectedDensityCurves.size(); ++sample) {
                const auto& rgb = dirCouplers.precorrectedDensityCurves[sample];
                dirB.linear[sample] = rgb[2];
                dirG.linear[sample] = rgb[1];
                dirR.linear[sample] = rgb[0];
            }
            if (!upload_curve_locked(resources, resources.dirDensB, dirB, cudaStreamOpaque, &lock, "direct DIR densB", outError) ||
                !upload_curve_locked(resources, resources.dirDensG, dirG, cudaStreamOpaque, &lock, "direct DIR densG", outError) ||
                !upload_curve_locked(resources, resources.dirDensR, dirR, cudaStreamOpaque, &lock, "direct DIR densR", outError)) {
                return false;
            }
        } else {
            if (!retire_curve_locked(resources, resources.dirDensB, cudaStreamOpaque, "disabled direct DIR densB", outError) ||
                !retire_curve_locked(resources, resources.dirDensG, cudaStreamOpaque, "disabled direct DIR densG", outError) ||
                !retire_curve_locked(resources, resources.dirDensR, cudaStreamOpaque, "disabled direct DIR densR", outError)) {
                return false;
            }
        }

        const Spectral::SpectralTables& exposureTables = *request.exposureTables;
        const int exposureK = exposureTables.K;
        if (!upload_array_locked(resources, resources.tablesAx, resources.tablesK, exposureTables.Ax.data(), exposureK, cudaStreamOpaque, &lock, "direct tablesAx", outError) ||
            !upload_array_locked(resources, resources.tablesAy, resources.tablesK, exposureTables.Ay.data(), exposureK, cudaStreamOpaque, &lock, "direct tablesAy", outError) ||
            !upload_array_locked(resources, resources.tablesAz, resources.tablesK, exposureTables.Az.data(), exposureK, cudaStreamOpaque, &lock, "direct tablesAz", outError) ||
            !upload_array_locked(resources, resources.tablesIllum, resources.tablesK, exposureTables.illum.data(), exposureK, cudaStreamOpaque, &lock, "direct tablesIllum", outError)) {
            return false;
        }
        resources.tablesK = exposureK;
        std::copy_n(request.spdSInv, 9, resources.spdSInv);
        std::copy_n(request.filmRawConfig->refIllumWhiteXYZ, 3, resources.refIllumWhiteXYZ);

        Spectral::SpectralContext& context = Spectral::context();
        if (filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025) {
            const int n = context.hanSpectra.size;
            const int k = context.hanSpectra.numSamples;
            if (!context.hanatosAvailable.load(std::memory_order_acquire) ||
                n <= 0 || k != Spectral::kNumSamples || context.hanSpectra.data.empty()) {
                outError = "selected Hanatos resource family is unavailable";
                return false;
            }
            const int lutCount = n * n * k;
            if (!upload_array_locked(resources, resources.hanatosLut, resources.hanatosN * resources.hanatosN * k, context.hanSpectra.data.data(), lutCount, cudaStreamOpaque, &lock, "direct Hanatos LUT", outError)) {
                return false;
            }
            resources.hanatosN = n;

            std::vector<float> integrated;
            if (!Precompute::build_direct_hanatos_integrated_lut_cpu(
                    context,
                    filmRaw,
                    request.filmRawConfig->refIllumWhiteXYZ,
                    integrated,
                    outError)) {
                return false;
            }
            const int integratedCount = n * n * 4;
            if (!upload_array_locked(resources, resources.hanatosLutIntegrated, resources.hanatosNIntegrated * resources.hanatosNIntegrated * 4, integrated.data(), integratedCount, cudaStreamOpaque, &lock, "direct Hanatos integrated LUT", outError)) {
                return false;
            }
            resources.hanatosNIntegrated = n;
            resources.hanatosIntegratedKeyHash = filmRaw.hanatosLutHash;
            if (resources.mallettBasis) {
                const std::size_t bytes = static_cast<std::size_t>(resources.mallettBasisK) * 3u * sizeof(float);
                if (!retire_ptr_locked(resources, resources.mallettBasis, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "unselected Mallett basis", outError)) {
                    return false;
                }
                resources.mallettBasis = nullptr;
                resources.mallettBasisK = 0;
            }
        } else {
            const int k = context.mallettBasis.rows;
            if (!context.mallettAvailable.load(std::memory_order_acquire) ||
                k != Spectral::kNumSamples || context.mallettBasis.cols != 3 ||
                context.mallettBasis.data.empty()) {
                outError = "selected Mallett resource family is unavailable";
                return false;
            }
            const int count = k * 3;
            if (!upload_array_locked(resources, resources.mallettBasis, resources.mallettBasisK * 3, context.mallettBasis.data.data(), count, cudaStreamOpaque, &lock, "direct Mallett basis", outError)) {
                return false;
            }
            resources.mallettBasisK = k;
            if (resources.hanatosLut) {
                const std::size_t bytes = static_cast<std::size_t>(resources.hanatosN) *
                                          static_cast<std::size_t>(resources.hanatosN) * Spectral::kNumSamples * sizeof(float);
                if (!retire_ptr_locked(resources, resources.hanatosLut, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "unselected Hanatos LUT", outError)) {
                    return false;
                }
                resources.hanatosLut = nullptr;
                resources.hanatosN = 0;
            }
            if (resources.hanatosLutIntegrated) {
                const std::size_t bytes = static_cast<std::size_t>(resources.hanatosNIntegrated) *
                                          static_cast<std::size_t>(resources.hanatosNIntegrated) * 4u * sizeof(float);
                if (!retire_ptr_locked(resources, resources.hanatosLutIntegrated, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "unselected Hanatos integrated LUT", outError)) {
                    return false;
                }
                resources.hanatosLutIntegrated = nullptr;
                resources.hanatosNIntegrated = 0;
                resources.hanatosIntegratedKeyHash = 0;
            }
        }

        const Spectral::SpectralTables& mediumTables = *request.scannerTables;
        Resources::DeviceScanMedium& scan = printRoute ? resources.scanPrint : resources.scanNegative;
        const int scanK = mediumTables.K;
        if (!upload_array_locked(resources, scan.tables.epsC, scan.tables.K, mediumTables.epsC.data(), scanK, cudaStreamOpaque, &lock, "direct medium epsC", outError) ||
            !upload_array_locked(resources, scan.tables.epsM, scan.tables.K, mediumTables.epsM.data(), scanK, cudaStreamOpaque, &lock, "direct medium epsM", outError) ||
            !upload_array_locked(resources, scan.tables.epsY, scan.tables.K, mediumTables.epsY.data(), scanK, cudaStreamOpaque, &lock, "direct medium epsY", outError) ||
            !upload_array_locked(resources, scan.tables.Ax, scan.tables.K, mediumTables.Ax.data(), scanK, cudaStreamOpaque, &lock, "direct medium Ax", outError) ||
            !upload_array_locked(resources, scan.tables.Ay, scan.tables.K, mediumTables.Ay.data(), scanK, cudaStreamOpaque, &lock, "direct medium Ay", outError) ||
            !upload_array_locked(resources, scan.tables.Az, scan.tables.K, mediumTables.Az.data(), scanK, cudaStreamOpaque, &lock, "direct medium Az", outError)) {
            return false;
        }
        if (mediumTables.hasBaseline) {
            if (!upload_array_locked(resources, scan.tables.baseDensityMin, scan.tables.K, mediumTables.baseDensityMin.data(), scanK, cudaStreamOpaque, &lock, "direct medium baseDensityMin", outError)) {
                return false;
            }
        } else if (scan.tables.baseDensityMin) {
            const std::size_t bytes = static_cast<std::size_t>(scan.tables.K) * sizeof(float);
            if (!retire_ptr_locked(resources, scan.tables.baseDensityMin, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "direct medium baseDensityMin", outError)) {
                return false;
            }
            scan.tables.baseDensityMin = nullptr;
        }
        scan.tables.K = scanK;
        scan.tables.hasBaseline = mediumTables.hasBaseline ? 1 : 0;
        scan.tables.invYn = mediumTables.invYn;
        scan.mediumIsNegative = printRoute ? 0 : 1;
        for (int channel = 0; channel < 3; ++channel) {
            scan.min_cmy[channel] = printRoute
                                        ? densityBounds.dataMinCmy[static_cast<std::size_t>(channel)]
                                        : -densityBounds.dataMinCmy[static_cast<std::size_t>(channel)];
            scan.inv_max_cmy[channel] = densityBounds.invSpanCmy[static_cast<std::size_t>(channel)];
        }

        Scanner::ScannerMediumRuntime medium{};
        medium.medium = printRoute ? Scanner::ScannerMedium::Print : Scanner::ScannerMedium::Negative;
        medium.tables = request.scannerTables;
        medium.color = request.scannerColor;
        for (int channel = 0; channel < 3; ++channel) {
            medium.range.min_cmy[channel] = scan.min_cmy[channel];
            medium.range.inv_max_cmy[channel] = scan.inv_max_cmy[channel];
            medium.range.max_cmy[channel] =
                scan.inv_max_cmy[channel] > 0.0f ? 1.0f / scan.inv_max_cmy[channel] : 0.0f;
        }

        Precompute::CanonicalScanLutCpu lutCpu;
        lock.unlock();
        const bool lutBuilt = Precompute::build_canonical_scan_lut_cpu(
            medium,
            scannerDescriptor.lutResolution,
            lutCpu,
            outError);
        lock.lock();
        if (!validate_resource_owner_locked(resources, outError, false) || !lutBuilt) {
            return false;
        }
        Resources::DeviceSpectralLut& lut = printRoute ? resources.scanPrintLut : resources.scanNegativeLut;
        struct NextCanonicalScanLut {
            float* log2PchipXYZ = nullptr;
            float* slopeC = nullptr;
            float* slopeM = nullptr;
            float* slopeY = nullptr;
            float* cellMin = nullptr;
            float* cellMax = nullptr;
        } next;
        auto free_next = [&]() {
            cudaFree(next.log2PchipXYZ);
            cudaFree(next.slopeC);
            cudaFree(next.slopeM);
            cudaFree(next.slopeY);
            cudaFree(next.cellMin);
            cudaFree(next.cellMax);
            next = {};
        };
        auto upload_next = [&](float*& destination, const std::vector<float>& source, const char* label) {
            void* raw = nullptr;
            if (!alloc_and_upload_bytes(
                    raw,
                    source.data(),
                    source.size() * sizeof(float),
                    cudaStreamOpaque,
                    label,
                    outError)) {
                return false;
            }
            destination = static_cast<float*>(raw);
            return true;
        };

        lock.unlock();
        const bool canonicalUploaded =
            upload_next(next.log2PchipXYZ, lutCpu.log2XYZ, "direct scan PCHIP log2 XYZ") &&
            upload_next(next.slopeC, lutCpu.slopeC, "direct scan PCHIP C slopes") &&
            upload_next(next.slopeM, lutCpu.slopeM, "direct scan PCHIP M slopes") &&
            upload_next(next.slopeY, lutCpu.slopeY, "direct scan PCHIP Y slopes") &&
            upload_next(next.cellMin, lutCpu.cellMin, "direct scan PCHIP cell minima") &&
            upload_next(next.cellMax, lutCpu.cellMax, "direct scan PCHIP cell maxima");
        lock.lock();
        if (!validate_resource_owner_locked(resources, outError, false) || !canonicalUploaded) {
            free_next();
            return false;
        }

        const std::size_t oldVoxelBytes =
            static_cast<std::size_t>(lut.res) * lut.res * lut.res * 3u * sizeof(float);
        const std::size_t oldCellRes = lut.res > 0u ? static_cast<std::size_t>(lut.res - 1u) : 0u;
        const std::size_t oldCellBytes = oldCellRes * oldCellRes * oldCellRes * 3u * sizeof(float);
        auto retire_old = [&](float*& pointer, std::size_t bytes, const char* label) {
            if (!pointer) {
                return true;
            }
            if (!retire_ptr_locked(
                    resources,
                    pointer,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    label,
                    outError)) {
                return false;
            }
            pointer = nullptr;
            return true;
        };
        lut.hash = 0;
        if (!retire_old(lut.log2PchipXYZ, oldVoxelBytes, "direct scan PCHIP log2 XYZ") ||
            !retire_old(lut.slopeC, oldVoxelBytes, "direct scan PCHIP C slopes") ||
            !retire_old(lut.slopeM, oldVoxelBytes, "direct scan PCHIP M slopes") ||
            !retire_old(lut.slopeY, oldVoxelBytes, "direct scan PCHIP Y slopes") ||
            !retire_old(lut.cellMin, oldCellBytes, "direct scan PCHIP cell minima") ||
            !retire_old(lut.cellMax, oldCellBytes, "direct scan PCHIP cell maxima")) {
            free_next();
            return false;
        }
        lut.log2PchipXYZ = next.log2PchipXYZ;
        lut.slopeC = next.slopeC;
        lut.slopeM = next.slopeM;
        lut.slopeY = next.slopeY;
        lut.cellMin = next.cellMin;
        lut.cellMax = next.cellMax;
        next = {};
        lut.res = scannerDescriptor.lutResolution;
        lut.hash = scannerDescriptor.hash;

        resources.directFinalSensitivityHash = filmRaw.finalSensitivityHash;
        resources.directDensityCurvesHash = filmDevelop.normalizedDensityCurvesHash;
        resources.directDensityLayersHash = wantDensityLayers ? filmDevelop.densityCurvesLayersHash : 0;
        resources.directDirHash = dirCouplers.active ? dirCouplers.hash : 0;
        resources.directDensityBoundsHash = densityBounds.hash;
        resources.directScannerDescriptorHash = scannerDescriptor.hash;
        resources.directSelectedMethod = filmRaw.rgbToRawMethod;
        ++resources.directUploadCounter;
        resources.uploadedBuildCounter = 0;
        resources.uploadedCoreHash = 0;
        resources.uploadedDirHash = 0;
        return true;
#endif
    }

    bool prepare_direct_resources(
        Resources& resources,
        const DirectResourcePreparation& request,
        void* cudaStreamOpaque,
        std::string& outError) {
        return prepare_focused_route_resources(
            resources,
            request,
            false,
            cudaStreamOpaque,
            outError);
    }

    bool prepare_print_route_resources(
        Resources& resources,
        const PrintRouteResourcePreparation& request,
        void* cudaStreamOpaque,
        std::string& outError) {
        return prepare_focused_route_resources(
            resources,
            request,
            true,
            cudaStreamOpaque,
            outError);
    }
    namespace {

        template <typename T>
        void hash_print_descriptor_value(std::uint64_t& hash, const T& value) {
            Hash::hash_bytes_update(hash, &value, sizeof(value));
        }

        void hash_print_descriptor_cmy(std::uint64_t& hash, const CmyCcTriplet& value) {
            hash_print_descriptor_value(hash, value.c);
            hash_print_descriptor_value(hash, value.m);
            hash_print_descriptor_value(hash, value.y);
        }

        std::uint64_t hash_profile_density_curves(const Profiles::SpektrafilmPrintData& data) {
            if (data.logExposure.empty() || data.densityCurves.empty() ||
                data.logExposure.size() != data.densityCurves.size()) {
                return 0;
            }
            const Hash::FloatSpanHash curves = Hash::hash_float_span_with_nan_mask(
                &data.densityCurves[0][0],
                data.densityCurves.size() * 3u);
            return Hash::hash_uint64_values({Hash::hash_float_span(data.logExposure.data(), data.logExposure.size()),
                                             curves.valueHash,
                                             curves.nanMaskHash});
        }

        std::uint64_t hash_profile_sensitivities(const Profiles::SpektrafilmPrintData& data) {
            const Hash::FloatSpanHash sensitivities = Hash::hash_float_span_with_nan_mask(
                &data.linearSensitivity[0][0],
                data.linearSensitivity.size() * 3u);
            return Hash::hash_uint64_values({sensitivities.valueHash,
                                             sensitivities.nanMaskHash});
        }

        std::uint64_t hash_profile_film_density_tables(const Profiles::SpektrafilmFilmData& data) {
            const Hash::FloatSpanHash channelDensity = Hash::hash_float_span_with_nan_mask(
                &data.channelDensity[0][0],
                data.channelDensity.size() * 3u);
            const Hash::FloatSpanHash baseDensity = Hash::hash_float_span_with_nan_mask(
                data.baseDensity.data(),
                data.baseDensity.size());
            return Hash::hash_uint64_values({Hash::hash_float_span(data.wavelengths.data(), data.wavelengths.size()),
                                             channelDensity.valueHash,
                                             channelDensity.nanMaskHash,
                                             baseDensity.valueHash,
                                             baseDensity.nanMaskHash});
        }

        bool print_curve_ready(const DeviceCurveView& curve, int expectedSamples, bool requireX) {
            return curve.y && (!requireX || curve.x) && curve.n == expectedSamples &&
                   curve.domainBegin >= 0 && curve.domainEnd >= curve.domainBegin &&
                   curve.domainEnd < curve.n;
        }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        const Spectral::Curve* select_print_illuminant(
            const JuicerAssets::IlluminantFilterCurveSet& curves,
            const std::string& key) {
            if (key == "D65")
                return &curves.d65;
            if (key == "D55")
                return &curves.d55;
            if (key == "D50")
                return &curves.d50;
            if (key == "TH-KG3")
                return &curves.tungstenKg3;
            if (key == "TH-KG3-L")
                return &curves.tungstenKg3Lens;
            if (key == "T")
                return &curves.tungsten;
            if (key == "K75P")
                return &curves.kinoton75P;
            return nullptr;
        }

        bool copy_source_illuminant(
            JuicerAssets::Library& assets,
            const std::string& key,
            std::uint64_t expectedAssetVersion,
            std::array<float, Spectral::kNumSamples>& out,
            std::string& diagnostic) {
            if (expectedAssetVersion != JuicerAssets::Library::kProcessAssetVersion) {
                diagnostic =
                    "ResourceDescriptorMismatch phase=4B field=source_illuminant_asset_version";
                return false;
            }
            if (key == "EQUAL") {
                out.fill(1.0f);
                return true;
            }
            const JuicerAssets::IlluminantFilterCurveSet& curves =
                assets.illuminant_filter_curves();
            if (curves.version != expectedAssetVersion) {
                diagnostic =
                    "ResourceDescriptorMismatch phase=4B field=source_illuminant_asset_version";
                return false;
            }
            const Spectral::Curve* selected = select_print_illuminant(curves, key);
            if (!selected ||
                selected->linear.size() != out.size() ||
                selected->lambda_nm.size() != out.size()) {
                diagnostic = "MissingRequiredResource phase=4B field=print_illuminant key=" + key;
                return false;
            }
            std::copy(selected->linear.begin(), selected->linear.end(), out.begin());
            float energy = 0.0f;
            const bool valid = std::all_of(out.begin(), out.end(), [&energy](float value) {
                                   if (std::isfinite(value) && value >= 0.0f) {
                                       energy += value;
                                       return true;
                                   }
                                   return false;
                               }) &&
                               energy > 0.0f;
            if (!valid) {
                diagnostic =
                    "MalformedRequiredResource phase=4B field=print_illuminant key=" + key;
            }
            return valid;
        }

        bool build_dichroic_curves(
            JuicerAssets::Library& assets,
            const DichroicResourceIdentity& identity,
            std::array<std::array<float, Spectral::kNumSamples>, 3>& out,
            std::string& diagnostic) {
            if (identity.kind == Spektrafilm::DichroicResourceKind::MeasuredCsv) {
                const JuicerAssets::MeasuredDichroicCurveResult measured =
                    assets.measured_dichroic_curves(identity.setKey);
                if (!measured.valid ||
                    measured.hash != identity.hash ||
                    measured.resourceHashesCmy != identity.resourceHashesCmy) {
                    diagnostic = measured.diagnostic.empty()
                                     ? "ResourceDescriptorMismatch phase=4B field=measured_dichroic"
                                     : measured.diagnostic;
                    return false;
                }
                out = measured.transmittanceCmy;
                return true;
            }

            for (int sample = 0; sample < Spectral::kNumSamples; ++sample) {
                const float wavelength = Spectral::gShape.wavelengths[sample];
                const float y =
                    0.5f * std::erf((wavelength - identity.customEdgesNm[0]) /
                                    identity.customTransitionsNm[0]) +
                    0.5f;
                const float mErf =
                    wavelength <= 550.0f
                        ? -std::erf((wavelength - identity.customEdgesNm[1]) /
                                    identity.customTransitionsNm[1])
                        : std::erf((wavelength - identity.customEdgesNm[2]) /
                                   identity.customTransitionsNm[2]);
                const float c =
                    -0.5f * std::erf((wavelength - identity.customEdgesNm[3]) /
                                     identity.customTransitionsNm[3]) +
                    0.5f;
                out[0][static_cast<std::size_t>(sample)] = c;
                out[1][static_cast<std::size_t>(sample)] = 0.5f * mErf + 0.5f;
                out[2][static_cast<std::size_t>(sample)] = y;
            }
            return true;
        }

        bool derive_filtered_print_illuminant(
            JuicerAssets::Library& assets,
            const PrintRecipe& recipe,
            std::uint64_t sourceIlluminantAssetVersion,
            const CmyCcTriplet& cc,
            std::array<float, Spectral::kNumSamples>& out,
            std::string& diagnostic) {
            std::array<float, Spectral::kNumSamples> source{};
            std::array<std::array<float, Spectral::kNumSamples>, 3> filters{};
            if (!copy_source_illuminant(
                    assets,
                    recipe.illuminant.key,
                    sourceIlluminantAssetVersion,
                    source,
                    diagnostic) ||
                !build_dichroic_curves(assets, recipe.filters.dichroic, filters, diagnostic)) {
                return false;
            }
            const std::array<float, 3> transmittance{{std::pow(10.0f, -cc.c / 100.0f),
                                                      std::pow(10.0f, -cc.m / 100.0f),
                                                      std::pow(10.0f, -cc.y / 100.0f)}};
            for (std::size_t sample = 0; sample < out.size(); ++sample) {
                float total = source[sample];
                for (std::size_t channel = 0; channel < filters.size(); ++channel) {
                    const float dimmed =
                        1.0f - (1.0f - filters[channel][sample]) *
                                   (1.0f - transmittance[channel]);
                    total *= dimmed;
                }
                if (!std::isfinite(total) || total < 0.0f) {
                    diagnostic = "MalformedRequiredResource phase=4B field=filtered_print_illuminant";
                    return false;
                }
                out[sample] = total;
            }
            const float energy = std::accumulate(out.begin(), out.end(), 0.0f);
            if (!(std::isfinite(energy) && energy > 0.0f)) {
                diagnostic = "MalformedRequiredResource phase=4B field=filtered_print_illuminant";
                return false;
            }
            return true;
        }

        float sample_profile_density_curve(
            float logExposure,
            const std::vector<float>& logExposureAxis,
            const std::vector<std::array<float, 3>>& densityCurves,
            std::size_t channel) {
            if (logExposureAxis.empty() || densityCurves.empty()) {
                return 0.0f;
            }
            if (logExposure <= logExposureAxis.front()) {
                return densityCurves.front()[channel];
            }
            if (logExposure >= logExposureAxis.back()) {
                return densityCurves.back()[channel];
            }
            const auto upper =
                std::upper_bound(logExposureAxis.begin(), logExposureAxis.end(), logExposure);
            const std::size_t hi = static_cast<std::size_t>(upper - logExposureAxis.begin());
            const std::size_t lo = hi - 1u;
            const float span = logExposureAxis[hi] - logExposureAxis[lo];
            const float t = span > 0.0f ? (logExposure - logExposureAxis[lo]) / span : 0.0f;
            const float a = densityCurves[lo][channel];
            const float b = densityCurves[hi][channel];
            return a + t * (b - a);
        }

        double density_to_light_sample_spektrafilm(double density, double illuminant) {
            const double transmitted = std::pow(10.0, -density) * illuminant;
            return std::isnan(transmitted) ? 0.0 : transmitted;
        }

        bool derive_reference_white_xyz(
            const std::array<float, Spectral::kNumSamples>& illuminant,
            float out[3]) {
            const Spectral::SpectralContext& context = Spectral::context();
            if (context.xBar.linear.size() != illuminant.size() ||
                context.yBar.linear.size() != illuminant.size() ||
                context.zBar.linear.size() != illuminant.size()) {
                return false;
            }
            double xyz[3] = {0.0, 0.0, 0.0};
            for (std::size_t sample = 0; sample < illuminant.size(); ++sample) {
                const double source = static_cast<double>(illuminant[sample]);
                xyz[0] += source * static_cast<double>(context.xBar.linear[sample]);
                xyz[1] += source * static_cast<double>(context.yBar.linear[sample]);
                xyz[2] += source * static_cast<double>(context.zBar.linear[sample]);
            }
            if (!(std::isfinite(xyz[0]) &&
                  std::isfinite(xyz[1]) &&
                  std::isfinite(xyz[2]) &&
                  xyz[1] > 0.0)) {
                return false;
            }
            out[0] = static_cast<float>(xyz[0] / xyz[1]);
            out[1] = 1.0f;
            out[2] = static_cast<float>(xyz[2] / xyz[1]);
            return true;
        }

        int spektrafilm_reflect_index(int index, int size) {
            if (size <= 1) {
                return 0;
            }
            if (index < 0) {
                return -index;
            }
            if (index >= size) {
                return 2 * (size - 1) - index;
            }
            return index;
        }

        double spektrafilm_mitchell_weight(double t) {
            constexpr double kB = 1.0 / 3.0;
            constexpr double kC = 1.0 / 3.0;
            const double x = std::abs(t);
            if (x < 1.0) {
                return (1.0 / 6.0) *
                       ((12.0 - 9.0 * kB - 6.0 * kC) * x * x * x +
                        (-18.0 + 12.0 * kB + 6.0 * kC) * x * x +
                        (6.0 - 2.0 * kB));
            }
            if (x < 2.0) {
                return (1.0 / 6.0) *
                       ((-kB - 6.0 * kC) * x * x * x +
                        (6.0 * kB + 30.0 * kC) * x * x +
                        (-12.0 * kB - 48.0 * kC) * x +
                        (8.0 * kB + 24.0 * kC));
            }
            return 0.0;
        }

        // NOLINTBEGIN(bugprone-easily-swappable-parameters)
        float sample_hanatos_integrated_cubic_host(
            const std::vector<float>& lut,
            int size,
            int channel,
            float tcC,
            float tcM) {
            auto coordinate = [size](float normalized, int& base, double& fraction) {
                const double value =
                    static_cast<double>(std::clamp(normalized, 0.0f, 1.0f)) *
                    static_cast<double>(size - 1);
                if (value >= static_cast<double>(size - 1)) {
                    base = size - 2;
                    fraction = 1.0;
                    return;
                }
                base = static_cast<int>(std::floor(value));
                fraction = value - static_cast<double>(base);
            };

            int cBase = 0;
            int mBase = 0;
            double cFraction = 0.0;
            double mFraction = 0.0;
            coordinate(tcC, cBase, cFraction);
            coordinate(tcM, mBase, mFraction);
            const double wc[4] = {
                spektrafilm_mitchell_weight(cFraction + 1.0),
                spektrafilm_mitchell_weight(cFraction),
                spektrafilm_mitchell_weight(cFraction - 1.0),
                spektrafilm_mitchell_weight(cFraction - 2.0)};
            const double wm[4] = {
                spektrafilm_mitchell_weight(mFraction + 1.0),
                spektrafilm_mitchell_weight(mFraction),
                spektrafilm_mitchell_weight(mFraction - 1.0),
                spektrafilm_mitchell_weight(mFraction - 2.0)};

            double value = 0.0;
            double weightSum = 0.0;
            for (int dc = 0; dc < 4; ++dc) {
                const int c = spektrafilm_reflect_index(cBase - 1 + dc, size);
                for (int dm = 0; dm < 4; ++dm) {
                    const int m = spektrafilm_reflect_index(mBase - 1 + dm, size);
                    const double weight = wc[dc] * wm[dm];
                    const std::size_t index =
                        (static_cast<std::size_t>(c) * static_cast<std::size_t>(size) +
                         static_cast<std::size_t>(m)) *
                            4u +
                        static_cast<std::size_t>(channel);
                    weightSum += weight;
                    value += weight * static_cast<double>(lut[index]);
                }
            }
            return static_cast<float>(weightSum != 0.0 ? value / weightSum : 0.0);
        }
        // NOLINTEND(bugprone-easily-swappable-parameters)

        bool derive_film_raw_for_midgray(
            const RenderRecipe& recipe,
            const std::array<float, Spectral::kNumSamples>& filmIlluminant,
            float exposureEv,
            std::array<double, 3>& out,
            std::string& diagnostic) {
            out = {};
            const double source = 0.184 * std::exp2(static_cast<double>(exposureEv));
            if (!(std::isfinite(source) && source >= 0.0)) {
                diagnostic = "MalformedRequiredResource phase=4B field=print_balance_midgray_source";
                return false;
            }

            if (recipe.filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Mallett2019) {
                const NpyFloat2D& basis = Spectral::context().mallettBasis;
                if (basis.rows != Spectral::kNumSamples ||
                    basis.cols != 3 ||
                    basis.data.size() !=
                        static_cast<std::size_t>(Spectral::kNumSamples) * 3u) {
                    diagnostic = "MissingRequiredResource phase=4B field=mallett_basis";
                    return false;
                }
                for (std::size_t sample = 0; sample < filmIlluminant.size(); ++sample) {
                    const std::size_t basisOffset = sample * 3u;
                    const double spectrum =
                        source *
                        static_cast<double>(
                            basis.data[basisOffset] +
                            basis.data[basisOffset + 1u] +
                            basis.data[basisOffset + 2u]) *
                        static_cast<double>(filmIlluminant[sample]);
                    for (std::size_t channel = 0; channel < out.size(); ++channel) {
                        out[channel] += spectrum *
                                        static_cast<double>(
                                            recipe.filmRaw.finalSensitivity[sample][channel]);
                    }
                }
                for (double& raw : out) {
                    raw *= static_cast<double>(recipe.filmRaw.mallettGreenMidgrayScale);
                }
            } else if (recipe.filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Hanatos2025) {
                float referenceWhiteXYZ[3] = {};
                if (!derive_reference_white_xyz(filmIlluminant, referenceWhiteXYZ)) {
                    diagnostic =
                        "MalformedRequiredResource phase=4B field=film_reference_illuminant_white";
                    return false;
                }
                std::vector<float> integratedLut;
                if (!Precompute::build_direct_hanatos_integrated_lut_cpu(
                        Spectral::context(),
                        recipe.filmRaw,
                        referenceWhiteXYZ,
                        integratedLut,
                        diagnostic)) {
                    return false;
                }

                const float sourceRgb[3] = {
                    static_cast<float>(source),
                    static_cast<float>(source),
                    static_cast<float>(source)};
                float sourceXYZ[3] = {};
                Spectral::kRGB_to_XYZ_sRGB_Rec709.mul(sourceRgb, sourceXYZ);
                float adaptedXYZ[3] = {};
                Spectral::ChromaticAdaptationWhites whites{};
                whites.source = Spectral::gDWG_WhitePoint_XYZ;
                whites.destination = referenceWhiteXYZ;
                Spectral::chromatic_adapt_XYZ_CAT02(
                    sourceXYZ,
                    whites,
                    adaptedXYZ);
                const float brightness = adaptedXYZ[0] + adaptedXYZ[1] + adaptedXYZ[2];
                const float denominator = std::max(brightness, 1e-10f);
                const float x = std::clamp(adaptedXYZ[0] / denominator, 0.0f, 1.0f);
                const float y = std::clamp(adaptedXYZ[1] / denominator, 0.0f, 1.0f);
                float tcC = 0.0f;
                float tcM = 0.0f;
                Spectral::tri2quad(x, y, tcC, tcM);
                const int size = Spectral::context().hanSpectra.size;
                for (std::size_t channel = 0; channel < out.size(); ++channel) {
                    out[channel] =
                        static_cast<double>(brightness) *
                        static_cast<double>(sample_hanatos_integrated_cubic_host(
                            integratedLut,
                            size,
                            static_cast<int>(channel),
                            tcC,
                            tcM));
                }
            } else {
                diagnostic = "ResourceDescriptorMismatch phase=4B field=rgb_to_raw_method";
                return false;
            }

            const bool valid = std::all_of(out.begin(), out.end(), [](double raw) {
                return std::isfinite(raw) && raw >= 0.0;
            });
            if (!valid) {
                diagnostic = "MalformedRequiredResource phase=4B field=print_balance_film_raw";
            }
            return valid;
        }

        bool derive_print_raw_for_midgray(
            JuicerAssets::Library& assets,
            const RenderRecipe& recipe,
            std::uint64_t filmReferenceIlluminantAssetVersion,
            const std::array<float, Spectral::kNumSamples>& mainIlluminant,
            float exposureEv,
            float& outFactor,
            std::string& diagnostic) {
            const Profiles::ValidatedFilmProfile& film = *recipe.profileRoute.filmProfile;
            const Profiles::ValidatedPrintProfile& print = *recipe.profileRoute.printProfile;
            std::array<float, Spectral::kNumSamples> filmIlluminant{};
            if (!copy_source_illuminant(
                    assets,
                    film.info.referenceIlluminant.value,
                    filmReferenceIlluminantAssetVersion,
                    filmIlluminant,
                    diagnostic)) {
                return false;
            }
            std::array<double, 3> filmRaw{};
            if (!derive_film_raw_for_midgray(
                    recipe,
                    filmIlluminant,
                    exposureEv,
                    filmRaw,
                    diagnostic)) {
                return false;
            }
            std::array<float, 3> densityCmy{};
            for (std::size_t channel = 0; channel < densityCmy.size(); ++channel) {
                const float logRaw = static_cast<float>(std::log10(filmRaw[channel] + 1e-10));
                const float gamma = recipe.filmDevelop.densityCurveGamma[channel];
                if (!std::isfinite(gamma) || !(gamma > 0.0f)) {
                    diagnostic =
                        "ResourceDescriptorMismatch phase=4B field=film_density_curve_gamma";
                    return false;
                }
                densityCmy[channel] = sample_profile_density_curve(
                    logRaw * gamma,
                    recipe.filmDevelop.logExposure,
                    recipe.filmDevelop.authoredDensityCurves,
                    channel);
            }

            std::array<double, 3> printRaw{};
            for (std::size_t sample = 0; sample < mainIlluminant.size(); ++sample) {
                double density = static_cast<double>(film.data.baseDensity[sample]);
                for (std::size_t channel = 0; channel < densityCmy.size(); ++channel) {
                    density += static_cast<double>(densityCmy[channel]) *
                               static_cast<double>(film.data.channelDensity[sample][channel]);
                }
                const double light = density_to_light_sample_spektrafilm(
                    density,
                    static_cast<double>(mainIlluminant[sample]));
                for (std::size_t channel = 0; channel < printRaw.size(); ++channel) {
                    const float sensitivity = print.data.linearSensitivity[sample][channel];
                    if (std::isfinite(sensitivity)) {
                        printRaw[channel] += light * static_cast<double>(sensitivity);
                    }
                }
            }
            double meanLogRaw = 0.0;
            for (double raw : printRaw) {
                if (!(std::isfinite(raw) && raw >= 0.0)) {
                    diagnostic = "MalformedRequiredResource phase=4B field=print_balance_midgray";
                    return false;
                }
                meanLogRaw += std::log(std::max(1e-10, raw));
            }
            const double geometricMean = std::exp(meanLogRaw / 3.0);
            outFactor = static_cast<float>(1.0 / geometricMean);
            return std::isfinite(outFactor) && outFactor > 0.0f;
        }

        bool derive_preflash_raw(
            const Profiles::ValidatedFilmProfile& film,
            const Profiles::ValidatedPrintProfile& print,
            const std::array<float, Spectral::kNumSamples>& preflashIlluminant,
            std::array<float, 3>& out,
            std::string& diagnostic) {
            std::array<double, 3> raw{};
            for (std::size_t sample = 0; sample < preflashIlluminant.size(); ++sample) {
                const double light = density_to_light_sample_spektrafilm(
                    static_cast<double>(film.data.baseDensity[sample]),
                    static_cast<double>(preflashIlluminant[sample]));
                for (std::size_t channel = 0; channel < raw.size(); ++channel) {
                    const float sensitivity = print.data.linearSensitivity[sample][channel];
                    if (std::isfinite(sensitivity)) {
                        raw[channel] += light * static_cast<double>(sensitivity);
                    }
                }
            }
            for (std::size_t channel = 0; channel < raw.size(); ++channel) {
                if (!std::isfinite(raw[channel]) || raw[channel] < 0.0) {
                    diagnostic = "MalformedRequiredResource phase=4B field=preflash_raw";
                    return false;
                }
                out[channel] = static_cast<float>(raw[channel]);
            }
            return true;
        }
#endif

    } // namespace

    bool build_print_resource_descriptors(
        const RenderRecipe& recipe,
        PrintResourceDescriptors& out,
        std::string& diagnostic) {
        diagnostic.clear();
        out = PrintResourceDescriptors{};
        if (!recipe.printStructuralReady ||
            !Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute) ||
            !recipe.profileRoute.filmProfile ||
            !recipe.profileRoute.printProfile ||
            recipe.print.hash == 0 ||
            recipe.print.filters.hash == 0 ||
            recipe.print.exposure.hash == 0 ||
            recipe.print.illuminant.hash == 0) {
            diagnostic = "ResourceDescriptorMismatch phase=4B field=print_recipe";
            return false;
        }

        const Profiles::ValidatedFilmProfile& film = *recipe.profileRoute.filmProfile;
        const Profiles::ValidatedPrintProfile& print = *recipe.profileRoute.printProfile;
        PrintProfileTablesDescriptor& profile = out.profileTables;
        profile.printProfileAssetVersionToken = recipe.profileRoute.printProfileAssetVersionToken;
        profile.densityCurvesHash = hash_profile_density_curves(print.data);
        profile.sensitivitiesHash = hash_profile_sensitivities(print.data);
        profile.densitySampleCount = static_cast<std::uint32_t>(print.data.logExposure.size());
        profile.spectralSampleCount = static_cast<std::uint32_t>(print.data.linearSensitivity.size());
        profile.hash = Hash::kFnvOffset;
        hash_print_descriptor_value(profile.hash, PrintProfileTablesDescriptor::kSchemaVersion);
        hash_print_descriptor_value(profile.hash, profile.printProfileAssetVersionToken);
        hash_print_descriptor_value(profile.hash, profile.densityCurvesHash);
        hash_print_descriptor_value(profile.hash, profile.sensitivitiesHash);
        hash_print_descriptor_value(profile.hash, profile.densitySampleCount);
        hash_print_descriptor_value(profile.hash, profile.spectralSampleCount);

        PrintFilmDensityTablesDescriptor& filmDensity = out.filmDensityTables;
        filmDensity.filmProfileAssetVersionToken = recipe.profileRoute.filmProfileAssetVersionToken;
        filmDensity.densityTablesHash = hash_profile_film_density_tables(film.data);
        filmDensity.spectralSampleCount = static_cast<std::uint32_t>(film.data.channelDensity.size());
        filmDensity.hash = Hash::kFnvOffset;
        hash_print_descriptor_value(filmDensity.hash, PrintFilmDensityTablesDescriptor::kSchemaVersion);
        hash_print_descriptor_value(filmDensity.hash, filmDensity.filmProfileAssetVersionToken);
        hash_print_descriptor_value(filmDensity.hash, filmDensity.densityTablesHash);
        hash_print_descriptor_value(filmDensity.hash, filmDensity.spectralSampleCount);

        auto build_illuminant = [&](FilteredPrintIlluminantDescriptor& descriptor,
                                    const CmyCcTriplet& cc,
                                    bool preflash) {
            descriptor.sourceIlluminantAssetVersionToken =
                JuicerAssets::Library::kProcessAssetVersion;
            descriptor.printIlluminantHash = recipe.print.illuminant.hash;
            descriptor.dichroicResourceHash = recipe.print.filters.dichroic.hash;
            descriptor.cmyCc = cc;
            descriptor.preflash = preflash;
            descriptor.hash = Hash::kFnvOffset;
            hash_print_descriptor_value(
                descriptor.hash,
                FilteredPrintIlluminantDescriptor::kSchemaVersion);
            hash_print_descriptor_value(
                descriptor.hash,
                descriptor.sourceIlluminantAssetVersionToken);
            hash_print_descriptor_value(descriptor.hash, descriptor.printIlluminantHash);
            hash_print_descriptor_value(descriptor.hash, descriptor.dichroicResourceHash);
            hash_print_descriptor_cmy(descriptor.hash, descriptor.cmyCc);
            hash_print_descriptor_value(descriptor.hash, descriptor.preflash);
        };
        build_illuminant(out.mainIlluminant, recipe.print.filters.mainCmyCc, false);

        out.preflashActive = recipe.print.exposure.preflashExposure > 0.0f;
        if (out.preflashActive) {
            build_illuminant(out.preflashIlluminant, recipe.print.filters.preflashCmyCc, true);
            out.preflashRaw.filmProfileAssetVersionToken =
                recipe.profileRoute.filmProfileAssetVersionToken;
            out.preflashRaw.printProfileTablesHash = profile.hash;
            out.preflashRaw.filteredPreflashIlluminantHash = out.preflashIlluminant.hash;
            out.preflashRaw.hash = Hash::kFnvOffset;
            hash_print_descriptor_value(out.preflashRaw.hash, PrintPreflashRawDescriptor::kSchemaVersion);
            hash_print_descriptor_value(out.preflashRaw.hash, out.preflashRaw.filmProfileAssetVersionToken);
            hash_print_descriptor_value(out.preflashRaw.hash, out.preflashRaw.printProfileTablesHash);
            hash_print_descriptor_value(out.preflashRaw.hash, out.preflashRaw.filteredPreflashIlluminantHash);
        }

        out.balance.filmProfileAssetVersionToken = recipe.profileRoute.filmProfileAssetVersionToken;
        out.balance.filmReferenceIlluminantAssetVersionToken =
            JuicerAssets::Library::kProcessAssetVersion;
        out.balance.filmRawRecipeHash = recipe.filmRaw.hash;
        out.balance.filmDevelopRecipeHash = recipe.filmDevelop.hash;
        out.balance.printProfileTablesHash = profile.hash;
        out.balance.filteredMainIlluminantHash = out.mainIlluminant.hash;
        out.balance.normalizationMode = recipe.print.exposure.normalizationMode;
        out.balance.cameraExposureCompensationEv =
            recipe.print.exposure.printExposureCompensation
                ? recipe.print.exposure.cameraExposureCompensationEv
                : 0.0f;
        out.balance.hash = Hash::kFnvOffset;
        hash_print_descriptor_value(out.balance.hash, PrintBalanceDescriptor::kSchemaVersion);
        hash_print_descriptor_value(out.balance.hash, out.balance.filmProfileAssetVersionToken);
        hash_print_descriptor_value(
            out.balance.hash,
            out.balance.filmReferenceIlluminantAssetVersionToken);
        hash_print_descriptor_value(out.balance.hash, out.balance.filmRawRecipeHash);
        hash_print_descriptor_value(out.balance.hash, out.balance.filmDevelopRecipeHash);
        hash_print_descriptor_value(out.balance.hash, out.balance.printProfileTablesHash);
        hash_print_descriptor_value(out.balance.hash, out.balance.filteredMainIlluminantHash);
        hash_print_descriptor_value(out.balance.hash, out.balance.normalizationMode);
        hash_print_descriptor_value(out.balance.hash, out.balance.cameraExposureCompensationEv);

        out.hash = Hash::hash_uint64_values({profile.hash,
                                             filmDensity.hash,
                                             out.mainIlluminant.hash,
                                             out.preflashIlluminant.hash,
                                             out.preflashRaw.hash,
                                             out.balance.hash});
        if (profile.densityCurvesHash == 0 ||
            profile.sensitivitiesHash == 0 ||
            profile.densitySampleCount == 0 ||
            profile.spectralSampleCount != Spectral::kNumSamples ||
            profile.hash == 0 ||
            filmDensity.filmProfileAssetVersionToken == 0 ||
            filmDensity.densityTablesHash == 0 ||
            filmDensity.spectralSampleCount != Spectral::kNumSamples ||
            filmDensity.hash == 0 ||
            out.mainIlluminant.sourceIlluminantAssetVersionToken == 0 ||
            out.mainIlluminant.hash == 0 ||
            out.balance.filmReferenceIlluminantAssetVersionToken == 0 ||
            out.balance.hash == 0 ||
            out.balance.filmRawRecipeHash == 0 ||
            out.balance.filmDevelopRecipeHash == 0 ||
            out.hash == 0 ||
            (out.preflashActive &&
             (out.preflashIlluminant.hash == 0 || out.preflashRaw.hash == 0))) {
            diagnostic = "ResourceDescriptorMismatch phase=4B field=print_resource_descriptors";
            return false;
        }
        return true;
    }

    bool prepare_print_resources(
        Resources& resources,
        const PrintResourcePreparation& request,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)request;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        outError.clear();
        if (!request.recipe || !request.assets) {
            outError = "MissingRequiredResource phase=4B field=print_preparation_request";
            return false;
        }
        PrintResourceDescriptors descriptors{};
        if (!build_print_resource_descriptors(*request.recipe, descriptors, outError)) {
            return false;
        }
        const Profiles::ValidatedFilmProfile& film = *request.recipe->profileRoute.filmProfile;
        const Profiles::ValidatedPrintProfile& print = *request.recipe->profileRoute.printProfile;

        std::lock_guard<std::mutex> servingUpdateLock(resources.servingUpdateMutex);
        std::unique_lock<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }
        const bool filmDensityHit =
            resources.printFilmDensityTablesDescriptorHash == descriptors.filmDensityTables.hash &&
            resources.printFilmDensityTables.K == Spectral::kNumSamples &&
            resources.printFilmDensityTables.epsC &&
            resources.printFilmDensityTables.epsM &&
            resources.printFilmDensityTables.epsY &&
            resources.printFilmDensityTables.hasBaseline &&
            resources.printFilmDensityTables.baseDensityMin;
        const bool profileHit =
            resources.printProfileTablesDescriptorHash == descriptors.profileTables.hash &&
            resources.printDcC.x && resources.printDcM.x && resources.printDcY.x &&
            resources.printSensC.y && resources.printSensM.y && resources.printSensY.y;
        const auto positive_finite_energy =
            [](const std::array<float, Spectral::kNumSamples>& values) {
                float energy = 0.0f;
                for (float value : values) {
                    if (!(std::isfinite(value) && value >= 0.0f)) {
                        return false;
                    }
                    energy += value;
                }
                return std::isfinite(energy) && energy > 0.0f;
            };
        const bool mainHit =
            resources.printMainIlluminantDescriptorHash == descriptors.mainIlluminant.hash &&
            resources.printIllumFiltered &&
            resources.printIllumK == Spectral::kNumSamples &&
            resources.printIllumFilteredHostValid &&
            positive_finite_energy(resources.printIllumFilteredHost);
        const bool preflashIlluminantHit =
            !descriptors.preflashActive ||
            (resources.printPreflashIlluminantDescriptorHash == descriptors.preflashIlluminant.hash &&
             resources.printPreflashIllumFiltered &&
             resources.printPreflashIllumK == Spectral::kNumSamples &&
             resources.printPreflashIllumFilteredHostValid &&
             positive_finite_energy(resources.printPreflashIllumFilteredHost));
        const bool preflashRawHit =
            !descriptors.preflashActive ||
            (resources.printPreflashRawDescriptorHash == descriptors.preflashRaw.hash &&
             resources.printPreflashValid);
        const bool balanceHit =
            resources.printBalanceDescriptorHash == descriptors.balance.hash &&
            std::isfinite(resources.printBalanceFactorMidgray) &&
            std::isfinite(resources.printBalanceFactorMidgrayComp) &&
            std::isfinite(resources.printBalanceNormalizer) &&
            resources.printBalanceNormalizer > 0.0f;
        auto trace_main_illuminant = [&](const char* cacheOutcome,
                                         const char* mainIlluminantOutcome) {
            if (!JTRACE_ENABLED(3)) {
                return;
            }
            const std::uint64_t hostIlluminantChecksum =
                resources.printIllumFilteredHostValid
                    ? Hash::hash_float_span(resources.printIllumFilteredHost)
                    : 0;
            std::string msg;
            msg.reserve(512);
            msg = "event=print_resource_preparation cacheOutcome=";
            msg += cacheOutcome;
            msg += " mainIlluminantOutcome=";
            msg += mainIlluminantOutcome;
            msg += " filteredMainIlluminantDescriptorHash=";
            msg += std::to_string(descriptors.mainIlluminant.hash);
            msg += " preparedMainIlluminantDescriptorHash=";
            msg += std::to_string(resources.printMainIlluminantDescriptorHash);
            msg += " descriptorCmyCc(C/M/Y)=";
            msg += std::to_string(descriptors.mainIlluminant.cmyCc.c);
            msg += "/";
            msg += std::to_string(descriptors.mainIlluminant.cmyCc.m);
            msg += "/";
            msg += std::to_string(descriptors.mainIlluminant.cmyCc.y);
            msg += " sourceIlluminantAssetVersionToken=";
            msg += std::to_string(
                descriptors.mainIlluminant.sourceIlluminantAssetVersionToken);
            msg += " sourceIlluminantDescriptorHash=";
            msg += std::to_string(descriptors.mainIlluminant.printIlluminantHash);
            msg += " dichroicResourceHash=";
            msg += std::to_string(descriptors.mainIlluminant.dichroicResourceHash);
            msg += " hostIlluminantChecksum=";
            msg += std::to_string(hostIlluminantChecksum);
            JTRACE_VERBOSE("PHASE4C_BALANCE", msg);
        };
        if (filmDensityHit && profileHit && mainHit && preflashIlluminantHit && preflashRawHit &&
            balanceHit &&
            resources.printPreparationDescriptorHash == descriptors.hash) {
            trace_main_illuminant("hit", "hit");
        }
        if (filmDensityHit && profileHit && mainHit && preflashIlluminantHit && preflashRawHit &&
            balanceHit &&
            resources.printPreparationDescriptorHash == descriptors.hash) {
            return true;
        }

        std::array<float, Spectral::kNumSamples> mainIlluminant{};
        std::array<float, Spectral::kNumSamples> preflashIlluminant{};
        std::array<float, 3> preflashRaw{};
        if (mainHit && !balanceHit) {
            mainIlluminant = resources.printIllumFilteredHost;
        }
        if (descriptors.preflashActive && preflashIlluminantHit && !preflashRawHit) {
            preflashIlluminant = resources.printPreflashIllumFilteredHost;
        }
        float factorMidgray = resources.printBalanceFactorMidgray;
        float factorMidgrayComp = resources.printBalanceFactorMidgrayComp;
        float normalizer = resources.printBalanceNormalizer;

        lock.unlock();
        if (!mainHit &&
            !derive_filtered_print_illuminant(
                *request.assets,
                request.recipe->print,
                descriptors.mainIlluminant.sourceIlluminantAssetVersionToken,
                request.recipe->print.filters.mainCmyCc,
                mainIlluminant,
                outError)) {
            return false;
        }
        if (descriptors.preflashActive && !preflashIlluminantHit &&
            !derive_filtered_print_illuminant(
                *request.assets,
                request.recipe->print,
                descriptors.preflashIlluminant.sourceIlluminantAssetVersionToken,
                request.recipe->print.filters.preflashCmyCc,
                preflashIlluminant,
                outError)) {
            return false;
        }
        if (!balanceHit) {
            if (!derive_print_raw_for_midgray(
                    *request.assets,
                    *request.recipe,
                    descriptors.balance.filmReferenceIlluminantAssetVersionToken,
                    mainIlluminant,
                    0.0f,
                    factorMidgray,
                    outError) ||
                !derive_print_raw_for_midgray(
                    *request.assets,
                    *request.recipe,
                    descriptors.balance.filmReferenceIlluminantAssetVersionToken,
                    mainIlluminant,
                    descriptors.balance.cameraExposureCompensationEv,
                    factorMidgrayComp,
                    outError)) {
                return false;
            }
            normalizer = Spektrafilm::print_exposure_normalizer(
                request.recipe->print.exposure.normalizationMode,
                factorMidgray,
                factorMidgrayComp);
            if (!(std::isfinite(normalizer) && normalizer > 0.0f)) {
                outError = "MalformedRequiredResource phase=4B field=print_exposure_normalizer";
                return false;
            }
        }
        if (descriptors.preflashActive && !preflashRawHit &&
            !derive_preflash_raw(film, print, preflashIlluminant, preflashRaw, outError)) {
            return false;
        }
        lock.lock();
        if (!validate_resource_owner_locked(resources, outError, false)) {
            return false;
        }

        if (!filmDensityHit) {
            std::array<float, Spectral::kNumSamples> epsC{};
            std::array<float, Spectral::kNumSamples> epsM{};
            std::array<float, Spectral::kNumSamples> epsY{};
            for (std::size_t sample = 0; sample < film.data.channelDensity.size(); ++sample) {
                epsC[sample] = film.data.channelDensity[sample][0];
                epsM[sample] = film.data.channelDensity[sample][1];
                epsY[sample] = film.data.channelDensity[sample][2];
            }
            Resources::DeviceSpectralTables& tables = resources.printFilmDensityTables;
            const int currentK = tables.K;
            if (!upload_array_locked(resources, tables.epsC, currentK, epsC.data(), Spectral::kNumSamples, cudaStreamOpaque, &lock, "phase4B print film density epsC", outError) ||
                !upload_array_locked(resources, tables.epsM, currentK, epsM.data(), Spectral::kNumSamples, cudaStreamOpaque, &lock, "phase4B print film density epsM", outError) ||
                !upload_array_locked(resources, tables.epsY, currentK, epsY.data(), Spectral::kNumSamples, cudaStreamOpaque, &lock, "phase4B print film density epsY", outError) ||
                !upload_array_locked(resources, tables.baseDensityMin, currentK, film.data.baseDensity.data(), Spectral::kNumSamples, cudaStreamOpaque, &lock, "phase4B print film density base", outError)) {
                return false;
            }
            tables.K = Spectral::kNumSamples;
            tables.hasBaseline = 1;
            tables.invYn = 1.0f;
            resources.printFilmDensityTablesDescriptorHash = descriptors.filmDensityTables.hash;
        }
        if (!profileHit) {
            Spectral::Curve dcC;
            Spectral::Curve dcM;
            Spectral::Curve dcY;
            dcC.lambda_nm = print.data.logExposure;
            dcM.lambda_nm = print.data.logExposure;
            dcY.lambda_nm = print.data.logExposure;
            dcC.linear.resize(print.data.densityCurves.size());
            dcM.linear.resize(print.data.densityCurves.size());
            dcY.linear.resize(print.data.densityCurves.size());
            std::vector<float> sensC(Spectral::kNumSamples);
            std::vector<float> sensM(Spectral::kNumSamples);
            std::vector<float> sensY(Spectral::kNumSamples);
            for (std::size_t sample = 0; sample < print.data.densityCurves.size(); ++sample) {
                dcC.linear[sample] = print.data.densityCurves[sample][0];
                dcM.linear[sample] = print.data.densityCurves[sample][1];
                dcY.linear[sample] = print.data.densityCurves[sample][2];
            }
            for (int sample = 0; sample < Spectral::kNumSamples; ++sample) {
                const auto& cmy = print.data.linearSensitivity[static_cast<std::size_t>(sample)];
                sensC[static_cast<std::size_t>(sample)] = cmy[0];
                sensM[static_cast<std::size_t>(sample)] = cmy[1];
                sensY[static_cast<std::size_t>(sample)] = cmy[2];
            }
            if (!upload_curve_locked(resources, resources.printDcC, dcC, cudaStreamOpaque, &lock, "phase4B print dcC", outError) ||
                !upload_curve_locked(resources, resources.printDcM, dcM, cudaStreamOpaque, &lock, "phase4B print dcM", outError) ||
                !upload_curve_locked(resources, resources.printDcY, dcY, cudaStreamOpaque, &lock, "phase4B print dcY", outError) ||
                !upload_spectral_samples_locked(resources, resources.printSensC, sensC, cudaStreamOpaque, &lock, "phase4B print sensC", outError) ||
                !upload_spectral_samples_locked(resources, resources.printSensM, sensM, cudaStreamOpaque, &lock, "phase4B print sensM", outError) ||
                !upload_spectral_samples_locked(resources, resources.printSensY, sensY, cudaStreamOpaque, &lock, "phase4B print sensY", outError)) {
                return false;
            }
            resources.printProfileTablesDescriptorHash = descriptors.profileTables.hash;
        }
        if (!mainHit) {
            if (!upload_array_locked(
                    resources,
                    resources.printIllumFiltered,
                    resources.printIllumK,
                    mainIlluminant.data(),
                    Spectral::kNumSamples,
                    cudaStreamOpaque,
                    &lock,
                    "phase4B filtered main print illuminant",
                    outError)) {
                return false;
            }
            resources.printIllumK = Spectral::kNumSamples;
            resources.printIllumFilteredHost = mainIlluminant;
            resources.printIllumFilteredHostValid = true;
            resources.printMainIlluminantDescriptorHash = descriptors.mainIlluminant.hash;
        }
        if (descriptors.preflashActive) {
            if (!preflashIlluminantHit &&
                !upload_array_locked(
                    resources,
                    resources.printPreflashIllumFiltered,
                    resources.printPreflashIllumK,
                    preflashIlluminant.data(),
                    Spectral::kNumSamples,
                    cudaStreamOpaque,
                    &lock,
                    "phase4B filtered preflash print illuminant",
                    outError)) {
                return false;
            }
            resources.printPreflashIllumK = Spectral::kNumSamples;
            resources.printPreflashIllumFilteredHost = preflashIlluminant;
            resources.printPreflashIllumFilteredHostValid = true;
            resources.printPreflashIlluminantDescriptorHash = descriptors.preflashIlluminant.hash;
            if (!preflashRawHit) {
                std::copy(preflashRaw.begin(), preflashRaw.end(), resources.printPreflashRaw);
                resources.printPreflashValid = true;
                resources.printPreflashRawDescriptorHash = descriptors.preflashRaw.hash;
            }
        } else {
            if (resources.printPreflashIllumFiltered) {
                const std::size_t bytes =
                    static_cast<std::size_t>(std::max(0, resources.printPreflashIllumK)) * sizeof(float);
                if (!retire_ptr_locked(
                        resources,
                        resources.printPreflashIllumFiltered,
                        bytes,
                        Resources::RetireKind::DeviceFree,
                        cudaStreamOpaque,
                        "disabled phase4B preflash illuminant",
                        outError)) {
                    return false;
                }
                resources.printPreflashIllumFiltered = nullptr;
            }
            resources.printPreflashIllumK = 0;
            resources.printPreflashIllumFilteredHost.fill(0.0f);
            resources.printPreflashIllumFilteredHostValid = false;
            resources.printPreflashIlluminantDescriptorHash = 0;
            resources.printPreflashRaw[0] = 0.0f;
            resources.printPreflashRaw[1] = 0.0f;
            resources.printPreflashRaw[2] = 0.0f;
            resources.printPreflashValid = false;
            resources.printPreflashRawDescriptorHash = 0;
        }
        if (!balanceHit) {
            resources.printBalanceFactorMidgray = factorMidgray;
            resources.printBalanceFactorMidgrayComp = factorMidgrayComp;
            resources.printBalanceNormalizer = normalizer;
            resources.printBalanceDescriptorHash = descriptors.balance.hash;
        }
        resources.printPreparationDescriptorHash = descriptors.hash;
        ++resources.printPreparationCounter;
        trace_main_illuminant("rebuilt", mainHit ? "retained" : "rebuilt");
        return true;
#endif
    }

    bool pack_print_cuda_payloads(
        const PrintRecipe& recipe,
        const PrintPreparedView& prepared,
        float routeCorrectionScale,
        PrintCudaPayloadPack& out,
        std::string& diagnostic) {
        diagnostic.clear();
        out = PrintCudaPayloadPack{};
        if (!prepared.active ||
            prepared.preparationHash == 0 ||
            prepared.filmDensityTablesHash == 0 ||
            prepared.filmDensityTables.K != Spectral::kNumSamples ||
            !prepared.filmDensityTables.epsC ||
            !prepared.filmDensityTables.epsM ||
            !prepared.filmDensityTables.epsY ||
            !prepared.filmDensityTables.hasBaseline ||
            !prepared.filmDensityTables.baseDensityMin ||
            prepared.profileTablesHash == 0 ||
            prepared.mainIlluminantHash == 0 ||
            prepared.balanceHash == 0 ||
            !prepared.mainIlluminant ||
            prepared.spectralSampleCount != Spectral::kNumSamples ||
            !print_curve_ready(prepared.printSensC, Spectral::kNumSamples, false) ||
            !print_curve_ready(prepared.printSensM, Spectral::kNumSamples, false) ||
            !print_curve_ready(prepared.printSensY, Spectral::kNumSamples, false) ||
            !print_curve_ready(prepared.printDcC, prepared.printDcC.n, true) ||
            !print_curve_ready(prepared.printDcM, prepared.printDcC.n, true) ||
            !print_curve_ready(prepared.printDcY, prepared.printDcC.n, true)) {
            diagnostic = "ResourceDescriptorMismatch phase=4B field=print_prepared_view";
            return false;
        }
        if (prepared.preflashActive &&
            (!prepared.preflashIlluminant ||
             prepared.preflashIlluminantHash == 0 ||
             prepared.preflashRawHash == 0)) {
            diagnostic = "MissingRequiredResource phase=4B field=preflash_prepared_view";
            return false;
        }
        if (!std::isfinite(recipe.exposure.printExposure) ||
            recipe.exposure.printExposure < 0.0f ||
            !std::isfinite(prepared.normalizer) ||
            !(prepared.normalizer > 0.0f) ||
            !std::isfinite(routeCorrectionScale) ||
            !(routeCorrectionScale > 0.0f)) {
            diagnostic =
                "ResourceDescriptorMismatch phase=4B field=print_exposure_scale";
            return false;
        }
        if (prepared.preflashActive) {
            if (!std::isfinite(recipe.exposure.preflashExposure) ||
                recipe.exposure.preflashExposure < 0.0f) {
                diagnostic =
                    "ResourceDescriptorMismatch phase=4B field=preflash_exposure";
                return false;
            }
            for (float raw : prepared.preflashRawCmy) {
                if (!std::isfinite(raw) || raw < 0.0f) {
                    diagnostic =
                        "ResourceDescriptorMismatch phase=4B field=preflash_raw";
                    return false;
                }
            }
        }

        out.expose.active = 1;
        out.expose.negTables = prepared.filmDensityTables;
        out.expose.printIllumFiltered = prepared.mainIlluminant;
        out.expose.printIllumK = prepared.spectralSampleCount;
        out.expose.printSensC = prepared.printSensC;
        out.expose.printSensM = prepared.printSensM;
        out.expose.printSensY = prepared.printSensY;
        out.expose.printExposure = recipe.exposure.printExposure;
        out.expose.routeCorrectionScale = routeCorrectionScale;
        out.expose.printPreflashExposure =
            prepared.preflashActive ? recipe.exposure.preflashExposure : 0.0f;
        out.expose.printMidgrayFactor = prepared.normalizer;
        if (prepared.preflashActive) {
            std::copy_n(prepared.preflashRawCmy, 3, out.expose.printPreflashRaw);
        }
        out.develop.printDcC = prepared.printDcC;
        out.develop.printDcM = prepared.printDcM;
        out.develop.printDcY = prepared.printDcY;
        out.printRawScale = recipe.exposure.printExposure * prepared.normalizer;
        out.scalingOrder = recipe.exposure.scalingOrder;
        out.preparationHash = prepared.preparationHash;
        if (!(std::isfinite(out.printRawScale) && out.printRawScale >= 0.0f)) {
            diagnostic = "MalformedRequiredResource phase=4B field=print_raw_scale";
            return false;
        }
        return true;
    }

    // Cuda/JuicerCudaResourcesScratch.cpp
    //
    // Included by JuicerCudaResources.cpp (single-TU split).
    enum class SharedGaussianKind : std::uint8_t {
        Standard = 0,
        Halation = 1,
        SpatialDir = 2
    };

    struct SharedGaussianSpec {
        SharedGaussianKind kind = SharedGaussianKind::Standard;
        int radius = 0;
        float sigma = 0.0f;
    };

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)

    struct SharedGaussianKey {
        int deviceId = -1;
        void* contextOpaque = nullptr;
        SharedGaussianKind kind = SharedGaussianKind::Standard;
        int radius = 0;
        std::uint32_t sigmaBits = 0;

        bool operator==(const SharedGaussianKey& other) const noexcept {
            return deviceId == other.deviceId &&
                   contextOpaque == other.contextOpaque &&
                   kind == other.kind &&
                   radius == other.radius &&
                   sigmaBits == other.sigmaBits;
        }
    };

    struct SharedGaussianKeyHash {
        std::size_t operator()(const SharedGaussianKey& key) const noexcept {
            const std::size_t hDevice = std::hash<int>{}(key.deviceId);
            const std::size_t hContext = std::hash<std::uintptr_t>{}(
                reinterpret_cast<std::uintptr_t>(key.contextOpaque));
            const std::size_t hKind = std::hash<std::uint8_t>{}(
                static_cast<std::uint8_t>(key.kind));
            const std::size_t hRadius = std::hash<int>{}(key.radius);
            const std::size_t hSigma = std::hash<std::uint32_t>{}(key.sigmaBits);
            std::size_t h = hDevice;
            h ^= hContext + 0x9e3779b9u + (h << 6u) + (h >> 2u);
            h ^= hKind + 0x9e3779b9u + (h << 6u) + (h >> 2u);
            h ^= hRadius + 0x9e3779b9u + (h << 6u) + (h >> 2u);
            h ^= hSigma + 0x9e3779b9u + (h << 6u) + (h >> 2u);
            return h;
        }
    };

    struct SharedGaussianEntry {
        float* weights = nullptr;
        int radius = 0;
        float sigma = 0.0f;
        int capacity = 0;
        std::uint64_t id = 0;
    };

    struct SharedGaussianCacheState {
        std::mutex mutex;
        std::unordered_map<SharedGaussianKey, SharedGaussianEntry, SharedGaussianKeyHash> byKey;
    };

    static SharedGaussianCacheState& shared_gaussian_cache_state() {
        static SharedGaussianCacheState state;
        return state;
    }

    static std::uint32_t gaussian_sigma_bits(float sigma) noexcept {
        float canonical = sigma;
        if (canonical == 0.0f) {
            canonical = 0.0f;
        }
        std::uint32_t bits = 0;
        std::memcpy(&bits, &canonical, sizeof(bits));
        return bits;
    }

    static void hash_shared_gaussian_bytes(
        std::uint64_t& h,
        const void* data,
        std::size_t numBytes) noexcept {
        const auto* bytes = static_cast<const std::uint8_t*>(data);
        for (std::size_t i = 0; i < numBytes; ++i) {
            h ^= static_cast<std::uint64_t>(bytes[i]);
            h *= Hash::kFnvPrime;
        }
    }

    static std::uint64_t make_shared_gaussian_id(const SharedGaussianKey& key) noexcept {
        std::uint64_t h = Hash::kFnvOffset;
        hash_shared_gaussian_bytes(h, &key.deviceId, sizeof(key.deviceId));
        const std::uintptr_t contextBits =
            reinterpret_cast<std::uintptr_t>(key.contextOpaque);
        hash_shared_gaussian_bytes(h, &contextBits, sizeof(contextBits));
        const std::uint8_t kindValue = static_cast<std::uint8_t>(key.kind);
        hash_shared_gaussian_bytes(h, &kindValue, sizeof(kindValue));
        hash_shared_gaussian_bytes(h, &key.radius, sizeof(key.radius));
        hash_shared_gaussian_bytes(h, &key.sigmaBits, sizeof(key.sigmaBits));
        return h;
    }

    static SharedGaussianKey make_shared_gaussian_key(
        const Resources& resources,
        SharedGaussianSpec spec) noexcept {
        SharedGaussianKey key{};
        key.deviceId = resources.deviceId;
        key.contextOpaque = resources.ownerContextOpaque;
        key.kind = spec.kind;
        key.radius = spec.radius;
        key.sigmaBits = gaussian_sigma_bits(spec.sigma);
        return key;
    }

    static void build_gaussian_weights_cpu(
        int radius,
        float sigma,
        std::vector<float>& outWeights) {
        outWeights.clear();
        if (radius <= 0 || !std::isfinite(sigma) || sigma <= 0.0f) {
            return;
        }
        const std::size_t radiusSize = static_cast<std::size_t>(radius);
        outWeights.resize(radiusSize * 2u + 1u);
        const double s2 = static_cast<double>(sigma) * static_cast<double>(sigma) * 2.0;
        double wsum = 0.0;
        std::size_t weightIndex = 0;
        for (int i = -radius; i <= radius; ++i, ++weightIndex) {
            const double iDouble = static_cast<double>(i);
            const double w = std::exp(-(iDouble * iDouble) / s2);
            outWeights[weightIndex] = static_cast<float>(w);
            wsum += w;
        }
        const double invW = (wsum != 0.0) ? (1.0 / wsum) : 0.0;
        for (float& w : outWeights) {
            w = static_cast<float>(static_cast<double>(w) * invW);
        }
    }

    static bool acquire_shared_gaussian_entry(
        const SharedGaussianKey& key,
        SharedGaussianSpec spec,
        const std::vector<float>& cpuWeights,
        SharedGaussianEntry& outEntry,
        std::string& outError) {
        outError.clear();
        SharedGaussianCacheState& cache = shared_gaussian_cache_state();
        {
            std::lock_guard<std::mutex> lock(cache.mutex);
            const auto it = cache.byKey.find(key);
            if (it != cache.byKey.end()) {
                outEntry = it->second;
                return true;
            }
        }

        float* dWeights = nullptr;
        const std::size_t bytes = cpuWeights.size() * sizeof(float);
        const cudaError_t allocErr = cudaMalloc(reinterpret_cast<void**>(&dWeights), bytes);
        if (allocErr != cudaSuccess || !dWeights) {
            outError = std::string("cudaMalloc(shared gaussian kernel) failed: ") + (cudaGetErrorString(allocErr) ? cudaGetErrorString(allocErr) : "(unknown)");
            dWeights = nullptr;
            return false;
        }
        const cudaError_t copyErr = cudaMemcpy(
            dWeights,
            cpuWeights.data(),
            bytes,
            cudaMemcpyHostToDevice);
        if (copyErr != cudaSuccess) {
            outError = std::string("cudaMemcpy(shared gaussian kernel) failed: ") + (cudaGetErrorString(copyErr) ? cudaGetErrorString(copyErr) : "(unknown)");
            cudaFree(dWeights);
            return false;
        }

        std::lock_guard<std::mutex> lock(cache.mutex);
        const auto existingIt = cache.byKey.find(key);
        if (existingIt != cache.byKey.end()) {
            cudaFree(dWeights);
            outEntry = existingIt->second;
            return true;
        }

        SharedGaussianEntry entry{};
        entry.weights = dWeights;
        entry.radius = spec.radius;
        entry.sigma = spec.sigma;
        entry.capacity = static_cast<int>(cpuWeights.size());
        entry.id = make_shared_gaussian_id(key);
        cache.byKey.emplace(key, entry);
        outEntry = entry;
        return true;
    }

    static bool shared_gaussian_entry_matches_cache(
        const SharedGaussianKey& key,
        const Resources::DeviceGaussianKernel& kernel) {
        if (kernel.sharedKernelId == 0 || !kernel.weights) {
            return false;
        }
        SharedGaussianCacheState& cache = shared_gaussian_cache_state();
        std::lock_guard<std::mutex> lock(cache.mutex);
        const auto it = cache.byKey.find(key);
        if (it == cache.byKey.end()) {
            return false;
        }
        return it->second.id == kernel.sharedKernelId &&
               it->second.weights == kernel.weights;
    }

    static const char* to_cstr(ResourceManager::AllocatorBackendMode mode) noexcept {
        switch (mode) {
            case ResourceManager::AllocatorBackendMode::CudaMalloc:
                return "cuda_malloc";
            case ResourceManager::AllocatorBackendMode::AsyncPool:
                return "async_pool";
            case ResourceManager::AllocatorBackendMode::Slab:
                return "slab";
            default:
                return "unknown";
        }
    }

    static ResourceManager::AllocatorBackendMode scratch_allocator_backend_mode_locked(
        const Resources& resources) noexcept {
        ResourceManager::DeviceContextKey key{};
        key.deviceId = resources.deviceId;
        key.contextOpaque = resources.ownerContextOpaque;
        return ResourceManager::query_allocator_backend_mode(key);
    }

    static bool is_async_device_ptr_tracked_locked(const Resources& resources, const void* ptr) noexcept {
        return ptr &&
               resources.asyncDeviceAllocPointers.find(const_cast<void*>(ptr)) !=
                   resources.asyncDeviceAllocPointers.end();
    }

    static bool track_async_device_ptr_locked(Resources& resources, void* ptr, bool asyncAllocated) noexcept {
        if (!ptr) {
            return true;
        }
        try {
            if (asyncAllocated) {
                resources.asyncDeviceAllocPointers.insert(ptr);
            } else {
                resources.asyncDeviceAllocPointers.erase(ptr);
            }
            return true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            if (asyncAllocated) {
                const cudaError_t freeErr = device_free_async_compat(ptr, nullptr);
                if (freeErr == cudaSuccess) {
                    (void)cudaStreamSynchronize(nullptr);
                } else {
                    (void)cudaFree(ptr);
                }
            }
            return false;
        }
    }

    static void untrack_async_device_ptr_locked(Resources& resources, void* ptr) noexcept {
        if (!ptr) {
            return;
        }
        resources.asyncDeviceAllocPointers.erase(ptr);
    }

    static cudaError_t device_free_async_compat(void* ptr, cudaStream_t stream) noexcept {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
        return cudaFreeAsync(ptr, stream);
#else
        (void)stream;
        return cudaErrorNotSupported;
#endif
    }

    static void free_tracked_device_ptr_locked(
        Resources& resources,
        void*& ptr,
        void* cudaStreamOpaque) noexcept {
        if (!ptr) {
            return;
        }
        const bool asyncTracked = is_async_device_ptr_tracked_locked(resources, ptr);
        if (asyncTracked) {
            const cudaStream_t stream =
                cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
            const cudaError_t asyncErr = device_free_async_compat(ptr, stream);
            if (asyncErr != cudaSuccess) {
                (void)cudaFree(ptr);
            } else if (!cudaStreamOpaque) {
                (void)cudaStreamSynchronize(nullptr);
            }
        } else {
            (void)cudaFree(ptr);
        }
        untrack_async_device_ptr_locked(resources, ptr);
        ptr = nullptr;
    }

    template <typename T>
    static void free_tracked_device_ptr_locked(
        Resources& resources,
        T*& ptr,
        void* cudaStreamOpaque) noexcept {
        void* raw = reinterpret_cast<void*>(ptr);
        free_tracked_device_ptr_locked(resources, raw, cudaStreamOpaque);
        ptr = reinterpret_cast<T*>(raw);
    }

    static bool allocate_scratch_device_ptr_locked(
        Resources& resources,
        void*& outPtr,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
        outPtr = nullptr;
        if (bytes == 0) {
            outError = std::string(label ? label : "scratch") + " bytes invalid";
            return false;
        }

        const ResourceManager::AllocatorBackendMode backendMode =
            scratch_allocator_backend_mode_locked(resources);
        const bool preferAsync = (backendMode == ResourceManager::AllocatorBackendMode::AsyncPool);

        cudaError_t asyncErr = cudaSuccess;
        if (preferAsync) {
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 11020)
            const cudaStream_t stream = cudaStreamOpaque ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque) : nullptr;
            asyncErr = cudaMallocAsync(&outPtr, bytes, stream);
            if (asyncErr == cudaSuccess && outPtr) {
                if (!track_async_device_ptr_locked(resources, outPtr, true)) {
                    outPtr = nullptr;
                    outError = "async scratch allocation tracking failed";
                    return false;
                }
                return true;
            }
            outPtr = nullptr;
#else
            asyncErr = cudaErrorNotSupported;
#endif
        }

        const cudaError_t allocErr = cudaMalloc(&outPtr, bytes);
        if (allocErr == cudaSuccess && outPtr) {
            if (!track_async_device_ptr_locked(resources, outPtr, false)) {
                cudaFree(outPtr);
                outPtr = nullptr;
                outError = "scratch allocation tracking failed";
                return false;
            }
            if (preferAsync && JTRACE_ENABLED(2)) {
                std::ostringstream oss;
                oss << "event=alloc_fallback"
                    << " path=scratch"
                    << " label=" << (label ? label : "scratch")
                    << " backend_requested=" << to_cstr(backendMode)
                    << " async_error=" << (cudaGetErrorString(asyncErr) ? cudaGetErrorString(asyncErr) : "(unknown)");
                JTRACE("MSALC", oss.str());
            }
            return true;
        }

        if (preferAsync) {
            outError = std::string("scratch alloc failed (async+cuda_malloc) [") + (label ? label : "scratch") + "]: async=" + (cudaGetErrorString(asyncErr) ? cudaGetErrorString(asyncErr) : "(unknown)") + ", cuda_malloc=" + (cudaGetErrorString(allocErr) ? cudaGetErrorString(allocErr) : "(unknown)");
        } else {
            outError = std::string("cudaMalloc(") + (label ? label : "scratch") + ") failed: " + (cudaGetErrorString(allocErr) ? cudaGetErrorString(allocErr) : "(unknown)");
        }
        outPtr = nullptr;
        return false;
    }

    template <typename T>
    static bool allocate_scratch_device_ptr_locked(
        Resources& resources,
        T*& outPtr,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
        void* raw = nullptr;
        const bool ok = allocate_scratch_device_ptr_locked(
            resources,
            raw,
            bytes,
            cudaStreamOpaque,
            label,
            outError);
        outPtr = reinterpret_cast<T*>(raw);
        return ok;
    }
#endif

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
        if (resources.tablesIllum) {
            cudaFree(resources.tablesIllum);
            resources.tablesIllum = nullptr;
        }
#endif
        resources.tablesK = 0;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool retire_tables_locked(Resources& resources, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t bytes = static_cast<size_t>(std::max(0, resources.tablesK)) * sizeof(float);
        if (resources.tablesAx) {
            if (!retire_ptr_locked(resources, resources.tablesAx, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.tablesAx = nullptr;
        }
        if (resources.tablesAy) {
            if (!retire_ptr_locked(resources, resources.tablesAy, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.tablesAy = nullptr;
        }
        if (resources.tablesAz) {
            if (!retire_ptr_locked(resources, resources.tablesAz, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.tablesAz = nullptr;
        }
        if (resources.tablesIllum) {
            if (!retire_ptr_locked(resources, resources.tablesIllum, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError)) {
                return false;
            }
            resources.tablesIllum = nullptr;
        }
        resources.tablesK = 0;
        return true;
#endif
    }
#endif

    static void free_spectral_tables(Resources::DeviceSpectralTables& t) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (t.epsC) {
            cudaFree(t.epsC);
            t.epsC = nullptr;
        }
        if (t.epsM) {
            cudaFree(t.epsM);
            t.epsM = nullptr;
        }
        if (t.epsY) {
            cudaFree(t.epsY);
            t.epsY = nullptr;
        }
        if (t.Ax) {
            cudaFree(t.Ax);
            t.Ax = nullptr;
        }
        if (t.Ay) {
            cudaFree(t.Ay);
            t.Ay = nullptr;
        }
        if (t.Az) {
            cudaFree(t.Az);
            t.Az = nullptr;
        }
        if (t.baseDensityMin) {
            cudaFree(t.baseDensityMin);
            t.baseDensityMin = nullptr;
        }
#endif
        t.K = 0;
        t.hasBaseline = 0;
        t.invYn = 1.0f;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool retire_spectral_tables_locked(Resources& resources, Resources::DeviceSpectralTables& t, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)t;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t bytes = static_cast<size_t>(std::max(0, t.K)) * sizeof(float);
        if (t.epsC) {
            if (!retire_ptr_locked(resources, t.epsC, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError))
                return false;
            t.epsC = nullptr;
        }
        if (t.epsM) {
            if (!retire_ptr_locked(resources, t.epsM, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError))
                return false;
            t.epsM = nullptr;
        }
        if (t.epsY) {
            if (!retire_ptr_locked(resources, t.epsY, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError))
                return false;
            t.epsY = nullptr;
        }
        if (t.Ax) {
            if (!retire_ptr_locked(resources, t.Ax, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError))
                return false;
            t.Ax = nullptr;
        }
        if (t.Ay) {
            if (!retire_ptr_locked(resources, t.Ay, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError))
                return false;
            t.Ay = nullptr;
        }
        if (t.Az) {
            if (!retire_ptr_locked(resources, t.Az, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError))
                return false;
            t.Az = nullptr;
        }
        if (t.baseDensityMin) {
            if (!retire_ptr_locked(resources, t.baseDensityMin, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError))
                return false;
            t.baseDensityMin = nullptr;
        }
        t.K = 0;
        t.hasBaseline = 0;
        t.invYn = 1.0f;
        return true;
#endif
    }
#endif

    static void free_scan_medium(Resources::DeviceScanMedium& m) noexcept {
        free_spectral_tables(m.tables);
        m.mediumIsNegative = 1;
        m.min_cmy[0] = m.min_cmy[1] = m.min_cmy[2] = 0.0f;
        m.inv_max_cmy[0] = m.inv_max_cmy[1] = m.inv_max_cmy[2] = 1.0f;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool retire_scan_medium_locked(Resources& resources, Resources::DeviceScanMedium& m, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)m;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!retire_spectral_tables_locked(resources, m.tables, cudaStreamOpaque, label, outError)) {
            return false;
        }
        m.mediumIsNegative = 1;
        m.min_cmy[0] = m.min_cmy[1] = m.min_cmy[2] = 0.0f;
        m.inv_max_cmy[0] = m.inv_max_cmy[1] = m.inv_max_cmy[2] = 1.0f;
        return true;
#endif
    }
#endif

    static void free_scan_lut(Resources::DeviceSpectralLut& lut) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (lut.log2PchipXYZ) {
            cudaFree(lut.log2PchipXYZ);
            lut.log2PchipXYZ = nullptr;
        }
        if (lut.slopeC) {
            cudaFree(lut.slopeC);
            lut.slopeC = nullptr;
        }
        if (lut.slopeM) {
            cudaFree(lut.slopeM);
            lut.slopeM = nullptr;
        }
        if (lut.slopeY) {
            cudaFree(lut.slopeY);
            lut.slopeY = nullptr;
        }
        if (lut.cellMin) {
            cudaFree(lut.cellMin);
            lut.cellMin = nullptr;
        }
        if (lut.cellMax) {
            cudaFree(lut.cellMax);
            lut.cellMax = nullptr;
        }
#endif
        lut.res = 0;
        lut.hash = 0;
    }

    static void free_gaussian_kernel(Resources::DeviceGaussianKernel& k) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        if (k.weights) {
            if (k.sharedKernelId == 0) {
                cudaFree(k.weights);
            }
            k.weights = nullptr;
        }
#endif
        k.radius = 0;
        k.sigma = 0.0f;
        k.capacity = 0;
        k.sharedKernelId = 0;
    }

    static void free_optics_scratch(Resources& resources, Resources::DeviceOpticsScratch& s, void* cudaStreamOpaque = nullptr) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        free_tracked_device_ptr_locked(resources, s.rgbR, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.rgbG, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.rgbB, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.blurred, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.aux, cudaStreamOpaque);
        free_tracked_device_ptr_locked(
            resources,
            s.grainFrameUniforms,
            cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.grainTmp, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.grainTmpShared, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.gateMask, cudaStreamOpaque);
#endif
        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
        s.capacityElements = 0;
        s.gateWidth = 0;
        s.gateHeight = 0;
        s.gateMaskCapacityElements = 0;
        s.gateMaskHash = 0;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool retire_optics_scratch_locked(Resources& resources, Resources::DeviceOpticsScratch& s, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)s;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t planeBytes = s.capacityElements * sizeof(float);
        if (s.rgbR) {
            if (!retire_ptr_locked(resources, s.rgbR, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.rgbR = nullptr;
        }
        if (s.rgbG) {
            if (!retire_ptr_locked(resources, s.rgbG, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.rgbG = nullptr;
        }
        if (s.rgbB) {
            if (!retire_ptr_locked(resources, s.rgbB, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.rgbB = nullptr;
        }
        if (s.blurred) {
            if (!retire_ptr_locked(resources, s.blurred, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.blurred = nullptr;
        }
        if (s.aux) {
            if (!retire_ptr_locked(resources, s.aux, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.aux = nullptr;
        }
        if (s.grainFrameUniforms) {
            if (!retire_ptr_locked(
                    resources,
                    s.grainFrameUniforms,
                    sizeof(GrainFrameUniforms),
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    label,
                    outError,
                    true)) {
                return false;
            }
            s.grainFrameUniforms = nullptr;
        }
        if (s.grainTmp) {
            if (!retire_ptr_locked(resources, s.grainTmp, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.grainTmp = nullptr;
        }
        if (s.grainTmpShared) {
            if (!retire_ptr_locked(resources, s.grainTmpShared, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.grainTmpShared = nullptr;
        }
        const size_t gateBytes = s.gateMaskCapacityElements * sizeof(float);
        if (s.gateMask) {
            if (!retire_ptr_locked(resources, s.gateMask, gateBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.gateMask = nullptr;
        }

        s.tmp = nullptr;
        s.width = 0;
        s.height = 0;
        s.capacityElements = 0;
        s.gateWidth = 0;
        s.gateHeight = 0;
        s.gateMaskCapacityElements = 0;
        s.gateMaskHash = 0;
        return true;
#endif
    }
#endif

    static void free_spatial_dir_scratch(Resources& resources, Resources::DeviceSpatialDirScratch& s, void* cudaStreamOpaque = nullptr) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        free_tracked_device_ptr_locked(resources, s.rawCorrectionY, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.rawCorrectionM, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.rawCorrectionC, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.filteredCorrectionY, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.filteredCorrectionM, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.filteredCorrectionC, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.filterTempM, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.filterTempC, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.logRawB, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.logRawG, cudaStreamOpaque);
        free_tracked_device_ptr_locked(resources, s.logRawR, cudaStreamOpaque);
#endif
        s.filterTemp = nullptr;
        s.width = 0;
        s.height = 0;
        s.capacityElements = 0;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool retire_spatial_dir_scratch_locked(Resources& resources, Resources::DeviceSpatialDirScratch& s, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)s;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const size_t bytes = s.capacityElements * sizeof(float);
        if (s.rawCorrectionY) {
            if (!retire_ptr_locked(resources, s.rawCorrectionY, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.rawCorrectionY = nullptr;
        }
        if (s.rawCorrectionM) {
            if (!retire_ptr_locked(resources, s.rawCorrectionM, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.rawCorrectionM = nullptr;
        }
        if (s.rawCorrectionC) {
            if (!retire_ptr_locked(resources, s.rawCorrectionC, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.rawCorrectionC = nullptr;
        }
        if (s.filteredCorrectionY) {
            if (!retire_ptr_locked(resources, s.filteredCorrectionY, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.filteredCorrectionY = nullptr;
        }
        if (s.filteredCorrectionM) {
            if (!retire_ptr_locked(resources, s.filteredCorrectionM, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.filteredCorrectionM = nullptr;
        }
        if (s.filteredCorrectionC) {
            if (!retire_ptr_locked(resources, s.filteredCorrectionC, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.filteredCorrectionC = nullptr;
        }
        if (s.filterTempM) {
            if (!retire_ptr_locked(resources, s.filterTempM, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.filterTempM = nullptr;
        }
        if (s.filterTempC) {
            if (!retire_ptr_locked(resources, s.filterTempC, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.filterTempC = nullptr;
        }
        if (s.logRawB) {
            if (!retire_ptr_locked(resources, s.logRawB, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.logRawB = nullptr;
        }
        if (s.logRawG) {
            if (!retire_ptr_locked(resources, s.logRawG, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.logRawG = nullptr;
        }
        if (s.logRawR) {
            if (!retire_ptr_locked(resources, s.logRawR, bytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label, outError, true))
                return false;
            s.logRawR = nullptr;
        }
        s.filterTemp = nullptr;
        s.width = 0;
        s.height = 0;
        s.capacityElements = 0;
        return true;
#endif
    }
#endif

    static void free_shared_tmp_plane(Resources& resources) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        free_tracked_device_ptr_locked(
            resources,
            resources.sharedTmpPlane,
            nullptr);
#endif
        resources.sharedTmpPlane = nullptr;
        resources.sharedTmpWidth = 0;
        resources.sharedTmpHeight = 0;
        resources.sharedTmpCapacityElements = 0;
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool optics_base_live_locked(const Resources& resources) noexcept {
        const auto& scratch = resources.scannerScratch;
        return scratch.rgbR || scratch.rgbG || scratch.rgbB ||
               scratch.grainFrameUniforms;
    }

    static bool optics_any_live_locked(const Resources::DeviceOpticsScratch& scratch) noexcept {
        return scratch.rgbR || scratch.rgbG || scratch.rgbB ||
               scratch.blurred || scratch.aux ||
               scratch.grainTmp || scratch.grainTmpShared ||
               scratch.grainFrameUniforms ||
               scratch.gateMask;
    }

    static std::size_t optics_stage_live_bytes_locked(
        const Resources::DeviceOpticsScratch& scratch) noexcept {
        const std::size_t planeBytes = scratch.capacityElements * sizeof(float);
        std::size_t bytes = 0;
        auto add_plane = [&](const float* ptr) noexcept {
            if (ptr) {
                bytes += planeBytes;
            }
        };
        add_plane(scratch.rgbR);
        add_plane(scratch.rgbG);
        add_plane(scratch.rgbB);
        add_plane(scratch.blurred);
        add_plane(scratch.aux);
        if (scratch.grainFrameUniforms) {
            bytes += sizeof(GrainFrameUniforms);
        }
        add_plane(scratch.grainTmp);
        add_plane(scratch.grainTmpShared);
        if (scratch.gateMask) {
            bytes += scratch.gateMaskCapacityElements * sizeof(float);
        }
        return bytes;
    }

    static bool spatial_dir_base_live_locked(const Resources& resources) noexcept {
        const auto& scratch = resources.spatialDirScratch;
        return scratch.rawCorrectionY || scratch.rawCorrectionM || scratch.rawCorrectionC ||
               scratch.filteredCorrectionY || scratch.filteredCorrectionM ||
               scratch.filteredCorrectionC || scratch.filterTempM || scratch.filterTempC ||
               scratch.logRawB || scratch.logRawG || scratch.logRawR;
    }

    static std::size_t spatial_dir_stage_live_bytes_locked(
        const Resources::DeviceSpatialDirScratch& scratch) noexcept {
        const std::size_t planeBytes = scratch.capacityElements * sizeof(float);
        std::size_t bytes = 0;
        auto add_plane = [&](const float* ptr) noexcept {
            if (ptr) {
                bytes += planeBytes;
            }
        };
        add_plane(scratch.rawCorrectionY);
        add_plane(scratch.rawCorrectionM);
        add_plane(scratch.rawCorrectionC);
        add_plane(scratch.filteredCorrectionY);
        add_plane(scratch.filteredCorrectionM);
        add_plane(scratch.filteredCorrectionC);
        add_plane(scratch.filterTempM);
        add_plane(scratch.filterTempC);
        add_plane(scratch.logRawB);
        add_plane(scratch.logRawG);
        add_plane(scratch.logRawR);
        return bytes;
    }

    static bool spatial_dir_scratch_has_required_roles_locked(
        const Resources::DeviceSpatialDirScratch& scratch,
        const Spektrafilm::DirScratchPlaneRoles& planeRoles) noexcept {
        const bool hasThreeRaw =
            scratch.rawCorrectionY && scratch.rawCorrectionM && scratch.rawCorrectionC;
        const bool hasStreamedRaw =
            scratch.rawCorrectionY && scratch.rawCorrectionM == nullptr && scratch.rawCorrectionC == nullptr;
        const bool rawCorrectionMatch =
            (planeRoles.rawCorrectionPlanes == 3 && hasThreeRaw) ||
            (planeRoles.rawCorrectionPlanes == 1 && hasStreamedRaw);
        const bool tier1Base =
            rawCorrectionMatch &&
            scratch.filteredCorrectionY && scratch.filteredCorrectionM && scratch.filteredCorrectionC;
        if (!tier1Base) {
            return false;
        }
        const bool hasAliasedForwardTemps =
            scratch.filterTempM && scratch.filterTempC;
        const bool hasLowScratchPairTemps =
            scratch.filterTempM && scratch.filterTempC == nullptr;
        const bool hasNoChannelTemps =
            scratch.filterTempM == nullptr && scratch.filterTempC == nullptr;
        const bool channelTempsMatch =
            (planeRoles.filterTempPlanes == 1 && hasNoChannelTemps) ||
            (planeRoles.filterTempPlanes == 2 && hasLowScratchPairTemps) ||
            (planeRoles.filterTempPlanes == 3 && hasAliasedForwardTemps);
        const bool cachedLogRawMatch =
            (planeRoles.cachedLogRawPlanes == 0 &&
             scratch.logRawB == nullptr && scratch.logRawG == nullptr && scratch.logRawR == nullptr) ||
            (planeRoles.cachedLogRawPlanes == 2 &&
             scratch.logRawB && scratch.logRawG && scratch.logRawR == nullptr) ||
            (planeRoles.cachedLogRawPlanes == 3 &&
             scratch.logRawB && scratch.logRawG && scratch.logRawR);
        return channelTempsMatch && cachedLogRawMatch;
    }

    static bool spatial_dir_scratch_has_retained_final_roles_locked(
        const Resources::DeviceSpatialDirScratch& scratch,
        const Spektrafilm::DirScratchPlaneRoles& planeRoles) noexcept {
        if (!scratch.filteredCorrectionY || !scratch.filteredCorrectionM ||
            !scratch.filteredCorrectionC) {
            return false;
        }
        const bool cachedLogRawCompatible =
            (planeRoles.cachedLogRawPlanes == 0 &&
             scratch.logRawB == nullptr && scratch.logRawG == nullptr && scratch.logRawR == nullptr) ||
            (planeRoles.cachedLogRawPlanes == 2 && scratch.logRawR == nullptr) ||
            (planeRoles.cachedLogRawPlanes == 3);
        if (!cachedLogRawCompatible) {
            return false;
        }
        const bool filterTempsCompatible =
            planeRoles.filterTempPlanes == 3 ||
            (planeRoles.filterTempPlanes == 2 && !scratch.filterTempC) ||
            (!scratch.filterTempM && !scratch.filterTempC);
        return filterTempsCompatible;
    }

    static bool spatial_dir_request_needs_shared_tmp_plane(
        const Spektrafilm::DirScratchPlaneRoles& planeRoles) noexcept {
        return planeRoles.filterTempPlanes > 0;
    }

    static bool retire_shared_tmp_plane_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!resources.sharedTmpPlane) {
            return true;
        }

        const std::size_t bytes = resources.sharedTmpCapacityElements * sizeof(float);
        if (!retire_ptr_locked(
                resources,
                resources.sharedTmpPlane,
                bytes,
                Resources::RetireKind::DeviceFree,
                cudaStreamOpaque,
                label ? label : "shared tmp plane",
                outError,
                true)) {
            return false;
        }
        resources.sharedTmpPlane = nullptr;
        resources.sharedTmpWidth = 0;
        resources.sharedTmpHeight = 0;
        resources.sharedTmpCapacityElements = 0;
        resources.scannerScratch.tmp = nullptr;
        resources.spatialDirScratch.filterTemp = nullptr;
        return true;
#endif
    }

    static bool retire_orphaned_shared_tmp_plane_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!resources.sharedTmpPlane) {
            return true;
        }
        if (optics_base_live_locked(resources) || spatial_dir_base_live_locked(resources)) {
            return true;
        }
        return retire_shared_tmp_plane_locked(
            resources,
            cudaStreamOpaque,
            label ? label : "scratch normalization shared tmp plane",
            outError);
#endif
    }

    static bool shared_tmp_has_live_reference_locked(const Resources& resources) noexcept {
        const float* sharedTmp = resources.sharedTmpPlane;
        if (!sharedTmp) {
            return false;
        }
        if (resources.spatialDirScratch.filterTemp == sharedTmp) {
            return true;
        }
        return resources.scannerScratch.tmp == sharedTmp &&
               optics_any_live_locked(resources.scannerScratch);
    }

    static bool retire_unreferenced_shared_tmp_plane_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!resources.sharedTmpPlane || shared_tmp_has_live_reference_locked(resources)) {
            return true;
        }
        return retire_shared_tmp_plane_locked(
            resources,
            cudaStreamOpaque,
            label ? label : "unreferenced shared tmp plane",
            outError);
#endif
    }

    static bool retire_spatial_dir_build_scratch_locked(
        Resources& resources,
        Resources::DeviceSpatialDirScratch& scratch,
        void* cudaStreamOpaque,
        SpatialDirBuildScratchReleaseStats& outStats,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)scratch;
        (void)cudaStreamOpaque;
        (void)outStats;
        outError = "CUDA is not enabled";
        return false;
#else
        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        auto retire_plane = [&](float*& ptr, const char* label, std::size_t& retiredBytes) {
            if (!ptr) {
                return true;
            }
            if (bytes == 0) {
                outError = std::string(label ? label : "spatial DIR build scratch") +
                           " capacity is invalid";
                return false;
            }
            if (!retire_ptr_locked(
                    resources,
                    ptr,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    label ? label : "spatial DIR build scratch",
                    outError,
                    true)) {
                return false;
            }
            ptr = nullptr;
            retiredBytes += bytes;
            return true;
        };

        if (!retire_plane(
                scratch.rawCorrectionY,
                "retained spatial DIR build rawCorrectionY",
                outStats.rawCorrectionRetiredBytes) ||
            !retire_plane(
                scratch.rawCorrectionM,
                "retained spatial DIR build rawCorrectionM",
                outStats.rawCorrectionRetiredBytes) ||
            !retire_plane(
                scratch.rawCorrectionC,
                "retained spatial DIR build rawCorrectionC",
                outStats.rawCorrectionRetiredBytes) ||
            !retire_plane(
                scratch.filterTempM,
                "retained spatial DIR build filterTempM",
                outStats.filterTempRetiredBytes) ||
            !retire_plane(
                scratch.filterTempC,
                "retained spatial DIR build filterTempC",
                outStats.filterTempRetiredBytes)) {
            return false;
        }

        scratch.filterTemp = nullptr;
        return true;
#endif
    }

    static bool retire_spatial_dir_cached_log_raw_locked(
        Resources& resources,
        Resources::DeviceSpatialDirScratch& scratch,
        void* cudaStreamOpaque,
        SpatialDirCachedLogRawReleaseStats& outStats,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)scratch;
        (void)cudaStreamOpaque;
        (void)outStats;
        outError = "CUDA is not enabled";
        return false;
#else
        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        auto retire_plane = [&](float*& ptr, const char* label) {
            if (!ptr) {
                return true;
            }
            if (bytes == 0) {
                outError = std::string(label ? label : "spatial DIR cached log raw") +
                           " capacity is invalid";
                return false;
            }
            if (!retire_ptr_locked(
                    resources,
                    ptr,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    label ? label : "spatial DIR cached log raw",
                    outError,
                    true)) {
                return false;
            }
            ptr = nullptr;
            outStats.cachedLogRawRetiredBytes += bytes;
            return true;
        };

        return retire_plane(scratch.logRawB, "retained spatial DIR final logRawB") &&
               retire_plane(scratch.logRawG, "retained spatial DIR final logRawG") &&
               retire_plane(scratch.logRawR, "retained spatial DIR final logRawR");
#endif
    }

    static bool retire_optics_gate_mask_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceOpticsScratch& scratch = resources.scannerScratch;
        if (!scratch.gateMask) {
            return true;
        }

        const std::size_t bytes = scratch.gateMaskCapacityElements * sizeof(float);
        if (!retire_ptr_locked(
                resources,
                scratch.gateMask,
                bytes,
                Resources::RetireKind::DeviceFree,
                cudaStreamOpaque,
                "scratch normalization optics gate mask",
                outError,
                true)) {
            return false;
        }
        scratch.gateMask = nullptr;
        scratch.gateWidth = 0;
        scratch.gateHeight = 0;
        scratch.gateMaskCapacityElements = 0;
        scratch.gateMaskHash = 0;
        return true;
#endif
    }

    static bool retire_optics_grain_shared_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceOpticsScratch& scratch = resources.scannerScratch;
        if (!scratch.grainTmpShared) {
            return true;
        }

        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        if (!retire_ptr_locked(
                resources,
                scratch.grainTmpShared,
                bytes,
                Resources::RetireKind::DeviceFree,
                cudaStreamOpaque,
                "scratch normalization optics grain shared",
                outError,
                true)) {
            return false;
        }
        scratch.grainTmpShared = nullptr;
        return true;
#endif
    }

    static bool retire_optics_grain_layer_work_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceOpticsScratch& scratch = resources.scannerScratch;
        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        if (scratch.grainTmp) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.grainTmp,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization optics grain layer work",
                    outError,
                    true)) {
                return false;
            }
            scratch.grainTmp = nullptr;
        }
        return true;
#endif
    }

    static bool retire_optics_aux_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceOpticsScratch& scratch = resources.scannerScratch;
        if (!scratch.aux) {
            return true;
        }

        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        if (!retire_ptr_locked(
                resources,
                scratch.aux,
                bytes,
                Resources::RetireKind::DeviceFree,
                cudaStreamOpaque,
                "scratch normalization optics aux",
                outError,
                true)) {
            return false;
        }
        scratch.aux = nullptr;
        return true;
#endif
    }

    static bool retire_optics_blurred_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceOpticsScratch& scratch = resources.scannerScratch;
        if (!scratch.blurred) {
            return true;
        }

        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        if (!retire_ptr_locked(
                resources,
                scratch.blurred,
                bytes,
                Resources::RetireKind::DeviceFree,
                cudaStreamOpaque,
                "scratch normalization optics blurred",
                outError,
                true)) {
            return false;
        }
        scratch.blurred = nullptr;
        return true;
#endif
    }

    static bool retire_spatial_dir_base_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceSpatialDirScratch& scratch = resources.spatialDirScratch;
        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        if (scratch.rawCorrectionY) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.rawCorrectionY,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization spatial dir base",
                    outError,
                    true)) {
                return false;
            }
            scratch.rawCorrectionY = nullptr;
        }
        if (scratch.rawCorrectionM) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.rawCorrectionM,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization spatial dir base",
                    outError,
                    true)) {
                return false;
            }
            scratch.rawCorrectionM = nullptr;
        }
        if (scratch.rawCorrectionC) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.rawCorrectionC,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization spatial dir base",
                    outError,
                    true)) {
                return false;
            }
            scratch.rawCorrectionC = nullptr;
        }
        float** filteredPlanes[3] = {
            &scratch.filteredCorrectionY,
            &scratch.filteredCorrectionM,
            &scratch.filteredCorrectionC};
        for (float** filteredPlane : filteredPlanes) {
            if (*filteredPlane) {
                if (!retire_ptr_locked(
                        resources,
                        *filteredPlane,
                        bytes,
                        Resources::RetireKind::DeviceFree,
                        cudaStreamOpaque,
                        "scratch normalization spatial dir filtered correction",
                        outError,
                        true)) {
                    return false;
                }
                *filteredPlane = nullptr;
            }
        }
        float** channelTempPlanes[2] = {&scratch.filterTempM, &scratch.filterTempC};
        for (float** tempPlane : channelTempPlanes) {
            if (*tempPlane) {
                if (!retire_ptr_locked(
                        resources,
                        *tempPlane,
                        bytes,
                        Resources::RetireKind::DeviceFree,
                        cudaStreamOpaque,
                        "scratch normalization spatial dir channel temp",
                        outError,
                        true)) {
                    return false;
                }
                *tempPlane = nullptr;
            }
        }
        float** logRawPlanes[3] = {&scratch.logRawB, &scratch.logRawG, &scratch.logRawR};
        for (float** logRawPlane : logRawPlanes) {
            if (*logRawPlane) {
                if (!retire_ptr_locked(
                        resources,
                        *logRawPlane,
                        bytes,
                        Resources::RetireKind::DeviceFree,
                        cudaStreamOpaque,
                        "scratch normalization spatial dir log raw",
                        outError,
                        true)) {
                    return false;
                }
                *logRawPlane = nullptr;
            }
        }
        scratch.filterTemp = nullptr;
        scratch.width = 0;
        scratch.height = 0;
        scratch.capacityElements = 0;
        return retire_orphaned_shared_tmp_plane_locked(
            resources,
            cudaStreamOpaque,
            "scratch normalization spatial dir shared tmp",
            outError);
#endif
    }

    static bool retire_optics_base_candidate_locked(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        Resources::DeviceOpticsScratch& scratch = resources.scannerScratch;
        const std::size_t bytes = scratch.capacityElements * sizeof(float);
        if (scratch.rgbR) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.rgbR,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization optics base",
                    outError,
                    true)) {
                return false;
            }
            scratch.rgbR = nullptr;
        }
        if (scratch.rgbG) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.rgbG,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization optics base",
                    outError,
                    true)) {
                return false;
            }
            scratch.rgbG = nullptr;
        }
        if (scratch.rgbB) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.rgbB,
                    bytes,
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization optics base",
                    outError,
                    true)) {
                return false;
            }
            scratch.rgbB = nullptr;
        }
        if (scratch.grainFrameUniforms) {
            if (!retire_ptr_locked(
                    resources,
                    scratch.grainFrameUniforms,
                    sizeof(GrainFrameUniforms),
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "scratch normalization grain frame uniforms",
                    outError,
                    true)) {
                return false;
            }
            scratch.grainFrameUniforms = nullptr;
        }

        if (!optics_any_live_locked(scratch)) {
            scratch.tmp = nullptr;
            scratch.width = 0;
            scratch.height = 0;
            scratch.capacityElements = 0;
            scratch.gateWidth = 0;
            scratch.gateHeight = 0;
            scratch.gateMaskCapacityElements = 0;
            scratch.gateMaskHash = 0;
        }
        return retire_orphaned_shared_tmp_plane_locked(
            resources,
            cudaStreamOpaque,
            "scratch normalization optics shared tmp",
            outError);
#endif
    }

#endif

    bool retire_scratch_policy_candidate(
        Resources& resources,
        ResourceManager::ScratchPolicyCandidate candidate,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)candidate;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }
        if (resources.retainedScratchLeaseGeneration != 0) {
            return true;
        }

        switch (candidate) {
            case ResourceManager::ScratchPolicyCandidate::OpticsGateMask:
                return retire_optics_gate_mask_candidate_locked(resources, cudaStreamOpaque, outError);
            case ResourceManager::ScratchPolicyCandidate::OpticsGrainShared:
                return retire_optics_grain_shared_candidate_locked(resources, cudaStreamOpaque, outError);
            case ResourceManager::ScratchPolicyCandidate::OpticsGrainLayerWork:
                return retire_optics_grain_layer_work_candidate_locked(resources, cudaStreamOpaque, outError);
            case ResourceManager::ScratchPolicyCandidate::OpticsAux:
                return retire_optics_aux_candidate_locked(resources, cudaStreamOpaque, outError);
            case ResourceManager::ScratchPolicyCandidate::OpticsBlurred:
                return retire_optics_blurred_candidate_locked(resources, cudaStreamOpaque, outError);
            case ResourceManager::ScratchPolicyCandidate::SpatialDirBase:
                return retire_spatial_dir_base_candidate_locked(resources, cudaStreamOpaque, outError);
            case ResourceManager::ScratchPolicyCandidate::OpticsBase:
                return retire_optics_base_candidate_locked(resources, cudaStreamOpaque, outError);
            default:
                outError = "scratch normalization candidate is invalid";
                return false;
        }
#endif
    }

    bool retire_orphaned_shared_tmp_plane(
        Resources& resources,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }
        if (resources.retainedScratchLeaseGeneration != 0) {
            return true;
        }
        return retire_orphaned_shared_tmp_plane_locked(
            resources,
            cudaStreamOpaque,
            "scratch normalization orphaned shared tmp",
            outError);
#endif
    }

    bool try_acquire_retained_frame_scratch_lease(
        Resources& resources,
        std::uint64_t leaseGeneration,
        void* cudaStreamOpaque,
        bool& outAcquired,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)leaseGeneration;
        (void)cudaStreamOpaque;
        outAcquired = false;
        outError = "CUDA is not enabled";
        return false;
#else
        outError.clear();
        outAcquired = false;
        if (leaseGeneration == 0) {
            outError = "retained frame scratch lease generation is invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        if (resources.retainedScratchLeaseGeneration == leaseGeneration) {
            outAcquired = true;
            return true;
        }
        if (resources.retainedScratchLeaseGeneration != 0) {
            return true;
        }
        if (!wait_for_frame_use_events_locked(
                resources,
                cudaStreamOpaque,
                "retained frame scratch lease",
                outError)) {
            return false;
        }
        resources.retainedScratchLeaseGeneration = leaseGeneration;
        outAcquired = true;
        return true;
#endif
    }

    bool release_retained_frame_scratch_lease(
        Resources& resources,
        std::uint64_t leaseGeneration,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)leaseGeneration;
        outError = "CUDA is not enabled";
        return false;
#else
        outError.clear();
        if (leaseGeneration == 0) {
            outError = "retained frame scratch lease generation is invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (resources.retainedScratchLeaseGeneration == 0) {
            return true;
        }
        if (resources.retainedScratchLeaseGeneration != leaseGeneration) {
            outError = "retained frame scratch lease generation mismatch";
            return false;
        }
        resources.retainedScratchLeaseGeneration = 0;
        return true;
#endif
    }

    bool release_retained_spatial_dir_build_scratch_stage(
        Resources& resources,
        std::uint64_t leaseGeneration,
        void* cudaStreamOpaque,
        SpatialDirBuildScratchReleaseStats& outStats,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)leaseGeneration;
        (void)cudaStreamOpaque;
        outStats = SpatialDirBuildScratchReleaseStats{};
        outError = "CUDA is not enabled";
        return false;
#else
        outStats = SpatialDirBuildScratchReleaseStats{};
        outError.clear();
        if (leaseGeneration == 0) {
            outError = "retained spatial DIR build scratch lease generation is invalid";
            return false;
        }

        bool syncBeforeReap = false;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }
            if (resources.retainedScratchLeaseGeneration != leaseGeneration) {
                outError = "retained spatial DIR build scratch lease generation mismatch";
                return false;
            }

            outStats.pendingScratchBytesBefore = resources.retireScratchBytes;
            if (!retire_spatial_dir_build_scratch_locked(
                    resources,
                    resources.spatialDirScratch,
                    cudaStreamOpaque,
                    outStats,
                    outError)) {
                return false;
            }

            const std::size_t sharedTmpBytesBefore =
                resources.sharedTmpPlane ? resources.sharedTmpCapacityElements * sizeof(float) : 0;
            if (!retire_unreferenced_shared_tmp_plane_locked(
                    resources,
                    cudaStreamOpaque,
                    "retained spatial DIR build shared tmp plane",
                    outError)) {
                return false;
            }
            if (sharedTmpBytesBefore > 0 && !resources.sharedTmpPlane) {
                outStats.sharedTmpRetiredBytes = sharedTmpBytesBefore;
            }

            syncBeforeReap =
                outStats.pendingScratchBytesBefore > 0 ||
                outStats.rawCorrectionRetiredBytes > 0 ||
                outStats.filterTempRetiredBytes > 0 ||
                outStats.sharedTmpRetiredBytes > 0;
        }

        if (!syncBeforeReap) {
            return true;
        }

        const cudaStream_t stream = cudaStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                                        : nullptr;
        const cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            outError = std::string("cudaStreamSynchronize(retained spatial DIR build scratch) failed: ") +
                       (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
            return false;
        }
        return reap_retired_allocations(resources, outStats.reclaimedBytes, outError);
#endif
    }

    bool release_retained_spatial_dir_scratch_stage(
        Resources& resources,
        std::uint64_t leaseGeneration,
        void* cudaStreamOpaque,
        SpatialDirStageReleaseStats& outStats,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)leaseGeneration;
        (void)cudaStreamOpaque;
        outStats = SpatialDirStageReleaseStats{};
        outError = "CUDA is not enabled";
        return false;
#else
        outStats = SpatialDirStageReleaseStats{};
        outError.clear();
        if (leaseGeneration == 0) {
            outError = "retained spatial DIR stage lease generation is invalid";
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }
            if (resources.retainedScratchLeaseGeneration != leaseGeneration) {
                outError = "retained spatial DIR stage lease generation mismatch";
                return false;
            }
            outStats.retiredBytes = spatial_dir_stage_live_bytes_locked(resources.spatialDirScratch);
            if (outStats.retiredBytes == 0) {
                return true;
            }
            if (!retire_spatial_dir_scratch_locked(
                    resources,
                    resources.spatialDirScratch,
                    cudaStreamOpaque,
                    "retained spatial DIR stage",
                    outError)) {
                return false;
            }
        }

        const cudaStream_t stream = cudaStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                                        : nullptr;
        const cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            outError = std::string("cudaStreamSynchronize(retained spatial DIR stage) failed: ") +
                       (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
            return false;
        }
        return reap_retired_allocations(resources, outStats.reclaimedBytes, outError);
#endif
    }

    bool release_retained_spatial_dir_cached_log_raw_stage(
        Resources& resources,
        std::uint64_t leaseGeneration,
        void* cudaStreamOpaque,
        SpatialDirCachedLogRawReleaseStats& outStats,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)leaseGeneration;
        (void)cudaStreamOpaque;
        outStats = SpatialDirCachedLogRawReleaseStats{};
        outError = "CUDA is not enabled";
        return false;
#else
        outStats = SpatialDirCachedLogRawReleaseStats{};
        outError.clear();
        if (leaseGeneration == 0) {
            outError = "retained spatial DIR cached log raw lease generation is invalid";
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }
            if (resources.retainedScratchLeaseGeneration != leaseGeneration) {
                outError = "retained spatial DIR cached log raw lease generation mismatch";
                return false;
            }

            outStats.pendingScratchBytesBefore = resources.retireScratchBytes;
            if (!retire_spatial_dir_cached_log_raw_locked(
                    resources,
                    resources.spatialDirScratch,
                    cudaStreamOpaque,
                    outStats,
                    outError)) {
                return false;
            }
        }

        if (outStats.pendingScratchBytesBefore == 0 &&
            outStats.cachedLogRawRetiredBytes == 0) {
            return true;
        }

        const cudaStream_t stream = cudaStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                                        : nullptr;
        const cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            outError = std::string("cudaStreamSynchronize(retained spatial DIR cached log raw) failed: ") +
                       (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
            return false;
        }
        return reap_retired_allocations(resources, outStats.reclaimedBytes, outError);
#endif
    }

    bool ensure_retained_spatial_dir_cached_log_raw_stage(
        Resources& resources,
        std::uint64_t leaseGeneration,
        const ResourceManager::ScratchRequestDescriptor& scratchRequest,
        void* cudaStreamOpaque,
        SpatialDirCachedLogRawStageStats& outStats,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)leaseGeneration;
        (void)scratchRequest;
        (void)cudaStreamOpaque;
        outStats = SpatialDirCachedLogRawStageStats{};
        outError = "CUDA is not enabled";
        return false;
#else
        outStats = SpatialDirCachedLogRawStageStats{};
        outError.clear();
        if (leaseGeneration == 0) {
            outError = "retained spatial DIR cached log raw stage lease generation is invalid";
            return false;
        }
        if (!ResourceManager::scratch_request_descriptor_is_valid(scratchRequest)) {
            outError = "retained spatial DIR cached log raw stage request invalid";
            return false;
        }
        if (scratchRequest.requestedWidth <= 0 || scratchRequest.requestedHeight <= 0) {
            outError = "retained spatial DIR cached log raw stage dimensions invalid";
            return false;
        }

        const std::size_t widthElements = static_cast<std::size_t>(scratchRequest.requestedWidth);
        const std::size_t heightElements = static_cast<std::size_t>(scratchRequest.requestedHeight);
        if (widthElements > (std::numeric_limits<std::size_t>::max() / heightElements)) {
            outError = "retained spatial DIR cached log raw stage element count overflow";
            return false;
        }
        const std::size_t requiredElements = widthElements * heightElements;
        if (requiredElements == 0 ||
            requiredElements > (std::numeric_limits<std::size_t>::max() / sizeof(float))) {
            outError = "retained spatial DIR cached log raw stage byte count overflow";
            return false;
        }
        const std::size_t bytes = requiredElements * sizeof(float);

        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }
        if (resources.retainedScratchLeaseGeneration != leaseGeneration) {
            outError = "retained spatial DIR cached log raw stage lease generation mismatch";
            return false;
        }

        Resources::DeviceSpatialDirScratch& scratch = resources.spatialDirScratch;
        outStats.pendingScratchBytesBefore = resources.retireScratchBytes;
        if (scratch.width != scratchRequest.requestedWidth ||
            scratch.height != scratchRequest.requestedHeight ||
            scratch.capacityElements < requiredElements ||
            !scratch.filteredCorrectionY || !scratch.filteredCorrectionM ||
            !scratch.filteredCorrectionC) {
            outError = "retained spatial DIR cached log raw stage missing filtered correction residency";
            return false;
        }
        if (scratch.logRawB && scratch.logRawG && scratch.logRawR) {
            return true;
        }

        auto ensure_plane = [&](float*& ptr, const char* label) {
            if (ptr) {
                return true;
            }
            if (!allocate_scratch_device_ptr_locked(
                    resources,
                    ptr,
                    bytes,
                    cudaStreamOpaque,
                    label,
                    outError)) {
                return false;
            }
            outStats.cachedLogRawAllocatedBytes += bytes;
            return true;
        };

        if (!ensure_plane(scratch.logRawB, "retained spatial DIR final logRawB") ||
            !ensure_plane(scratch.logRawG, "retained spatial DIR final logRawG") ||
            !ensure_plane(scratch.logRawR, "retained spatial DIR final logRawR")) {
            free_tracked_device_ptr_locked(resources, scratch.logRawB, cudaStreamOpaque);
            free_tracked_device_ptr_locked(resources, scratch.logRawG, cudaStreamOpaque);
            free_tracked_device_ptr_locked(resources, scratch.logRawR, cudaStreamOpaque);
            return false;
        }
        return true;
#endif
    }

    bool reclaim_large_scratch_transition(
        Resources& resources,
        const ResourceManager::ScratchRequestDescriptor& scratchRequest,
        void* cudaStreamOpaque,
        LargeScratchTransitionReclaimStats& outStats,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)scratchRequest;
        (void)cudaStreamOpaque;
        outStats = LargeScratchTransitionReclaimStats{};
        outError = "CUDA is not enabled";
        return false;
#else
        outStats = LargeScratchTransitionReclaimStats{};
        outError.clear();
        if (!ResourceManager::scratch_request_descriptor_is_valid(scratchRequest)) {
            outError = "large scratch transition request descriptor is invalid";
            return false;
        }
        const std::size_t requestedWidth = static_cast<std::size_t>(scratchRequest.requestedWidth);
        const std::size_t requestedHeight = static_cast<std::size_t>(scratchRequest.requestedHeight);
        if (requestedHeight != 0 &&
            requestedWidth > (std::numeric_limits<std::size_t>::max() / requestedHeight)) {
            outError = "large scratch transition element count overflow";
            return false;
        }
        const std::size_t requiredElements = requestedWidth * requestedHeight;

        bool syncBeforeReap = false;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }
            outStats.pendingScratchBytesBefore = resources.retireScratchBytes;
            if (resources.retainedScratchLeaseGeneration != 0) {
                return true;
            }

            Resources::DeviceOpticsScratch& optics = resources.scannerScratch;
            if (optics_any_live_locked(optics)) {
                const bool baseLive = optics.rgbR && optics.rgbG && optics.rgbB;
                const bool capacityMatch = optics.capacityElements >= requiredElements;
                const bool retireOptics =
                    !scratchRequest.needOptics ||
                    !capacityMatch ||
                    (scratchRequest.aliasScannerRgbFromSpatialDirFiltered ? baseLive : !baseLive);
                if (retireOptics) {
                    outStats.opticsRetiredBytes = optics_stage_live_bytes_locked(optics);
                    if (!retire_optics_scratch_locked(
                            resources,
                            optics,
                            cudaStreamOpaque,
                            "large scratch transition optics scratch",
                            outError)) {
                        return false;
                    }
                }
            }

            Resources::DeviceSpatialDirScratch& spatialDir = resources.spatialDirScratch;
            const std::size_t spatialDirBytesBefore =
                spatial_dir_stage_live_bytes_locked(spatialDir);
            if (spatialDirBytesBefore > 0) {
                const bool capacityMatch = spatialDir.capacityElements >= requiredElements;
                const bool rolesMatch =
                    scratchRequest.needSpatialDir &&
                    spatial_dir_scratch_has_required_roles_locked(
                        spatialDir,
                        scratchRequest.spatialDirPlaneRoles);
                const bool retireSpatialDir =
                    !scratchRequest.needSpatialDir || !capacityMatch || !rolesMatch;
                if (retireSpatialDir) {
                    outStats.spatialDirRetiredBytes = spatialDirBytesBefore;
                    if (!retire_spatial_dir_scratch_locked(
                            resources,
                            spatialDir,
                            cudaStreamOpaque,
                            "large scratch transition spatial DIR scratch",
                            outError)) {
                        return false;
                    }
                }
            }

            const std::size_t sharedTmpBytesBefore =
                resources.sharedTmpPlane ? resources.sharedTmpCapacityElements * sizeof(float) : 0;
            if (!retire_orphaned_shared_tmp_plane_locked(
                    resources,
                    cudaStreamOpaque,
                    "large scratch transition shared tmp plane",
                    outError)) {
                return false;
            }
            if (sharedTmpBytesBefore > 0 && !resources.sharedTmpPlane) {
                outStats.sharedTmpRetiredBytes = sharedTmpBytesBefore;
            }

            syncBeforeReap =
                outStats.pendingScratchBytesBefore > 0 ||
                outStats.opticsRetiredBytes > 0 ||
                outStats.spatialDirRetiredBytes > 0 ||
                outStats.sharedTmpRetiredBytes > 0;
        }

        if (!syncBeforeReap) {
            return true;
        }

        const cudaStream_t stream = cudaStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                                        : nullptr;
        const cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            outError = std::string("cudaStreamSynchronize(large scratch transition) failed: ") +
                       (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
            return false;
        }
        return reap_retired_allocations(resources, outStats.reclaimedBytes, outError);
#endif
    }

    bool reclaim_retained_optics_for_admission_retry(
        Resources& resources,
        std::uint64_t leaseGeneration,
        const ResourceManager::ScratchRequestDescriptor& scratchRequest,
        void* cudaStreamOpaque,
        AdmissionRetryOpticsReclaimStats& outStats,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)leaseGeneration;
        (void)scratchRequest;
        (void)cudaStreamOpaque;
        outStats = AdmissionRetryOpticsReclaimStats{};
        outError = "CUDA is not enabled";
        return false;
#else
        outStats = AdmissionRetryOpticsReclaimStats{};
        outError.clear();
        if (leaseGeneration == 0) {
            outError = "admission retry optics reclaim lease generation is invalid";
            return false;
        }
        if (!ResourceManager::scratch_request_descriptor_is_valid(scratchRequest)) {
            outError = "admission retry optics reclaim request descriptor is invalid";
            return false;
        }
        if (!scratchRequest.needSpatialDir) {
            outError = "admission retry optics reclaim requires spatial DIR scratch";
            return false;
        }

        bool syncBeforeReap = false;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }
            outStats.pendingScratchBytesBefore = resources.retireScratchBytes;
            if (resources.retainedScratchLeaseGeneration != 0 &&
                resources.retainedScratchLeaseGeneration != leaseGeneration) {
                return true;
            }

            Resources::DeviceOpticsScratch& optics = resources.scannerScratch;
            if (optics_any_live_locked(optics)) {
                outStats.opticsRetiredBytes = optics_stage_live_bytes_locked(optics);
                if (!retire_optics_scratch_locked(
                        resources,
                        optics,
                        cudaStreamOpaque,
                        "admission retry optics scratch",
                        outError)) {
                    return false;
                }
            }

            const std::size_t sharedTmpBytesBefore =
                resources.sharedTmpPlane ? resources.sharedTmpCapacityElements * sizeof(float) : 0;
            if (!retire_orphaned_shared_tmp_plane_locked(
                    resources,
                    cudaStreamOpaque,
                    "admission retry shared tmp plane",
                    outError)) {
                return false;
            }
            if (sharedTmpBytesBefore > 0 && !resources.sharedTmpPlane) {
                outStats.sharedTmpRetiredBytes = sharedTmpBytesBefore;
            }

            syncBeforeReap =
                outStats.pendingScratchBytesBefore > 0 ||
                outStats.opticsRetiredBytes > 0 ||
                outStats.sharedTmpRetiredBytes > 0;
        }

        if (!syncBeforeReap) {
            return true;
        }

        const cudaStream_t stream = cudaStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                                        : nullptr;
        const cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            outError = std::string("cudaStreamSynchronize(admission retry optics reclaim) failed: ") +
                       (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
            return false;
        }
        return reap_retired_allocations(resources, outStats.reclaimedBytes, outError);
#endif
    }

    bool shed_retained_scratch_after_frame(
        Resources& resources,
        void* cudaStreamOpaque,
        PostFrameScratchShedStats& outStats,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)cudaStreamOpaque;
        outStats = PostFrameScratchShedStats{};
        outError = "CUDA is not enabled";
        return false;
#else
        outStats = PostFrameScratchShedStats{};
        outError.clear();

        bool syncBeforeReap = false;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }
            outStats.pendingScratchBytesBefore = resources.retireScratchBytes;
            if (resources.retainedScratchLeaseGeneration != 0) {
                return true;
            }

            Resources::DeviceOpticsScratch& optics = resources.scannerScratch;
            if (optics_any_live_locked(optics)) {
                outStats.opticsRetiredBytes = optics_stage_live_bytes_locked(optics);
                if (!retire_optics_scratch_locked(
                        resources,
                        optics,
                        cudaStreamOpaque,
                        "post-frame optics scratch",
                        outError)) {
                    return false;
                }
            }

            Resources::DeviceSpatialDirScratch& spatialDir = resources.spatialDirScratch;
            outStats.spatialDirRetiredBytes = spatial_dir_stage_live_bytes_locked(spatialDir);
            if (outStats.spatialDirRetiredBytes > 0 &&
                !retire_spatial_dir_scratch_locked(
                    resources,
                    spatialDir,
                    cudaStreamOpaque,
                    "post-frame spatial DIR scratch",
                    outError)) {
                return false;
            }

            const std::size_t sharedTmpBytesBefore =
                resources.sharedTmpPlane ? resources.sharedTmpCapacityElements * sizeof(float) : 0;
            if (!retire_orphaned_shared_tmp_plane_locked(
                    resources,
                    cudaStreamOpaque,
                    "post-frame shared tmp plane",
                    outError)) {
                return false;
            }
            if (sharedTmpBytesBefore > 0 && !resources.sharedTmpPlane) {
                outStats.sharedTmpRetiredBytes = sharedTmpBytesBefore;
            }

            syncBeforeReap =
                outStats.pendingScratchBytesBefore > 0 ||
                outStats.opticsRetiredBytes > 0 ||
                outStats.spatialDirRetiredBytes > 0 ||
                outStats.sharedTmpRetiredBytes > 0;
        }

        if (!syncBeforeReap) {
            return true;
        }

        const cudaStream_t stream = cudaStreamOpaque
                                        ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
                                        : nullptr;
        const cudaError_t syncErr = cudaStreamSynchronize(stream);
        if (syncErr != cudaSuccess) {
            outError = std::string("cudaStreamSynchronize(post-frame scratch shed) failed: ") +
                       (cudaGetErrorString(syncErr) ? cudaGetErrorString(syncErr) : "(unknown)");
            return false;
        }
        return reap_retired_allocations(resources, outStats.reclaimedBytes, outError);
#endif
    }

    bool retire_frame_scratch_allocation(
        Resources& resources,
        void* ptr,
        std::size_t bytes,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)ptr;
        (void)bytes;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (!ptr) {
            return true;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }
        return retire_ptr_locked(
            resources,
            ptr,
            bytes,
            Resources::RetireKind::DeviceFree,
            cudaStreamOpaque,
            label ? label : "frame scratch",
            outError,
            true);
#endif
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool ensure_shared_tmp_plane_locked(Resources& resources, int width, int height, void* cudaStreamOpaque, const char* label, std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)width;
        (void)height;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (width <= 0 || height <= 0) {
            outError = std::string(label ? label : "shared tmp") + " dimensions invalid";
            return false;
        }

        const size_t requiredElements = static_cast<size_t>(width) * static_cast<size_t>(height);
        if (resources.sharedTmpPlane &&
            resources.sharedTmpCapacityElements >= requiredElements) {
            resources.sharedTmpWidth = width;
            resources.sharedTmpHeight = height;
            return true;
        }

        if (resources.sharedTmpPlane) {
            const size_t oldBytes = resources.sharedTmpCapacityElements * sizeof(float);
            if (!retire_ptr_locked(resources, resources.sharedTmpPlane, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, label ? label : "shared tmp", outError, true)) {
                return false;
            }
            resources.sharedTmpPlane = nullptr;
        }
        resources.sharedTmpWidth = 0;
        resources.sharedTmpHeight = 0;
        resources.sharedTmpCapacityElements = 0;

        const size_t bytes = requiredElements * sizeof(float);
        if (!allocate_scratch_device_ptr_locked(
                resources,
                resources.sharedTmpPlane,
                bytes,
                cudaStreamOpaque,
                label ? label : "shared tmp",
                outError)) {
            resources.sharedTmpPlane = nullptr;
            return false;
        }

        resources.sharedTmpWidth = width;
        resources.sharedTmpHeight = height;
        resources.sharedTmpCapacityElements = requiredElements;
        return true;
#endif
    }
#endif

    bool ensure_optics_scratch(
        Resources& resources,
        const ResourceManager::ScratchRequestDescriptor& request,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)request;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        const int width = request.requestedWidth;
        const int height = request.requestedHeight;
        const bool needBlurredScratch = request.needBlurred;
        const bool aliasScannerRgbFromSpatialDirFiltered =
            request.aliasScannerRgbFromSpatialDirFiltered;
        const bool needAuxScratch = request.needAux;
        const bool needSharedTmpScratch = request.needSharedTmp;
        const bool needGrainFrameUniforms = request.needGrainFrameUniforms;
        const bool needGrainLayerWorkScratch = request.needGrainLayerWork;
        const bool needGrainSharedScratch = request.needGrainShared;
        const bool needGateMask = request.needGateMask;
        if (width <= 0 || height <= 0) {
            outError = "optics scratch dimensions invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        const size_t requiredElements = static_cast<size_t>(width) * static_cast<size_t>(height);
        const bool dimsMatch = (resources.scannerScratch.width == width && resources.scannerScratch.height == height);
        const bool capacityMatch = resources.scannerScratch.capacityElements >= requiredElements;
        const bool needScannerRgbScratch = !aliasScannerRgbFromSpatialDirFiltered;
        const bool haveBase =
            !needScannerRgbScratch ||
            (resources.scannerScratch.rgbR && resources.scannerScratch.rgbG &&
             resources.scannerScratch.rgbB);

        if (!capacityMatch || !haveBase) {
            if (resources.scannerScratch.rgbR || resources.scannerScratch.rgbG || resources.scannerScratch.rgbB ||
                resources.scannerScratch.blurred || resources.scannerScratch.aux || resources.scannerScratch.grainTmp ||
                resources.scannerScratch.grainTmpShared ||
                resources.scannerScratch.grainFrameUniforms ||
                resources.scannerScratch.gateMask) {
                if (!retire_optics_scratch_locked(resources, resources.scannerScratch, cudaStreamOpaque, "optics scratch", outError)) {
                    return false;
                }
            } else {
                free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
            }

            const size_t bytes = requiredElements * sizeof(float);
            if (needScannerRgbScratch) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.rgbR,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.rgbR",
                        outError)) {
                    free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                    return false;
                }
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.rgbG,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.rgbG",
                        outError)) {
                    free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                    return false;
                }
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.rgbB,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.rgbB",
                        outError)) {
                    free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                    return false;
                }
            }

            resources.scannerScratch.capacityElements = requiredElements;
        }
        resources.scannerScratch.width = width;
        resources.scannerScratch.height = height;
        if (!dimsMatch) {
            resources.scannerScratch.gateMaskHash = 0;
        }

        if (needSharedTmpScratch) {
            if (!ensure_shared_tmp_plane_locked(resources, width, height, cudaStreamOpaque, "shared tmp plane", outError)) {
                free_optics_scratch(resources, resources.scannerScratch, cudaStreamOpaque);
                return false;
            }
        }
        resources.scannerScratch.tmp = needSharedTmpScratch ? resources.sharedTmpPlane : nullptr;
        const size_t planeBytes = resources.scannerScratch.capacityElements * sizeof(float);

        if (!needScannerRgbScratch) {
            auto retire_rgb_plane = [&](float*& ptr, const char* label) {
                if (!ptr) {
                    return true;
                }
                if (!retire_ptr_locked(
                        resources,
                        ptr,
                        planeBytes,
                        Resources::RetireKind::DeviceFree,
                        cudaStreamOpaque,
                        label,
                        outError,
                        true)) {
                    return false;
                }
                ptr = nullptr;
                return true;
            };
            if (!retire_rgb_plane(resources.scannerScratch.rgbR, "scannerScratch.rgbR alias release") ||
                !retire_rgb_plane(resources.scannerScratch.rgbG, "scannerScratch.rgbG alias release") ||
                !retire_rgb_plane(resources.scannerScratch.rgbB, "scannerScratch.rgbB alias release")) {
                return false;
            }
        }

        if (needBlurredScratch) {
            if (!resources.scannerScratch.blurred) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.blurred,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.blurred",
                        outError)) {
                    return false;
                }
            }
        } else {
            if (resources.scannerScratch.blurred) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.blurred, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "unsharp scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.blurred = nullptr;
            }
        }

        if (needAuxScratch) {
            if (!resources.scannerScratch.aux) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.aux,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.aux",
                        outError)) {
                    return false;
                }
            }
        } else {
            if (resources.scannerScratch.aux) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.aux, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.aux = nullptr;
            }
        }

        if (needGrainFrameUniforms) {
            if (!resources.scannerScratch.grainFrameUniforms) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainFrameUniforms,
                        sizeof(GrainFrameUniforms),
                        cudaStreamOpaque,
                        "scannerScratch.grainFrameUniforms",
                        outError)) {
                    return false;
                }
            }
        } else if (resources.scannerScratch.grainFrameUniforms) {
            if (!retire_ptr_locked(
                    resources,
                    resources.scannerScratch.grainFrameUniforms,
                    sizeof(GrainFrameUniforms),
                    Resources::RetireKind::DeviceFree,
                    cudaStreamOpaque,
                    "grain frame uniforms",
                    outError,
                    true)) {
                return false;
            }
            resources.scannerScratch.grainFrameUniforms = nullptr;
        }

        if (needGrainLayerWorkScratch) {
            if (!resources.scannerScratch.grainTmp) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainTmp,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.grainTmp",
                        outError)) {
                    return false;
                }
            }
        } else {
            if (resources.scannerScratch.grainTmp) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.grainTmp, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain mix scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.grainTmp = nullptr;
            }
        }

        if (needGrainSharedScratch) {
            if (!resources.scannerScratch.grainTmpShared) {
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.grainTmpShared,
                        planeBytes,
                        cudaStreamOpaque,
                        "scannerScratch.grainTmpShared",
                        outError)) {
                    return false;
                }
            }
        } else {
            if (resources.scannerScratch.grainTmpShared) {
                if (!retire_ptr_locked(resources, resources.scannerScratch.grainTmpShared, planeBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "grain shared scratch", outError, true)) {
                    return false;
                }
                resources.scannerScratch.grainTmpShared = nullptr;
            }
        }

        const int gateWidth = (width + 1) / 2;
        const int gateHeight = (height + 1) / 2;
        if (needGateMask) {
            const size_t requiredGateElements = static_cast<size_t>(gateWidth) * static_cast<size_t>(gateHeight);
            const bool gateDimsMatch = (resources.scannerScratch.gateWidth == gateWidth &&
                                        resources.scannerScratch.gateHeight == gateHeight);
            const bool gateCapacityMatch =
                resources.scannerScratch.gateMaskCapacityElements >= requiredGateElements;
            if (!resources.scannerScratch.gateMask || !gateCapacityMatch) {
                if (resources.scannerScratch.gateMask) {
                    const size_t oldBytes =
                        resources.scannerScratch.gateMaskCapacityElements * sizeof(float);
                    if (!retire_ptr_locked(resources, resources.scannerScratch.gateMask, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "gate defect mask", outError, true)) {
                        return false;
                    }
                    resources.scannerScratch.gateMask = nullptr;
                }
                const size_t bytes = requiredGateElements * sizeof(float);
                if (!allocate_scratch_device_ptr_locked(
                        resources,
                        resources.scannerScratch.gateMask,
                        bytes,
                        cudaStreamOpaque,
                        "scannerScratch.gateMask",
                        outError)) {
                    return false;
                }
                resources.scannerScratch.gateMaskCapacityElements = requiredGateElements;
            }
            if (!gateDimsMatch) {
                resources.scannerScratch.gateMaskHash = 0;
            }
            resources.scannerScratch.gateWidth = gateWidth;
            resources.scannerScratch.gateHeight = gateHeight;
        } else if (resources.scannerScratch.gateMask) {
            const size_t oldBytes =
                resources.scannerScratch.gateMaskCapacityElements * sizeof(float);
            if (!retire_ptr_locked(resources, resources.scannerScratch.gateMask, oldBytes, Resources::RetireKind::DeviceFree, cudaStreamOpaque, "gate defect mask", outError, true)) {
                return false;
            }
            resources.scannerScratch.gateMask = nullptr;
            resources.scannerScratch.gateWidth = 0;
            resources.scannerScratch.gateHeight = 0;
            resources.scannerScratch.gateMaskCapacityElements = 0;
            resources.scannerScratch.gateMaskHash = 0;
        }

        return true;
#endif
    }

    bool ensure_spatial_dir_scratch(
        Resources& resources,
        const ResourceManager::ScratchRequestDescriptor& request,
        void* cudaStreamOpaque,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)request;
        (void)cudaStreamOpaque;
        outError = "CUDA is not enabled";
        return false;
#else
        const int width = request.requestedWidth;
        const int height = request.requestedHeight;
        const Spektrafilm::DirScratchTier scratchTier = request.spatialDirScratchTier;
        const Spektrafilm::DirScratchPlaneRoles& planeRoles = request.spatialDirPlaneRoles;
        if (width <= 0 || height <= 0) {
            outError = "spatial DIR scratch dimensions invalid";
            return false;
        }
        if (!Spektrafilm::spatial_dir_roles_match_tier(scratchTier, planeRoles)) {
            outError = "spatial DIR scratch request tier/roles invalid";
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        Resources::DeviceSpatialDirScratch& scratch = resources.spatialDirScratch;
        const size_t requiredElements = static_cast<size_t>(width) * static_cast<size_t>(height);
        const bool dimsMatch = (scratch.width == width && scratch.height == height);
        const bool capacityMatch = scratch.capacityElements >= requiredElements;
        const bool needSharedTmp = spatial_dir_request_needs_shared_tmp_plane(planeRoles);
        const bool haveRequiredRoles =
            spatial_dir_scratch_has_required_roles_locked(scratch, planeRoles);
        if (dimsMatch && capacityMatch && haveRequiredRoles) {
            if (needSharedTmp) {
                if (!ensure_shared_tmp_plane_locked(
                        resources,
                        width,
                        height,
                        cudaStreamOpaque,
                        "shared tmp plane",
                        outError)) {
                    return false;
                }
                scratch.filterTemp = resources.sharedTmpPlane;
            } else {
                scratch.filterTemp = nullptr;
            }
            return true;
        }

        const bool haveRetainedFinalRoles =
            spatial_dir_scratch_has_retained_final_roles_locked(scratch, planeRoles);
        if (dimsMatch && capacityMatch && haveRetainedFinalRoles) {
            const size_t bytes = requiredElements * sizeof(float);
            auto ensure_plane = [&](float*& ptr, const char* label) {
                if (ptr) {
                    return true;
                }
                return allocate_scratch_device_ptr_locked(
                    resources,
                    ptr,
                    bytes,
                    cudaStreamOpaque,
                    label,
                    outError);
            };
            if (!ensure_plane(scratch.rawCorrectionY, "spatial DIR rawCorrectionY") ||
                (planeRoles.rawCorrectionPlanes == 3 &&
                 (!ensure_plane(scratch.rawCorrectionM, "spatial DIR rawCorrectionM") ||
                  !ensure_plane(scratch.rawCorrectionC, "spatial DIR rawCorrectionC"))) ||
                (planeRoles.filterTempPlanes == 2 &&
                 !ensure_plane(scratch.filterTempM, "spatial DIR filterTempM")) ||
                (planeRoles.filterTempPlanes == 3 &&
                 (!ensure_plane(scratch.filterTempM, "spatial DIR filterTempM") ||
                  !ensure_plane(scratch.filterTempC, "spatial DIR filterTempC"))) ||
                (planeRoles.cachedLogRawPlanes >= 1 &&
                 !ensure_plane(scratch.logRawB, "spatial DIR logRawB")) ||
                (planeRoles.cachedLogRawPlanes >= 2 &&
                 !ensure_plane(scratch.logRawG, "spatial DIR logRawG")) ||
                (planeRoles.cachedLogRawPlanes >= 3 &&
                 !ensure_plane(scratch.logRawR, "spatial DIR logRawR"))) {
                free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
                return false;
            }
            if (needSharedTmp) {
                if (!ensure_shared_tmp_plane_locked(
                        resources,
                        width,
                        height,
                        cudaStreamOpaque,
                        "shared tmp plane",
                        outError)) {
                    free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
                    return false;
                }
                scratch.filterTemp = resources.sharedTmpPlane;
            } else {
                scratch.filterTemp = nullptr;
            }
            scratch.width = width;
            scratch.height = height;
            scratch.capacityElements = requiredElements;
            return true;
        }

        if (scratch.rawCorrectionY || scratch.rawCorrectionM || scratch.rawCorrectionC ||
            scratch.filteredCorrectionY || scratch.filteredCorrectionM ||
            scratch.filteredCorrectionC || scratch.filterTempM || scratch.filterTempC ||
            scratch.logRawB || scratch.logRawG || scratch.logRawR) {
            if (!capacityMatch || !haveRequiredRoles) {
                if (!retire_spatial_dir_scratch_locked(resources, scratch, cudaStreamOpaque, "spatial DIR scratch", outError)) {
                    return false;
                }
            }
        }

        if (capacityMatch && haveRequiredRoles) {
            scratch.width = width;
            scratch.height = height;
            if (needSharedTmp) {
                if (!ensure_shared_tmp_plane_locked(
                        resources,
                        width,
                        height,
                        cudaStreamOpaque,
                        "shared tmp plane",
                        outError)) {
                    return false;
                }
                scratch.filterTemp = resources.sharedTmpPlane;
            } else {
                scratch.filterTemp = nullptr;
            }
            return true;
        }

        const size_t bytes = requiredElements * sizeof(float);
        if (!allocate_scratch_device_ptr_locked(
                resources,
                scratch.rawCorrectionY,
                bytes,
                cudaStreamOpaque,
                "spatial DIR rawCorrectionY",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (!allocate_scratch_device_ptr_locked(
                resources,
                scratch.filteredCorrectionY,
                bytes,
                cudaStreamOpaque,
                "spatial DIR filteredCorrectionY",
                outError) ||
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.filteredCorrectionM,
                bytes,
                cudaStreamOpaque,
                "spatial DIR filteredCorrectionM",
                outError) ||
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.filteredCorrectionC,
                bytes,
                cudaStreamOpaque,
                "spatial DIR filteredCorrectionC",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (planeRoles.rawCorrectionPlanes == 3) {
            if (!allocate_scratch_device_ptr_locked(
                    resources,
                    scratch.rawCorrectionM,
                    bytes,
                    cudaStreamOpaque,
                    "spatial DIR rawCorrectionM",
                    outError) ||
                !allocate_scratch_device_ptr_locked(
                    resources,
                    scratch.rawCorrectionC,
                    bytes,
                    cudaStreamOpaque,
                    "spatial DIR rawCorrectionC",
                    outError)) {
                free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
                return false;
            }
        }
        if (planeRoles.filterTempPlanes >= 2 &&
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.filterTempM,
                bytes,
                cudaStreamOpaque,
                "spatial DIR filterTempM",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (planeRoles.filterTempPlanes == 3 &&
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.filterTempC,
                bytes,
                cudaStreamOpaque,
                "spatial DIR filterTempC",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (planeRoles.cachedLogRawPlanes >= 1 &&
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.logRawB,
                bytes,
                cudaStreamOpaque,
                "spatial DIR logRawB",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (planeRoles.cachedLogRawPlanes >= 2 &&
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.logRawG,
                bytes,
                cudaStreamOpaque,
                "spatial DIR logRawG",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (planeRoles.cachedLogRawPlanes >= 3 &&
            !allocate_scratch_device_ptr_locked(
                resources,
                scratch.logRawR,
                bytes,
                cudaStreamOpaque,
                "spatial DIR logRawR",
                outError)) {
            free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
            return false;
        }
        if (needSharedTmp) {
            if (!ensure_shared_tmp_plane_locked(
                    resources,
                    width,
                    height,
                    cudaStreamOpaque,
                    "shared tmp plane",
                    outError)) {
                free_spatial_dir_scratch(resources, scratch, cudaStreamOpaque);
                return false;
            }
            scratch.filterTemp = resources.sharedTmpPlane;
        } else {
            scratch.filterTemp = nullptr;
        }

        scratch.width = width;
        scratch.height = height;
        scratch.capacityElements = requiredElements;
        return true;
#endif
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    static bool retire_or_clear_gaussian_kernel_locked(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)kernel;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        if (kernel.weights) {
            if (kernel.sharedKernelId == 0) {
                const std::size_t bytes =
                    static_cast<std::size_t>(std::max(0, kernel.capacity)) * sizeof(float);
                if (!retire_ptr_locked(
                        resources,
                        kernel.weights,
                        bytes,
                        Resources::RetireKind::DeviceFree,
                        cudaStreamOpaque,
                        label,
                        outError)) {
                    return false;
                }
            }
            kernel.weights = nullptr;
        }
        kernel.radius = 0;
        kernel.sigma = 0.0f;
        kernel.capacity = 0;
        kernel.sharedKernelId = 0;
        return true;
#endif
    }
#endif

    static bool ensure_shared_gaussian_kernel(
        Resources& resources,
        Resources::DeviceGaussianKernel& kernel,
        SharedGaussianSpec spec,
        void* cudaStreamOpaque,
        const char* label,
        std::string& outError) {
#if !defined(JUICER_ENABLE_CUDA) || defined(__APPLE__)
        (void)resources;
        (void)kernel;
        (void)spec;
        (void)cudaStreamOpaque;
        (void)label;
        outError = "CUDA is not enabled";
        return false;
#else
        const float sigma = spec.sigma;
        const int radius = spec.radius;
        const bool sigmaOk = std::isfinite(sigma) && sigma > 0.0f;
        if (!sigmaOk || radius <= 0) {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }
            return retire_or_clear_gaussian_kernel_locked(
                resources,
                kernel,
                cudaStreamOpaque,
                label,
                outError);
        }

        SharedGaussianKey key{};
        std::uint64_t kernelId = 0;
        {
            std::lock_guard<std::mutex> lock(resources.m);
            reap_retire_queue_locked(resources);
            if (!validate_resource_owner_locked(resources, outError, true)) {
                return false;
            }
            key = make_shared_gaussian_key(resources, spec);
            kernelId = make_shared_gaussian_id(key);
            const bool same =
                kernel.weights &&
                kernel.sharedKernelId == kernelId &&
                kernel.radius == radius &&
                std::fabs(kernel.sigma - sigma) <= 1e-6f &&
                shared_gaussian_entry_matches_cache(key, kernel);
            if (same) {
                return true;
            }
        }

        std::vector<float> cpuWeights;
        build_gaussian_weights_cpu(radius, sigma, cpuWeights);
        if (cpuWeights.empty()) {
            outError = std::string(label ? label : "gaussian kernel") + " weights build failed";
            return false;
        }

        SharedGaussianEntry sharedEntry{};
        if (!acquire_shared_gaussian_entry(
                key,
                spec,
                cpuWeights,
                sharedEntry,
                outError)) {
            return false;
        }

        std::lock_guard<std::mutex> lock(resources.m);
        reap_retire_queue_locked(resources);
        if (!validate_resource_owner_locked(resources, outError, true)) {
            return false;
        }

        key = make_shared_gaussian_key(resources, spec);
        kernelId = make_shared_gaussian_id(key);
        if (kernelId != sharedEntry.id) {
            if (!acquire_shared_gaussian_entry(
                    key,
                    spec,
                    cpuWeights,
                    sharedEntry,
                    outError)) {
                return false;
            }
            kernelId = sharedEntry.id;
        }

        const bool same =
            kernel.weights &&
            kernel.sharedKernelId == kernelId &&
            kernel.radius == radius &&
            std::fabs(kernel.sigma - sigma) <= 1e-6f &&
            shared_gaussian_entry_matches_cache(key, kernel);
        if (same) {
            return true;
        }

        if (!retire_or_clear_gaussian_kernel_locked(
                resources,
                kernel,
                cudaStreamOpaque,
                label,
                outError)) {
            return false;
        }

        kernel.weights = sharedEntry.weights;
        kernel.radius = sharedEntry.radius;
        kernel.sigma = sharedEntry.sigma;
        kernel.capacity = 0;
        kernel.sharedKernelId = sharedEntry.id;
        return true;
#endif
    }

    bool ensure_spatial_dir_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
        constexpr int kMaxRadius = 2048;
        const bool sigmaOk = std::isfinite(sigma) && sigma > 0.0f;
        const int radiusRaw = sigmaOk
                                  ? std::max(1, static_cast<int>(std::ceil(3.0f * sigma)))
                                  : 0;
        if (radiusRaw > kMaxRadius) {
            outError = "spatial DIR kernel radius exceeds prepared-frame limit";
            return false;
        }
        const int radius = radiusRaw;
        SharedGaussianSpec spec{};
        spec.kind = SharedGaussianKind::SpatialDir;
        spec.radius = radius;
        spec.sigma = sigma;
        return ensure_shared_gaussian_kernel(
            resources,
            kernel,
            spec,
            cudaStreamOpaque,
            "spatial DIR kernel",
            outError);
    }

    bool ensure_gaussian_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
        constexpr int kMaxRadius = 75;
        const bool sigmaOk = std::isfinite(sigma) && sigma > 0.0f;
        const int radiusRaw = sigmaOk
                                  ? JuicerGaussian::scipy_gaussian_radius(sigma, 4.0f)
                                  : 0;
        const int radius = std::min(radiusRaw, kMaxRadius);
        SharedGaussianSpec spec{};
        spec.kind = SharedGaussianKind::Standard;
        spec.radius = radius;
        spec.sigma = sigma;
        return ensure_shared_gaussian_kernel(
            resources,
            kernel,
            spec,
            cudaStreamOpaque,
            "gaussian kernel",
            outError);
    }

    bool ensure_halation_kernel(Resources& resources, Resources::DeviceGaussianKernel& kernel, float sigma, void* cudaStreamOpaque, std::string& outError) {
        constexpr int kMaxRadius = 75;
        const bool sigmaOk = std::isfinite(sigma) && sigma > 0.0f;
        const int radiusRaw = sigmaOk
                                  ? JuicerGaussian::scipy_gaussian_radius(sigma, 7.0f)
                                  : 0;
        const int radius = std::min(radiusRaw, kMaxRadius);
        SharedGaussianSpec spec{};
        spec.kind = SharedGaussianKind::Halation;
        spec.radius = radius;
        spec.sigma = sigma;
        return ensure_shared_gaussian_kernel(
            resources,
            kernel,
            spec,
            cudaStreamOpaque,
            "halation kernel",
            outError);
    }

    void purge_shared_gaussian_kernels_for_context(int deviceId, void* contextOpaque) noexcept {
        try {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
            if (deviceId < 0 || contextOpaque == nullptr) {
                return;
            }

            std::vector<float*> toFree;
            {
                SharedGaussianCacheState& cache = shared_gaussian_cache_state();
                std::lock_guard<std::mutex> lock(cache.mutex);
                toFree.reserve(cache.byKey.size());
                for (auto it = cache.byKey.begin(); it != cache.byKey.end();) {
                    if (it->first.deviceId == deviceId &&
                        it->first.contextOpaque == contextOpaque) {
                        if (it->second.weights) {
                            toFree.push_back(it->second.weights);
                        }
                        it = cache.byKey.erase(it);
                        continue;
                    }
                    ++it;
                }
            }

            if (toFree.empty()) {
                return;
            }

            int previousDevice = -1;
            const cudaError_t prevErr = cudaGetDevice(&previousDevice);
            const bool havePreviousDevice = (prevErr == cudaSuccess && previousDevice >= 0);
            const bool needRestore = havePreviousDevice && previousDevice != deviceId;
            (void)cudaSetDevice(deviceId);

            for (float* ptr : toFree) {
                if (ptr) {
                    (void)cudaFree(ptr);
                }
            }

            if (needRestore) {
                (void)cudaSetDevice(previousDevice);
            }
#else
            (void)deviceId;
            (void)contextOpaque;
#endif
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }
} // namespace JuicerCuda
