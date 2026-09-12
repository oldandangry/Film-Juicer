#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <windows.h>

#include "Cuda/JuicerCudaFilmPayloads.h"
#include "Cuda/JuicerCudaDeviceLedger.h"
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/Diffusion/JuicerCudaDiffusion.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "GammaValidationCuda.h"
#include "SpectralProcessing.h"
#include "JuicerState.h"
#include "OutputColor.h"
#include "ProcessRoot.h"
#include "Scanner.h"
#include "nlohmann/json.hpp"

extern "C" cudaError_t juicer_cuda_print_focused_pipeline(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_negative_direct_pipeline(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_direct_focused_capture_density(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_build_direct_spatial_dir(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    JuicerCuda::SpatialDirBuildRequest request);

extern "C" cudaError_t juicer_cuda_build_direct_spatial_dir_cached_log_raw(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    float* logRawB,
    float* logRawG,
    float* logRawR,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_direct_focused_scan_linear_density_rgb(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    const float* filmDustTransmittance,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_direct_focused_scanner_post_output(
    const JuicerCuda::DirectPipelineRunParams* hParams,
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

extern "C" cudaError_t juicer_cuda_print_focused_capture_density(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_continue_from_capture_density(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    const float* filmDustTransmittance,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_enlarger_linear_exposure(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    JuicerCuda::EnlargerPrintLinearExposurePlanes output,
    const float* filmDustTransmittance,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_develop_from_enlarger_linear(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    JuicerCuda::EnlargerPrintLinearExposurePlanes input,
    float* dDensityC,
    float* dDensityM,
    float* dDensityY,
    void* cudaStreamOpaque);

extern "C" cudaError_t juicer_cuda_print_focused_scan_linear_density_rgb(
    const JuicerCuda::PrintPipelineRunParams* hParams,
    const float* dDensityC,
    const float* dDensityM,
    const float* dDensityY,
    float* dRgbR,
    float* dRgbG,
    float* dRgbB,
    float* dTmp,
    float* dScratchBlurred,
    int glareOriginX,
    int glareOriginY,
    std::uint64_t glareSeed,
    float glarePercent,
    float glareRoughness,
    const float* dGlareKernel,
    int glareRadius,
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

namespace {

    struct Arguments {
        std::string caseGroup;
        std::filesystem::path resourceRoot;
        std::filesystem::path inputFile;
        std::filesystem::path outputDirectory;
        int width = 0;
        int height = 0;
    };

    Arguments parse_arguments(int argc, char** argv) {
        Arguments arguments;
        for (int index = 1; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (index + 1 >= argc) {
                throw std::runtime_error("missing value for " + std::string(option));
            }
            const std::string value(argv[++index]);
            if (option == "--case-group") {
                arguments.caseGroup = value;
            } else if (option == "--resource-root") {
                arguments.resourceRoot = value;
            } else if (option == "--input") {
                arguments.inputFile = value;
            } else if (option == "--output-dir") {
                arguments.outputDirectory = value;
            } else if (option == "--width") {
                arguments.width = std::stoi(value);
            } else if (option == "--height") {
                arguments.height = std::stoi(value);
            } else {
                throw std::runtime_error("unknown option " + std::string(option));
            }
        }
        const bool routeCase = arguments.caseGroup == "default-paper" ||
                               arguments.caseGroup == "default-routes" ||
                               arguments.caseGroup == "route-probe" ||
                               arguments.caseGroup == "candidate-route-probe" ||
                               arguments.caseGroup == "routes" ||
                               arguments.caseGroup == "lifetime" ||
                               arguments.caseGroup == "performance";
        const bool probeCase = arguments.caseGroup == "sampling-dir" ||
                               arguments.caseGroup == "print-backend" ||
                               arguments.caseGroup == "state";
        if ((!routeCase && !probeCase) || arguments.resourceRoot.empty() ||
            arguments.inputFile.empty() || arguments.outputDirectory.empty() ||
            (routeCase && (arguments.width <= 0 || arguments.height <= 0))) {
            throw std::runtime_error(
                "expected --case-group default-paper|default-routes|route-probe|candidate-route-probe|routes|lifetime|performance|sampling-dir|print-backend|state "
                "--resource-root PATH --input PATH --output-dir PATH and route dimensions");
        }
        return arguments;
    }

    std::filesystem::path executable_path() {
        std::wstring buffer(32768, L'\0');
        const DWORD length = GetModuleFileNameW(
            nullptr,
            buffer.data(),
            static_cast<DWORD>(buffer.size()));
        if (length == 0 || length >= buffer.size()) {
            throw std::runtime_error("GetModuleFileNameW failed");
        }
        buffer.resize(length);
        return std::filesystem::path(buffer);
    }

    void verify_runtime_resource_root(const std::filesystem::path& supplied) {
        const std::filesystem::path expected =
            executable_path().parent_path().parent_path() / "Resources";
        const auto expectedCanonical = std::filesystem::weakly_canonical(expected);
        const auto suppliedCanonical = std::filesystem::weakly_canonical(supplied);
        if (expectedCanonical != suppliedCanonical ||
            !std::filesystem::is_regular_file(
                expectedCanonical / "profiles" / "kodak_portra_400.json") ||
            !std::filesystem::is_regular_file(
                expectedCanonical / "profiles" / "kodak_portra_endura.json")) {
            throw std::runtime_error(
                "staged executable-relative Resources root is missing or mismatched");
        }
    }

    std::vector<float> read_float_file(
        const std::filesystem::path& path,
        std::size_t expectedCount) {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) {
            throw std::runtime_error("unable to open input file " + path.string());
        }
        const std::streamsize byteCount = stream.tellg();
        const std::size_t expectedBytes = expectedCount * sizeof(float);
        if (byteCount < 0 || static_cast<std::size_t>(byteCount) != expectedBytes) {
            throw std::runtime_error("input file has unexpected byte count");
        }
        stream.seekg(0);
        std::vector<float> values(expectedCount);
        stream.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(expectedBytes));
        if (!stream) {
            throw std::runtime_error("unable to read complete input file");
        }
        return values;
    }

    void write_float_file(
        const std::filesystem::path& path,
        const std::vector<float>& values) {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("unable to create output file " + path.string());
        }
        stream.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
        if (!stream) {
            throw std::runtime_error("unable to write complete output file");
        }
    }

    nlohmann::json read_json_file(const std::filesystem::path& path) {
        std::ifstream stream(path);
        if (!stream) {
            throw std::runtime_error("unable to open JSON input " + path.string());
        }
        nlohmann::json document;
        stream >> document;
        if (!stream) {
            throw std::runtime_error("unable to parse JSON input " + path.string());
        }
        return document;
    }

    std::vector<float> float_vector(const nlohmann::json& values) {
        if (!values.is_array() || values.empty()) {
            throw std::runtime_error("expected a nonempty numeric array");
        }
        std::vector<float> result;
        result.reserve(values.size());
        for (const auto& value : values) {
            const double converted = value.get<double>();
            if (!std::isfinite(converted) ||
                converted < -static_cast<double>(std::numeric_limits<float>::max()) ||
                converted > static_cast<double>(std::numeric_limits<float>::max())) {
                throw std::runtime_error("probe value is not representable as Float32");
            }
            result.push_back(static_cast<float>(converted));
        }
        return result;
    }

    template <typename T>
    class DeviceBuffer final {
    public:
        explicit DeviceBuffer(std::size_t count) : _count(count) {
            if (count == 0) {
                throw std::runtime_error("zero-sized CUDA allocation requested");
            }
            const cudaError_t status = cudaMalloc(
                reinterpret_cast<void**>(&_pointer),
                count * sizeof(T));
            if (status != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaMalloc failed: ") + cudaGetErrorString(status));
            }
        }

        ~DeviceBuffer() {
            if (_pointer) {
                (void)cudaFree(_pointer);
            }
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        T* get() const noexcept {
            return _pointer;
        }

        std::size_t bytes() const noexcept {
            return _count * sizeof(T);
        }

    private:
        T* _pointer = nullptr;
        std::size_t _count = 0;
    };

    void require_cuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string(operation) + " failed: " + cudaGetErrorString(status));
        }
    }

    template <typename T>
    void upload(
        const DeviceBuffer<T>& destination,
        const std::vector<T>& source,
        cudaStream_t stream) {
        require_cuda(
            cudaMemcpyAsync(
                destination.get(),
                source.data(),
                destination.bytes(),
                cudaMemcpyHostToDevice,
                stream),
            "probe upload");
    }

    template <typename T>
    std::vector<T> download(
        const DeviceBuffer<T>& source,
        std::size_t count,
        cudaStream_t stream) {
        std::vector<T> result(count);
        require_cuda(
            cudaMemcpyAsync(
                result.data(),
                source.get(),
                source.bytes(),
                cudaMemcpyDeviceToHost,
                stream),
            "probe download");
        require_cuda(cudaStreamSynchronize(stream), "probe completion");
        return result;
    }

    float finite_rgba_maximum_absolute_difference(
        std::string_view caseId,
        std::string_view arrayName,
        const std::vector<float>& split,
        const std::vector<float>& fused,
        int componentCount) {
        if (split.empty() || split.size() != fused.size() || componentCount <= 0 ||
            split.size() % static_cast<std::size_t>(componentCount) != 0u) {
            throw std::runtime_error(
                "split/fused shape mismatch case=" + std::string(caseId) +
                " array=" + std::string(arrayName) + " split_count=" +
                std::to_string(split.size()) + " fused_count=" +
                std::to_string(fused.size()) + " components=" +
                std::to_string(componentCount));
        }
        float maximumDifference = 0.0f;
        for (std::size_t index = 0; index < split.size(); ++index) {
            const float splitValue = split[index];
            const float fusedValue = fused[index];
            if (!std::isfinite(splitValue) || !std::isfinite(fusedValue)) {
                throw std::runtime_error(
                    "split/fused nonfinite value case=" + std::string(caseId) +
                    " array=" + std::string(arrayName) + " sample=" +
                    std::to_string(index / static_cast<std::size_t>(componentCount)) +
                    " component=" +
                    std::to_string(index % static_cast<std::size_t>(componentCount)) +
                    " split=" + std::to_string(splitValue) + " fused=" +
                    std::to_string(fusedValue));
            }
            maximumDifference = std::max(
                maximumDifference,
                std::abs(splitValue - fusedValue));
        }
        return maximumDifference;
    }

    void copy_float3(float destination[3], const float source[3]) {
        std::memcpy(destination, source, 3u * sizeof(float));
    }

    void copy_float9(float destination[9], const float source[9]) {
        std::memcpy(destination, source, 9u * sizeof(float));
    }

    void bind_scan_tables(
        JuicerCuda::ScanTablesPayload& output,
        const JuicerCuda::Resources::DeviceScanMedium& input) {
        output.epsC = input.tables.epsC;
        output.epsM = input.tables.epsM;
        output.epsY = input.tables.epsY;
        output.Ax = input.tables.Ax;
        output.Ay = input.tables.Ay;
        output.Az = input.tables.Az;
        output.baseDensityMin = input.tables.baseDensityMin;
        output.K = input.tables.K;
        output.hasBaseline = input.tables.hasBaseline;
        output.invYn = input.tables.invYn;
        output.mediumIsNegative = input.mediumIsNegative;
        copy_float3(output.min_cmy, input.min_cmy);
        copy_float3(output.inv_max_cmy, input.inv_max_cmy);
    }

    void bind_scan_stage(
        const JuicerProcess::Root::PreparedCudaFrame::FocusedPreparedView& prepared,
        const Scanner::ScannerColorCorrectionDescriptor& correction,
        JuicerCuda::ScanStagePayload& output) {
        if (!prepared.scanMedium || !prepared.scanLut || !prepared.scannerColor) {
            throw std::runtime_error("focused scanner resources are incomplete");
        }
        bind_scan_tables(output.scanTables, *prepared.scanMedium);
        output.scannerUseLut = 1;
        output.scanLutLog2PchipXYZ = prepared.scanLut->log2PchipXYZ;
        output.scanLutPchipSlopeC = prepared.scanLut->slopeC;
        output.scanLutPchipSlopeM = prepared.scanLut->slopeM;
        output.scanLutPchipSlopeY = prepared.scanLut->slopeY;
        output.scanLutPchipCellMin = prepared.scanLut->cellMin;
        output.scanLutPchipCellMax = prepared.scanLut->cellMax;
        output.scanLutRes = static_cast<int>(prepared.scanLut->res);

        const Scanner::ColorRuntime& color = *prepared.scannerColor;
        copy_float9(output.scanColor.cat02, color.cat02);
        copy_float9(output.scanColor.xyzToRgb, color.xyzToRgb);
        copy_float3(output.scanColor.illuminantXYZ, color.illuminantXYZ);
        output.scanColor.encoding.outputColorSpaceIndex =
            OutputEncoding::toIndex(color.encoding.colorSpace);
        output.scanColor.encoding.applyCctfEncoding =
            color.encoding.applyCctfEncoding ? 1 : 0;
        output.scanColor.encoding.inputIsOutputSpace =
            color.encoding.inputIsOutputSpace ? 1 : 0;
        const auto& outputSpace =
            GeneratedColorSpaces::get(color.encoding.colorSpace);
        output.scanColor.encoding.cctf.kind =
            static_cast<int>(outputSpace.cctf.kind);
        output.scanColor.encoding.cctf.gamma = outputSpace.cctf.gamma;
        output.scanColor.encoding.cctf.a = outputSpace.cctf.a;
        output.scanColor.encoding.cctf.b = outputSpace.cctf.b;
        output.scanColor.encoding.cctf.c = outputSpace.cctf.c;
        output.scanColor.encoding.cctf.d = outputSpace.cctf.d;
        output.scanColor.encoding.cctf.linearCutoff =
            outputSpace.cctf.linearCutoff;
        const OutputEncoding::Matrix3x3 dwgToOutput =
            OutputEncoding::dwg_to_output_matrix(color.encoding.colorSpace);
        copy_float9(output.scanColor.encoding.dwgToOutput, dwgToOutput.m);
        output.correctionActive = correction.active ? 1 : 0;
        output.correctionSlope = correction.xyzSlope;
        output.correctionOffset = correction.xyzOffset;
    }

    struct CudaContext final {
        CudaContext() {
            require_cuda(cudaSetDevice(0), "cudaSetDevice");
            require_cuda(cudaFree(nullptr), "CUDA runtime initialization");
            CUcontext context = nullptr;
            if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context) {
                throw std::runtime_error("cuCtxGetCurrent did not return the primary context");
            }
            key.deviceId = 0;
            key.contextOpaque = context;
            require_cuda(
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags");
        }

        ~CudaContext() {
            if (stream) {
                (void)cudaStreamDestroy(stream);
            }
        }

        JuicerCuda::ResourceManager::DeviceContextKey key{};
        cudaStream_t stream = nullptr;
    };

    bool verify_print_develop_observer_padded_stride(
        const JuicerCuda::PrintDevelopPayload& develop,
        cudaStream_t stream) {
        constexpr int kWidth = 3;
        constexpr int kHeight = 2;
        constexpr std::size_t kStride = 5;
        constexpr std::size_t kCount =
            static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight);
        constexpr std::size_t kPaddedCount =
            kStride * static_cast<std::size_t>(kHeight);
        const std::array<std::vector<float>, 3> active{
            std::vector<float>{0.0f, 1.0e-8f, 0.125f, 0.5f, 1.0f, 2.0f},
            std::vector<float>{0.03125f, 0.25f, 0.75f, 1.5f, 3.0f, 6.0f},
            std::vector<float>{0.0625f, 0.375f, 0.875f, 1.75f, 4.0f, 8.0f}};
        std::array<std::vector<float>, 3> padded{
            std::vector<float>(kPaddedCount, -1234.0f),
            std::vector<float>(kPaddedCount, 2345.0f),
            std::vector<float>(kPaddedCount, -3456.0f)};
        for (std::size_t channel = 0; channel < active.size(); ++channel) {
            for (int y = 0; y < kHeight; ++y) {
                for (int x = 0; x < kWidth; ++x) {
                    padded[channel][static_cast<std::size_t>(y) * kStride +
                                    static_cast<std::size_t>(x)] =
                        active[channel][static_cast<std::size_t>(y) * kWidth +
                                        static_cast<std::size_t>(x)];
                }
            }
        }

        DeviceBuffer<float> paddedR(kPaddedCount);
        DeviceBuffer<float> paddedG(kPaddedCount);
        DeviceBuffer<float> paddedB(kPaddedCount);
        DeviceBuffer<float> compactR(kCount);
        DeviceBuffer<float> compactG(kCount);
        DeviceBuffer<float> compactB(kCount);
        DeviceBuffer<float> paddedLogR(kCount);
        DeviceBuffer<float> paddedLogG(kCount);
        DeviceBuffer<float> paddedLogB(kCount);
        DeviceBuffer<float> compactLogR(kCount);
        DeviceBuffer<float> compactLogG(kCount);
        DeviceBuffer<float> compactLogB(kCount);
        DeviceBuffer<float> paddedDensityC(kCount);
        DeviceBuffer<float> paddedDensityM(kCount);
        DeviceBuffer<float> paddedDensityY(kCount);
        DeviceBuffer<float> compactDensityC(kCount);
        DeviceBuffer<float> compactDensityM(kCount);
        DeviceBuffer<float> compactDensityY(kCount);
        DeviceBuffer<float> unitGamma(kCount);
        upload(paddedR, padded[0], stream);
        upload(paddedG, padded[1], stream);
        upload(paddedB, padded[2], stream);
        upload(compactR, active[0], stream);
        upload(compactG, active[1], stream);
        upload(compactB, active[2], stream);
        const std::vector<float> unitGammaHost(kCount, 1.0f);
        upload(unitGamma, unitGammaHost, stream);

        GammaValidation::PrintDevelopLogExposureLaunch paddedLaunch{};
        paddedLaunch.linearExposure = {
            paddedR.get(),
            paddedG.get(),
            paddedB.get(),
            kStride};
        paddedLaunch.logExposureR = paddedLogR.get();
        paddedLaunch.logExposureG = paddedLogG.get();
        paddedLaunch.logExposureB = paddedLogB.get();
        paddedLaunch.width = kWidth;
        paddedLaunch.height = kHeight;
        require_cuda(
            gamma_validation_launch_print_develop_log_exposure(
                &paddedLaunch,
                stream),
            "padded print-development observer launch");

        GammaValidation::PrintDevelopLogExposureLaunch compactLaunch{};
        compactLaunch.linearExposure = {
            compactR.get(),
            compactG.get(),
            compactB.get(),
            static_cast<std::size_t>(kWidth)};
        compactLaunch.logExposureR = compactLogR.get();
        compactLaunch.logExposureG = compactLogG.get();
        compactLaunch.logExposureB = compactLogB.get();
        compactLaunch.width = kWidth;
        compactLaunch.height = kHeight;
        require_cuda(
            gamma_validation_launch_print_develop_log_exposure(
                &compactLaunch,
                stream),
            "compact print-development observer launch");

        const auto launch_curve = [&](const JuicerCuda::DeviceCurveView& curve,
                                      const DeviceBuffer<float>& logExposure,
                                      DeviceBuffer<float>& density) {
            GammaValidation::CurveProbeLaunch launch{};
            launch.curve = curve;
            launch.queries = logExposure.get();
            launch.gammaFactors = unitGamma.get();
            launch.results = density.get();
            launch.count = static_cast<int>(kCount);
            require_cuda(
                gamma_validation_launch_curve_probe(&launch, stream),
                "padded print-development curve sample");
        };
        launch_curve(develop.printDcC, paddedLogR, paddedDensityC);
        launch_curve(develop.printDcM, paddedLogG, paddedDensityM);
        launch_curve(develop.printDcY, paddedLogB, paddedDensityY);
        launch_curve(develop.printDcC, compactLogR, compactDensityC);
        launch_curve(develop.printDcM, compactLogG, compactDensityM);
        launch_curve(develop.printDcY, compactLogB, compactDensityY);

        const std::array<std::vector<float>, 6> paddedResults{
            download(paddedLogR, kCount, stream),
            download(paddedLogG, kCount, stream),
            download(paddedLogB, kCount, stream),
            download(paddedDensityC, kCount, stream),
            download(paddedDensityM, kCount, stream),
            download(paddedDensityY, kCount, stream)};
        const std::array<std::vector<float>, 6> compactResults{
            download(compactLogR, kCount, stream),
            download(compactLogG, kCount, stream),
            download(compactLogB, kCount, stream),
            download(compactDensityC, kCount, stream),
            download(compactDensityM, kCount, stream),
            download(compactDensityY, kCount, stream)};
        for (std::size_t plane = 0; plane < paddedResults.size(); ++plane) {
            for (std::size_t sample = 0; sample < kCount; ++sample) {
                if (!std::isfinite(paddedResults[plane][sample]) ||
                    !std::isfinite(compactResults[plane][sample]) ||
                    paddedResults[plane][sample] != compactResults[plane][sample]) {
                    throw std::runtime_error(
                        "padded print-development observer mismatch plane=" +
                        std::to_string(plane) + " sample=" +
                        std::to_string(sample) + " padded=" +
                        std::to_string(paddedResults[plane][sample]) +
                        " compact=" +
                        std::to_string(compactResults[plane][sample]));
                }
            }
        }
        return true;
    }

    struct DiffusionStageBinding {
        Spektrafilm::PlanLayout layout{};
        Spektrafilm::DiffusionStageTileGeometry geometry{};
        Spektrafilm::DiffusionFrameDomain fullFrame{};
        JuicerCuda::Diffusion::SpectrumPackageView spectra{};
        JuicerCuda::Diffusion::ExecutionWorkspaceView execution{};
        JuicerCuda::Diffusion::StagePlaneSet stagePlanes{};
    };

    bool same_diffusion_spectrum_key(
        const Spektrafilm::DiffusionSpectrumKey& left,
        const Spektrafilm::DiffusionSpectrumKey& right) noexcept {
        return left.sampleHash == right.sampleHash && left.extent == right.extent &&
               left.hash == right.hash;
    }

    bool same_diffusion_plan_key(
        const Spektrafilm::DiffusionPlanKey& left,
        const Spektrafilm::DiffusionPlanKey& right) noexcept {
        return left.extent == right.extent && left.hash == right.hash;
    }

    bool bind_enlarger_diffusion_stage(
        const Spektrafilm::DiffusionFrameSetDescriptor& frameSet,
        const JuicerCuda::Diffusion::DiffusionPreparedView& prepared,
        DiffusionStageBinding& out,
        std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        if (!frameSet.enlarger || !prepared.active ||
            prepared.executionDescriptor.frameSetHash != frameSet.hash ||
            prepared.executionDescriptor.stageCount == 0 ||
            prepared.executionDescriptor.stageCount >
                prepared.executionDescriptor.stages.size() ||
            prepared.executionDescriptor.uniqueSpectrumCount == 0 ||
            prepared.executionDescriptor.uniqueSpectrumCount >
                prepared.executionDescriptor.spectrumKeys.size() ||
            prepared.spectrumCount !=
                prepared.executionDescriptor.uniqueSpectrumCount) {
            diagnostic =
                "MissingRequiredResource component=diffusion stage=enlarger field=execution_descriptor";
            return false;
        }
        const Spektrafilm::DiffusionStageTileGeometry* stageGeometry = nullptr;
        for (std::size_t index = 0;
             index < prepared.executionDescriptor.stageCount;
             ++index) {
            const auto& candidate = prepared.executionDescriptor.stages[index];
            if (candidate.stage !=
                Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear) {
                continue;
            }
            if (stageGeometry) {
                diagnostic =
                    "ResourceDescriptorMismatch component=diffusion stage=enlarger field=duplicate_stage";
                return false;
            }
            stageGeometry = &candidate;
        }
        if (!stageGeometry ||
            stageGeometry->stageDescriptorHash != frameSet.enlarger->hash ||
            stageGeometry->spectrumKeyIndex >=
                prepared.executionDescriptor.uniqueSpectrumCount ||
            stageGeometry->spectrumKeyIndex >= prepared.spectrumCount) {
            diagnostic =
                "ResourceDescriptorMismatch component=diffusion stage=enlarger field=stage_geometry";
            return false;
        }
        const std::size_t spectrumIndex = stageGeometry->spectrumKeyIndex;
        if (!same_diffusion_spectrum_key(
                prepared.executionDescriptor.spectrumKeys[spectrumIndex],
                prepared.spectra[spectrumIndex].key) ||
            !same_diffusion_plan_key(
                prepared.executionDescriptor.planKey,
                prepared.execution.planKey)) {
            diagnostic =
                "ResourceDescriptorMismatch component=diffusion stage=enlarger field=resource_key";
            return false;
        }
        const auto& spectra = prepared.spectra[spectrumIndex].spectra;
        const auto& execution = prepared.execution.execution;
        const auto& planes = prepared.execution.stagePlanes;
        if (!spectra.red || !spectra.green || !spectra.blue ||
            !execution.transformBuffer || execution.r2cPlan == 0 ||
            execution.c2rPlan == 0 || !planes.redSensitive ||
            !planes.greenSensitive || !planes.blueSensitive ||
            !planes.auxiliary || planes.redSensitive == planes.greenSensitive ||
            planes.redSensitive == planes.blueSensitive ||
            planes.redSensitive == planes.auxiliary ||
            planes.greenSensitive == planes.blueSensitive ||
            planes.greenSensitive == planes.auxiliary ||
            planes.blueSensitive == planes.auxiliary ||
            planes.rowStrideFloats <
                static_cast<std::size_t>(frameSet.fullFrame.width)) {
            diagnostic =
                "MissingRequiredResource component=diffusion stage=enlarger field=execution_workspace";
            return false;
        }
        out.layout = prepared.executionDescriptor.layout;
        out.geometry = *stageGeometry;
        out.fullFrame = frameSet.fullFrame;
        out.spectra = spectra;
        out.execution = execution;
        out.stagePlanes = planes;
        return true;
    }

    JuicerCuda::EnlargerPrintLinearExposurePlanes enlarger_linear_planes(
        const JuicerCuda::Diffusion::StagePlaneSet& planes) noexcept {
        return {
            planes.redSensitive,
            planes.greenSensitive,
            planes.blueSensitive,
            planes.rowStrideFloats};
    }

    ParamSnapshot route_snapshot(
        Spektrafilm::ScanRoute route,
        std::string filmProfileKey,
        std::string printProfileKey = "kodak_portra_endura") {
        ParamSnapshot snapshot;
        snapshot.scanRoute = route;
        snapshot.filmProfileKey = std::move(filmProfileKey);
        snapshot.printProfileKey = std::move(printProfileKey);
        snapshot.inputColorSpace = Spectral::inputColorSpaceToIndex(
            Spectral::InputColorSpace::SRGB_Rec709);
        snapshot.inputCctfDecoding = 0;
        snapshot.cameraAutoExposureEnabled = 0;
        snapshot.cameraFilterOverride = false;
        snapshot.couplersActive = 1;
        snapshot.couplersDiffusionSizeUm = 0.0;
        snapshot.scatterHalationControls = ScatterHalationControls{};
        snapshot.grainControls.active = false;
        snapshot.glareActive = false;
        snapshot.scannerLensBlurSigmaPx = 0.0;
        snapshot.scannerUnsharpMask = {0.0, 0.0};
        snapshot.scannerBlackCorrection = 0;
        snapshot.scannerWhiteCorrection = 0;
        snapshot.scannerUseLut = 1;
        snapshot.scannerLutResolution = 17;
        snapshot.outputColorSpace =
            OutputEncoding::toIndex(OutputEncoding::ColorSpace::sRGB);
        snapshot.outputCctfEncoding = 1;
        return snapshot;
    }

    ParamSnapshot default_paper_snapshot() {
        return route_snapshot(
            Spektrafilm::ScanRoute::NegativePrintScan,
            "kodak_portra_400");
    }

    template <typename Snapshot>
    bool assign_retained_film_gamma(Snapshot& snapshot, float factor) {
        if constexpr (requires { snapshot.filmGammaFactor = factor; }) {
            snapshot.filmGammaFactor = factor;
            return true;
        }
        return false;
    }

    struct InjectedPrintCurves {
        std::string tableId;
        float filmGamma = 1.0f;
        double printGamma = 1.0;
        std::vector<float> axis;
        std::vector<float> densityCmyInterleaved;
    };

    const char* profile_polarity_key(
        Spektrafilm::ProfilePolarity polarity) noexcept {
        switch (polarity) {
            case Spektrafilm::ProfilePolarity::Negative:
                return "negative";
            case Spektrafilm::ProfilePolarity::Positive:
                return "positive";
            case Spektrafilm::ProfilePolarity::Unsupported:
                return "unsupported";
            default:
                return "unknown";
        }
    }

    const char* diffusion_family_key(
        Spektrafilm::DiffusionFilterFamily family) noexcept {
        switch (family) {
            case Spektrafilm::DiffusionFilterFamily::Glimmerglass:
                return "glimmerglass";
            case Spektrafilm::DiffusionFilterFamily::BlackProMist:
                return "black_pro_mist";
            case Spektrafilm::DiffusionFilterFamily::ProMist:
                return "pro_mist";
            case Spektrafilm::DiffusionFilterFamily::Cinebloom:
                return "cinebloom";
            default:
                return "unknown";
        }
    }

    const char* print_normalization_mode_key(
        Spektrafilm::PrintNormalizationMode mode) noexcept {
        switch (mode) {
            case Spektrafilm::PrintNormalizationMode::None:
                return "none";
            case Spektrafilm::PrintNormalizationMode::CompensationOnly:
                return "compensation_only";
            case Spektrafilm::PrintNormalizationMode::NormalizeOnly:
                return "normalize_only";
            case Spektrafilm::PrintNormalizationMode::NormalizeAndCompensate:
                return "normalize_and_compensate";
            default:
                return "unknown";
        }
    }

    nlohmann::json capture_effective_settings(
        const RenderRecipe& recipe,
        int width,
        int height,
        double pixelSizeUm,
        const Spektrafilm::SpatialDirDescriptor* spatialDir,
        const Spektrafilm::DiffusionFrameSetDescriptor* diffusionFrameSet,
        const Scanner::ScannerPostEffectsDescriptor& scannerPost,
        const InjectedPrintCurves* injectedPrintCurves) {
        const auto inputSpace = Spectral::inputColorSpaceFromIndex(
            recipe.filmRaw.inputColorSpace);
        const auto outputSpace = OutputEncoding::colorSpaceFromIndex(
            recipe.scannerOutput.outputColorSpace);
        const bool spatialDirActive =
            spatialDir && spatialDir->hash != 0 &&
            spatialDir->support ==
                Spektrafilm::DirDescriptorSupport::Supported;
        const bool cameraDiffusionActive =
            diffusionFrameSet && diffusionFrameSet->camera.has_value();
        const bool enlargerDiffusionActive =
            diffusionFrameSet && diffusionFrameSet->enlarger.has_value();
        const auto& scatterHalation = recipe.spatialOptics.scatterHalation;
        nlohmann::json settings = {
            {"scan_route", static_cast<int>(recipe.profileRoute.scanRoute)},
            {"scan_route_key",
             Spektrafilm::scan_route_key(recipe.profileRoute.scanRoute)},
            {"capture_polarity",
             profile_polarity_key(recipe.profileRoute.capturePolarity)},
            {"frame_width", width},
            {"frame_height", height},
            {"camera_film_format_long_edge_mm",
             recipe.filmRaw.filmFormatLongEdgeMm},
            {"pixel_size_um", pixelSizeUm},
            {"input_color_space",
             Spectral::kInputColorSpaceLabels[Spectral::inputColorSpaceToIndex(inputSpace)]},
            {"input_cctf_decoding", recipe.filmRaw.inputCctfDecoding},
            {"output_color_space",
             OutputEncoding::kColorSpaceLabels[OutputEncoding::toIndex(outputSpace)]},
            {"output_cctf_encoding",
             recipe.scannerOutput.outputCctfEncoding},
            {"auto_exposure", recipe.filmRaw.autoExposureEnabled},
            {"manual_camera_exposure_compensation_ev",
             recipe.filmRaw.manualExposureCompensationEv},
            {"film_gamma",
             injectedPrintCurves
                 ? nlohmann::json(injectedPrintCurves->filmGamma)
                 : nlohmann::json(
                       recipe.filmDevelop.densityCurveGamma[0])},
            {"film_gamma_recipe_rgb",
             recipe.filmDevelop.densityCurveGamma},
            {"print_gamma",
             injectedPrintCurves
                 ? nlohmann::json(injectedPrintCurves->printGamma)
                 : nlohmann::json(recipe.print.develop.gammaFactor)},
            {"print_gamma_recipe_factor", recipe.print.develop.gammaFactor},
            {"print_curve_binding",
             injectedPrintCurves ? "validation_injected" : "recipe_owned"},
            {"dir_active", recipe.dirCouplers.active},
            {"dir_diffusion_size_um", recipe.dirCouplers.diffusionSizeUm},
            {"spatial_dir_active", spatialDirActive},
            {"spatial_dir_descriptor_hash",
             spatialDir ? spatialDir->hash : 0},
            {"spatial_dir_descriptor_support",
             spatialDir ? Spektrafilm::to_cstr(spatialDir->support)
                        : "not_built"},
            {"spatial_dir_gaussian_sigma_px",
             spatialDir ? spatialDir->gaussianSigmaPixels : 0.0f},
            {"visual_grain", recipe.visualGrain.active},
            {"scatter_active", scatterHalation.scatterActive},
            {"halation_active", scatterHalation.backReflectionActive},
            {"scatter_halation",
             scatterHalation.scatterActive ||
                 scatterHalation.backReflectionActive},
            {"camera_lens_blur_sigma_um",
             recipe.spatialOptics.cameraLensBlur.sigmaUm},
            {"camera_diffusion_active", cameraDiffusionActive},
            {"camera_diffusion_recipe_active",
             recipe.spatialOptics.cameraDiffusion.resolved.active},
            {"enlarger_diffusion_active", enlargerDiffusionActive},
            {"enlarger_diffusion_recipe_active",
             recipe.spatialOptics.enlargerDiffusion.resolved.active},
            {"print_settings_applicable",
             Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute)},
            {"print_exposure", recipe.print.exposure.printExposure},
            {"print_preflash_exposure",
             recipe.print.exposure.preflashExposure},
            {"print_exposure_compensation",
             recipe.print.exposure.printExposureCompensation},
            {"normalize_print_exposure",
             recipe.print.exposure.normalizePrintExposure},
            {"print_normalization_mode",
             print_normalization_mode_key(
                 recipe.print.exposure.normalizationMode)},
            {"glare", scannerPost.glareActive},
            {"glare_percent", scannerPost.glarePercent},
            {"glare_roughness", scannerPost.glareRoughness},
            {"glare_blur_sigma_px", scannerPost.glareBlurSigmaPx},
            {"scanner_black_correction",
             recipe.scannerOutput.blackCorrection},
            {"scanner_white_correction",
             recipe.scannerOutput.whiteCorrection},
            {"scanner_lens_blur_sigma_px", scannerPost.lensBlurSigmaPx},
            {"scanner_unsharp_sigma_px", scannerPost.unsharpSigmaPx},
            {"scanner_unsharp_amount", scannerPost.unsharpAmount},
            {"scanner_lut_resolution", recipe.scannerOutput.lutResolution}};

        const auto add_diffusion = [&settings](
                                       std::string_view prefix,
                                       const DiffusionFilterOpticsRecipe& optics,
                                       const std::optional<
                                           Spektrafilm::DiffusionStageFrameDescriptor>&
                                           stage) {
            const std::string base(prefix);
            const auto& resolved = optics.resolved;
            settings[base + "_diffusion_family"] =
                diffusion_family_key(resolved.family);
            settings[base + "_diffusion_scatter_fraction"] =
                resolved.scatterFraction;
            settings[base + "_diffusion_group_weights_core_halo_bloom"] =
                resolved.groupWeightsCoreHaloBloom;
            settings[base + "_diffusion_group_center_lambda_um"] =
                resolved.groupCenterLambdaUm;
            settings[base + "_diffusion_effective_warmth"] =
                resolved.effectiveWarmth;
            settings[base + "_diffusion_spatial_scale"] =
                resolved.spatialScale;
            settings[base + "_diffusion_recipe_hash"] = optics.hash;
            settings[base + "_diffusion_descriptor_hash"] =
                stage ? stage->hash : 0;
            settings[base + "_diffusion_radius_px"] =
                stage ? stage->sample.radiusPixels : 0;
            settings[base + "_diffusion_descriptor_pixel_size_um"] =
                stage ? stage->sample.pixelSizeUm : 0.0;
        };
        const std::optional<Spektrafilm::DiffusionStageFrameDescriptor>
            noDiffusionStage;
        add_diffusion(
            "camera",
            recipe.spatialOptics.cameraDiffusion,
            diffusionFrameSet ? diffusionFrameSet->camera
                              : noDiffusionStage);
        add_diffusion(
            "enlarger",
            recipe.spatialOptics.enlargerDiffusion,
            diffusionFrameSet ? diffusionFrameSet->enlarger
                              : noDiffusionStage);
        return settings;
    }

    std::vector<float> flatten_triplets(const nlohmann::json& rows) {
        if (!rows.is_array() || rows.empty()) {
            throw std::runtime_error("expected nonempty triplet rows");
        }
        std::vector<float> result;
        result.reserve(rows.size() * 3u);
        for (const auto& row : rows) {
            if (!row.is_array() || row.size() != 3u) {
                throw std::runtime_error("expected a three-channel probe row");
            }
            const std::vector<float> values = float_vector(row);
            result.insert(result.end(), values.begin(), values.end());
        }
        return result;
    }

    nlohmann::json triplet_rows(const std::vector<float>& values) {
        if (values.size() % 3u != 0u) {
            throw std::runtime_error("triplet result has an invalid size");
        }
        nlohmann::json result = nlohmann::json::array();
        for (std::size_t offset = 0; offset < values.size(); offset += 3u) {
            result.push_back({values[offset], values[offset + 1], values[offset + 2]});
        }
        return result;
    }

    void write_capture_json(
        const std::filesystem::path& outputDirectory,
        const nlohmann::json& result) {
        std::filesystem::create_directories(outputDirectory);
        std::ofstream stream(outputDirectory / "capture.json", std::ios::trunc);
        stream << std::setw(2) << result << '\n';
        if (!stream) {
            throw std::runtime_error("unable to write capture manifest");
        }
    }

    nlohmann::json run_curve_probe_case(
        const nlohmann::json& input,
        CudaContext& cuda) {
        const std::vector<float> axis = float_vector(input.at("axis_float32"));
        const std::vector<float> density = float_vector(input.at("density_float32"));
        const std::vector<float> queries = float_vector(input.at("queries_float32"));
        const std::vector<float> factors = float_vector(input.at("gamma_float32"));
        if (axis.size() != density.size() || queries.size() != factors.size()) {
            throw std::runtime_error("curve probe arrays have incompatible sizes");
        }
        DeviceBuffer<float> deviceAxis(axis.size());
        DeviceBuffer<float> deviceDensity(density.size());
        DeviceBuffer<float> deviceQueries(queries.size());
        DeviceBuffer<float> deviceFactors(factors.size());
        DeviceBuffer<float> deviceResults(queries.size());
        upload(deviceAxis, axis, cuda.stream);
        upload(deviceDensity, density, cuda.stream);
        upload(deviceQueries, queries, cuda.stream);
        upload(deviceFactors, factors, cuda.stream);
        GammaValidation::CurveProbeLaunch launch{};
        launch.curve.x = deviceAxis.get();
        launch.curve.y = deviceDensity.get();
        launch.curve.n = static_cast<int>(axis.size());
        launch.curve.domainBegin = input.value("domain_begin", 0);
        launch.curve.domainEnd = input.value(
            "domain_end",
            static_cast<int>(axis.size()) - 1);
        launch.queries = deviceQueries.get();
        launch.gammaFactors = deviceFactors.get();
        launch.results = deviceResults.get();
        launch.count = static_cast<int>(queries.size());
        require_cuda(
            gamma_validation_launch_curve_probe(&launch, cuda.stream),
            "curve probe launch");
        nlohmann::json result;
        result["id"] = input.at("id");
        result["classification"] = input.at("classification");
        result["actual_float32"] = download(
            deviceResults,
            queries.size(),
            cuda.stream);
        result["expected_float64"] = input.at("expected_float64");
        return result;
    }

    nlohmann::json run_dir_probe_case(
        const nlohmann::json& input,
        CudaContext& cuda) {
        const std::vector<float> axis = float_vector(input.at("axis_float32"));
        const auto& densityRows = input.at("curve_density_bgr_float32");
        if (!densityRows.is_array() || densityRows.size() != 3u) {
            throw std::runtime_error("DIR probe requires three B/G/R curves");
        }
        std::array<std::vector<float>, 3> curves;
        for (int channel = 0; channel < 3; ++channel) {
            curves[channel] = float_vector(densityRows.at(channel));
            if (curves[channel].size() != axis.size()) {
                throw std::runtime_error("DIR curve length does not match its axis");
            }
        }
        const std::vector<float> logExposure =
            flatten_triplets(input.at("log_exposure_bgr_float32"));
        const std::vector<float> initialDensity =
            flatten_triplets(input.at("initial_density_bgr_float32"));
        if (logExposure.size() != initialDensity.size()) {
            throw std::runtime_error("DIR probe input lengths differ");
        }
        const std::size_t count = logExposure.size() / 3u;
        DeviceBuffer<float> deviceAxis(axis.size());
        std::array<DeviceBuffer<float>, 3> deviceCurves{
            DeviceBuffer<float>(axis.size()),
            DeviceBuffer<float>(axis.size()),
            DeviceBuffer<float>(axis.size())};
        DeviceBuffer<float> deviceLogExposure(logExposure.size());
        DeviceBuffer<float> deviceInitialDensity(initialDensity.size());
        DeviceBuffer<float> deviceCorrected(logExposure.size());
        DeviceBuffer<float> deviceFinalDensity(logExposure.size());
        upload(deviceAxis, axis, cuda.stream);
        for (int channel = 0; channel < 3; ++channel) {
            upload(deviceCurves[channel], curves[channel], cuda.stream);
        }
        upload(deviceLogExposure, logExposure, cuda.stream);
        upload(deviceInitialDensity, initialDensity, cuda.stream);

        GammaValidation::DirProbeLaunch launch{};
        launch.logExposureBgr = deviceLogExposure.get();
        launch.initialDensityBgr = deviceInitialDensity.get();
        launch.correctedLogExposureBgr = deviceCorrected.get();
        launch.finalDensityBgr = deviceFinalDensity.get();
        launch.dir.active = 1;
        launch.dir.positive = input.at("positive").get<bool>() ? 1 : 0;
        const std::vector<float> matrix = float_vector(input.at("matrix_float32"));
        const std::vector<float> maxima = float_vector(input.at("dmax_float32"));
        const std::vector<float> factors = float_vector(input.at("gamma_bgr_float32"));
        if (matrix.size() != 9u || maxima.size() != 3u || factors.size() != 3u) {
            throw std::runtime_error("DIR payload arrays have invalid sizes");
        }
        std::copy(matrix.begin(), matrix.end(), launch.dir.M);
        std::copy(maxima.begin(), maxima.end(), launch.dir.dMax);
        std::copy(factors.begin(), factors.end(), launch.gammaFactorsBgr);
        for (int channel = 0; channel < 3; ++channel) {
            launch.densityCurvesBgr[channel].x = deviceAxis.get();
            launch.densityCurvesBgr[channel].y = deviceCurves[channel].get();
            launch.densityCurvesBgr[channel].n = static_cast<int>(axis.size());
            launch.densityCurvesBgr[channel].domainBegin = 0;
            launch.densityCurvesBgr[channel].domainEnd =
                static_cast<int>(axis.size()) - 1;
        }
        launch.count = static_cast<int>(count);
        require_cuda(
            gamma_validation_launch_dir_probe(&launch, cuda.stream),
            "DIR probe launch");
        const std::vector<float> corrected = download(
            deviceCorrected,
            logExposure.size(),
            cuda.stream);
        const std::vector<float> finalDensity = download(
            deviceFinalDensity,
            logExposure.size(),
            cuda.stream);
        nlohmann::json result;
        result["id"] = input.at("id");
        result["classification"] = input.at("classification");
        result["actual_corrected_log_exposure_bgr_float32"] =
            triplet_rows(corrected);
        result["actual_final_density_bgr_float32"] = triplet_rows(finalDensity);
        result["expected_corrected_log_exposure_bgr_float64"] =
            input.at("expected_corrected_log_exposure_bgr_float64");
        result["expected_final_density_bgr_float64"] =
            input.at("expected_final_density_bgr_float64");
        return result;
    }

    nlohmann::json run_sampling_dir(const Arguments& arguments) {
        const nlohmann::json input = read_json_file(arguments.inputFile);
        CudaContext cuda;
        nlohmann::json result;
        result["case"] = "sampling-dir";
        result["curve_cases"] = nlohmann::json::array();
        result["dir_cases"] = nlohmann::json::array();
        for (const auto& probe : input.at("curve_cases")) {
            result["curve_cases"].push_back(run_curve_probe_case(probe, cuda));
        }
        for (const auto& probe : input.at("dir_cases")) {
            result["dir_cases"].push_back(run_dir_probe_case(probe, cuda));
        }
        write_capture_json(arguments.outputDirectory, result);
        return result;
    }

    Profiles::PrintDensityModel print_density_model_from_probe(
        const nlohmann::json& input) {
        Profiles::PrintDensityModel model{};
        const auto& centers = input.at("centers");
        const auto& amplitudes = input.at("amplitudes");
        const auto& sigmas = input.at("sigmas");
        for (std::size_t channel = 0; channel < 3u; ++channel) {
            for (std::size_t layer = 0; layer < 3u; ++layer) {
                model.centers[channel][layer] =
                    centers.at(channel).at(layer).get<double>();
                model.amplitudes[channel][layer] =
                    amplitudes.at(channel).at(layer).get<double>();
                model.sigmas[channel][layer] =
                    sigmas.at(channel).at(layer).get<double>();
            }
        }
        return model;
    }

    template <typename Profile>
    nlohmann::json selected_print_model_contract(const Profile& profile) {
        if constexpr (requires {
                          profile.sourceLogExposure.size();
                          profile.densityModel.centers[0][0];
                          profile.densityModel.amplitudes[0][0];
                          profile.densityModel.sigmas[0][0];
                      }) {
            if (profile.sourceLogExposure.size() !=
                    profile.data.logExposure.size() ||
                profile.sourceLogExposure.empty()) {
                throw std::runtime_error(
                    "selected print model double axis does not match the Float32 axis");
            }
            return {
                {"source_double_axis_count", profile.sourceLogExposure.size()},
                {"float32_axis_count", profile.data.logExposure.size()},
                {"model_shape", std::array<int, 2>{3, 3}},
                {"asset_version_token", profile.assetVersionToken}};
        }
        throw std::runtime_error(
            "selected print profile does not own the required density model and source double axis");
    }

    void write_json_document(
        const std::filesystem::path& path,
        const nlohmann::json& document) {
        std::ofstream stream(path, std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("unable to create JSON output " + path.string());
        }
        stream << document.dump(2) << '\n';
        if (!stream) {
            throw std::runtime_error("unable to write JSON output " + path.string());
        }
    }

    Spektrafilm::PrintRecipeBuildResult build_print_recipe_for_profile(
        const std::shared_ptr<const Profiles::ValidatedFilmProfile>& filmProfile,
        const std::shared_ptr<const Profiles::ValidatedPrintProfile>& printProfile,
        double gammaFactor) {
        Spektrafilm::PrintRecipeBuildInput input{};
        input.film.filmProfileKey = filmProfile->info.stock;
        input.film.scanRoute = Spektrafilm::ScanRoute::NegativePrintScan;
        input.film.filmProfile = filmProfile;
        input.film.dirCouplers.active = false;
        const auto& illuminant =
            JuicerProcess::root().assets().illuminant_filter_curves().d55.linear;
        if (illuminant.size() != input.film.referenceIlluminant.size()) {
            throw std::runtime_error("D55 reference illuminant has the wrong shape");
        }
        std::copy(
            illuminant.begin(),
            illuminant.end(),
            input.film.referenceIlluminant.begin());
        input.film.referenceIlluminantValid = true;
        input.printProfileKey = printProfile->info.stock;
        input.printProfile = printProfile;
        input.printGammaFactor = gammaFactor;
        input.glareActive = false;
        input.scannerUnsharpSigmaPx = 0.0f;
        input.scannerUnsharpAmount = 0.0f;
        return Spektrafilm::build_print_render_recipe(input);
    }

    std::string write_model_ingestion_case(
        const Arguments& arguments,
        std::string_view caseId,
        nlohmann::json document,
        bool nonfiniteCenterLiteral,
        Spektrafilm::ProfileCatalog& catalog) {
        const std::filesystem::path caseRoot =
            arguments.outputDirectory / "model-ingestion" / caseId;
        const std::filesystem::path profilesRoot = caseRoot / "profiles";
        std::filesystem::create_directories(profilesRoot);
        std::filesystem::copy_file(
            arguments.resourceRoot / "profiles" / "kodak_portra_400.json",
            profilesRoot / "kodak_portra_400.json");
        std::filesystem::copy_file(
            arguments.resourceRoot / "profiles" / "kodak_portra_endura.json",
            profilesRoot / "kodak_portra_endura.json");
        catalog = Spektrafilm::build_profile_catalog(caseRoot.string());
        if (!catalog.valid) {
            throw std::runtime_error(
                "model-ingestion catalog setup failed: " + catalog.failure);
        }

        const std::filesystem::path printPath =
            profilesRoot / "kodak_portra_endura.json";
        if (!nonfiniteCenterLiteral) {
            write_json_document(printPath, document);
            return printPath.string();
        }
        constexpr std::string_view kMarker = "\"__NONFINITE_CENTER__\"";
        std::string text = document.dump(2);
        const std::size_t marker = text.find(kMarker);
        if (marker == std::string::npos) {
            throw std::runtime_error("nonfinite model-ingestion marker was not found");
        }
        text.replace(marker, kMarker.size(), "1e400");
        std::ofstream stream(printPath, std::ios::trunc);
        stream << text << '\n';
        if (!stream) {
            throw std::runtime_error("unable to write nonfinite model-ingestion case");
        }
        return printPath.string();
    }

    nlohmann::json run_model_ingestion_contracts(
        const Arguments& arguments,
        const std::shared_ptr<const Profiles::ValidatedFilmProfile>& filmProfile,
        const std::shared_ptr<const Profiles::ValidatedPrintProfile>& originalPrintProfile) {
        const nlohmann::json fixture = read_json_file(
            arguments.inputFile.parent_path() / "model_ingestion.json");
        const nlohmann::json original = read_json_file(
            arguments.resourceRoot / "profiles" / "kodak_portra_endura.json");
        nlohmann::json result = nlohmann::json::object();
        for (const auto& fixtureCase : fixture.at("cases")) {
            const std::string id = fixtureCase.at("id").get<std::string>();
            nlohmann::json mutated = original;
            bool nonfiniteCenterLiteral = false;
            if (id == "missing-model" ||
                id == "malformed-unused-print-model-on-direct") {
                mutated["data"].erase("density_curves_model");
            } else if (id == "wrong-model-tag") {
                mutated["data"]["density_curves_model"]["model_type"] =
                    "gaussians";
            } else if (id == "wrong-centers-shape") {
                auto& centers =
                    mutated["data"]["density_curves_model"]["centers"];
                centers.erase(centers.begin() + 2);
            } else if (id == "nonfinite-center") {
                mutated["data"]["density_curves_model"]["centers"][0][0] =
                    "__NONFINITE_CENTER__";
                nonfiniteCenterLiteral = true;
            } else if (id == "nonpositive-sigma") {
                mutated["data"]["density_curves_model"]["sigmas"][0][0] = 0.0;
            } else if (id == "float32-overflow-derived-table") {
                mutated["data"]["density_curves_model"]["amplitudes"][0][0] =
                    3.5e38;
            } else if (id == "source-double-axis-digest-change") {
                const double authored = mutated["data"]["log_exposure"][128].get<double>();
                mutated["data"]["log_exposure"][128] =
                    std::nextafter(authored, std::numeric_limits<double>::infinity());
            } else if (id == "duplicate-axis") {
                mutated["data"]["log_exposure"][128] =
                    mutated["data"]["log_exposure"][127];
            } else {
                throw std::runtime_error("unknown model-ingestion fixture case " + id);
            }

            Spektrafilm::ProfileCatalog catalog;
            (void)write_model_ingestion_case(
                arguments,
                id,
                std::move(mutated),
                nonfiniteCenterLiteral,
                catalog);
            Profiles::ProfileAssetStore store;
            if (id == "malformed-unused-print-model-on-direct") {
                const Profiles::SelectedProfileResult selected =
                    store.selected_profiles_for_route(
                        catalog,
                        Profiles::SelectedProfileRequest{
                            "kodak_portra_400",
                            "kodak_portra_endura",
                            Spektrafilm::ScanRoute::NegativeDirectScan});
                if (!selected.valid || !selected.filmProfile || selected.printProfile) {
                    throw std::runtime_error(
                        "direct route parsed or required an unused malformed print model");
                }
                result[id] = "direct-selection-accepted-without-print-load";
                continue;
            }

            std::string diagnostic;
            const auto loaded = store.load_print_profile_by_key(
                catalog,
                "kodak_portra_endura",
                &diagnostic);
            if (id == "missing-model" || id == "wrong-model-tag" ||
                id == "wrong-centers-shape" || id == "nonfinite-center" ||
                id == "nonpositive-sigma") {
                const std::string expectedField =
                    id == "missing-model"
                        ? "data.density_curves_model"
                    : id == "wrong-model-tag"
                        ? "data.density_curves_model.model_type"
                    : id == "wrong-centers-shape"
                        ? "data.density_curves_model.centers"
                    : id == "nonfinite-center"
                        ? "data.density_curves_model.centers[0][0]"
                        : "data.density_curves_model.sigmas[0][0]";
                if (loaded ||
                    diagnostic.find("profile=kodak_portra_endura") ==
                        std::string::npos ||
                    diagnostic.find("field=" + expectedField) ==
                        std::string::npos) {
                    throw std::runtime_error(
                        "malformed print model did not fail at its exact selected field for " +
                        id + ": " + diagnostic);
                }
                result[id] = diagnostic;
                continue;
            }
            if (!loaded) {
                throw std::runtime_error(
                    "accepted model-ingestion case failed to load for " + id + ": " +
                    diagnostic);
            }
            if (id == "float32-overflow-derived-table") {
                const Spektrafilm::PrintRecipeBuildResult built =
                    build_print_recipe_for_profile(filmProfile, loaded, 1.0);
                if (built.valid ||
                    built.diagnostic.find("field=print_development_table") ==
                        std::string::npos) {
                    throw std::runtime_error(
                        "Float32-overflow print table was published successfully");
                }
                result[id] = built.diagnostic;
            } else if (id == "source-double-axis-digest-change") {
                if (loaded->data.logExposure != originalPrintProfile->data.logExposure ||
                    loaded->sourceLogExposure == originalPrintProfile->sourceLogExposure ||
                    loaded->assetVersionToken == originalPrintProfile->assetVersionToken) {
                    throw std::runtime_error(
                        "adjacent source double did not change print asset identity while retaining the Float32 axis");
                }
                result[id] = loaded->assetVersionToken;
            } else {
                const Spektrafilm::PrintRecipeBuildResult built =
                    build_print_recipe_for_profile(filmProfile, loaded, 1.0);
                if (!built.valid) {
                    throw std::runtime_error(
                        "accepted model-ingestion recipe failed for " + id + ": " +
                        built.diagnostic);
                }
                result[id] = "accepted";
            }
        }

        nlohmann::json signedAndZero = original;
        signedAndZero["data"]["density_curves_model"]["amplitudes"][0] =
            nlohmann::json::array({-0.25, 0.0, 0.25});
        signedAndZero["data"]["density_curves_model"]["amplitudes"][1] =
            nlohmann::json::array({0.0, 0.0, 0.0});
        Spektrafilm::ProfileCatalog signedCatalog;
        (void)write_model_ingestion_case(
            arguments,
            "signed-zero-amplitudes",
            std::move(signedAndZero),
            false,
            signedCatalog);
        Profiles::ProfileAssetStore signedStore;
        std::string signedDiagnostic;
        const auto signedProfile = signedStore.load_print_profile_by_key(
            signedCatalog,
            "kodak_portra_endura",
            &signedDiagnostic);
        if (!signedProfile ||
            !build_print_recipe_for_profile(filmProfile, signedProfile, 1.0).valid) {
            throw std::runtime_error(
                "finite signed/zero print-model amplitudes were rejected: " +
                signedDiagnostic);
        }
        result["signed-zero-amplitudes"] = "accepted";
        return result;
    }

    nlohmann::json run_print_backend(const Arguments& arguments) {
        const nlohmann::json input = read_json_file(arguments.inputFile);
        CudaContext cuda;
        JuicerProcess::root().ensure_bootstrap();
        const JuicerAssets::SelectedProfileResult selected =
            JuicerProcess::root().assets().selected_profiles_for_route(
                JuicerAssets::SelectedProfileRequest{
                    "kodak_portra_400",
                    "kodak_portra_endura",
                    Spektrafilm::ScanRoute::NegativePrintScan});
        if (!selected.valid || !selected.printProfile) {
            throw std::runtime_error(
                "selected print profile loading failed: " + selected.diagnostic);
        }
        nlohmann::json result;
        result["case"] = "print-backend";
        result["selected_print_model"] =
            selected_print_model_contract(*selected.printProfile);
        result["model_ingestion"] =
            run_model_ingestion_contracts(
                arguments,
                selected.filmProfile,
                selected.printProfile);
        result["cdf_cases"] = nlohmann::json::array();
        for (const auto& probe : input.at("cdf_cases")) {
            const Profiles::PrintDensityModel model =
                print_density_model_from_probe(probe);
            const Spektrafilm::ProfilePolarity polarity =
                probe.at("positive").get<bool>()
                    ? Spektrafilm::ProfilePolarity::Positive
                    : Spektrafilm::ProfilePolarity::Negative;
            const double gamma = probe.at("gamma").get<double>();
            const double logExposure = probe.at("log_exposure").get<double>();
            const std::array<double, 3> actualDouble =
                Spektrafilm::evaluate_print_density_sample(
                    model,
                    gamma,
                    polarity,
                    logExposure);
            nlohmann::json row;
            row["id"] = probe.at("id");
            row["actual_float64"] = actualDouble;
            row["actual_float32"] = {
                static_cast<float>(actualDouble[0]),
                static_cast<float>(actualDouble[1]),
                static_cast<float>(actualDouble[2])};
            row["expected_float64"] = probe.at("expected_float64");
            row["expected_float32"] = probe.at("expected_float32");
            result["cdf_cases"].push_back(std::move(row));
        }
        result["sampler_cases"] = nlohmann::json::array();
        for (const auto& probe : input.at("sampler_cases")) {
            const std::vector<float> axis = float_vector(probe.at("axis_float32"));
            const auto& densityRows = probe.at("density_cmy_float32");
            const auto& queryRows = probe.at("queries_cmy_float32");
            if (densityRows.size() != axis.size()) {
                throw std::runtime_error("print sampler table has the wrong length");
            }
            const std::vector<float> densityInterleaved = flatten_triplets(densityRows);
            ParamSnapshot snapshot = default_paper_snapshot();
            snapshot.printProfileKey = probe.at("stock").get<std::string>();
            snapshot.printGammaFactor = probe.at("gamma").get<double>();
            FocusedRenderStateBuildProduct product;
            std::string diagnostic;
            if (!build_print_render_state_product(snapshot, product, diagnostic)) {
                throw std::runtime_error(
                    "production print recipe build failed for " +
                    probe.at("id").get<std::string>() + ": " + diagnostic);
            }
            if (product.recipe.print.develop.gammaFactor !=
                    snapshot.printGammaFactor ||
                product.recipe.print.develop.densityCurvesHash == 0 ||
                product.recipe.print.develop.densityCurves.size() != axis.size() ||
                !product.recipe.profileRoute.printProfile ||
                product.recipe.profileRoute.printProfile->data.logExposure != axis) {
                throw std::runtime_error(
                    "production print recipe has incomplete derived-table identity for " +
                    probe.at("id").get<std::string>());
            }
            for (std::size_t sample = 0; sample < axis.size(); ++sample) {
                for (std::size_t channel = 0; channel < 3u; ++channel) {
                    if (product.recipe.print.develop.densityCurves[sample][channel] !=
                        densityInterleaved[sample * 3u + channel]) {
                        throw std::runtime_error(
                            "production print recipe Float32 table diverged from the pinned conversion for " +
                            probe.at("id").get<std::string>());
                    }
                }
            }
            const std::vector<float> queryInterleaved = flatten_triplets(queryRows);
            const std::size_t queryCount = queryRows.size();
            std::vector<float> actual(queryCount * 3u);
            for (int channel = 0; channel < 3; ++channel) {
                std::vector<float> density(axis.size());
                std::vector<float> queries(queryCount);
                std::vector<float> factors(queryCount, 1.0f);
                for (std::size_t index = 0; index < axis.size(); ++index) {
                    density[index] = densityInterleaved[index * 3u + channel];
                }
                for (std::size_t index = 0; index < queryCount; ++index) {
                    queries[index] = queryInterleaved[index * 3u + channel];
                }
                nlohmann::json channelInput;
                channelInput["id"] = probe.at("id");
                channelInput["classification"] = "independent-float32-table-sampling";
                channelInput["axis_float32"] = axis;
                channelInput["density_float32"] = density;
                channelInput["queries_float32"] = queries;
                channelInput["gamma_float32"] = factors;
                channelInput["domain_begin"] = 0;
                channelInput["domain_end"] = static_cast<int>(axis.size()) - 1;
                channelInput["expected_float64"] = nlohmann::json::array();
                const nlohmann::json channelResult =
                    run_curve_probe_case(channelInput, cuda);
                const std::vector<float> channelActual =
                    channelResult.at("actual_float32").get<std::vector<float>>();
                for (std::size_t index = 0; index < queryCount; ++index) {
                    actual[index * 3u + channel] = channelActual[index];
                }
            }
            nlohmann::json row;
            row["id"] = probe.at("id");
            row["actual_float32"] = triplet_rows(actual);
            row["expected_float64"] = probe.at("expected_float64");
            row["recipe_density_curves_hash"] =
                product.recipe.print.develop.densityCurvesHash;
            row["recipe_hash"] = product.recipe.hash;
            result["sampler_cases"].push_back(std::move(row));
        }
        ParamSnapshot unitSnapshot = default_paper_snapshot();
        ParamSnapshot nearUnitSnapshot = unitSnapshot;
        nearUnitSnapshot.printGammaFactor = std::nextafter(1.0, 2.0);
        FocusedRenderStateBuildProduct unitProduct;
        FocusedRenderStateBuildProduct nearUnitProduct;
        std::string unitDiagnostic;
        std::string nearUnitDiagnostic;
        if (!build_print_render_state_product(
                unitSnapshot,
                unitProduct,
                unitDiagnostic) ||
            !build_print_render_state_product(
                nearUnitSnapshot,
                nearUnitProduct,
                nearUnitDiagnostic) ||
            unitProduct.recipe.hash == nearUnitProduct.recipe.hash ||
            unitProduct.recipe.print.develop.densityCurves !=
                nearUnitProduct.recipe.print.develop.densityCurves) {
            throw std::runtime_error(
                "exact near-one print gamma does not preserve distinct recipe identity over a shared Float32 table");
        }
        if (hash_params(unitSnapshot) == hash_params(nearUnitSnapshot)) {
            throw std::runtime_error(
                "exact near-one print gamma values share a print snapshot hash");
        }
        ParamSnapshot gammaSnapshot = unitSnapshot;
        gammaSnapshot.printGammaFactor = 1.25;
        FocusedRenderStateBuildProduct gammaProduct;
        std::string gammaDiagnostic;
        if (!build_print_render_state_product(
                gammaSnapshot,
                gammaProduct,
                gammaDiagnostic)) {
            throw std::runtime_error(
                "nonunit print descriptor product build failed: " + gammaDiagnostic);
        }
        JuicerCuda::PrintResourceDescriptors unitDescriptors{};
        JuicerCuda::PrintResourceDescriptors gammaDescriptors{};
        std::string descriptorDiagnostic;
        if (!JuicerCuda::build_print_resource_descriptors(
                unitProduct.recipe,
                unitDescriptors,
                descriptorDiagnostic) ||
            !JuicerCuda::build_print_resource_descriptors(
                gammaProduct.recipe,
                gammaDescriptors,
                descriptorDiagnostic)) {
            throw std::runtime_error(
                "print descriptor derivation failed: " + descriptorDiagnostic);
        }
        if (JuicerCuda::PrintProfileTablesDescriptor::kSchemaVersion != 2u ||
            JuicerCuda::PrintBalanceDescriptor::kSchemaVersion != 2u ||
            JuicerCuda::PrintPreflashRawDescriptor::kSchemaVersion != 2u ||
            unitDescriptors.profileTables.densityCurvesHash !=
                unitProduct.recipe.print.develop.densityCurvesHash ||
            gammaDescriptors.profileTables.densityCurvesHash !=
                gammaProduct.recipe.print.develop.densityCurvesHash ||
            unitDescriptors.profileTables.densitySampleCount !=
                unitProduct.recipe.print.develop.densityCurves.size() ||
            gammaDescriptors.profileTables.hash == unitDescriptors.profileTables.hash ||
            gammaDescriptors.balance.printProfileAssetVersionToken !=
                gammaProduct.recipe.profileRoute.printProfileAssetVersionToken ||
            unitDescriptors.balance.hash != gammaDescriptors.balance.hash) {
            throw std::runtime_error(
                "resolved print development did not own table identity while stock balance stayed invariant");
        }
        ParamSnapshot preflashUnitSnapshot = unitSnapshot;
        ParamSnapshot preflashGammaSnapshot = gammaSnapshot;
        preflashUnitSnapshot.printPreflashExposure = 0.1;
        preflashGammaSnapshot.printPreflashExposure = 0.1;
        FocusedRenderStateBuildProduct preflashUnitProduct;
        FocusedRenderStateBuildProduct preflashGammaProduct;
        if (!build_print_render_state_product(
                preflashUnitSnapshot,
                preflashUnitProduct,
                descriptorDiagnostic) ||
            !build_print_render_state_product(
                preflashGammaSnapshot,
                preflashGammaProduct,
                descriptorDiagnostic)) {
            throw std::runtime_error(
                "preflash print product build failed: " + descriptorDiagnostic);
        }
        JuicerCuda::PrintResourceDescriptors preflashUnitDescriptors{};
        JuicerCuda::PrintResourceDescriptors preflashGammaDescriptors{};
        if (!JuicerCuda::build_print_resource_descriptors(
                preflashUnitProduct.recipe,
                preflashUnitDescriptors,
                descriptorDiagnostic) ||
            !JuicerCuda::build_print_resource_descriptors(
                preflashGammaProduct.recipe,
                preflashGammaDescriptors,
                descriptorDiagnostic) ||
            !preflashUnitDescriptors.preflashActive ||
            !preflashGammaDescriptors.preflashActive ||
            preflashUnitDescriptors.preflashRaw.hash == 0 ||
            preflashUnitDescriptors.preflashRaw.hash !=
                preflashGammaDescriptors.preflashRaw.hash ||
            preflashUnitDescriptors.balance.hash !=
                preflashGammaDescriptors.balance.hash ||
            preflashUnitDescriptors.profileTables.hash ==
                preflashGammaDescriptors.profileTables.hash) {
            throw std::runtime_error(
                "print-only gamma changed stock balance/preflash identity or failed to change resolved tables");
        }
        for (double factor : {
                 std::nextafter(0.5, 0.0),
                 std::nextafter(2.0, std::numeric_limits<double>::infinity()),
                 std::numeric_limits<double>::quiet_NaN(),
                 std::numeric_limits<double>::infinity()}) {
            ParamSnapshot invalidSnapshot = unitSnapshot;
            invalidSnapshot.printGammaFactor = factor;
            FocusedRenderStateBuildProduct invalidProduct;
            std::string invalidDiagnostic;
            if (build_print_render_state_product(
                    invalidSnapshot,
                    invalidProduct,
                    invalidDiagnostic) ||
                invalidDiagnostic.find("field=print_gamma_factor") ==
                    std::string::npos) {
                throw std::runtime_error(
                    "unsupported print gamma did not fail at the owning recipe boundary");
            }
        }
        ParamSnapshot directUnit = route_snapshot(
            Spektrafilm::ScanRoute::NegativeDirectScan,
            "kodak_portra_400");
        ParamSnapshot directUnusedInvalid = directUnit;
        directUnusedInvalid.printGammaFactor =
            std::numeric_limits<double>::quiet_NaN();
        FocusedRenderStateBuildProduct directProduct;
        std::string directDiagnostic;
        if (hash_params(directUnit) != hash_params(directUnusedInvalid) ||
            !build_direct_render_state_product(
                directUnusedInvalid,
                directProduct,
                directDiagnostic)) {
            throw std::runtime_error(
                "unused invalid print gamma affected direct recipe identity or construction");
        }
        result["near_one_recipe_identity_distinct"] = true;
        result["near_one_float32_table_shared"] = true;
        result["print_gamma_factor_bounds_enforced"] = true;
        result["direct_unused_print_gamma_ignored"] = true;
        result["print_descriptor_uses_resolved_curve_content"] = true;
        result["stock_balance_identity_is_print_gamma_invariant"] = true;
        result["stock_preflash_identity_is_print_gamma_invariant"] = true;
        write_capture_json(arguments.outputDirectory, result);
        return result;
    }

    nlohmann::json run_direct_route(
        const Arguments& arguments,
        const ParamSnapshot& snapshot,
        std::string_view caseId,
        std::uint64_t submissionOrdinal) {
        constexpr int kComponents = 4;
        const std::size_t pixelCount =
            static_cast<std::size_t>(arguments.width) * arguments.height;
        const std::size_t packedCount = pixelCount * kComponents;
        const std::size_t packedRowBytes =
            static_cast<std::size_t>(arguments.width) * kComponents * sizeof(float);
        const std::vector<float> input =
            read_float_file(arguments.inputFile, packedCount);

        CudaContext cuda;
        JuicerProcess::root().ensure_bootstrap();
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        if (!build_direct_render_state_product(snapshot, product, diagnostic)) {
            throw std::runtime_error("direct route build failed: " + diagnostic);
        }
        Scanner::ScannerSpectralLutDescriptor scannerDescriptor{};
        if (!Scanner::build_direct_scanner_spectral_lut_descriptor(
                Scanner::DirectScannerSpectralLutDescriptorInput{
                    &product.recipe.profileRoute,
                    &product.recipe.densityBounds,
                    &product.recipe.scannerOutput},
                scannerDescriptor,
                diagnostic)) {
            throw std::runtime_error("direct scanner descriptor failed: " + diagnostic);
        }
        Scanner::ScannerPostEffectsDescriptor scannerPost{};
        if (!Scanner::build_scanner_post_effects_descriptor(
                product.recipe.scannerOutput,
                scannerPost,
                diagnostic)) {
            throw std::runtime_error(
                "direct scanner post-effects descriptor failed: " +
                diagnostic);
        }
        const double longEdgePixels =
            static_cast<double>(std::max(arguments.width, arguments.height));
        const float pixelSizeUm = static_cast<float>(
            (static_cast<double>(product.recipe.filmRaw.filmFormatLongEdgeMm) *
             1000.0) /
            longEdgePixels);
        Spektrafilm::SpatialDirDescriptor spatialDir{};
        if (!Spektrafilm::build_spatial_dir_descriptor(
                product.recipe.dirCouplers,
                pixelSizeUm,
                Spektrafilm::DirFrameExtent{
                    0,
                    0,
                    arguments.width,
                    arguments.height},
                Spektrafilm::DirFrameExtent{
                    0,
                    0,
                    arguments.width,
                    arguments.height},
                Spektrafilm::scan_route_label(
                    product.recipe.profileRoute.scanRoute),
                spatialDir)) {
            throw std::runtime_error("direct spatial DIR descriptor failed");
        }
        JuicerProcess::Root::CudaFramePreparationRequest preparation{};
        preparation.recipe = &product.recipe;
        preparation.exposureTables = &product.payload.exposureTables;
        preparation.spdSInv = product.payload.spdSInv.data();
        preparation.filmRawConfig = &product.payload.filmRawConfig;
        preparation.scannerTables = &product.payload.scannerTables;
        preparation.scannerColor = &product.payload.scannerColor;
        preparation.scannerLutDescriptor = &scannerDescriptor;
        preparation.spatialDirDescriptor = &spatialDir;
        preparation.requestedWidth = arguments.width;
        preparation.requestedHeight = arguments.height;

        JuicerCuda::ResourceManager::SubmissionSnapshot submission{};
        submission.instanceToken.value = 0x47414d4d410002ull;
        submission.frameToken.value = submissionOrdinal;
        submission.snapshotId = submissionOrdinal;
        submission.deviceContextKey = cuda.key;
        submission.contextEpoch = 1;
        submission.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
            product.payload.uploadCoreHash,
            product.recipe.dirCouplers.hash,
            product.payload.scannerHash,
            0);
        const auto prepareStart = std::chrono::steady_clock::now();
        JuicerProcess::Root::PreparedCudaFrame frame =
            JuicerProcess::root().prepare_cuda_frame(
                cuda.key,
                submission,
                preparation,
                {},
                cuda.stream,
                diagnostic);
        const auto prepareEnd = std::chrono::steady_clock::now();
        if (!frame.active()) {
            throw std::runtime_error("direct CUDA preparation failed: " + diagnostic);
        }
        const auto prepared = frame.focused_resources();
        if (!prepared.active ||
            prepared.densityBoundsHash != product.recipe.densityBounds.hash ||
            prepared.scannerDescriptorHash != scannerDescriptor.hash) {
            frame.abort();
            throw std::runtime_error("prepared direct identities do not match the recipe");
        }
        const auto focusedWorkspace = frame.workspace_lease();
        if (spatialDir.hash != 0) {
            if (!focusedWorkspace.active()) {
                frame.abort();
                throw std::runtime_error(
                    "direct spatial DIR workspace lease is inactive");
            }
            if (!frame.prepare_spatial_dir_resources(
                    spatialDir,
                    focusedWorkspace,
                    cuda.stream,
                    diagnostic)) {
                frame.abort();
                throw std::runtime_error(
                    "direct spatial DIR resource preparation failed: " +
                    diagnostic);
            }
        }
        Scanner::ScannerColorCorrectionDescriptor scannerCorrection{};
        if (!Scanner::build_direct_scanner_color_correction_descriptor(
                product.recipe,
                product.payload.scannerTables,
                scannerCorrection,
                diagnostic)) {
            frame.abort();
            throw std::runtime_error("direct scanner correction failed: " + diagnostic);
        }
        JuicerCuda::FilmPayloadPack filmPayloads{};
        if (!JuicerCuda::pack_film_payloads(
                product.recipe.filmRaw,
                product.recipe.filmDevelop,
                product.recipe.dirCouplers,
                product.recipe.densityBounds,
                prepared.film,
                nullptr,
                scannerCorrection.exposureScale,
                filmPayloads,
                diagnostic)) {
            frame.abort();
            throw std::runtime_error("direct film payload packing failed: " + diagnostic);
        }

        DeviceBuffer<float> source(packedCount);
        DeviceBuffer<float> stagedOutput(packedCount);
        DeviceBuffer<float> fusedOutput(packedCount);
        DeviceBuffer<float> densityC(pixelCount);
        DeviceBuffer<float> densityM(pixelCount);
        DeviceBuffer<float> densityY(pixelCount);
        DeviceBuffer<float> capturedC(pixelCount);
        DeviceBuffer<float> capturedM(pixelCount);
        DeviceBuffer<float> capturedY(pixelCount);
        DeviceBuffer<float> linearR(pixelCount);
        DeviceBuffer<float> linearG(pixelCount);
        DeviceBuffer<float> linearB(pixelCount);
        require_cuda(
            cudaMemcpyAsync(
                source.get(),
                input.data(),
                source.bytes(),
                cudaMemcpyHostToDevice,
                cuda.stream),
            "direct input upload");
        JuicerCuda::DirectPipelineRunParams run{};
        run.src = source.get();
        run.srcRowBytes = packedRowBytes;
        run.dst = stagedOutput.get();
        run.dstRowBytes = packedRowBytes;
        run.width = arguments.width;
        run.height = arguments.height;
        run.nComponents = kComponents;
        run.filmRaw = filmPayloads.filmRaw;
        run.filmExpose = filmPayloads.filmExposure;
        run.filmDevelop = filmPayloads.filmDevelop;
        bind_scan_stage(prepared, scannerCorrection, run.scanStage);
        if (spatialDir.hash != 0) {
            const auto scratch =
                frame.spatial_dir_scratch(focusedWorkspace);
            const auto resources = frame.spatial_dir_resources(
                focusedWorkspace,
                spatialDir.hash);
            if (!scratch.active || !resources.active) {
                frame.abort();
                throw std::runtime_error(
                    "direct spatial DIR prepared views are inactive");
            }
            JuicerCuda::SpatialDirBuildRequest request{};
            request.planes.rawCorrectionY = scratch.rawCorrectionY;
            request.planes.rawCorrectionM = scratch.rawCorrectionM;
            request.planes.rawCorrectionC = scratch.rawCorrectionC;
            request.planes.filteredCorrectionY = scratch.filteredCorrectionY;
            request.planes.filteredCorrectionM = scratch.filteredCorrectionM;
            request.planes.filteredCorrectionC = scratch.filteredCorrectionC;
            request.planes.filterTemp = scratch.filterTemp;
            request.planes.filterTempM = scratch.filterTempM;
            request.planes.filterTempC = scratch.filterTempC;
            request.gaussian.kernel = resources.gaussian.weights;
            request.gaussian.radius = resources.gaussian.radius;
            request.gaussian.sigma = resources.gaussian.sigma;
            request.gaussian.weight = spatialDir.gaussianWeight;
            for (int index = 0; index < 3; ++index) {
                request.tails[index].kernel =
                    resources.exponential[index].weights;
                request.tails[index].radius =
                    resources.exponential[index].radius;
                request.tails[index].sigma =
                    resources.exponential[index].sigma;
                request.tails[index].weight =
                    spatialDir.exponentialWeights[index];
            }
            request.streamOpaque = cuda.stream;
            require_cuda(
                juicer_cuda_build_direct_spatial_dir(&run, request),
                "direct spatial DIR build launch");
            if (!frame.stage_spatial_dir_cached_log_raw_for_final_develop(
                    focusedWorkspace,
                    cuda.stream,
                    diagnostic)) {
                frame.abort();
                throw std::runtime_error(
                    "direct spatial DIR scratch transition failed: " +
                    diagnostic);
            }
            const auto finalScratch =
                frame.spatial_dir_scratch(focusedWorkspace);
            if (!finalScratch.filteredCorrectionY ||
                !finalScratch.filteredCorrectionM ||
                !finalScratch.filteredCorrectionC) {
                frame.abort();
                throw std::runtime_error(
                    "direct spatial DIR filtered correction is missing");
            }
            if (finalScratch.logRawB || finalScratch.logRawG ||
                finalScratch.logRawR) {
                if (!finalScratch.logRawB || !finalScratch.logRawG ||
                    !finalScratch.logRawR) {
                    frame.abort();
                    throw std::runtime_error(
                        "direct spatial DIR cached log-raw planes are incomplete");
                }
                require_cuda(
                    juicer_cuda_build_direct_spatial_dir_cached_log_raw(
                        &run,
                        finalScratch.logRawB,
                        finalScratch.logRawG,
                        finalScratch.logRawR,
                        cuda.stream),
                    "direct spatial DIR cached log-raw launch");
            }
            run.filmDevelop.spatialDir.active = 1;
            run.filmDevelop.spatialDir.filteredCorrectionY =
                finalScratch.filteredCorrectionY;
            run.filmDevelop.spatialDir.filteredCorrectionM =
                finalScratch.filteredCorrectionM;
            run.filmDevelop.spatialDir.filteredCorrectionC =
                finalScratch.filteredCorrectionC;
            run.filmDevelop.spatialDir.logRawB = finalScratch.logRawB;
            run.filmDevelop.spatialDir.logRawG = finalScratch.logRawG;
            run.filmDevelop.spatialDir.logRawR = finalScratch.logRawR;
        }
        std::string scanError;
        if (!frame.prepare_scan_error_stage(
                run.scanStage.scanErrorFlag,
                cuda.stream,
                scanError)) {
            frame.abort();
            throw std::runtime_error("direct scan error preparation failed: " + scanError);
        }

        const auto renderStart = std::chrono::steady_clock::now();
        require_cuda(
            juicer_cuda_direct_focused_capture_density(
                &run,
                densityC.get(),
                densityM.get(),
                densityY.get(),
                cuda.stream),
            "direct capture density launch");
        require_cuda(
            cudaMemcpyAsync(
                capturedC.get(),
                densityC.get(),
                densityC.bytes(),
                cudaMemcpyDeviceToDevice,
                cuda.stream),
            "direct C capture");
        require_cuda(
            cudaMemcpyAsync(
                capturedM.get(),
                densityM.get(),
                densityM.bytes(),
                cudaMemcpyDeviceToDevice,
                cuda.stream),
            "direct M capture");
        require_cuda(
            cudaMemcpyAsync(
                capturedY.get(),
                densityY.get(),
                densityY.bytes(),
                cudaMemcpyDeviceToDevice,
                cuda.stream),
            "direct Y capture");
        require_cuda(
            juicer_cuda_direct_focused_scan_linear_density_rgb(
                &run,
                densityC.get(),
                densityM.get(),
                densityY.get(),
                linearR.get(),
                linearG.get(),
                linearB.get(),
                nullptr,
                cuda.stream),
            "direct scanner-linear launch");
        require_cuda(
            juicer_cuda_direct_focused_scanner_post_output(
                &run,
                linearR.get(),
                linearG.get(),
                linearB.get(),
                nullptr,
                nullptr,
                0,
                nullptr,
                0,
                0.0f,
                nullptr,
                nullptr,
                nullptr,
                0,
                0,
                cuda.stream),
            "direct scanner output launch");
        JuicerCuda::DirectPipelineRunParams fusedRun = run;
        fusedRun.dst = fusedOutput.get();
        require_cuda(
            juicer_cuda_negative_direct_pipeline(&fusedRun, cuda.stream),
            "complete focused direct launch");
        if (!frame.finalize_scan_error_stage(
                run.scanStage.scanErrorFlag,
                cuda.stream,
                scanError)) {
            frame.abort();
            throw std::runtime_error("direct scan error finalization failed: " + scanError);
        }

        std::vector<float> density(pixelCount * 3u);
        std::vector<float> linear(pixelCount * 3u);
        std::vector<float> output(packedCount);
        std::vector<float> fused(packedCount);
        const auto copyPlane = [&](std::vector<float>& destination,
                                   std::size_t channel,
                                   const DeviceBuffer<float>& sourcePlane) {
            require_cuda(
                cudaMemcpyAsync(
                    destination.data() + channel * pixelCount,
                    sourcePlane.get(),
                    sourcePlane.bytes(),
                    cudaMemcpyDeviceToHost,
                    cuda.stream),
                "direct capture download");
        };
        copyPlane(density, 0, capturedC);
        copyPlane(density, 1, capturedM);
        copyPlane(density, 2, capturedY);
        copyPlane(linear, 0, linearR);
        copyPlane(linear, 1, linearG);
        copyPlane(linear, 2, linearB);
        require_cuda(
            cudaMemcpyAsync(
                output.data(),
                stagedOutput.get(),
                stagedOutput.bytes(),
                cudaMemcpyDeviceToHost,
                cuda.stream),
            "direct staged output download");
        require_cuda(
            cudaMemcpyAsync(
                fused.data(),
                fusedOutput.get(),
                fusedOutput.bytes(),
                cudaMemcpyDeviceToHost,
                cuda.stream),
            "direct fused output download");
        if (!frame.record_use(cuda.stream, diagnostic)) {
            frame.abort();
            throw std::runtime_error("direct use record failed: " + diagnostic);
        }
        if (!frame.finish(cuda.stream, diagnostic)) {
            throw std::runtime_error("direct frame finish failed: " + diagnostic);
        }
        require_cuda(cudaStreamSynchronize(cuda.stream), "direct route completion");
        const auto renderEnd = std::chrono::steady_clock::now();
        const float fusedMaximumDifference =
            finite_rgba_maximum_absolute_difference(
                caseId,
                "output_rgba_interleaved",
                output,
                fused,
                kComponents);
        std::filesystem::create_directories(arguments.outputDirectory);
        write_float_file(arguments.outputDirectory / "film_density_cmy_planar.f32", density);
        write_float_file(arguments.outputDirectory / "scanner_linear_rgb_planar.f32", linear);
        write_float_file(arguments.outputDirectory / "output_rgba_interleaved.f32", output);
        write_float_file(arguments.outputDirectory / "fused_output_rgba_interleaved.f32", fused);
        nlohmann::json result;
        result["case"] = std::string(caseId);
        result["width"] = arguments.width;
        result["height"] = arguments.height;
        result["film_profile_key"] = product.recipe.profileRoute.filmProfileKey;
        result["recipe_hash"] = product.recipe.hash;
        result["film_profile_asset_version_token"] =
            product.recipe.profileRoute.filmProfileAssetVersionToken;
        result["scanner_descriptor_hash"] = scannerDescriptor.hash;
        result["spatial_dir_descriptor_hash"] = spatialDir.hash;
        result["fused_maximum_absolute_difference"] = fusedMaximumDifference;
        result["fused_path_applicable"] = true;
        result["prepare_elapsed_ms"] =
            std::chrono::duration<double, std::milli>(prepareEnd - prepareStart).count();
        result["render_and_capture_elapsed_ms"] =
            std::chrono::duration<double, std::milli>(renderEnd - renderStart).count();
        result["scan_route"] = static_cast<int>(snapshot.scanRoute);
        result["settings"] = capture_effective_settings(
            product.recipe,
            arguments.width,
            arguments.height,
            pixelSizeUm,
            &spatialDir,
            nullptr,
            scannerPost,
            nullptr);
        write_capture_json(arguments.outputDirectory, result);
        return result;
    }

    nlohmann::json run_print_route(
        const Arguments& arguments,
        const ParamSnapshot& snapshot,
        std::string_view caseId,
        std::uint64_t submissionOrdinal,
        const InjectedPrintCurves* injectedPrintCurves = nullptr) {
        constexpr int kComponents = 4;
        const std::size_t pixelCount =
            static_cast<std::size_t>(arguments.width) * arguments.height;
        const std::size_t packedCount = pixelCount * kComponents;
        const std::size_t packedRowBytes =
            static_cast<std::size_t>(arguments.width) * kComponents * sizeof(float);
        const std::vector<float> input =
            read_float_file(arguments.inputFile, packedCount);

        CudaContext cuda;
        JuicerProcess::root().ensure_bootstrap();

        const auto recipeBuildStart = std::chrono::steady_clock::now();
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        if (!build_print_render_state_product(
                snapshot,
                product,
                diagnostic)) {
            throw std::runtime_error("print route build failed: " + diagnostic);
        }
        const auto recipeBuildEnd = std::chrono::steady_clock::now();
        JuicerCuda::PrintResourceDescriptors printDescriptors{};
        if (!JuicerCuda::build_print_resource_descriptors(
                product.recipe,
                printDescriptors,
                diagnostic)) {
            throw std::runtime_error(
                "print resource descriptor build failed: " + diagnostic);
        }
        if (injectedPrintCurves) {
            const std::array<float, 3> expectedGamma{
                injectedPrintCurves->filmGamma,
                injectedPrintCurves->filmGamma,
                injectedPrintCurves->filmGamma};
            if (product.recipe.filmDevelop.densityCurveGamma != expectedGamma) {
                throw std::runtime_error(
                    "film gamma snapshot value did not reach the print recipe");
            }
            if (printDescriptors.balance.filmDevelopRecipeHash !=
                product.recipe.filmDevelop.hash) {
                throw std::runtime_error(
                    "print balance descriptor lost film development identity");
            }
        }

        Scanner::ScannerSpectralLutDescriptor scannerDescriptor{};
        if (!Scanner::build_print_scanner_spectral_lut_descriptor(
                Scanner::PrintScannerSpectralLutDescriptorInput{
                    &product.recipe.profileRoute,
                    &product.recipe.densityBounds,
                    &product.recipe.scannerOutput},
                scannerDescriptor,
                diagnostic)) {
            throw std::runtime_error("scanner descriptor build failed: " + diagnostic);
        }
        Scanner::ScannerPostEffectsDescriptor scannerPost{};
        if (!Scanner::build_scanner_post_effects_descriptor(
                product.recipe.scannerOutput,
                scannerPost,
                diagnostic)) {
            throw std::runtime_error(
                "scanner post-effects descriptor failed: " + diagnostic);
        }

        const double longEdgePixels =
            static_cast<double>(std::max(arguments.width, arguments.height));
        const double pixelSizeUm =
            (static_cast<double>(product.recipe.filmRaw.filmFormatLongEdgeMm) *
             1000.0) /
            longEdgePixels;
        std::optional<Spektrafilm::DiffusionFrameSetDescriptor>
            diffusionFrameSet;
        if (!Spektrafilm::build_diffusion_frame_set_descriptor(
                product.recipe.spatialOptics,
                product.recipe.profileRoute.scanRoute,
                pixelSizeUm,
                Spektrafilm::DiffusionFrameDomain{
                    0,
                    0,
                    arguments.width,
                    arguments.height},
                diffusionFrameSet,
                diagnostic)) {
            throw std::runtime_error(
                "diffusion frame descriptor failed: " + diagnostic);
        }

        JuicerProcess::Root::CudaFramePreparationRequest preparation{};
        preparation.recipe = &product.recipe;
        preparation.exposureTables = &product.payload.exposureTables;
        preparation.spdSInv = product.payload.spdSInv.data();
        preparation.filmRawConfig = &product.payload.filmRawConfig;
        preparation.scannerTables = &product.payload.scannerTables;
        preparation.scannerColor = &product.payload.scannerColor;
        preparation.scannerLutDescriptor = &scannerDescriptor;
        preparation.diffusionFrameSetDescriptor =
            diffusionFrameSet ? &*diffusionFrameSet : nullptr;
        preparation.requestedWidth = arguments.width;
        preparation.requestedHeight = arguments.height;

        JuicerCuda::ResourceManager::SubmissionSnapshot submission{};
        submission.instanceToken.value = 0x47414d4d410001ull;
        submission.frameToken.value = submissionOrdinal;
        submission.snapshotId = submissionOrdinal;
        submission.deviceContextKey = cuda.key;
        submission.contextEpoch = 1;
        submission.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
            product.payload.uploadCoreHash,
            product.recipe.dirCouplers.hash,
            product.payload.scannerHash,
            0);

        const auto prepareStart = std::chrono::steady_clock::now();
        JuicerProcess::Root::PreparedCudaFrame frame =
            JuicerProcess::root().prepare_cuda_frame(
                cuda.key,
                submission,
                preparation,
                {},
                cuda.stream,
                diagnostic);
        const auto prepareEnd = std::chrono::steady_clock::now();
        if (!frame.active()) {
            throw std::runtime_error("CUDA preparation failed: " + diagnostic);
        }

        const auto prepared = frame.focused_resources();
        const auto preparedPrint = frame.print_resources();
        if (!prepared.active || !preparedPrint.active ||
            prepared.densityBoundsHash != product.recipe.densityBounds.hash ||
            prepared.scannerDescriptorHash != scannerDescriptor.hash) {
            frame.abort();
            throw std::runtime_error("prepared route identities do not match the recipe");
        }
        DiffusionStageBinding enlargerDiffusion{};
        if (diffusionFrameSet && diffusionFrameSet->enlarger) {
            if (!bind_enlarger_diffusion_stage(
                    *diffusionFrameSet,
                    frame.diffusion_resources(),
                    enlargerDiffusion,
                    diagnostic)) {
                frame.abort();
                throw std::runtime_error(
                    "enlarger diffusion binding failed: " + diagnostic);
            }
        }

        Scanner::ScannerColorCorrectionDescriptor scannerCorrection{};
        Scanner::PrintCorrectionDerivationInput correctionInput{};
        correctionInput.recipe = &product.recipe;
        correctionInput.scannerTables = &product.payload.scannerTables;
        correctionInput.mainIlluminant = preparedPrint.mainIlluminantHost;
        correctionInput.spectralSampleCount = preparedPrint.spectralSampleCount;
        correctionInput.preflashRawCmy = preparedPrint.preflashRawCmy;
        correctionInput.normalizer = preparedPrint.normalizer;
        if (!Scanner::build_print_scanner_color_correction_descriptor(
                correctionInput,
                scannerCorrection,
                diagnostic)) {
            frame.abort();
            throw std::runtime_error("scanner correction build failed: " + diagnostic);
        }

        JuicerCuda::FilmPayloadPack filmPayloads{};
        if (!JuicerCuda::pack_film_payloads(
                product.recipe.filmRaw,
                product.recipe.filmDevelop,
                product.recipe.dirCouplers,
                product.recipe.enlargerFilmBounds,
                prepared.film,
                nullptr,
                1.0f,
                filmPayloads,
                diagnostic)) {
            frame.abort();
            throw std::runtime_error("film payload packing failed: " + diagnostic);
        }
        if (injectedPrintCurves &&
            (filmPayloads.filmDevelop.gammaFactorB !=
                 injectedPrintCurves->filmGamma ||
             filmPayloads.filmDevelop.gammaFactorG !=
                 injectedPrintCurves->filmGamma ||
             filmPayloads.filmDevelop.gammaFactorR !=
                 injectedPrintCurves->filmGamma)) {
            frame.abort();
            throw std::runtime_error(
                "film gamma recipe value did not reach the packed B/G/R payload");
        }
        JuicerCuda::PrintCudaPayloadPack printPayloads{};
        if (!JuicerCuda::pack_print_cuda_payloads(
                product.recipe.print,
                preparedPrint,
                scannerCorrection.exposureScale,
                printPayloads,
                diagnostic)) {
            frame.abort();
            throw std::runtime_error("print payload packing failed: " + diagnostic);
        }

        DeviceBuffer<float> source(packedCount);
        DeviceBuffer<float> stagedOutput(packedCount);
        DeviceBuffer<float> fusedOutput(packedCount);
        DeviceBuffer<float> densityC(pixelCount);
        DeviceBuffer<float> densityM(pixelCount);
        DeviceBuffer<float> densityY(pixelCount);
        DeviceBuffer<float> filmC(pixelCount);
        DeviceBuffer<float> filmM(pixelCount);
        DeviceBuffer<float> filmY(pixelCount);
        DeviceBuffer<float> printC(pixelCount);
        DeviceBuffer<float> printM(pixelCount);
        DeviceBuffer<float> printY(pixelCount);
        DeviceBuffer<float> linearR(pixelCount);
        DeviceBuffer<float> linearG(pixelCount);
        DeviceBuffer<float> linearB(pixelCount);
        DeviceBuffer<float> printLogExposureR(pixelCount);
        DeviceBuffer<float> printLogExposureG(pixelCount);
        DeviceBuffer<float> printLogExposureB(pixelCount);
        std::unique_ptr<DeviceBuffer<float>> injectedAxis;
        std::unique_ptr<DeviceBuffer<float>> injectedDensityC;
        std::unique_ptr<DeviceBuffer<float>> injectedDensityM;
        std::unique_ptr<DeviceBuffer<float>> injectedDensityY;
        require_cuda(
            cudaMemcpyAsync(
                source.get(),
                input.data(),
                source.bytes(),
                cudaMemcpyHostToDevice,
                cuda.stream),
            "input upload");

        JuicerCuda::PrintPipelineRunParams run{};
        run.src = source.get();
        run.srcRowBytes = packedRowBytes;
        run.dst = stagedOutput.get();
        run.dstRowBytes = packedRowBytes;
        run.width = arguments.width;
        run.height = arguments.height;
        run.nComponents = kComponents;
        run.filmRaw = filmPayloads.filmRaw;
        run.filmExpose = filmPayloads.filmExposure;
        run.filmDevelop = filmPayloads.filmDevelop;
        run.printExpose = printPayloads.expose;
        run.printDevelop = printPayloads.develop;
        bind_scan_stage(prepared, scannerCorrection, run.scanStage);
        if (injectedPrintCurves) {
            const std::size_t sampleCount = injectedPrintCurves->axis.size();
            const auto& selectedAxis =
                product.recipe.profileRoute.printProfile->data.logExposure;
            if (sampleCount == 0 ||
                injectedPrintCurves->densityCmyInterleaved.size() !=
                    sampleCount * 3u ||
                selectedAxis.size() != sampleCount ||
                sampleCount > static_cast<std::size_t>(
                                  std::numeric_limits<int>::max())) {
                frame.abort();
                throw std::runtime_error(
                    "injected print curve shape does not match its axis");
            }
            std::vector<float> densityC(sampleCount);
            std::vector<float> densityM(sampleCount);
            std::vector<float> densityY(sampleCount);
            for (std::size_t sample = 0; sample < sampleCount; ++sample) {
                const float axis = injectedPrintCurves->axis[sample];
                if (!std::isfinite(axis) ||
                    axis != selectedAxis[sample] ||
                    (sample > 0 && axis < injectedPrintCurves->axis[sample - 1])) {
                    frame.abort();
                    throw std::runtime_error(
                        "injected print curve axis does not match the selected Float32 upload axis");
                }
                densityC[sample] =
                    injectedPrintCurves->densityCmyInterleaved[sample * 3u];
                densityM[sample] =
                    injectedPrintCurves->densityCmyInterleaved[sample * 3u + 1u];
                densityY[sample] =
                    injectedPrintCurves->densityCmyInterleaved[sample * 3u + 2u];
                if (!std::isfinite(densityC[sample]) ||
                    !std::isfinite(densityM[sample]) ||
                    !std::isfinite(densityY[sample])) {
                    frame.abort();
                    throw std::runtime_error(
                        "injected print curve contains a nonfinite density");
                }
            }
            injectedAxis =
                std::make_unique<DeviceBuffer<float>>(sampleCount);
            injectedDensityC =
                std::make_unique<DeviceBuffer<float>>(sampleCount);
            injectedDensityM =
                std::make_unique<DeviceBuffer<float>>(sampleCount);
            injectedDensityY =
                std::make_unique<DeviceBuffer<float>>(sampleCount);
            upload(*injectedAxis, injectedPrintCurves->axis, cuda.stream);
            upload(*injectedDensityC, densityC, cuda.stream);
            upload(*injectedDensityM, densityM, cuda.stream);
            upload(*injectedDensityY, densityY, cuda.stream);
            const int count = static_cast<int>(sampleCount);
            run.printDevelop.printDcC =
                {injectedAxis->get(), injectedDensityC->get(), count, 0, count - 1};
            run.printDevelop.printDcM =
                {injectedAxis->get(), injectedDensityM->get(), count, 0, count - 1};
            run.printDevelop.printDcY =
                {injectedAxis->get(), injectedDensityY->get(), count, 0, count - 1};
        }

        std::string scanError;
        if (!frame.prepare_scan_error_stage(
                run.scanStage.scanErrorFlag,
                cuda.stream,
                scanError)) {
            frame.abort();
            throw std::runtime_error("scan error preparation failed: " + scanError);
        }

        const bool fusedApplicable = !diffusionFrameSet.has_value();
        const bool developmentObserverActive =
            diffusionFrameSet && diffusionFrameSet->enlarger.has_value();
        bool developmentObserverExact = false;
        bool developmentObserverPaddedStrideVerified = false;
        std::unique_ptr<DeviceBuffer<float>> observedPrintC;
        std::unique_ptr<DeviceBuffer<float>> observedPrintM;
        std::unique_ptr<DeviceBuffer<float>> observedPrintY;
        std::unique_ptr<DeviceBuffer<float>> observerUnitGamma;
        if (developmentObserverActive) {
            observedPrintC = std::make_unique<DeviceBuffer<float>>(pixelCount);
            observedPrintM = std::make_unique<DeviceBuffer<float>>(pixelCount);
            observedPrintY = std::make_unique<DeviceBuffer<float>>(pixelCount);
            observerUnitGamma =
                std::make_unique<DeviceBuffer<float>>(pixelCount);
            const std::vector<float> unitGamma(pixelCount, 1.0f);
            upload(*observerUnitGamma, unitGamma, cuda.stream);
            developmentObserverPaddedStrideVerified =
                verify_print_develop_observer_padded_stride(
                    run.printDevelop,
                    cuda.stream);
        }
        cudaEvent_t fusedStart = nullptr;
        cudaEvent_t fusedEnd = nullptr;
        if (fusedApplicable) {
            require_cuda(cudaEventCreate(&fusedStart), "fused timing start event creation");
            require_cuda(cudaEventCreate(&fusedEnd), "fused timing end event creation");
        }
        const auto renderStart = std::chrono::steady_clock::now();
        require_cuda(
            juicer_cuda_print_focused_capture_density(
                &run,
                densityC.get(),
                densityM.get(),
                densityY.get(),
                cuda.stream),
            "capture-film density launch");
        require_cuda(
            cudaMemcpyAsync(
                filmC.get(),
                densityC.get(),
                densityC.bytes(),
                cudaMemcpyDeviceToDevice,
                cuda.stream),
            "film C capture");
        require_cuda(
            cudaMemcpyAsync(
                filmM.get(),
                densityM.get(),
                densityM.bytes(),
                cudaMemcpyDeviceToDevice,
                cuda.stream),
            "film M capture");
        require_cuda(
            cudaMemcpyAsync(
                filmY.get(),
                densityY.get(),
                densityY.bytes(),
                cudaMemcpyDeviceToDevice,
                cuda.stream),
            "film Y capture");
        if (diffusionFrameSet && diffusionFrameSet->enlarger) {
            frame.mark_diffusion_work_enqueued();
            JuicerCuda::EnlargerPrintLinearExposurePlanes linearExposure =
                enlarger_linear_planes(enlargerDiffusion.stagePlanes);
            require_cuda(
                juicer_cuda_print_focused_enlarger_linear_exposure(
                    &run,
                    densityC.get(),
                    densityM.get(),
                    densityY.get(),
                    linearExposure,
                    nullptr,
                    cuda.stream),
                "print enlarger-linear exposure launch");
            JuicerCuda::Diffusion::StageLaunchRequest diffusionLaunch{};
            diffusionLaunch.layout = enlargerDiffusion.layout;
            diffusionLaunch.geometry = enlargerDiffusion.geometry;
            diffusionLaunch.fullFrame = enlargerDiffusion.fullFrame;
            diffusionLaunch.spectra = enlargerDiffusion.spectra;
            diffusionLaunch.execution = enlargerDiffusion.execution;
            diffusionLaunch.planes = &enlargerDiffusion.stagePlanes;
            diffusionLaunch.stream = cuda.stream;
            const JuicerCuda::Diffusion::LaunchResult diffusionResult =
                JuicerCuda::Diffusion::launch_stage(diffusionLaunch);
            if (!diffusionResult.ok()) {
                frame.abort();
                throw std::runtime_error(
                    "enlarger diffusion launch failed at " +
                    std::string(
                        diffusionResult.stage ? diffusionResult.stage : "unknown"));
            }
            linearExposure = enlarger_linear_planes(enlargerDiffusion.stagePlanes);
            GammaValidation::PrintDevelopLogExposureLaunch observerLaunch{};
            observerLaunch.linearExposure = linearExposure;
            observerLaunch.logExposureR = printLogExposureR.get();
            observerLaunch.logExposureG = printLogExposureG.get();
            observerLaunch.logExposureB = printLogExposureB.get();
            observerLaunch.width = arguments.width;
            observerLaunch.height = arguments.height;
            require_cuda(
                gamma_validation_launch_print_develop_log_exposure(
                    &observerLaunch,
                    cuda.stream),
                "post-diffusion print-development observer launch");
            const auto launch_observer_curve =
                [&](const JuicerCuda::DeviceCurveView& curve,
                    const DeviceBuffer<float>& logExposure,
                    DeviceBuffer<float>& density) {
                    GammaValidation::CurveProbeLaunch launch{};
                    launch.curve = curve;
                    launch.queries = logExposure.get();
                    launch.gammaFactors = observerUnitGamma->get();
                    launch.results = density.get();
                    launch.count = static_cast<int>(pixelCount);
                    require_cuda(
                        gamma_validation_launch_curve_probe(
                            &launch,
                            cuda.stream),
                        "post-diffusion print-development curve sample");
                };
            launch_observer_curve(
                run.printDevelop.printDcC,
                printLogExposureR,
                *observedPrintC);
            launch_observer_curve(
                run.printDevelop.printDcM,
                printLogExposureG,
                *observedPrintM);
            launch_observer_curve(
                run.printDevelop.printDcY,
                printLogExposureB,
                *observedPrintY);
            require_cuda(
                juicer_cuda_print_focused_develop_from_enlarger_linear(
                    &run,
                    linearExposure,
                    densityC.get(),
                    densityM.get(),
                    densityY.get(),
                    cuda.stream),
                "print development from enlarger-linear exposure launch");
        } else {
            GammaValidation::PrintLogExposureLaunch printLogExposureLaunch{};
            printLogExposureLaunch.expose = run.printExpose;
            printLogExposureLaunch.densityC = densityC.get();
            printLogExposureLaunch.densityM = densityM.get();
            printLogExposureLaunch.densityY = densityY.get();
            printLogExposureLaunch.logExposureR = printLogExposureR.get();
            printLogExposureLaunch.logExposureG = printLogExposureG.get();
            printLogExposureLaunch.logExposureB = printLogExposureB.get();
            printLogExposureLaunch.count = static_cast<int>(pixelCount);
            require_cuda(
                gamma_validation_launch_print_log_exposure(
                    &printLogExposureLaunch,
                    cuda.stream),
                "non-diffused print-development input observer launch");
            require_cuda(
                juicer_cuda_print_focused_continue_from_capture_density(
                    &run,
                    densityC.get(),
                    densityM.get(),
                    densityY.get(),
                    nullptr,
                    cuda.stream),
                "print continuation launch");
        }
        require_cuda(
            cudaMemcpyAsync(
                printC.get(),
                densityC.get(),
                densityC.bytes(),
                cudaMemcpyDeviceToDevice,
                cuda.stream),
            "print C capture");
        require_cuda(
            cudaMemcpyAsync(
                printM.get(),
                densityM.get(),
                densityM.bytes(),
                cudaMemcpyDeviceToDevice,
                cuda.stream),
            "print M capture");
        require_cuda(
            cudaMemcpyAsync(
                printY.get(),
                densityY.get(),
                densityY.bytes(),
                cudaMemcpyDeviceToDevice,
                cuda.stream),
            "print Y capture");
        require_cuda(
            juicer_cuda_print_focused_scan_linear_density_rgb(
                &run,
                densityC.get(),
                densityM.get(),
                densityY.get(),
                linearR.get(),
                linearG.get(),
                linearB.get(),
                nullptr,
                nullptr,
                0,
                0,
                0,
                0.0f,
                0.0f,
                nullptr,
                0,
                cuda.stream),
            "scanner-linear launch");
        require_cuda(
            juicer_cuda_print_focused_scanner_post_output(
                &run,
                linearR.get(),
                linearG.get(),
                linearB.get(),
                nullptr,
                nullptr,
                0,
                nullptr,
                0,
                0.0f,
                nullptr,
                nullptr,
                nullptr,
                0,
                0,
                cuda.stream),
            "scanner output launch");

        if (fusedApplicable) {
            JuicerCuda::PrintPipelineRunParams fusedRun = run;
            fusedRun.dst = fusedOutput.get();
            require_cuda(
                cudaEventRecord(fusedStart, cuda.stream),
                "fused timing start event record");
            require_cuda(
                juicer_cuda_print_focused_pipeline(&fusedRun, cuda.stream),
                "complete focused print launch");
            require_cuda(
                cudaEventRecord(fusedEnd, cuda.stream),
                "fused timing end event record");
        }
        if (!frame.finalize_scan_error_stage(
                run.scanStage.scanErrorFlag,
                cuda.stream,
                scanError)) {
            frame.abort();
            throw std::runtime_error("scan error finalization failed: " + scanError);
        }

        std::vector<float> film(pixelCount * 3u);
        std::vector<float> print(pixelCount * 3u);
        std::vector<float> printLogExposure(pixelCount * 3u);
        std::vector<float> linear(pixelCount * 3u);
        std::vector<float> output(packedCount);
        std::vector<float> fused(packedCount);
        std::vector<float> observedPrint;
        if (developmentObserverActive) {
            observedPrint.resize(pixelCount * 3u);
        }
        const auto copy_plane = [&](std::vector<float>& destination,
                                    std::size_t channel,
                                    const DeviceBuffer<float>& sourcePlane) {
            require_cuda(
                cudaMemcpyAsync(
                    destination.data() + channel * pixelCount,
                    sourcePlane.get(),
                    sourcePlane.bytes(),
                    cudaMemcpyDeviceToHost,
                    cuda.stream),
                "capture download");
        };
        copy_plane(film, 0, filmC);
        copy_plane(film, 1, filmM);
        copy_plane(film, 2, filmY);
        copy_plane(print, 0, printC);
        copy_plane(print, 1, printM);
        copy_plane(print, 2, printY);
        if (developmentObserverActive) {
            copy_plane(observedPrint, 0, *observedPrintC);
            copy_plane(observedPrint, 1, *observedPrintM);
            copy_plane(observedPrint, 2, *observedPrintY);
        }
        copy_plane(printLogExposure, 0, printLogExposureR);
        copy_plane(printLogExposure, 1, printLogExposureG);
        copy_plane(printLogExposure, 2, printLogExposureB);
        copy_plane(linear, 0, linearR);
        copy_plane(linear, 1, linearG);
        copy_plane(linear, 2, linearB);
        require_cuda(
            cudaMemcpyAsync(
                output.data(),
                stagedOutput.get(),
                stagedOutput.bytes(),
                cudaMemcpyDeviceToHost,
                cuda.stream),
            "staged output download");
        if (fusedApplicable) {
            require_cuda(
                cudaMemcpyAsync(
                    fused.data(),
                    fusedOutput.get(),
                    fusedOutput.bytes(),
                    cudaMemcpyDeviceToHost,
                    cuda.stream),
                "fused output download");
        }
        if (!frame.record_use(cuda.stream, diagnostic)) {
            frame.abort();
            throw std::runtime_error("prepared-frame use record failed: " + diagnostic);
        }
        if (!frame.finish(cuda.stream, diagnostic)) {
            throw std::runtime_error("prepared-frame finish failed: " + diagnostic);
        }
        require_cuda(cudaStreamSynchronize(cuda.stream), "route completion");
        const auto renderEnd = std::chrono::steady_clock::now();
        if (developmentObserverActive) {
            if (observedPrint.size() != print.size()) {
                throw std::runtime_error(
                    "post-diffusion print-development observer shape mismatch case=" +
                    std::string(caseId));
            }
            for (std::size_t index = 0; index < print.size(); ++index) {
                const float observed = observedPrint[index];
                const float developed = print[index];
                if (!std::isfinite(observed) || !std::isfinite(developed) ||
                    observed != developed) {
                    throw std::runtime_error(
                        "post-diffusion print-development observer mismatch case=" +
                        std::string(caseId) + " channel=" +
                        std::to_string(index / pixelCount) + " sample=" +
                        std::to_string(index % pixelCount) + " observed=" +
                        std::to_string(observed) + " developed=" +
                        std::to_string(developed));
                }
            }
            developmentObserverExact = true;
        }
        float fusedGpuElapsedMs = 0.0f;
        if (fusedApplicable) {
            require_cuda(
                cudaEventElapsedTime(&fusedGpuElapsedMs, fusedStart, fusedEnd),
                "fused GPU elapsed time");
            require_cuda(cudaEventDestroy(fusedStart), "fused timing start event destroy");
            require_cuda(cudaEventDestroy(fusedEnd), "fused timing end event destroy");
        }

        nlohmann::json fusedMaximumDifference = nullptr;
        if (fusedApplicable) {
            fusedMaximumDifference = finite_rgba_maximum_absolute_difference(
                caseId,
                "output_rgba_interleaved",
                output,
                fused,
                kComponents);
        }
        std::filesystem::create_directories(arguments.outputDirectory);
        write_float_file(arguments.outputDirectory / "film_density_cmy_planar.f32", film);
        write_float_file(
            arguments.outputDirectory / "print_log_exposure_rgb_planar.f32",
            printLogExposure);
        write_float_file(arguments.outputDirectory / "print_density_cmy_planar.f32", print);
        write_float_file(arguments.outputDirectory / "scanner_linear_rgb_planar.f32", linear);
        write_float_file(arguments.outputDirectory / "output_rgba_interleaved.f32", output);
        if (fusedApplicable) {
            write_float_file(
                arguments.outputDirectory / "fused_output_rgba_interleaved.f32",
                fused);
        }

        nlohmann::json result;
        result["case"] = std::string(caseId);
        result["width"] = arguments.width;
        result["height"] = arguments.height;
        result["components"] = kComponents;
        result["film_density_layout"] = "C,M,Y planar float32 little-endian";
        result["print_density_layout"] = "C,M,Y planar float32 little-endian";
        result["print_log_exposure_layout"] =
            "R,G,B sensitive-layer planar float32 little-endian";
        result["print_log_exposure_boundary"] = "print_development_input";
        result["print_log_exposure_source"] =
            developmentObserverActive ? "post_diffusion_planes"
                                      : "non_diffused_continuation";
        result["print_development_observer_exact"] =
            developmentObserverActive
                ? nlohmann::json(developmentObserverExact)
                : nlohmann::json(nullptr);
        result["print_development_observer_padded_stride_verified"] =
            developmentObserverActive
                ? nlohmann::json(developmentObserverPaddedStrideVerified)
                : nlohmann::json(nullptr);
        result["scanner_linear_layout"] = "R,G,B planar float32 little-endian";
        result["output_layout"] = "R,G,B,A interleaved float32 little-endian";
        result["film_profile_key"] = product.recipe.profileRoute.filmProfileKey;
        result["print_profile_key"] = product.recipe.profileRoute.printProfileKey;
        result["recipe_hash"] = product.recipe.hash;
        result["film_profile_asset_version_token"] =
            product.recipe.profileRoute.filmProfileAssetVersionToken;
        result["print_profile_asset_version_token"] =
            product.recipe.profileRoute.printProfileAssetVersionToken;
        result["scanner_descriptor_hash"] = scannerDescriptor.hash;
        result["prepared_print_hash"] = preparedPrint.preparationHash;
        result["fused_maximum_absolute_difference"] = fusedMaximumDifference;
        result["fused_path_applicable"] = fusedApplicable;
        result["fused_gpu_elapsed_ms"] =
            fusedApplicable ? nlohmann::json(fusedGpuElapsedMs)
                            : nlohmann::json(nullptr);
        result["enlarger_diffusion_active"] =
            diffusionFrameSet && diffusionFrameSet->enlarger.has_value();
        result["prepare_elapsed_ms"] =
            std::chrono::duration<double, std::milli>(prepareEnd - prepareStart).count();
        result["recipe_build_elapsed_ms"] =
            std::chrono::duration<double, std::milli>(
                recipeBuildEnd - recipeBuildStart)
                .count();
        result["render_and_capture_elapsed_ms"] =
            std::chrono::duration<double, std::milli>(renderEnd - renderStart).count();
        result["settings"] = capture_effective_settings(
            product.recipe,
            arguments.width,
            arguments.height,
            pixelSizeUm,
            nullptr,
            diffusionFrameSet ? &*diffusionFrameSet : nullptr,
            scannerPost,
            injectedPrintCurves);
        result["film_develop_recipe_hash"] = product.recipe.filmDevelop.hash;
        result["print_develop_recipe_hash"] =
            product.recipe.print.develop.densityCurvesHash;
        result["print_gamma_recipe_factor"] =
            product.recipe.print.develop.gammaFactor;
        result["film_gamma_recipe_rgb"] =
            product.recipe.filmDevelop.densityCurveGamma;
        result["film_gamma_payload_bgr"] = {
            filmPayloads.filmDevelop.gammaFactorB,
            filmPayloads.filmDevelop.gammaFactorG,
            filmPayloads.filmDevelop.gammaFactorR};
        result["print_balance_film_develop_recipe_hash"] =
            printDescriptors.balance.filmDevelopRecipeHash;
        result["print_balance_descriptor_hash"] =
            printDescriptors.balance.hash;
        result["print_balance_normalizer"] = preparedPrint.normalizer;
        result["prepared_print_profile_tables_hash"] =
            preparedPrint.profileTablesHash;
        result["reference_print_curves_injected"] =
            injectedPrintCurves != nullptr;
        if (injectedPrintCurves) {
            result["injected_print_table_id"] = injectedPrintCurves->tableId;
            result["injected_print_sample_count"] =
                injectedPrintCurves->axis.size();
            result["injected_print_axis_first"] =
                injectedPrintCurves->axis.front();
            result["injected_print_axis_last"] =
                injectedPrintCurves->axis.back();
            result["injected_print_sampling_gamma_cmy"] =
                std::array<float, 3>{1.0f, 1.0f, 1.0f};
        }
        std::ofstream manifest(
            arguments.outputDirectory / "capture.json",
            std::ios::trunc);
        manifest << std::setw(2) << result << '\n';
        if (!manifest) {
            throw std::runtime_error("unable to write capture manifest");
        }
        return result;
    }

    nlohmann::json run_default_routes(const Arguments& arguments) {
        nlohmann::json result;
        result["case"] = "default-routes";
        result["routes"] = nlohmann::json::array();
        Arguments routeArguments = arguments;
        routeArguments.outputDirectory =
            arguments.outputDirectory / "negative-direct";
        result["routes"].push_back(run_direct_route(
            routeArguments,
            route_snapshot(
                Spektrafilm::ScanRoute::NegativeDirectScan,
                "kodak_portra_400"),
            "negative-direct",
            101));
        routeArguments.outputDirectory =
            arguments.outputDirectory / "negative-print";
        result["routes"].push_back(run_print_route(
            routeArguments,
            route_snapshot(
                Spektrafilm::ScanRoute::NegativePrintScan,
                "kodak_portra_400"),
            "negative-print",
            102));
        routeArguments.outputDirectory =
            arguments.outputDirectory / "positive-direct";
        result["routes"].push_back(run_direct_route(
            routeArguments,
            route_snapshot(
                Spektrafilm::ScanRoute::PositiveDirectScan,
                "kodak_ektachrome_100"),
            "positive-direct",
            103));
        routeArguments.outputDirectory =
            arguments.outputDirectory / "positive-print";
        result["routes"].push_back(run_print_route(
            routeArguments,
            route_snapshot(
                Spektrafilm::ScanRoute::PositivePrintScan,
                "kodak_ektachrome_100"),
            "positive-print",
            104));
        write_capture_json(arguments.outputDirectory, result);
        return result;
    }

    nlohmann::json run_film_gamma_wiring_contracts() {
        ParamSnapshot directUnit = route_snapshot(
            Spektrafilm::ScanRoute::NegativeDirectScan,
            "kodak_portra_400");
        ParamSnapshot directNonunit = directUnit;
        if (!assign_retained_film_gamma(directUnit, 1.0f) ||
            !assign_retained_film_gamma(directNonunit, 1.25f)) {
            throw std::runtime_error(
                "film gamma snapshot field is not available to the route builders");
        }
        if (hash_params(directUnit) == hash_params(directNonunit)) {
            throw std::runtime_error(
                "film gamma Float32 bits are absent from the snapshot hash");
        }

        FocusedRenderStateBuildProduct directProduct;
        std::string diagnostic;
        if (!build_direct_render_state_product(
                directNonunit,
                directProduct,
                diagnostic) ||
            directProduct.recipe.filmDevelop.densityCurveGamma !=
                std::array<float, 3>{1.25f, 1.25f, 1.25f}) {
            throw std::runtime_error(
                "nonunit film gamma did not reach the direct route recipe: " +
                diagnostic);
        }

        ParamSnapshot printNonunit = default_paper_snapshot();
        if (!assign_retained_film_gamma(printNonunit, 1.25f)) {
            throw std::runtime_error(
                "film gamma snapshot field is not available to the print route builder");
        }
        FocusedRenderStateBuildProduct printProduct;
        if (!build_print_render_state_product(
                printNonunit,
                printProduct,
                diagnostic) ||
            printProduct.recipe.filmDevelop.densityCurveGamma !=
                std::array<float, 3>{1.25f, 1.25f, 1.25f}) {
            throw std::runtime_error(
                "nonunit film gamma did not reach the print route recipe: " +
                diagnostic);
        }
        JuicerCuda::PrintResourceDescriptors printDescriptors{};
        if (!JuicerCuda::build_print_resource_descriptors(
                printProduct.recipe,
                printDescriptors,
                diagnostic) ||
            printDescriptors.balance.filmDevelopRecipeHash !=
                printProduct.recipe.filmDevelop.hash) {
            throw std::runtime_error(
                "nonunit film gamma did not reach print balance identity: " +
                diagnostic);
        }

        auto require_supported = [&](float factor) {
            ParamSnapshot snapshot = directUnit;
            (void)assign_retained_film_gamma(snapshot, factor);
            FocusedRenderStateBuildProduct product;
            std::string localDiagnostic;
            if (!build_direct_render_state_product(
                    snapshot,
                    product,
                    localDiagnostic)) {
                throw std::runtime_error(
                    "supported film gamma rejected: " + localDiagnostic);
            }
        };
        auto require_rejected = [&](float factor) {
            ParamSnapshot snapshot = directUnit;
            (void)assign_retained_film_gamma(snapshot, factor);
            FocusedRenderStateBuildProduct product;
            std::string localDiagnostic;
            if (build_direct_render_state_product(
                    snapshot,
                    product,
                    localDiagnostic) ||
                localDiagnostic.find("film_gamma_factor") == std::string::npos) {
                throw std::runtime_error(
                    "unsupported film gamma did not fail at the owning recipe boundary");
            }
        };
        require_supported(0.05f);
        require_supported(4.0f);
        require_rejected(std::nextafter(0.05f, 0.0f));
        require_rejected(std::nextafter(4.0f, 5.0f));

        ParamSnapshot adjacent = directUnit;
        (void)assign_retained_film_gamma(
            adjacent,
            std::nextafter(1.0f, 2.0f));
        if (hash_params(adjacent) == hash_params(directUnit)) {
            throw std::runtime_error(
                "adjacent retained Float32 film gamma values share a snapshot hash");
        }
        return {
            {"direct_nonunit_recipe_hash", directProduct.recipe.hash},
            {"direct_nonunit_film_develop_hash",
             directProduct.recipe.filmDevelop.hash},
            {"print_nonunit_recipe_hash", printProduct.recipe.hash},
            {"print_nonunit_film_develop_hash",
             printProduct.recipe.filmDevelop.hash},
            {"print_nonunit_balance_hash", printDescriptors.balance.hash},
            {"print_balance_film_develop_recipe_hash",
             printDescriptors.balance.filmDevelopRecipeHash},
            {"supported_range_endpoints_retained", true},
            {"out_of_range_values_rejected", true},
            {"adjacent_float32_hashes_distinct", true}};
    }

    nlohmann::json run_route_probe(
        const Arguments& arguments,
        bool injectReferencePrintCurves) {
        JuicerProcess::root().ensure_bootstrap();
        const nlohmann::json input = read_json_file(arguments.inputFile);
        const std::filesystem::path chartInput =
            input.at("chart_input").get<std::string>();
        nlohmann::json result;
        result["case"] = injectReferencePrintCurves
                             ? "route-probe"
                             : "candidate-route-probe";
        result["film_gamma_wiring"] = run_film_gamma_wiring_contracts();
        result["routes"] = nlohmann::json::array();

        Arguments routeArguments = arguments;
        routeArguments.inputFile = chartInput;
        std::uint64_t ordinal = 501;
        for (const auto& probe : input.at("cases")) {
            const std::string identifier = probe.at("id").get<std::string>();
            InjectedPrintCurves curves;
            curves.filmGamma =
                static_cast<float>(probe.at("film_gamma").get<double>());
            curves.printGamma = probe.at("print_gamma").get<double>();
            if (injectReferencePrintCurves) {
                curves.tableId =
                    probe.at("injected_print_table_id").get<std::string>();
                curves.axis = float_vector(probe.at("axis_float32"));
                curves.densityCmyInterleaved =
                    flatten_triplets(probe.at("density_cmy_float32"));
            }

            ParamSnapshot snapshot = route_snapshot(
                Spektrafilm::ScanRoute::NegativePrintScan,
                probe.at("film_profile").get<std::string>(),
                probe.at("print_profile").get<std::string>());
            if (!assign_retained_film_gamma(snapshot, curves.filmGamma)) {
                throw std::runtime_error(
                    "film gamma snapshot field is not available to route-probe");
            }
            if (!injectReferencePrintCurves) {
                snapshot.printGammaFactor = curves.printGamma;
            }
            routeArguments.outputDirectory =
                arguments.outputDirectory / identifier;
            result["routes"].push_back(run_print_route(
                routeArguments,
                snapshot,
                identifier,
                ordinal++,
                injectReferencePrintCurves ? &curves : nullptr));
        }

        auto find_route = [&](std::string_view identifier)
            -> const nlohmann::json& {
            for (const auto& route : result["routes"]) {
                if (route.at("case").get<std::string>() == identifier) {
                    return route;
                }
            }
            throw std::runtime_error(
                "route-probe is missing required case " +
                std::string(identifier));
        };
        for (std::string_view stock : {"paper", "cine"}) {
            const auto& filmOnly = find_route(
                std::string(stock) + "-film-1.25");
            const auto& printOnly = find_route(
                std::string(stock) + "-print-1.25");
            if (filmOnly.at("film_develop_recipe_hash") ==
                    printOnly.at("film_develop_recipe_hash") ||
                filmOnly.at("print_balance_descriptor_hash") ==
                    printOnly.at("print_balance_descriptor_hash") ||
                filmOnly.at("print_balance_normalizer") ==
                    printOnly.at("print_balance_normalizer")) {
                throw std::runtime_error(
                    "film gamma did not change the existing print balance and mid-gray path for " +
                    std::string(stock));
            }
        }
        result["film_gamma_changes_balance_and_midgray"] = true;
        result["injected_print_curves_are_validation_owned"] =
            injectReferencePrintCurves;
        result["candidate_print_curves_are_recipe_owned"] =
            !injectReferencePrintCurves;
        write_capture_json(arguments.outputDirectory, result);
        return result;
    }

    nlohmann::json run_routes(const Arguments& arguments) {
        nlohmann::json result;
        result["case"] = "routes";
        result["routes"] = nlohmann::json::array();
        Arguments routeArguments = arguments;
        std::uint64_t ordinal = 201;
        auto run_case = [&](std::string_view identifier,
                            ParamSnapshot snapshot) {
            routeArguments.outputDirectory =
                arguments.outputDirectory / std::string(identifier);
            result["routes"].push_back(run_print_route(
                routeArguments,
                snapshot,
                identifier,
                ordinal++));
        };

        ParamSnapshot snapshot = default_paper_snapshot();
        snapshot.cameraExposureCompensationEv = 1.0;
        snapshot.printExposureCompensation = 1;
        run_case("compensation-on-manual-plus-one", snapshot);

        snapshot.printExposureCompensation = 0;
        run_case("compensation-off-manual-plus-one", snapshot);

        snapshot = default_paper_snapshot();
        snapshot.scannerBlackCorrection = 1;
        snapshot.scannerWhiteCorrection = 1;
        run_case("scanner-black-white-correction-on", snapshot);

        snapshot = default_paper_snapshot();
        snapshot.printPreflashExposure = 0.1;
        run_case("preflash-0.1", snapshot);

        snapshot = default_paper_snapshot();
        snapshot.enlargerDiffusion.active = true;
        run_case("enlarger-diffusion-black-pro-mist-half", snapshot);

        snapshot = route_snapshot(
            Spektrafilm::ScanRoute::NegativePrintScan,
            "kodak_vision3_250d",
            "kodak_2383");
        run_case("cine-default", snapshot);

        snapshot = route_snapshot(
            Spektrafilm::ScanRoute::NegativeDirectScan,
            "kodak_portra_400");
        snapshot.couplersActive = 0;
        routeArguments.outputDirectory =
            arguments.outputDirectory / "dir-off-direct";
        result["routes"].push_back(run_direct_route(
            routeArguments,
            snapshot,
            "dir-off-direct",
            ordinal++));

        snapshot = route_snapshot(
            Spektrafilm::ScanRoute::NegativeDirectScan,
            "kodak_portra_400");
        snapshot.couplersDiffusionSizeUm = 2187.5;
        routeArguments.outputDirectory =
            arguments.outputDirectory / "spatial-dir-ramp-edge-direct";
        result["routes"].push_back(run_direct_route(
            routeArguments,
            snapshot,
            "spatial-dir-ramp-edge-direct",
            ordinal++));

        write_capture_json(arguments.outputDirectory, result);
        return result;
    }

    nlohmann::json run_state(const Arguments& arguments) {
        JuicerProcess::root().ensure_bootstrap();
        const nlohmann::json inputs = read_json_file(arguments.inputFile);
        InstanceState state;
        nlohmann::json result;
        result["case"] = "state";
        result["inputs"] = inputs;

        auto assignGamma = [](double film,
                              double print,
                              Spektrafilm::ScanRoute route,
                              ParamSnapshot& snapshot,
                              std::string& diagnostic) {
            snapshot.scanRoute = route;
            return set_gamma_snapshot_values(
                film,
                print,
                snapshot,
                diagnostic);
        };
        ParamSnapshot gammaSnapshot = route_snapshot(
            Spektrafilm::ScanRoute::NegativeDirectScan,
            "kodak_portra_400");
        std::string gammaDiagnostic;
        if (!assignGamma(
                Spektrafilm::kFilmGammaFactorMinimum,
                std::numeric_limits<double>::quiet_NaN(),
                Spektrafilm::ScanRoute::NegativeDirectScan,
                gammaSnapshot,
                gammaDiagnostic) ||
            gammaSnapshot.filmGammaFactor !=
                static_cast<float>(Spektrafilm::kFilmGammaFactorMinimum) ||
            !std::isnan(gammaSnapshot.printGammaFactor)) {
            throw std::runtime_error(
                "valid film endpoint or unused print gamma was not retained");
        }
        const ParamSnapshot directInvalidPrint = gammaSnapshot;
        const std::uint64_t directInvalidPrintHash =
            hash_params(directInvalidPrint);
        ParamSnapshot directUnitPrint = directInvalidPrint;
        directUnitPrint.printGammaFactor = 1.0;
        if (directInvalidPrintHash != hash_params(directUnitPrint)) {
            throw std::runtime_error(
                "unused direct-route print gamma changed executable identity");
        }
        if (assignGamma(
                1.0,
                std::numeric_limits<double>::quiet_NaN(),
                Spektrafilm::ScanRoute::NegativePrintScan,
                gammaSnapshot,
                gammaDiagnostic) ||
            gammaDiagnostic.find("field=print_gamma_factor") ==
                std::string::npos) {
            throw std::runtime_error(
                "route switch did not reject the retained invalid print gamma");
        }

        auto requireInvalidFilm = [&](double film) {
            ParamSnapshot invalid = route_snapshot(
                Spektrafilm::ScanRoute::NegativeDirectScan,
                "kodak_portra_400");
            std::string diagnostic;
            if (assignGamma(
                    film,
                    1.0,
                    invalid.scanRoute,
                    invalid,
                    diagnostic) ||
                diagnostic.find("field=film_gamma_factor") ==
                    std::string::npos) {
                throw std::runtime_error(
                    "invalid authored film gamma crossed the snapshot boundary");
            }
        };
        requireInvalidFilm(std::nextafter(
            Spektrafilm::kFilmGammaFactorMinimum,
            0.0));
        requireInvalidFilm(std::nextafter(
            Spektrafilm::kFilmGammaFactorMaximum,
            5.0));
        requireInvalidFilm(std::numeric_limits<double>::infinity());

        ParamSnapshot gammaEndpoints = default_paper_snapshot();
        if (!assignGamma(
                Spektrafilm::kFilmGammaFactorMaximum,
                Spektrafilm::kPrintGammaFactorMinimum,
                gammaEndpoints.scanRoute,
                gammaEndpoints,
                gammaDiagnostic) ||
            gammaEndpoints.filmGammaFactor !=
                static_cast<float>(Spektrafilm::kFilmGammaFactorMaximum) ||
            gammaEndpoints.printGammaFactor !=
                Spektrafilm::kPrintGammaFactorMinimum ||
            !assignGamma(
                Spektrafilm::kFilmGammaFactorMinimum,
                Spektrafilm::kPrintGammaFactorMaximum,
                gammaEndpoints.scanRoute,
                gammaEndpoints,
                gammaDiagnostic)) {
            throw std::runtime_error(
                "valid authored gamma endpoints were rejected");
        }
        auto requireInvalidPrint = [&](double print) {
            ParamSnapshot invalid = default_paper_snapshot();
            std::string diagnostic;
            if (assignGamma(
                    1.0,
                    print,
                    invalid.scanRoute,
                    invalid,
                    diagnostic) ||
                diagnostic.find("field=print_gamma_factor") ==
                    std::string::npos) {
                throw std::runtime_error(
                    "invalid authored print gamma crossed the snapshot boundary");
            }
        };
        requireInvalidPrint(std::nextafter(
            Spektrafilm::kPrintGammaFactorMinimum,
            0.0));
        requireInvalidPrint(std::nextafter(
            Spektrafilm::kPrintGammaFactorMaximum,
            3.0));
        requireInvalidPrint(-std::numeric_limits<double>::infinity());

        ParamSnapshot equalRetainedA = route_snapshot(
            Spektrafilm::ScanRoute::NegativeDirectScan,
            "kodak_portra_400");
        ParamSnapshot equalRetainedB = equalRetainedA;
        if (!assignGamma(
                1.0,
                1.0,
                equalRetainedA.scanRoute,
                equalRetainedA,
                gammaDiagnostic) ||
            !assignGamma(
                std::nextafter(1.0, 2.0),
                1.0,
                equalRetainedB.scanRoute,
                equalRetainedB,
                gammaDiagnostic) ||
            equalRetainedA.filmGammaFactor != equalRetainedB.filmGammaFactor ||
            hash_params(equalRetainedA) != hash_params(equalRetainedB)) {
            throw std::runtime_error(
                "equal retained Float32 film gamma did not reuse identity");
        }
        ParamSnapshot distinctRetained = equalRetainedA;
        if (!assignGamma(
                static_cast<double>(std::nextafter(1.0f, 2.0f)),
                1.0,
                distinctRetained.scanRoute,
                distinctRetained,
                gammaDiagnostic) ||
            hash_params(equalRetainedA) == hash_params(distinctRetained)) {
            throw std::runtime_error(
                "distinct retained Float32 film gamma reused identity");
        }

        ParamSnapshot nearOnePrintA = default_paper_snapshot();
        ParamSnapshot nearOnePrintB = nearOnePrintA;
        if (!assignGamma(
                1.0,
                1.0,
                nearOnePrintA.scanRoute,
                nearOnePrintA,
                gammaDiagnostic) ||
            !assignGamma(
                1.0,
                std::nextafter(1.0, 2.0),
                nearOnePrintB.scanRoute,
                nearOnePrintB,
                gammaDiagnostic) ||
            hash_params(nearOnePrintA) == hash_params(nearOnePrintB)) {
            throw std::runtime_error(
                "near-one retained print doubles did not keep exact identity");
        }
        result["gamma_snapshot_assignment"] = {
            {"authored_film_double_validated_before_float32", true},
            {"authored_range_endpoints_retained", true},
            {"equal_retained_film_identity_reused", true},
            {"distinct_retained_film_identity_changed", true},
            {"near_one_print_identity_distinct", true},
            {"direct_unused_invalid_print_retained", true},
            {"route_switch_rejects_invalid_print", true}};

        const PendingRenderAdmissionResult uninitialized =
            admit_pending_render_state(state);
        if (uninitialized.status !=
            PendingRenderAdmissionStatus::NeedsSnapshotAcquisition) {
            throw std::runtime_error("uninitialized state was unexpectedly admitted");
        }
        result["uninitialized_status"] =
            static_cast<int>(uninitialized.status);

        ParamSnapshot direct = route_snapshot(
            Spektrafilm::ScanRoute::NegativeDirectScan,
            "kodak_portra_400");
        {
            std::lock_guard<std::mutex> lock(state.pending.m);
            state.pending.value = PendingParamsState::Valid{
                direct,
                hash_params(direct)};
        }
        const PendingRenderAdmissionResult admittedDirect =
            admit_pending_render_state(state);
        if (admittedDirect.status !=
                PendingRenderAdmissionStatus::AdmittedDirect ||
            !admittedDirect.directState) {
            throw std::runtime_error(
                "valid direct snapshot was not admitted status=" +
                std::to_string(static_cast<int>(admittedDirect.status)) +
                " diagnostic=" + admittedDirect.diagnostic);
        }
        const std::shared_ptr<const DirectRenderState> retainedDirect =
            admittedDirect.directState;
        result["direct_hash"] = hash_params(direct);
        result["direct_build_counter"] = retainedDirect->buildCounter;

        ParamSnapshot directPrintEdit = direct;
        directPrintEdit.printGammaFactor = 1.75;
        {
            std::lock_guard<std::mutex> lock(state.pending.m);
            state.pending.value = PendingParamsState::Valid{
                directPrintEdit,
                hash_params(directPrintEdit)};
        }
        const PendingRenderAdmissionResult admittedUnusedPrintEdit =
            admit_pending_render_state(state);
        if (admittedUnusedPrintEdit.status !=
                PendingRenderAdmissionStatus::AdmittedDirect ||
            !admittedUnusedPrintEdit.directState ||
            admittedUnusedPrintEdit.directState->buildCounter !=
                retainedDirect->buildCounter) {
            throw std::runtime_error(
                "unused direct-route print edit rebuilt executable state");
        }
        result["direct_unused_print_edit_reused_build"] = true;

        {
            std::lock_guard<std::mutex> lock(state.pending.m);
            state.pending.value = PendingParamsState::InvalidSnapshotControls{
                "GammaBaselineInvalidApplicableControl"};
        }
        const PendingRenderAdmissionResult invalid =
            admit_pending_render_state(state);
        if (invalid.status !=
                PendingRenderAdmissionStatus::InvalidSnapshotControls ||
            invalid.diagnostic != "GammaBaselineInvalidApplicableControl") {
            throw std::runtime_error("invalid pending controls did not block admission");
        }
        result["invalid_blocks_old_publication"] = true;

        ParamSnapshot print = default_paper_snapshot();
        {
            std::lock_guard<std::mutex> lock(state.pending.m);
            state.pending.value = PendingParamsState::Valid{
                print,
                hash_params(print)};
        }
        const PendingRenderAdmissionResult admittedPrint =
            admit_pending_render_state(state);
        if (admittedPrint.status !=
                PendingRenderAdmissionStatus::AdmittedPrint ||
            !admittedPrint.printState ||
            !retainedDirect->recipe.directStructuralReady) {
            throw std::runtime_error(
                "route replacement did not retain admitted direct state");
        }
        result["print_hash"] = hash_params(print);
        result["print_build_counter"] = admittedPrint.printState->buildCounter;
        result["retained_admitted_direct_state_survived"] = true;
        write_capture_json(arguments.outputDirectory, result);
        return result;
    }

    nlohmann::json run_print_preflash_host_cache_transitions() {
        CudaContext cuda;
        JuicerProcess::root().ensure_bootstrap();
        ParamSnapshot unitSnapshot = default_paper_snapshot();
        unitSnapshot.printPreflashExposure = 0.1;
        ParamSnapshot gammaSnapshot = unitSnapshot;
        gammaSnapshot.printGammaFactor = 1.25;

        FocusedRenderStateBuildProduct unitProduct;
        FocusedRenderStateBuildProduct gammaProduct;
        std::string diagnostic;
        if (!build_print_render_state_product(
                unitSnapshot,
                unitProduct,
                diagnostic)) {
            throw std::runtime_error(
                "preflash host-cache unit product build failed: " + diagnostic);
        }
        if (!build_print_render_state_product(
                gammaSnapshot,
                gammaProduct,
                diagnostic)) {
            throw std::runtime_error(
                "preflash host-cache gamma product build failed: " + diagnostic);
        }
        JuicerCuda::PrintResourceDescriptors unitDescriptors{};
        JuicerCuda::PrintResourceDescriptors gammaDescriptors{};
        if (!JuicerCuda::build_print_resource_descriptors(
                unitProduct.recipe,
                unitDescriptors,
                diagnostic) ||
            !JuicerCuda::build_print_resource_descriptors(
                gammaProduct.recipe,
                gammaDescriptors,
                diagnostic)) {
            throw std::runtime_error(
                "preflash host-cache descriptor build failed: " + diagnostic);
        }
        if (!unitDescriptors.preflashActive ||
            !gammaDescriptors.preflashActive ||
            unitDescriptors.profileTables.hash ==
                gammaDescriptors.profileTables.hash ||
            unitDescriptors.preflashIlluminant.hash !=
                gammaDescriptors.preflashIlluminant.hash ||
            unitDescriptors.preflashRaw.hash != gammaDescriptors.preflashRaw.hash) {
            throw std::runtime_error(
                "preflash host-cache products do not isolate creative print tables");
        }

        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;
        require_cuda(
            cudaMemGetInfo(&freeBytes, &totalBytes),
            "preflash host-cache device-memory query");
        if (freeBytes == 0 || totalBytes == 0) {
            throw std::runtime_error(
                "preflash host-cache device-memory budget is empty");
        }
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger> ledger =
            JuicerCuda::DeviceAllocationLedger::create(
                JuicerCuda::DeviceLedgerBudget{
                    cuda.key.deviceId,
                    static_cast<std::uint64_t>(totalBytes)},
                diagnostic);
        if (!ledger ||
            !ledger->bind_or_validate_cap(
                static_cast<std::uint64_t>(
                    std::min(freeBytes, totalBytes)),
                diagnostic)) {
            throw std::runtime_error(
                "preflash host-cache ledger creation failed: " + diagnostic);
        }
        using ResourceOwner = std::unique_ptr<
            JuicerCuda::Resources,
            void (*)(JuicerCuda::Resources*)>;
        ResourceOwner resources(
            JuicerCuda::create(
                cuda.key,
                1,
                JuicerCuda::Resources::kAllocationOwnershipSchemaVersion,
                ledger,
                diagnostic),
            &JuicerCuda::destroy);
        if (!resources) {
            throw std::runtime_error(
                "preflash host-cache owner creation failed: " + diagnostic);
        }

        struct Observation {
            std::array<float, Spectral::kNumSamples> hostSpectrum{};
            std::array<float, 3> raw{};
            const float* deviceSpectrum = nullptr;
            const float* curveAxis = nullptr;
            const float* curveDensity = nullptr;
            std::uint64_t illuminantHash = 0;
            std::uint64_t rawHash = 0;
            std::uint64_t profileTablesHash = 0;
            int spectrumCount = 0;
            bool hostValid = false;
            bool rawValid = false;
        };
        const std::array<const FocusedRenderStateBuildProduct*, 5> products{
            &unitProduct,
            &gammaProduct,
            &gammaProduct,
            &unitProduct,
            &unitProduct};
        const std::array<std::uint64_t, 5> expectedCreativeHashes{
            unitDescriptors.profileTables.hash,
            gammaDescriptors.profileTables.hash,
            gammaDescriptors.profileTables.hash,
            unitDescriptors.profileTables.hash,
            unitDescriptors.profileTables.hash};
        std::array<Observation, 5> observations{};
        for (std::size_t step = 0; step < products.size(); ++step) {
            JuicerCuda::PrintResourcePreparation request{};
            request.recipe = &products[step]->recipe;
            request.assets = &JuicerProcess::root().assets();
            if (!JuicerCuda::prepare_print_resources(
                    *resources,
                    request,
                    cuda.stream,
                    diagnostic)) {
                throw std::runtime_error(
                    "preflash host-cache preparation step=" +
                    std::to_string(step) + " failed: " + diagnostic);
            }

            Observation& observation = observations[step];
            {
                std::lock_guard<std::mutex> lock(resources->m);
                observation.hostSpectrum =
                    resources->printPreflashIllumFilteredHost;
                std::copy_n(
                    resources->printPreflashRaw,
                    observation.raw.size(),
                    observation.raw.begin());
                observation.deviceSpectrum =
                    resources->printPreflashIllumFiltered;
                observation.curveAxis = resources->printDcC.x;
                observation.curveDensity = resources->printDcC.y;
                observation.illuminantHash =
                    resources->printPreflashIlluminantDescriptorHash;
                observation.rawHash =
                    resources->printPreflashRawDescriptorHash;
                observation.profileTablesHash =
                    resources->printProfileTablesDescriptorHash;
                observation.spectrumCount =
                    resources->printPreflashIllumK;
                observation.hostValid =
                    resources->printPreflashIllumFilteredHostValid;
                observation.rawValid = resources->printPreflashValid;
            }
            float energy = 0.0f;
            for (float value : observation.hostSpectrum) {
                if (!(std::isfinite(value) && value >= 0.0f)) {
                    throw std::runtime_error(
                        "preflash host-cache spectrum is invalid step=" +
                        std::to_string(step));
                }
                energy += value;
            }
            if (!observation.hostValid || !observation.rawValid ||
                !observation.deviceSpectrum || !observation.curveAxis ||
                !observation.curveDensity ||
                observation.spectrumCount != Spectral::kNumSamples ||
                !(std::isfinite(energy) && energy > 0.0f) ||
                observation.profileTablesHash != expectedCreativeHashes[step]) {
                throw std::runtime_error(
                    "preflash host-cache state is incomplete step=" +
                    std::to_string(step));
            }
            for (float value : observation.raw) {
                if (!std::isfinite(value)) {
                    throw std::runtime_error(
                        "preflash host-cache raw value is nonfinite step=" +
                        std::to_string(step));
                }
            }
            if (step > 0 &&
                (observation.hostSpectrum != observations[0].hostSpectrum ||
                 observation.raw != observations[0].raw ||
                 observation.deviceSpectrum != observations[0].deviceSpectrum ||
                 observation.illuminantHash != observations[0].illuminantHash ||
                 observation.rawHash != observations[0].rawHash)) {
                throw std::runtime_error(
                    "print gamma changed stock-anchored preflash cache step=" +
                    std::to_string(step));
            }
            std::array<float, Spectral::kNumSamples> deviceSpectrum{};
            require_cuda(
                cudaMemcpyAsync(
                    deviceSpectrum.data(),
                    observation.deviceSpectrum,
                    deviceSpectrum.size() * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    cuda.stream),
                "preflash host-cache device spectrum download");
            require_cuda(
                cudaStreamSynchronize(cuda.stream),
                "preflash host-cache device spectrum completion");
            if (deviceSpectrum != observation.hostSpectrum) {
                throw std::runtime_error(
                    "preflash host/device spectrum mismatch step=" +
                    std::to_string(step));
            }
        }
        if (observations[0].profileTablesHash ==
                observations[1].profileTablesHash ||
            observations[1].profileTablesHash !=
                observations[2].profileTablesHash ||
            observations[0].profileTablesHash !=
                observations[3].profileTablesHash ||
            observations[3].profileTablesHash !=
                observations[4].profileTablesHash) {
            throw std::runtime_error(
                "preflash host-cache creative table identity transition failed");
        }

        if (!JuicerCuda::drain_for_context_retire(*resources, diagnostic)) {
            throw std::runtime_error(
                "preflash host-cache owner drain failed: " + diagnostic);
        }
        resources.reset();
        const JuicerCuda::DeviceLedgerSnapshot ledgerAfter = ledger->snapshot();
        if (ledgerAfter.reservedBytes != 0 ||
            ledgerAfter.committedBytes != 0 ||
            ledgerAfter.retiringBytes != 0 ||
            ledgerAfter.chargedBytes != 0 || ledgerAfter.recordCount != 0) {
            throw std::runtime_error(
                "preflash host-cache owner retained ledger charges after destroy");
        }
        return {
            {"sequence", "1 -> 1.25 -> unchanged 1.25 -> 1 -> unchanged 1"},
            {"host_spectrum_valid_after_every_preparation", true},
            {"host_spectrum_exactly_stock_anchored", true},
            {"preflash_hashes_raw_and_storage_invariant", true},
            {"creative_table_identity_transition_verified", true},
            {"device_spectrum_matches_host_after_every_preparation", true},
            {"local_owner_drained", true},
            {"ledger_zero_after_destroy", true},
            {"unit_profile_tables_hash", observations[0].profileTablesHash},
            {"gamma_profile_tables_hash", observations[1].profileTablesHash},
            {"preflash_illuminant_hash", observations[0].illuminantHash},
            {"preflash_raw_hash", observations[0].rawHash},
            {"device_memory_total_bytes", totalBytes},
            {"device_memory_free_bytes", freeBytes}};
    }

    nlohmann::json run_print_table_resource_transitions() {
        CudaContext cuda;
        JuicerProcess::root().ensure_bootstrap();
        cudaStream_t streamB = nullptr;
        require_cuda(
            cudaStreamCreateWithFlags(&streamB, cudaStreamNonBlocking),
            "gamma transition stream creation");
        try {
            ParamSnapshot unitSnapshot = default_paper_snapshot();
            unitSnapshot.printPreflashExposure = 0.1;
            ParamSnapshot gammaSnapshot = unitSnapshot;
            gammaSnapshot.printGammaFactor = 1.25;
            FocusedRenderStateBuildProduct unitProduct;
            FocusedRenderStateBuildProduct gammaProduct;
            std::string diagnostic;
            if (!build_print_render_state_product(
                    unitSnapshot,
                    unitProduct,
                    diagnostic)) {
                throw std::runtime_error(
                    "unit gamma transition product build failed: " + diagnostic);
            }
            if (!build_print_render_state_product(
                    gammaSnapshot,
                    gammaProduct,
                    diagnostic)) {
                throw std::runtime_error(
                    "nonunit gamma transition product build failed: " +
                    diagnostic);
            }

            auto prepare = [&](const FocusedRenderStateBuildProduct& product,
                               cudaStream_t stream,
                               std::uint64_t ordinal,
                               JuicerCuda::PrintResourceDescriptors& descriptors) {
                if (!JuicerCuda::build_print_resource_descriptors(
                        product.recipe,
                        descriptors,
                        diagnostic)) {
                    throw std::runtime_error(
                        "gamma transition descriptor build failed: " + diagnostic);
                }
                Scanner::ScannerSpectralLutDescriptor scannerDescriptor{};
                if (!Scanner::build_print_scanner_spectral_lut_descriptor(
                        Scanner::PrintScannerSpectralLutDescriptorInput{
                            &product.recipe.profileRoute,
                            &product.recipe.densityBounds,
                            &product.recipe.scannerOutput},
                        scannerDescriptor,
                        diagnostic)) {
                    throw std::runtime_error(
                        "gamma transition scanner descriptor failed: " +
                        diagnostic);
                }
                JuicerProcess::Root::CudaFramePreparationRequest preparation{};
                preparation.recipe = &product.recipe;
                preparation.exposureTables = &product.payload.exposureTables;
                preparation.spdSInv = product.payload.spdSInv.data();
                preparation.filmRawConfig = &product.payload.filmRawConfig;
                preparation.scannerTables = &product.payload.scannerTables;
                preparation.scannerColor = &product.payload.scannerColor;
                preparation.scannerLutDescriptor = &scannerDescriptor;
                preparation.requestedWidth = 16;
                preparation.requestedHeight = 4;

                JuicerCuda::ResourceManager::SubmissionSnapshot submission{};
                submission.instanceToken.value = 0x47414d4d410004ull;
                submission.frameToken.value = ordinal;
                submission.snapshotId = ordinal;
                submission.deviceContextKey = cuda.key;
                submission.contextEpoch = 1;
                submission.keyDigests =
                    JuicerCuda::ResourceManager::make_key_digests(
                        product.payload.uploadCoreHash,
                        product.recipe.dirCouplers.hash,
                        product.payload.scannerHash,
                        0);
                JuicerProcess::Root::PreparedCudaFrame frame =
                    JuicerProcess::root().prepare_cuda_frame(
                        cuda.key,
                        submission,
                        preparation,
                        {},
                        stream,
                        diagnostic);
                if (!frame.active()) {
                    throw std::runtime_error(
                        "gamma transition preparation failed: " + diagnostic);
                }
                const JuicerCuda::PrintPreparedView view =
                    frame.print_resources();
                if (!view.active ||
                    view.profileTablesHash != descriptors.profileTables.hash ||
                    view.balanceHash != descriptors.balance.hash ||
                    view.preflashRawHash != descriptors.preflashRaw.hash) {
                    frame.abort();
                    throw std::runtime_error(
                        "gamma transition prepared identities do not match descriptors");
                }
                return frame;
            };

            auto sameCurveStorage = [](const JuicerCuda::DeviceCurveView& left,
                                       const JuicerCuda::DeviceCurveView& right) {
                return left.x == right.x && left.y == right.y && left.n == right.n;
            };
            auto sameDensityStorage = [&](const JuicerCuda::PrintPreparedView& left,
                                          const JuicerCuda::PrintPreparedView& right) {
                return sameCurveStorage(left.printDcC, right.printDcC) &&
                       sameCurveStorage(left.printDcM, right.printDcM) &&
                       sameCurveStorage(left.printDcY, right.printDcY);
            };
            auto distinctDensityStorage = [&](const JuicerCuda::PrintPreparedView& left,
                                              const JuicerCuda::PrintPreparedView& right) {
                return left.printDcC.x != right.printDcC.x &&
                       left.printDcC.y != right.printDcC.y &&
                       left.printDcM.x != right.printDcM.x &&
                       left.printDcM.y != right.printDcM.y &&
                       left.printDcY.x != right.printDcY.x &&
                       left.printDcY.y != right.printDcY.y;
            };
            auto samePreflashValues = [](const JuicerCuda::PrintPreparedView& left,
                                         const JuicerCuda::PrintPreparedView& right) {
                return left.preflashRawCmy[0] == right.preflashRawCmy[0] &&
                       left.preflashRawCmy[1] == right.preflashRawCmy[1] &&
                       left.preflashRawCmy[2] == right.preflashRawCmy[2] &&
                       left.normalizer == right.normalizer;
            };

            DeviceBuffer<float> query(1);
            DeviceBuffer<float> gammaFactor(1);
            DeviceBuffer<float> resultA(1);
            DeviceBuffer<float> resultB(1);
            DeviceBuffer<float> resultC(1);
            DeviceBuffer<float> resultD(1);
            DeviceBuffer<float> resultRecovered(1);
            DeviceBuffer<unsigned long long> delayElapsed(1);
            const std::vector<float> queryHost{
                unitProduct.recipe.profileRoute.printProfile->data.logExposure.front()};
            const std::vector<float> unitSamplingGamma{1.0f};
            upload(query, queryHost, cuda.stream);
            upload(gammaFactor, unitSamplingGamma, cuda.stream);

            auto launchSample = [&](const JuicerCuda::DeviceCurveView& curve,
                                    DeviceBuffer<float>& output,
                                    cudaStream_t stream) {
                GammaValidation::CurveProbeLaunch launch{};
                launch.curve = curve;
                launch.queries = query.get();
                launch.gammaFactors = gammaFactor.get();
                launch.results = output.get();
                launch.count = 1;
                require_cuda(
                    gamma_validation_launch_curve_probe(&launch, stream),
                    "gamma transition curve sample");
            };
            auto prepareScanStage = [&](
                                        JuicerProcess::Root::PreparedCudaFrame& frame,
                                        cudaStream_t stream) {
                int* scanErrorFlag = nullptr;
                if (!frame.prepare_scan_error_stage(
                        scanErrorFlag,
                        stream,
                        diagnostic)) {
                    throw std::runtime_error(
                        "gamma transition scan stage preparation failed: " +
                        diagnostic);
                }
                return scanErrorFlag;
            };
            auto finishFrame = [&](JuicerProcess::Root::PreparedCudaFrame& frame,
                                   cudaStream_t stream,
                                   int* scanErrorFlag) {
                if (!frame.finalize_scan_error_stage(
                        scanErrorFlag,
                        stream,
                        diagnostic) ||
                    !frame.record_use(stream, diagnostic) ||
                    !frame.finish(stream, diagnostic)) {
                    throw std::runtime_error(
                        "gamma transition frame finish failed: " + diagnostic);
                }
            };

            JuicerCuda::PrintResourceDescriptors unitDescriptors{};
            JuicerProcess::Root::PreparedCudaFrame warmup = prepare(
                unitProduct,
                cuda.stream,
                600,
                unitDescriptors);
            int* warmupScanError = prepareScanStage(warmup, cuda.stream);
            launchSample(
                warmup.print_resources().printDcC,
                resultC,
                cuda.stream);
            finishFrame(warmup, cuda.stream, warmupScanError);
            require_cuda(
                cudaStreamSynchronize(cuda.stream),
                "gamma transition event-pool warmup completion");
            JuicerProcess::Root::PreparedCudaFrame frameA = prepare(
                unitProduct,
                cuda.stream,
                601,
                unitDescriptors);
            const JuicerCuda::PrintPreparedView viewA = frameA.print_resources();
            int* scanErrorA = prepareScanStage(frameA, cuda.stream);
            constexpr unsigned long long kDelayCycles = 750'000'000ull;
            require_cuda(
                gamma_validation_launch_delay(
                    kDelayCycles,
                    delayElapsed.get(),
                    cuda.stream),
                "gamma transition queued delay");
            launchSample(viewA.printDcC, resultA, cuda.stream);
            if (!frameA.finalize_scan_error_stage(
                    scanErrorA,
                    cuda.stream,
                    diagnostic)) {
                throw std::runtime_error(
                    "gamma transition frame A scan finalization failed: " +
                    diagnostic);
            }
            const cudaError_t beforeFinishStatus = cudaStreamQuery(cuda.stream);
            if (!frameA.record_use(cuda.stream, diagnostic)) {
                throw std::runtime_error(
                    "gamma transition frame A use record failed: " + diagnostic);
            }
            const cudaError_t afterRecordStatus = cudaStreamQuery(cuda.stream);
            if (!frameA.finish(cuda.stream, diagnostic)) {
                throw std::runtime_error(
                    "gamma transition frame A finish failed: " + diagnostic);
            }
            const cudaError_t afterFinishStatus = cudaStreamQuery(cuda.stream);
            if (beforeFinishStatus != cudaErrorNotReady ||
                afterRecordStatus != cudaErrorNotReady ||
                afterFinishStatus != cudaErrorNotReady) {
                throw std::runtime_error(
                    "gamma transition frame A was not queued across finish before=" +
                    std::to_string(static_cast<int>(beforeFinishStatus)) +
                    " recorded=" +
                    std::to_string(static_cast<int>(afterRecordStatus)) +
                    " after=" +
                    std::to_string(static_cast<int>(afterFinishStatus)));
            }

            JuicerCuda::PrintResourceDescriptors gammaDescriptors{};
            JuicerProcess::Root::PreparedCudaFrame frameB = prepare(
                gammaProduct,
                streamB,
                602,
                gammaDescriptors);
            const JuicerCuda::PrintPreparedView viewB = frameB.print_resources();
            int* scanErrorB = prepareScanStage(frameB, streamB);
            if (!sameDensityStorage(viewA, viewB) ||
                viewA.profileTablesHash == viewB.profileTablesHash ||
                viewA.balanceHash != viewB.balanceHash ||
                viewA.preflashRawHash != viewB.preflashRawHash ||
                !samePreflashValues(viewA, viewB)) {
                frameB.abort();
                throw std::runtime_error(
                    "equal-size gamma transition violated storage reuse or stock-derived invariance");
            }
            launchSample(viewB.printDcC, resultB, streamB);
            finishFrame(frameB, streamB, scanErrorB);
            require_cuda(
                cudaStreamSynchronize(cuda.stream),
                "gamma transition frame A completion");
            require_cuda(
                cudaStreamSynchronize(streamB),
                "gamma transition frame B completion");

            JuicerCuda::PrintResourceDescriptors returnDescriptors{};
            JuicerProcess::Root::PreparedCudaFrame frameC = prepare(
                unitProduct,
                cuda.stream,
                603,
                returnDescriptors);
            const JuicerCuda::PrintPreparedView viewC = frameC.print_resources();
            int* scanErrorC = prepareScanStage(frameC, cuda.stream);
            if (!sameDensityStorage(viewB, viewC) ||
                viewC.profileTablesHash != unitDescriptors.profileTables.hash ||
                !samePreflashValues(viewB, viewC)) {
                frameC.abort();
                throw std::runtime_error(
                    "1 -> 1.25 -> 1 transition did not reuse storage and restore identity");
            }
            launchSample(viewC.printDcC, resultC, cuda.stream);
            finishFrame(frameC, cuda.stream, scanErrorC);
            require_cuda(
                cudaStreamSynchronize(cuda.stream),
                "gamma transition frame C completion");

            auto expandedProfile =
                std::make_shared<Profiles::ValidatedPrintProfile>(
                    *unitProduct.recipe.profileRoute.printProfile);
            const double expandedAxis =
                expandedProfile->sourceLogExposure.back() + 0.125;
            expandedProfile->sourceLogExposure.push_back(expandedAxis);
            expandedProfile->data.logExposure.push_back(
                static_cast<float>(expandedAxis));
            expandedProfile->data.densityCurves.push_back(
                expandedProfile->data.densityCurves.back());
            expandedProfile->assetVersionToken ^= 0x9e3779b97f4a7c15ull;
            if (expandedProfile->assetVersionToken == 0) {
                expandedProfile->assetVersionToken = 1;
            }
            const auto expandedProfileConst =
                std::static_pointer_cast<const Profiles::ValidatedPrintProfile>(
                    expandedProfile);
            const Spektrafilm::PrintRecipeBuildResult expandedBuild =
                build_print_recipe_for_profile(
                    unitProduct.recipe.profileRoute.filmProfile,
                    expandedProfileConst,
                    1.0);
            if (!expandedBuild.valid ||
                expandedBuild.recipe.print.develop.densityCurves.size() !=
                    unitProduct.recipe.print.develop.densityCurves.size() + 1u) {
                throw std::runtime_error(
                    "different-size print recipe build failed: " +
                    expandedBuild.diagnostic);
            }
            FocusedRenderStateBuildProduct expandedProduct = unitProduct;
            expandedProduct.recipe = expandedBuild.recipe;
            JuicerCuda::PrintResourceDescriptors expandedDescriptors{};
            JuicerProcess::Root::PreparedCudaFrame frameD = prepare(
                expandedProduct,
                streamB,
                604,
                expandedDescriptors);
            const JuicerCuda::PrintPreparedView viewD = frameD.print_resources();
            int* scanErrorD = prepareScanStage(frameD, streamB);
            if (!distinctDensityStorage(viewC, viewD) ||
                viewD.printDcC.n != viewC.printDcC.n + 1) {
                frameD.abort();
                throw std::runtime_error(
                    "different-size gamma table did not replace existing curve storage");
            }
            launchSample(viewD.printDcC, resultD, streamB);
            finishFrame(frameD, streamB, scanErrorD);
            require_cuda(
                cudaStreamSynchronize(streamB),
                "different-size gamma transition completion");

            JuicerCuda::PrintResourceDescriptors abortDescriptors{};
            JuicerProcess::Root::PreparedCudaFrame aborted = prepare(
                gammaProduct,
                cuda.stream,
                605,
                abortDescriptors);
            const JuicerCuda::PrintPreparedView abortedView =
                aborted.print_resources();
            aborted.abort();
            if (aborted.active()) {
                throw std::runtime_error(
                    "aborted gamma transition remained active");
            }
            require_cuda(
                cudaStreamSynchronize(cuda.stream),
                "aborted gamma transition completion");

            JuicerCuda::PrintResourceDescriptors recoveredDescriptors{};
            JuicerProcess::Root::PreparedCudaFrame recovered = prepare(
                unitProduct,
                streamB,
                606,
                recoveredDescriptors);
            const JuicerCuda::PrintPreparedView recoveredView =
                recovered.print_resources();
            int* recoveredScanError = prepareScanStage(recovered, streamB);
            if (!sameDensityStorage(abortedView, recoveredView) ||
                recoveredView.profileTablesHash !=
                    unitDescriptors.profileTables.hash ||
                !samePreflashValues(viewA, recoveredView)) {
                recovered.abort();
                throw std::runtime_error(
                    "gamma table preparation did not recover after abort");
            }
            launchSample(recoveredView.printDcC, resultRecovered, streamB);
            finishFrame(recovered, streamB, recoveredScanError);
            require_cuda(
                cudaStreamSynchronize(streamB),
                "recovered gamma transition completion");

            const auto readResult = [](const DeviceBuffer<float>& source) {
                float value = 0.0f;
                require_cuda(
                    cudaMemcpy(
                        &value,
                        source.get(),
                        sizeof(value),
                        cudaMemcpyDeviceToHost),
                    "gamma transition sample download");
                return value;
            };
            const float sampleA = readResult(resultA);
            const float sampleB = readResult(resultB);
            const float sampleC = readResult(resultC);
            const float sampleD = readResult(resultD);
            const float sampleRecovered = readResult(resultRecovered);
            unsigned long long measuredDelayCycles = 0;
            require_cuda(
                cudaMemcpy(
                    &measuredDelayCycles,
                    delayElapsed.get(),
                    sizeof(measuredDelayCycles),
                    cudaMemcpyDeviceToHost),
                "gamma transition delay download");
            const float expectedUnit =
                unitProduct.recipe.print.develop.densityCurves.front()[0];
            const float expectedGamma =
                gammaProduct.recipe.print.develop.densityCurves.front()[0];
            const float expectedExpanded =
                expandedProduct.recipe.print.develop.densityCurves.front()[0];
            if (measuredDelayCycles < kDelayCycles ||
                sampleA != expectedUnit || sampleB != expectedGamma ||
                sampleC != expectedUnit || sampleD != expectedExpanded ||
                sampleRecovered != expectedUnit) {
                throw std::runtime_error(
                    "queued print-table transitions sampled stale or overwritten values");
            }

            require_cuda(
                cudaStreamDestroy(streamB),
                "gamma transition stream destruction");
            streamB = nullptr;
            return {
                {"queued_a_finished_before_sync", true},
                {"queued_a_delay_clock_cycles", measuredDelayCycles},
                {"queued_a_old_table_sample_verified", true},
                {"queued_b_new_table_sample_verified", true},
                {"equal_size_storage_reused", true},
                {"round_trip_1_to_1_25_to_1_verified", true},
                {"different_size_storage_replaced", true},
                {"abort_recovery_verified", true},
                {"stock_balance_invariant", true},
                {"stock_preflash_invariant", true},
                {"unit_profile_tables_hash", viewA.profileTablesHash},
                {"gamma_profile_tables_hash", viewB.profileTablesHash},
                {"expanded_profile_tables_hash", viewD.profileTablesHash},
                {"unit_sample_count", viewA.printDcC.n},
                {"expanded_sample_count", viewD.printDcC.n}};
        } catch (...) {
            if (streamB) {
                (void)cudaStreamSynchronize(cuda.stream);
                (void)cudaStreamSynchronize(streamB);
                (void)cudaStreamDestroy(streamB);
            }
            throw;
        }
    }

    nlohmann::json abort_and_retire_print_preparation(
        std::uint64_t submissionOrdinal) {
        CudaContext cuda;
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        const ParamSnapshot snapshot = default_paper_snapshot();
        if (!build_print_render_state_product(snapshot, product, diagnostic)) {
            throw std::runtime_error(
                "lifetime print product build failed: " + diagnostic);
        }
        Scanner::ScannerSpectralLutDescriptor scannerDescriptor{};
        if (!Scanner::build_print_scanner_spectral_lut_descriptor(
                Scanner::PrintScannerSpectralLutDescriptorInput{
                    &product.recipe.profileRoute,
                    &product.recipe.densityBounds,
                    &product.recipe.scannerOutput},
                scannerDescriptor,
                diagnostic)) {
            throw std::runtime_error(
                "lifetime scanner descriptor failed: " + diagnostic);
        }
        JuicerProcess::Root::CudaFramePreparationRequest preparation{};
        preparation.recipe = &product.recipe;
        preparation.exposureTables = &product.payload.exposureTables;
        preparation.spdSInv = product.payload.spdSInv.data();
        preparation.filmRawConfig = &product.payload.filmRawConfig;
        preparation.scannerTables = &product.payload.scannerTables;
        preparation.scannerColor = &product.payload.scannerColor;
        preparation.scannerLutDescriptor = &scannerDescriptor;
        preparation.requestedWidth = 16;
        preparation.requestedHeight = 4;

        JuicerCuda::ResourceManager::SubmissionSnapshot submission{};
        submission.instanceToken.value = 0x47414d4d410003ull;
        submission.frameToken.value = submissionOrdinal;
        submission.snapshotId = submissionOrdinal;
        submission.deviceContextKey = cuda.key;
        submission.contextEpoch = 1;
        submission.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
            product.payload.uploadCoreHash,
            product.recipe.dirCouplers.hash,
            product.payload.scannerHash,
            0);
        JuicerProcess::Root::PreparedCudaFrame frame =
            JuicerProcess::root().prepare_cuda_frame(
                cuda.key,
                submission,
                preparation,
                {},
                cuda.stream,
                diagnostic);
        if (!frame.active()) {
            throw std::runtime_error(
                "lifetime abort preparation failed: " + diagnostic);
        }
        const std::uint64_t preparedHash =
            frame.print_resources().preparationHash;
        frame.abort();
        if (frame.active()) {
            throw std::runtime_error("aborted prepared frame remained active");
        }
        require_cuda(cudaStreamSynchronize(cuda.stream), "lifetime abort completion");
        if (!JuicerProcess::root().retire_idle_context(
                cuda.key.deviceId,
                cuda.key.contextOpaque,
                diagnostic)) {
            throw std::runtime_error(
                "idle context retirement failed after abort: " + diagnostic);
        }
        return {
            {"aborted_preparation_hash", preparedHash},
            {"aborted_frame_inactive", true},
            {"idle_context_retired", true}};
    }

    nlohmann::json run_lifetime(const Arguments& arguments) {
        nlohmann::json result;
        result["case"] = "lifetime";
        result["phase"] =
            "candidate print-table exact-context transition and existing route lifetime contracts";
        result["preflash_host_cache"] =
            run_print_preflash_host_cache_transitions();
        result["print_table_transitions"] =
            run_print_table_resource_transitions();
        Arguments routeArguments = arguments;
        routeArguments.outputDirectory = arguments.outputDirectory / "initial-print";
        result["initial_print"] = run_print_route(
            routeArguments,
            default_paper_snapshot(),
            "lifetime-initial-print",
            301);
        routeArguments.outputDirectory = arguments.outputDirectory / "direct-switch";
        result["direct_switch"] = run_direct_route(
            routeArguments,
            route_snapshot(
                Spektrafilm::ScanRoute::NegativeDirectScan,
                "kodak_portra_400"),
            "lifetime-direct-switch",
            302);
        result["abort_and_retire"] = abort_and_retire_print_preparation(303);
        routeArguments.outputDirectory = arguments.outputDirectory / "recovered-print";
        result["recovered_print"] = run_print_route(
            routeArguments,
            default_paper_snapshot(),
            "lifetime-recovered-print",
            304);
        write_capture_json(arguments.outputDirectory, result);
        return result;
    }

    nlohmann::json run_performance(const Arguments& arguments) {
        constexpr int kWarmupCount = 3;
        constexpr int kMeasuredCount = 11;
        nlohmann::json result;
        result["case"] = "performance";
        result["width"] = arguments.width;
        result["height"] = arguments.height;
        result["warmup_count"] = kWarmupCount;
        result["measured_count"] = kMeasuredCount;
        result["synchronization"] =
            "prepare_cuda_frame host interval; CUDA events around fused production launch; stream synchronized";
        result["capture_cost"] =
            "render_and_capture_elapsed_ms is invasive and includes split boundaries, downloads, and fused launch";

        Arguments routeArguments = arguments;
        routeArguments.outputDirectory = arguments.outputDirectory / "cold";
        result["cold"] = run_print_route(
            routeArguments,
            default_paper_snapshot(),
            "performance-cold",
            401);
        result["warmups"] = nlohmann::json::array();
        for (int index = 0; index < kWarmupCount; ++index) {
            routeArguments.outputDirectory =
                arguments.outputDirectory / "warmup" / std::to_string(index);
            result["warmups"].push_back(run_print_route(
                routeArguments,
                default_paper_snapshot(),
                "performance-warmup",
                402 + static_cast<std::uint64_t>(index)));
        }
        result["samples"] = nlohmann::json::array();
        for (int index = 0; index < kMeasuredCount; ++index) {
            routeArguments.outputDirectory =
                arguments.outputDirectory / "sample" / std::to_string(index);
            result["samples"].push_back(run_print_route(
                routeArguments,
                default_paper_snapshot(),
                "performance-sample",
                405 + static_cast<std::uint64_t>(index)));
        }

        auto runEditCycle = [&](const std::filesystem::path& category,
                                int index,
                                std::uint64_t& ordinal,
                                nlohmann::json& destination) {
            const std::filesystem::path cycleRoot =
                arguments.outputDirectory / category / std::to_string(index);

            ParamSnapshot printGamma = default_paper_snapshot();
            printGamma.printGammaFactor = 1.25;
            routeArguments.outputDirectory = cycleRoot / "print-1-to-1.25";
            destination["print_1_to_1_25"].push_back(run_print_route(
                routeArguments,
                printGamma,
                "performance-print-gamma-1-to-1.25",
                ordinal++));

            routeArguments.outputDirectory = cycleRoot / "print-1.25-to-1";
            destination["print_1_25_to_1"].push_back(run_print_route(
                routeArguments,
                default_paper_snapshot(),
                "performance-print-gamma-1.25-to-1",
                ordinal++));

            ParamSnapshot filmGamma = default_paper_snapshot();
            filmGamma.filmGammaFactor = 1.25f;
            routeArguments.outputDirectory = cycleRoot / "film-1-to-1.25";
            destination["film_1_to_1_25"].push_back(run_print_route(
                routeArguments,
                filmGamma,
                "performance-film-gamma-1-to-1.25",
                ordinal++));

            routeArguments.outputDirectory = cycleRoot / "film-1.25-to-1";
            destination["film_1_25_to_1"].push_back(run_print_route(
                routeArguments,
                default_paper_snapshot(),
                "performance-film-gamma-1.25-to-1",
                ordinal++));
        };
        auto makeEditRecord = []() {
            return nlohmann::json{
                {"print_1_to_1_25", nlohmann::json::array()},
                {"print_1_25_to_1", nlohmann::json::array()},
                {"film_1_to_1_25", nlohmann::json::array()},
                {"film_1_25_to_1", nlohmann::json::array()}};
        };
        result["gamma_edit_warmups"] = makeEditRecord();
        result["gamma_edit_samples"] = makeEditRecord();
        std::uint64_t ordinal = 500;
        for (int index = 0; index < kWarmupCount; ++index) {
            runEditCycle(
                "gamma-edit-warmup",
                index,
                ordinal,
                result["gamma_edit_warmups"]);
        }
        for (int index = 0; index < kMeasuredCount; ++index) {
            runEditCycle(
                "gamma-edit-sample",
                index,
                ordinal,
                result["gamma_edit_samples"]);
        }
        write_capture_json(arguments.outputDirectory, result);
        return result;
    }

} // namespace

int main(int argc, char** argv) {
    bool processRootAccessed = false;
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        verify_runtime_resource_root(arguments.resourceRoot);
        nlohmann::json result;
        if (arguments.caseGroup == "default-paper") {
            processRootAccessed = true;
            result = run_print_route(
                arguments,
                default_paper_snapshot(),
                "default-paper",
                1);
        } else if (arguments.caseGroup == "default-routes") {
            processRootAccessed = true;
            result = run_default_routes(arguments);
        } else if (arguments.caseGroup == "route-probe" ||
                   arguments.caseGroup == "candidate-route-probe") {
            processRootAccessed = true;
            result = run_route_probe(
                arguments,
                arguments.caseGroup == "route-probe");
        } else if (arguments.caseGroup == "routes") {
            processRootAccessed = true;
            result = run_routes(arguments);
        } else if (arguments.caseGroup == "lifetime") {
            processRootAccessed = true;
            result = run_lifetime(arguments);
        } else if (arguments.caseGroup == "performance") {
            processRootAccessed = true;
            result = run_performance(arguments);
        } else if (arguments.caseGroup == "sampling-dir") {
            result = run_sampling_dir(arguments);
        } else if (arguments.caseGroup == "print-backend") {
            processRootAccessed = true;
            result = run_print_backend(arguments);
        } else {
            processRootAccessed = true;
            result = run_state(arguments);
        }
        std::cout << result.dump() << '\n';
        if (processRootAccessed) {
            JuicerProcess::root().shutdown();
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        if (processRootAccessed) {
            JuicerProcess::root().shutdown();
        }
        return 2;
    }
}
