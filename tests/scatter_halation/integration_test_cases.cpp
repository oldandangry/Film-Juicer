#include "integration_test.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Cuda/Diffusion/JuicerCudaDiffusion.h"
#include "Cuda/Film/JuicerCudaScatterHalation.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "ProcessRoot.h"
#include "ProfileAssets.h"
#include "Scanner.h"
#include "ScatterHalation.h"
#include "nlohmann/json.hpp"

extern "C" cudaError_t
juicer_cuda_build_direct_spatial_dir_cached_log_raw_from_camera_film_linear(
    const JuicerCuda::DirectPipelineRunParams* hParams,
    JuicerCuda::CameraFilmLinearExposurePlanes cameraFilmLinear,
    float* logRawB,
    float* logRawG,
    float* logRawR,
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
    const JuicerCuda::GrainPayload* gateDefects,
    const JuicerCuda::GateWeavePayload* weave,
    const float* gateMask,
    int gateMaskWidth,
    int gateMaskHeight,
    void* cudaStreamOpaque);

namespace {
    using ScatterHalationValidation::Arguments;
    using ScatterHalationValidation::build_prepared_route_inputs;
    using ScatterHalationValidation::PreparedRouteInputs;
    using ScatterHalationValidation::Results;

    constexpr std::string_view kFixtureSchema =
        "spektrafilm-scatter-halation-reference.v1";
    constexpr std::string_view kReferenceCommit =
        "48645a2b4bf58c20b6a3b75c8022d0f462db754a";
    constexpr std::string_view kHostGate1RecipeHash = "4812af8cf93cb3af";
    constexpr std::uint64_t kProfileAssetVersionToken =
        16343725478736492178ULL;

    class Sha256 final {
    public:
        void update(const std::uint8_t* data, std::size_t size) {
            for (std::size_t index = 0; index < size; ++index) {
                _block[_blockSize++] = data[index];
                if (_blockSize == _block.size()) {
                    transform();
                    _bitCount += 512;
                    _blockSize = 0;
                }
            }
        }

        [[nodiscard]] std::string finish() {
            const std::uint64_t totalBits =
                _bitCount + static_cast<std::uint64_t>(_blockSize) * 8;
            _block[_blockSize++] = 0x80;
            if (_blockSize > 56) {
                while (_blockSize < _block.size()) {
                    _block[_blockSize++] = 0;
                }
                transform();
                _blockSize = 0;
            }
            while (_blockSize < 56) {
                _block[_blockSize++] = 0;
            }
            for (int shift = 56; shift >= 0; shift -= 8) {
                _block[_blockSize++] =
                    static_cast<std::uint8_t>(totalBits >> shift);
            }
            transform();

            std::ostringstream stream;
            stream << std::hex << std::setfill('0');
            for (std::uint32_t word : _state) {
                stream << std::setw(8) << word;
            }
            return stream.str();
        }

    private:
        static constexpr std::array<std::uint32_t, 64> kRoundConstants{{0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U}};

        void transform() {
            std::array<std::uint32_t, 64> schedule{};
            for (std::size_t index = 0; index < 16; ++index) {
                const std::size_t offset = index * 4;
                schedule[index] =
                    (static_cast<std::uint32_t>(_block[offset]) << 24) |
                    (static_cast<std::uint32_t>(_block[offset + 1]) << 16) |
                    (static_cast<std::uint32_t>(_block[offset + 2]) << 8) |
                    static_cast<std::uint32_t>(_block[offset + 3]);
            }
            for (std::size_t index = 16; index < schedule.size(); ++index) {
                const std::uint32_t s0 =
                    std::rotr(schedule[index - 15], 7) ^
                    std::rotr(schedule[index - 15], 18) ^
                    (schedule[index - 15] >> 3);
                const std::uint32_t s1 =
                    std::rotr(schedule[index - 2], 17) ^
                    std::rotr(schedule[index - 2], 19) ^
                    (schedule[index - 2] >> 10);
                schedule[index] = schedule[index - 16] + s0 +
                                  schedule[index - 7] + s1;
            }

            std::uint32_t a = _state[0];
            std::uint32_t b = _state[1];
            std::uint32_t c = _state[2];
            std::uint32_t d = _state[3];
            std::uint32_t e = _state[4];
            std::uint32_t f = _state[5];
            std::uint32_t g = _state[6];
            std::uint32_t h = _state[7];
            for (std::size_t index = 0; index < schedule.size(); ++index) {
                const std::uint32_t sum1 = std::rotr(e, 6) ^
                                           std::rotr(e, 11) ^
                                           std::rotr(e, 25);
                const std::uint32_t choose = (e & f) ^ (~e & g);
                const std::uint32_t temporary1 = h + sum1 + choose +
                                                 kRoundConstants[index] +
                                                 schedule[index];
                const std::uint32_t sum0 = std::rotr(a, 2) ^
                                           std::rotr(a, 13) ^
                                           std::rotr(a, 22);
                const std::uint32_t majority =
                    (a & b) ^ (a & c) ^ (b & c);
                const std::uint32_t temporary2 = sum0 + majority;
                h = g;
                g = f;
                f = e;
                e = d + temporary1;
                d = c;
                c = b;
                b = a;
                a = temporary1 + temporary2;
            }
            _state[0] += a;
            _state[1] += b;
            _state[2] += c;
            _state[3] += d;
            _state[4] += e;
            _state[5] += f;
            _state[6] += g;
            _state[7] += h;
        }

        std::array<std::uint8_t, 64> _block{};
        std::size_t _blockSize = 0;
        std::uint64_t _bitCount = 0;
        std::array<std::uint32_t, 8> _state{{0x6a09e667U,
                                             0xbb67ae85U,
                                             0x3c6ef372U,
                                             0xa54ff53aU,
                                             0x510e527fU,
                                             0x9b05688cU,
                                             0x1f83d9abU,
                                             0x5be0cd19U}};
    };

    std::string sha256(const std::uint8_t* data, std::size_t size) {
        Sha256 digest;
        digest.update(data, size);
        return digest.finish();
    }

    std::vector<std::uint8_t> read_binary(
        const std::filesystem::path& path) {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) {
            throw std::runtime_error("failed to open fixture binary " + path.string());
        }
        const std::streamsize size = stream.tellg();
        if (size < 0) {
            throw std::runtime_error("failed to size fixture binary " + path.string());
        }
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
        stream.seekg(0);
        if (!bytes.empty()) {
            stream.read(
                reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
        }
        if (!stream) {
            throw std::runtime_error("failed to read fixture binary " + path.string());
        }
        return bytes;
    }

    nlohmann::json read_manifest(const std::filesystem::path& path) {
        std::ifstream stream(path);
        if (!stream) {
            throw std::runtime_error("failed to open fixture manifest " + path.string());
        }
        nlohmann::json manifest;
        stream >> manifest;
        return manifest;
    }

    template <typename T>
    std::vector<T> read_plane(
        const nlohmann::json& plane,
        const std::vector<std::uint8_t>& binary,
        std::string_view expectedDtype,
        int width,
        int height) {
        if (std::endian::native != std::endian::little ||
            plane.at("dtype").get<std::string>() != expectedDtype ||
            plane.at("shape").at(0).get<int>() != height ||
            plane.at("shape").at(1).get<int>() != width) {
            throw std::runtime_error("fixture plane encoding/shape mismatch");
        }
        const std::size_t elementCount =
            plane.at("elementCount").get<std::size_t>();
        const std::size_t byteOffset = plane.at("byteOffset").get<std::size_t>();
        const std::size_t byteCount = elementCount * sizeof(T);
        if (elementCount != static_cast<std::size_t>(width) * height ||
            byteOffset > binary.size() || byteCount > binary.size() - byteOffset) {
            throw std::runtime_error("fixture plane segment is out of bounds");
        }
        const std::uint8_t* source = binary.data() + byteOffset;
        if (sha256(source, byteCount) != plane.at("sha256").get<std::string>()) {
            throw std::runtime_error("fixture plane checksum mismatch");
        }
        std::vector<T> values(elementCount);
        std::memcpy(values.data(), source, byteCount);
        return values;
    }

    template <typename T>
    using SemanticPlanes = std::array<std::vector<T>, 3>;

    template <typename T>
    SemanticPlanes<T> read_planes(
        const nlohmann::json& carrier,
        const std::vector<std::uint8_t>& binary,
        std::string_view dtype,
        int width,
        int height) {
        if (carrier.at("carrierEncoding").get<std::string>() !=
                "visual-top row-major planar semantic RGB" ||
            carrier.at("planes").size() != 3) {
            throw std::runtime_error("fixture carrier encoding mismatch");
        }
        SemanticPlanes<T> result;
        constexpr std::array<std::string_view, 3> channels{{"R", "G", "B"}};
        for (std::size_t channel = 0; channel < result.size(); ++channel) {
            const auto& plane = carrier.at("planes").at(channel);
            if (plane.at("semanticChannel").get<std::string>() != channels[channel]) {
                throw std::runtime_error("fixture semantic channel mismatch");
            }
            result[channel] = read_plane<T>(
                plane, binary, dtype, width, height);
        }
        return result;
    }

    std::uint32_t parse_bits(const nlohmann::json& value) {
        const std::string text = value.get<std::string>();
        if (text.size() != 8 ||
            !std::all_of(text.begin(), text.end(), [](char character) {
                return (character >= '0' && character <= '9') ||
                       (character >= 'a' && character <= 'f');
            })) {
            throw std::runtime_error("invalid canonical Float32 bit string");
        }
        return static_cast<std::uint32_t>(std::stoul(text, nullptr, 16));
    }

    float parse_float(const nlohmann::json& value) {
        return std::bit_cast<float>(parse_bits(value));
    }

    std::uint64_t parse_hash(const nlohmann::json& value) {
        const std::string text = value.get<std::string>();
        if (text.size() != 16 ||
            !std::all_of(text.begin(), text.end(), [](char character) {
                return (character >= '0' && character <= '9') ||
                       (character >= 'a' && character <= 'f');
            })) {
            throw std::runtime_error("invalid canonical recipe hash string");
        }
        return std::stoull(text, nullptr, 16);
    }

    struct ResolvedRow {
        ScatterHalationControls controls{};
        Profiles::ProfileDigest profile{};
        ScatterHalationOpticsRecipe recipe{};
        ScatterHalationFrameDescriptor descriptor{};
        float pixelSizeUm = 0.0f;
        int width = 0;
        int height = 0;
    };

    ResolvedRow resolve_row(const nlohmann::json& input) {
        ResolvedRow row;
        const auto& controls = input.at("controls");
        ScatterHalationRawControls raw;
        raw.active = controls.at("active").get<bool>();
        raw.scatterAmount = static_cast<double>(
            parse_float(controls.at("scatterAmountFloat32Bits")));
        raw.scatterSpatialScale = static_cast<double>(
            parse_float(controls.at("scatterSpatialScaleFloat32Bits")));
        raw.halationAmount = static_cast<double>(
            parse_float(controls.at("halationAmountFloat32Bits")));
        raw.halationSpatialScale = static_cast<double>(
            parse_float(controls.at("halationSpatialScaleFloat32Bits")));
        std::string diagnostic;
        if (!Spektrafilm::build_scatter_halation_controls(
                raw, row.controls, diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        const auto& profile = input.at("profile");
        for (std::size_t channel = 0; channel < 3; ++channel) {
            row.profile.halationFirstSigmaUm[channel] = parse_float(
                profile.at("halationFirstSigmaUmFloat32Bits").at(channel));
            row.profile.halationPrimaryAmount[channel] = parse_float(
                profile.at("halationPrimaryAmountFloat32Bits").at(channel));
        }
        if (!Spektrafilm::resolve_scatter_halation_recipe(
                row.controls, row.profile, row.recipe, diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        row.pixelSizeUm = parse_float(input.at("pixelSizeUmFloat32Bits"));
        row.width = input.at("fullFrameWidth").get<int>();
        row.height = input.at("fullFrameHeight").get<int>();
        std::optional<ScatterHalationFrameDescriptor> descriptor;
        if (!Spektrafilm::build_scatter_halation_frame_descriptor(
                row.recipe, row.pixelSizeUm, descriptor, diagnostic) ||
            !descriptor) {
            throw std::runtime_error(
                diagnostic.empty() ? "missing row descriptor" : diagnostic);
        }
        row.descriptor = *descriptor;
        return row;
    }

    template <typename T>
    std::vector<T> visual_to_storage(
        const std::vector<T>& visual,
        int width,
        int height) {
        std::vector<T> storage(visual.size());
        for (int visualY = 0; visualY < height; ++visualY) {
            const int storageY = height - 1 - visualY;
            std::copy_n(
                visual.begin() + static_cast<std::size_t>(visualY) * width,
                width,
                storage.begin() + static_cast<std::size_t>(storageY) * width);
        }
        return storage;
    }

    template <typename T>
    std::vector<T> storage_to_visual(
        const std::vector<T>& storage,
        int width,
        int height) {
        return visual_to_storage(storage, width, height);
    }

    void require_cuda(cudaError_t status, std::string_view operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string(operation) + ": " + cudaGetErrorString(status));
        }
    }

    class CudaAllocation final {
    public:
        explicit CudaAllocation(std::size_t bytes) {
            require_cuda(cudaMalloc(&_pointer, bytes), "cudaMalloc");
        }
        ~CudaAllocation() {
            if (_pointer) {
                cudaFree(_pointer);
            }
        }
        CudaAllocation(const CudaAllocation&) = delete;
        CudaAllocation& operator=(const CudaAllocation&) = delete;
        [[nodiscard]] float* floats() const noexcept {
            return static_cast<float*>(_pointer);
        }

    private:
        void* _pointer = nullptr;
    };

    class CudaStream final {
    public:
        CudaStream() {
            require_cuda(cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking),
                         "cudaStreamCreateWithFlags");
        }
        ~CudaStream() {
            if (_stream) {
                cudaStreamDestroy(_stream);
            }
        }
        CudaStream(const CudaStream&) = delete;
        CudaStream& operator=(const CudaStream&) = delete;
        [[nodiscard]] cudaStream_t get() const noexcept {
            return _stream;
        }

    private:
        cudaStream_t _stream = nullptr;
    };

    struct CandidateOutput {
        SemanticPlanes<float> finalExposure;
        std::optional<SemanticPlanes<float>> downstreamLog;
    };

    CandidateOutput run_candidate(
        const ResolvedRow& row,
        const SemanticPlanes<float>& source,
        bool downstream,
        int rowPadding = 0) {
        const std::size_t planeElements =
            static_cast<std::size_t>(row.width) * row.height;
        const std::size_t planeBytes = planeElements * sizeof(float);
        const std::size_t carrierStride = static_cast<std::size_t>(row.width) + rowPadding;
        const std::size_t carrierElements = carrierStride * row.height;
        const std::size_t carrierBytes = carrierElements * sizeof(float);
        constexpr std::uint32_t kPaddingBits = 0x7fc01234u;
        CudaAllocation allocation(3 * carrierBytes + (downstream ? 5 : 2) * planeBytes);
        CudaStream stream;
        float* const block = allocation.floats();
        std::array<float*, 3> devicePlanes{{block, block + carrierElements, block + 2 * carrierElements}};
        float* const temporary = block + 3 * carrierElements;
        float* const accumulation = temporary + planeElements;
        for (std::size_t channel = 0; channel < source.size(); ++channel) {
            const std::vector<float> storage =
                visual_to_storage(source[channel], row.width, row.height);
            std::vector<float> carrier(carrierElements, std::bit_cast<float>(kPaddingBits));
            for (int y = 0; y < row.height; ++y) {
                std::copy_n(storage.data() + static_cast<std::size_t>(y) * row.width,
                            row.width,
                            carrier.data() + static_cast<std::size_t>(y) * carrierStride);
            }
            require_cuda(
                cudaMemcpy(
                    devicePlanes[channel],
                    carrier.data(),
                    carrierBytes,
                    cudaMemcpyHostToDevice),
                "cudaMemcpy source");
        }

        JuicerCuda::ScatterHalationPreparedView view;
        view.descriptor = &row.descriptor;
        view.fullFrameWidth = row.width;
        view.fullFrameHeight = row.height;
        view.carrierSource =
            JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes;
        view.currentCarrier.redSensitive = devicePlanes[0];
        view.currentCarrier.greenSensitive = devicePlanes[1];
        view.currentCarrier.blueSensitive = devicePlanes[2];
        view.currentCarrier.rowStrideFloats = carrierStride;
        view.filterTemp = temporary;
        view.weightedAccumulation = accumulation;
        const JuicerCuda::ScatterHalationLaunchResult launch =
            JuicerCuda::launch_scatter_halation(view, stream.get());
        if (launch.status != cudaSuccess) {
            std::ostringstream detail;
            detail << "halation launch status=" << cudaGetErrorString(launch.status)
                   << " stage=" << static_cast<int>(launch.stage)
                   << " channel=" << static_cast<int>(launch.channel)
                   << " gaussian=" << launch.gaussianIndex;
            throw std::runtime_error(detail.str());
        }

        CandidateOutput output;
        if (downstream) {
            float* const logB = accumulation + planeElements;
            float* const logG = logB + planeElements;
            float* const logR = logG + planeElements;
            JuicerCuda::DirectPipelineRunParams params;
            params.width = row.width;
            params.height = row.height;
            params.nComponents = 3;
            params.filmExpose.routeCorrectionScale = 1.0f;
            require_cuda(
                juicer_cuda_build_direct_spatial_dir_cached_log_raw_from_camera_film_linear(
                    &params,
                    view.currentCarrier,
                    logB,
                    logG,
                    logR,
                    reinterpret_cast<void*>(stream.get())),
                "direct cached-log boundary");
            require_cuda(cudaStreamSynchronize(stream.get()), "downstream synchronize");
            output.downstreamLog.emplace();
            const std::array<float*, 3> logs{{logR, logG, logB}};
            for (std::size_t channel = 0; channel < logs.size(); ++channel) {
                std::vector<float> storage(planeElements);
                require_cuda(
                    cudaMemcpy(
                        storage.data(),
                        logs[channel],
                        planeBytes,
                        cudaMemcpyDeviceToHost),
                    "cudaMemcpy downstream");
                (*output.downstreamLog)[channel] =
                    storage_to_visual(storage, row.width, row.height);
            }
        } else {
            require_cuda(cudaStreamSynchronize(stream.get()), "halation synchronize");
        }

        for (std::size_t channel = 0; channel < devicePlanes.size(); ++channel) {
            std::vector<float> carrier(carrierElements);
            std::vector<float> storage(planeElements);
            require_cuda(
                cudaMemcpy(
                    carrier.data(),
                    devicePlanes[channel],
                    carrierBytes,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy final exposure");
            for (int y = 0; y < row.height; ++y) {
                const std::size_t offset = static_cast<std::size_t>(y) * carrierStride;
                std::copy_n(carrier.data() + offset, row.width, storage.data() + static_cast<std::size_t>(y) * row.width);
                for (std::size_t x = static_cast<std::size_t>(row.width); x < carrierStride; ++x) {
                    if (std::bit_cast<std::uint32_t>(carrier[offset + x]) != kPaddingBits) {
                        throw std::runtime_error("halation modified carrier padding");
                    }
                }
            }
            output.finalExposure[channel] =
                storage_to_visual(storage, row.width, row.height);
        }
        return output;
    }

    struct ErrorStats {
        bool finite = true;
        double mean = 0.0;
        double maximum = 0.0;
        double l1 = 0.0;
    };

    template <typename ActualValue, typename ExpectedValue>
    ErrorStats absolute_error(
        const std::vector<ActualValue>& actual,
        const std::vector<ExpectedValue>& expected,
        double scale = 1.0) {
        if (actual.size() != expected.size() || !(scale > 0.0)) {
            return {false, 0.0, 0.0, 0.0};
        }
        ErrorStats stats;
        for (std::size_t index = 0; index < actual.size(); ++index) {
            const double actualValue = static_cast<double>(actual[index]);
            const double expectedValue = static_cast<double>(expected[index]);
            if (!std::isfinite(actualValue) || !std::isfinite(expectedValue)) {
                stats.finite = false;
                continue;
            }
            const double error = std::abs(actualValue - expectedValue) / scale;
            stats.maximum = std::max(stats.maximum, error);
            stats.l1 += error;
        }
        stats.mean = actual.empty()
                         ? 0.0
                         : stats.l1 / static_cast<double>(actual.size());
        return stats;
    }

    double source_scale(const std::vector<float>& source) {
        double scale = 0.0;
        for (float value : source) {
            scale = std::max(scale, std::abs(static_cast<double>(value)));
        }
        return scale;
    }

    bool source_is_impulse_like(const SemanticPlanes<float>& source) {
        std::size_t nonzero = 0;
        for (const auto& plane : source) {
            nonzero += static_cast<std::size_t>(std::count_if(
                plane.begin(), plane.end(), [](float value) {
                    return value != 0.0f;
                }));
        }
        return nonzero <= 3;
    }

    std::string stats_detail(
        const std::array<ErrorStats, 3>& finalStats,
        const std::optional<std::array<ErrorStats, 3>>& deltaStats) {
        std::ostringstream stream;
        stream << std::scientific << std::setprecision(3);
        constexpr std::array<char, 3> names{{'R', 'G', 'B'}};
        for (std::size_t channel = 0; channel < names.size(); ++channel) {
            stream << names[channel] << "(mean=" << finalStats[channel].mean
                   << ",max=" << finalStats[channel].maximum
                   << ",l1=" << finalStats[channel].l1 << ')';
            if (deltaStats) {
                stream << " delta(mean=" << (*deltaStats)[channel].mean
                       << ",max=" << (*deltaStats)[channel].maximum
                       << ",l1=" << (*deltaStats)[channel].l1 << ')';
            }
            if (channel + 1 < names.size()) {
                stream << ' ';
            }
        }
        return stream.str();
    }

    bool compare_operator_row(
        const nlohmann::json& fixture,
        const SemanticPlanes<float>& source,
        const SemanticPlanes<double>& expected,
        const CandidateOutput& candidate,
        std::string& detail) {
        const std::string comparison = fixture.at("comparison").get<std::string>();
        const bool impulse = source_is_impulse_like(source);
        std::array<ErrorStats, 3> finalStats{};
        std::optional<std::array<ErrorStats, 3>> deltaStats;
        bool passed = true;
        if (comparison == "fir" || comparison == "yvv") {
            const double meanLimit = comparison == "fir" ? std::numeric_limits<double>::infinity() : 2.0e-4;
            const double maximumLimit = comparison == "fir" ? 2.0e-5 : 2.0e-3;
            const double impulseL1Limit = comparison == "fir" ? 2.0e-5 : 2.0e-3;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                finalStats[channel] = absolute_error(
                    candidate.finalExposure[channel], expected[channel]);
                passed = passed && finalStats[channel].finite &&
                         finalStats[channel].mean <= meanLimit &&
                         finalStats[channel].maximum <= maximumLimit &&
                         (!impulse || finalStats[channel].l1 <= impulseL1Limit);
            }
        } else if (comparison == "combined") {
            deltaStats.emplace();
            for (std::size_t channel = 0; channel < 3; ++channel) {
                const double scale = source_scale(source[channel]);
                if (scale == 0.0) {
                    const bool exactZero = std::all_of(
                        candidate.finalExposure[channel].begin(),
                        candidate.finalExposure[channel].end(),
                        [](float value) {
                            return value == 0.0f;
                        });
                    const bool referenceZero = std::all_of(
                        expected[channel].begin(),
                        expected[channel].end(),
                        [](double value) {
                            return value == 0.0;
                        });
                    finalStats[channel] = {exactZero && referenceZero, 0.0, 0.0, 0.0};
                    (*deltaStats)[channel] = finalStats[channel];
                    passed = passed && exactZero && referenceZero;
                    continue;
                }
                finalStats[channel] = absolute_error(
                    candidate.finalExposure[channel], expected[channel], scale);
                std::vector<float> candidateDelta(source[channel].size());
                std::vector<double> referenceDelta(source[channel].size());
                for (std::size_t index = 0; index < source[channel].size(); ++index) {
                    candidateDelta[index] = candidate.finalExposure[channel][index] -
                                            source[channel][index];
                    referenceDelta[index] = expected[channel][index] -
                                            static_cast<double>(source[channel][index]);
                }
                (*deltaStats)[channel] = absolute_error(
                    candidateDelta, referenceDelta, scale);
                const auto accepted = [&](const ErrorStats& stats) {
                    return stats.finite && stats.mean <= 3.0e-4 &&
                           stats.maximum <= 3.0e-3 &&
                           (!impulse || stats.l1 <= 3.0e-3);
                };
                passed = passed && accepted(finalStats[channel]) &&
                         accepted((*deltaStats)[channel]);
            }
        } else {
            throw std::runtime_error("unknown fixture comparison kind");
        }
        if (fixture.at("referenceOutsideUnitInterval").get<bool>()) {
            bool candidateOutside = false;
            for (const auto& plane : candidate.finalExposure) {
                candidateOutside = candidateOutside || std::any_of(
                                                           plane.begin(), plane.end(), [](float value) {
                                                               return value < 0.0f || value > 1.0f;
                                                           });
            }
            passed = passed && candidateOutside;
        }
        detail = stats_detail(finalStats, deltaStats);
        return passed;
    }

    bool compare_downstream(
        const SemanticPlanes<float>& source,
        const SemanticPlanes<double>& expected,
        const SemanticPlanes<float>& candidate,
        std::string& detail) {
        std::array<ErrorStats, 3> stats{};
        bool passed = true;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const double scale = source_scale(source[channel]);
            if (!(scale > 0.0)) {
                throw std::runtime_error(
                    "downstream fixture has zero source comparison scale");
            }
            stats[channel] = absolute_error(
                candidate[channel], expected[channel], scale);
            passed = passed && stats[channel].finite &&
                     stats[channel].mean <= 3.0e-4 &&
                     stats[channel].maximum <= 3.0e-3;
        }
        detail = stats_detail(stats, std::nullopt);
        return passed;
    }

    ScatterHalationFrameDescriptor build_descriptor(
        double spatialScale,
        float pixelSizeUm) {
        ScatterHalationRawControls raw;
        raw.active = true;
        raw.scatterAmount = 1.0;
        raw.scatterSpatialScale = spatialScale;
        raw.halationAmount = 0.0;
        raw.halationSpatialScale = 1.0;
        ScatterHalationControls controls;
        std::string diagnostic;
        if (!Spektrafilm::build_scatter_halation_controls(
                raw, controls, diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        Profiles::ProfileDigest profile;
        profile.halationFirstSigmaUm = {65.0f, 65.0f, 65.0f};
        profile.halationPrimaryAmount = {0.08f, 0.02f, 0.0f};
        ScatterHalationOpticsRecipe recipe;
        if (!Spektrafilm::resolve_scatter_halation_recipe(
                controls, profile, recipe, diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        std::optional<ScatterHalationFrameDescriptor> descriptor;
        if (!Spektrafilm::build_scatter_halation_frame_descriptor(
                recipe, pixelSizeUm, descriptor, diagnostic) ||
            !descriptor) {
            throw std::runtime_error(diagnostic);
        }
        return *descriptor;
    }

    void run_carrier_stride_rows(Results& results) {
        // Carrier layout must not alter the operator or leak poisoned padding
        // into the image, including complete and partial row batches and tiles.
        for (const auto extent : {std::array{1, 33}, std::array{33, 1}, std::array{19, 7}, std::array{19, 8}, std::array{19, 9}, std::array{19, 15}, std::array{19, 16}, std::array{19, 17}, std::array{31, 31}, std::array{32, 32}, std::array{33, 33}, std::array{63, 65}, std::array{65, 63}, std::array{129, 67}}) {
            const std::string name = "operator/carrier-stride/" +
                                     std::to_string(extent[0]) + "x" + std::to_string(extent[1]);
            try {
                ResolvedRow row;
                row.width = extent[0];
                row.height = extent[1];
                row.descriptor = build_descriptor(2.0, 0.5f);
                const std::size_t elements = static_cast<std::size_t>(row.width) * row.height;
                SemanticPlanes<float> source;
                for (std::size_t channel = 0; channel < source.size(); ++channel) {
                    source[channel].resize(elements);
                    for (std::size_t index = 0; index < elements; ++index) {
                        source[channel][index] = channel == 2
                                                     ? 0.0f
                                                     : 0.05f + static_cast<float>((index * (37 + channel) + 11) % 1021) / 255.0f;
                    }
                }
                const auto compact = run_candidate(row, source, false);
                const auto padded = run_candidate(row, source, false, 3);
                bool passed = true;
                for (std::size_t channel = 0; channel < source.size(); ++channel) {
                    for (std::size_t index = 0; index < elements; ++index) {
                        const float actual = padded.finalExposure[channel][index];
                        const float expected = compact.finalExposure[channel][index];
                        passed = passed && std::isfinite(actual) &&
                                 std::bit_cast<std::uint32_t>(actual) == std::bit_cast<std::uint32_t>(expected) &&
                                 (channel != 2 || actual == 0.0f);
                    }
                }
                results.record(name, passed, "bit-exact compact/padded carrier; padding intact; zero channel isolated");
            } catch (const std::exception& error) {
                results.record(name, false, error.what());
            }
        }
    }

    void run_reference_descriptor_rows(
        const nlohmann::json& manifest,
        Results& results) {
        const float below = std::nextafter(1.5f, 0.0f);
        const auto belowDescriptor = build_descriptor(
            static_cast<double>(below), 1.0f);
        const auto exactDescriptor = build_descriptor(1.5, 1.0f);
        results.record(
            "reference/descriptor/green-core-dispatch-seam",
            belowDescriptor.channels[1].core.kind ==
                    ScatterHalationGaussianKind::FirReflect &&
                exactDescriptor.channels[1].core.kind ==
                    ScatterHalationGaussianKind::YvvReplicate,
            "pixel_size_um=1 green_core_um=2 retained_scale=nextafter(1.5,0)/1.5");

        bool nativePassed = true;
        std::ostringstream nativeDetail;
        for (const auto& row : manifest.at("nativeDoubleAcquisitionRows")) {
            const std::string rawText = row.at("rawDoubleHex").get<std::string>();
            char* end = nullptr;
            const double raw = std::strtod(rawText.c_str(), &end);
            const float retained = static_cast<float>(raw);
            nativePassed = nativePassed && end && *end == '\0' &&
                           std::bit_cast<std::uint32_t>(retained) ==
                               parse_bits(row.at("retainedFloat32Bits"));
            const auto descriptor = build_descriptor(raw, 1.0f);
            const bool onto =
                row.at("id").get<std::string>().find("onto") != std::string::npos;
            nativePassed = nativePassed &&
                           descriptor.channels[1].core.kind ==
                               (onto ? ScatterHalationGaussianKind::YvvReplicate
                                     : ScatterHalationGaussianKind::FirReflect);
            nativeDetail << row.at("id").get<std::string>() << '='
                         << row.at("retainedFloat32Bits").get<std::string>() << ' ';
        }
        results.record(
            "reference/descriptor/native-double-one-time-narrowing",
            nativePassed,
            nativeDetail.str());

        bool formatRowsPassed = true;
        std::ostringstream formatDetail;
        for (int longEdgePixels : {1920, 3840, 7680}) {
            const float pixelSize =
                static_cast<float>(35000.0 / static_cast<double>(longEdgePixels));
            const auto descriptor = build_descriptor(1.0, pixelSize);
            formatRowsPassed = formatRowsPassed && descriptor.recipeHash != 0;
            formatDetail << longEdgePixels << ':' << pixelSize << ' ';
        }
        results.record(
            "reference/descriptor/35mm-pixel-size-rows",
            formatRowsPassed,
            formatDetail.str());

        const auto extreme = build_descriptor(
            2.0, std::numeric_limits<float>::denorm_min());
        bool extremePassed = extreme.recipeHash != 0;
        for (const auto& channel : extreme.channels) {
            extremePassed = extremePassed &&
                            std::isfinite(channel.core.B) &&
                            std::isfinite(channel.core.B1) &&
                            std::isfinite(channel.core.B2) &&
                            std::isfinite(channel.core.B3);
        }
        results.record(
            "reference/descriptor/min-positive-pixel-finite-schedule",
            extremePassed,
            "spatial_scale=2 pixel_size_um=denorm_min");
    }
} // namespace

namespace ScatterHalationValidation {
    void run_scanner_post_effect_cuda_rows(Results& results) {
        constexpr int kWidth = 5;
        constexpr int kHeight = 4;
        constexpr int kRadius = 2;
        constexpr float kAmount = 0.7f;
        constexpr std::array<float, kWidth * kHeight> kSource{{0.03f, 0.11f, 0.29f, 0.47f, 0.83f, 0.07f, 0.19f, 0.31f, 0.61f, 0.97f, 0.13f, 0.23f, 0.41f, 0.73f, 1.09f, 0.17f, 0.37f, 0.59f, 0.89f, 1.31f}};
        constexpr std::array<float, 2 * kRadius + 1> kKernel{{0.00962005683f,
                                                              0.205423697f,
                                                              0.569912492f,
                                                              0.205423697f,
                                                              0.00962005683f}};
        constexpr std::array<float, kWidth * kHeight> kExpected{{0.008215948f, 0.08396746f, 0.27832553f, 0.4252918f, 0.86518514f, 0.048741654f, 0.18746452f, 0.27515787f, 0.6005329f, 1.0301093f, 0.11236872f, 0.2057411f, 0.37625948f, 0.71667653f, 1.1366233f, 0.14732751f, 0.38531336f, 0.6033108f, 0.90200037f, 1.4113867f}};

        try {
            constexpr std::size_t kElements = kSource.size();
            constexpr std::size_t kPlaneBytes = kElements * sizeof(float);
            CudaAllocation planes(4 * kPlaneBytes);
            CudaAllocation kernel(kKernel.size() * sizeof(float));
            CudaAllocation pixels(
                2 * kElements * 3 * sizeof(float));
            CudaStream stream;
            float* const red = planes.floats();
            float* const green = red + kElements;
            float* const blue = green + kElements;
            float* const temporary = blue + kElements;
            for (float* plane : {red, green, blue}) {
                require_cuda(
                    cudaMemcpy(
                        plane,
                        kSource.data(),
                        kPlaneBytes,
                        cudaMemcpyHostToDevice),
                    "scanner border source upload");
            }
            require_cuda(
                cudaMemcpy(
                    kernel.floats(),
                    kKernel.data(),
                    kKernel.size() * sizeof(float),
                    cudaMemcpyHostToDevice),
                "scanner border kernel upload");

            JuicerCuda::DirectPipelineRunParams params;
            params.src = pixels.floats();
            params.srcRowBytes = kWidth * 3 * sizeof(float);
            params.dst = pixels.floats() + kElements * 3;
            params.dstRowBytes = kWidth * 3 * sizeof(float);
            params.width = kWidth;
            params.height = kHeight;
            params.nComponents = 3;
            require_cuda(
                juicer_cuda_direct_focused_scanner_post_output(
                    &params,
                    red,
                    green,
                    blue,
                    temporary,
                    nullptr,
                    0,
                    kernel.floats(),
                    kRadius,
                    kAmount,
                    nullptr,
                    nullptr,
                    nullptr,
                    0,
                    0,
                    reinterpret_cast<void*>(stream.get())),
                "scanner border launch");
            std::array<float, kElements> actual{};
            require_cuda(
                cudaMemcpyAsync(
                    actual.data(),
                    red,
                    kPlaneBytes,
                    cudaMemcpyDeviceToHost,
                    stream.get()),
                "scanner border output download");
            require_cuda(
                cudaStreamSynchronize(stream.get()),
                "scanner border completion");
            float maximumError = 0.0f;
            for (std::size_t index = 0; index < actual.size(); ++index) {
                maximumError = std::max(
                    maximumError,
                    std::abs(actual[index] - kExpected[index]));
            }
            results.record(
                "scanner-post/half-sample-reflect-border",
                maximumError <= 2.0e-6f,
                "maximum_absolute_error=" + std::to_string(maximumError));
        } catch (const std::exception& error) {
            results.record(
                "scanner-post/half-sample-reflect-border",
                false,
                error.what());
        }
    }

    void run_focused_cuda_reference_rows(
        const Arguments& arguments,
        Results& results) {
        try {
            const std::filesystem::path manifestPath =
                arguments.fixtureRoot / "manifest.json";
            const nlohmann::json manifest = read_manifest(manifestPath);
            if (manifest.at("schema").get<std::string>() != kFixtureSchema ||
                manifest.at("reference").at("commit").get<std::string>() !=
                    kReferenceCommit ||
                manifest.at("reference")
                        .at("hostGate1Identity")
                        .at("expectedRecipeHash")
                        .get<std::string>() != kHostGate1RecipeHash) {
                throw std::runtime_error("fixture provenance mismatch");
            }
            const std::filesystem::path binaryPath =
                arguments.fixtureRoot /
                manifest.at("binary").at("file").get<std::string>();
            const std::vector<std::uint8_t> binary = read_binary(binaryPath);
            if (binary.size() !=
                    manifest.at("binary").at("byteCount").get<std::size_t>() ||
                sha256(binary.data(), binary.size()) !=
                    manifest.at("binary").at("sha256").get<std::string>()) {
                throw std::runtime_error("fixture binary checksum mismatch");
            }
            results.record(
                "reference/fixture/provenance-and-container",
                true,
                "commit=" + std::string(kReferenceCommit) +
                    " rows=" + std::to_string(manifest.at("rows").size()) +
                    " sha256=" +
                    manifest.at("binary").at("sha256").get<std::string>());

            run_reference_descriptor_rows(manifest, results);
            run_carrier_stride_rows(results);

            std::size_t descriptorInputCount = 0;
            for (const auto& fixture : manifest.at("rows")) {
                const std::string identifier = fixture.at("id").get<std::string>();
                try {
                    const ResolvedRow row = resolve_row(fixture.at("recipeInput"));
                    const SemanticPlanes<float> source = read_planes<float>(
                        fixture.at("source"),
                        binary,
                        "<f4",
                        row.width,
                        row.height);
                    const SemanticPlanes<double> expected = read_planes<double>(
                        fixture.at("expected"),
                        binary,
                        "<f8",
                        row.width,
                        row.height);
                    const bool downstream = fixture.contains("downstreamExpected");
                    const CandidateOutput candidate =
                        run_candidate(row, source, downstream);
                    std::string detail;
                    const bool operatorPassed = compare_operator_row(
                        fixture, source, expected, candidate, detail);
                    results.record(
                        "reference/operator/" + identifier,
                        operatorPassed,
                        fixture.at("category").get<std::string>() + " " + detail);

                    if (identifier == "back_reflection_only") {
                        bool schedulePassed = row.descriptor.scatterAmount == 0.0f;
                        for (const auto& channel : row.descriptor.channels) {
                            schedulePassed = schedulePassed &&
                                             channel.totalStrength >= 0.0f;
                        }
                        results.record(
                            "reference/launch-schedule/back-reflection-only",
                            schedulePassed,
                            "scatter-clear=0 tail=0 core=0 scatter-finalize=0 "
                            "bounce=three-independent-per-active-channel "
                            "back-reflection-finalize=one-per-active-channel");
                    }

                    if (downstream) {
                        if (!candidate.downstreamLog) {
                            throw std::runtime_error("missing downstream candidate");
                        }
                        const SemanticPlanes<double> downstreamExpected =
                            read_planes<double>(
                                fixture.at("downstreamExpected"),
                                binary,
                                "<f8",
                                row.width,
                                row.height);
                        std::string downstreamDetail;
                        const bool downstreamPassed = compare_downstream(
                            source,
                            downstreamExpected,
                            *candidate.downstreamLog,
                            downstreamDetail);
                        results.record(
                            "reference/downstream/" + identifier,
                            downstreamPassed,
                            downstreamDetail);
                    }

                    if (fixture.contains("descriptorInput")) {
                        ++descriptorInputCount;
                        const auto& input = fixture.at("descriptorInput");
                        const ResolvedRow identity = resolve_row(input);
                        const std::uint64_t expectedHash =
                            parse_hash(input.at("expectedRecipeHash"));
                        const bool identityPassed =
                            expectedHash != 0 &&
                            identity.recipe.hash == expectedHash &&
                            identity.descriptor.recipeHash == expectedHash &&
                            input.at("profile")
                                    .at("filmProfileKey")
                                    .get<std::string>() == "kodak_portra_400" &&
                            input.at("profile")
                                    .at("filmProfileAssetVersionToken")
                                    .get<std::uint64_t>() ==
                                kProfileAssetVersionToken;
                        std::ostringstream identityDetail;
                        identityDetail << std::hex << std::setfill('0')
                                       << "recipe=0x" << std::setw(16)
                                       << identity.recipe.hash
                                       << " descriptor=0x" << std::setw(16)
                                       << identity.descriptor.recipeHash;
                        results.record(
                            "reference/descriptor-input/" + identifier,
                            identityPassed,
                            identityDetail.str());
                    }
                } catch (const std::exception& error) {
                    results.record(
                        "reference/operator/" + identifier,
                        false,
                        error.what());
                }
            }
            results.record(
                "reference/fixture/exactly-one-descriptor-input",
                descriptorInputCount == 1,
                "count=" + std::to_string(descriptorInputCount));

            std::vector<float> nonfiniteActual{std::numeric_limits<float>::quiet_NaN()};
            std::vector<double> finiteExpected{0.0};
            results.record(
                "reference/comparison/non-finite-rejected",
                !absolute_error(nonfiniteActual, finiteExpected).finite,
                "candidate NaN must fail every numerical gate");

            ScatterHalationFrameDescriptor bindingDescriptor;
            bindingDescriptor.recipeHash = 1;
            JuicerCuda::ScatterHalationPreparedView invalidView;
            invalidView.descriptor = &bindingDescriptor;
            invalidView.fullFrameWidth = 1;
            invalidView.fullFrameHeight = 1;
            const auto bindingFailure =
                JuicerCuda::launch_scatter_halation(invalidView, nullptr);
            results.record(
                "reference/launcher/mechanical-binding-rejected",
                bindingFailure.status == cudaErrorInvalidValue &&
                    bindingFailure.stage ==
                        JuicerCuda::ScatterHalationLaunchStage::Binding &&
                    bindingFailure.channel ==
                        JuicerCuda::ScatterHalationLaunchChannel::None &&
                    bindingFailure.gaussianIndex == -1,
                "closed result attributes only the binding observation boundary");
        } catch (const std::exception& error) {
            results.record("reference/fixture/fatal", false, error.what());
        }
    }
} // namespace ScatterHalationValidation

namespace {
    JuicerCuda::ResourceManager::DeviceContextKey current_context_key(
        int deviceIndex) {
        require_cuda(cudaSetDevice(deviceIndex), "cudaSetDevice");
        require_cuda(cudaFree(nullptr), "cudaFree(nullptr)");
        CUcontext context = nullptr;
        if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context) {
            throw std::runtime_error("active CUDA context unavailable");
        }
        return {deviceIndex, context};
    }

    JuicerCuda::Diffusion::StagePlaneSet run_camera_diffusion_from_source(
        JuicerProcess::Root::PreparedCudaFrame& frame,
        const Spektrafilm::DiffusionFrameSetDescriptor& frameSet,
        const SemanticPlanes<float>& source,
        int width,
        int height,
        cudaStream_t stream) {
        auto prepared = frame.diffusion_resources();
        if (!prepared.active || !frameSet.camera) {
            throw std::runtime_error("camera diffusion prepared view unavailable");
        }
        const Spektrafilm::DiffusionStageTileGeometry* geometry = nullptr;
        for (std::size_t index = 0;
             index < prepared.executionDescriptor.stageCount;
             ++index) {
            const auto& candidate =
                prepared.executionDescriptor.stages[index];
            if (candidate.stage ==
                Spektrafilm::DiffusionLinearStage::CameraFilmLinear) {
                geometry = &candidate;
                break;
            }
        }
        if (!geometry ||
            geometry->spectrumKeyIndex >= prepared.spectrumCount) {
            throw std::runtime_error("camera diffusion geometry unavailable");
        }
        auto planes = prepared.execution.stagePlanes;
        const std::size_t elements =
            static_cast<std::size_t>(width) * height;
        const std::size_t bytes = elements * sizeof(float);
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const std::vector<float> storage = visual_to_storage(
                source[channel], width, height);
            float* destination =
                channel == 0   ? planes.redSensitive
                : channel == 1 ? planes.greenSensitive
                               : planes.blueSensitive;
            require_cuda(
                cudaMemcpyAsync(
                    destination,
                    storage.data(),
                    bytes,
                    cudaMemcpyHostToDevice,
                    stream),
                "camera carrier upload");
        }
        require_cuda(
            cudaMemsetAsync(planes.auxiliary, 0, bytes, stream),
            "camera auxiliary clear");
        frame.mark_diffusion_work_enqueued();
        JuicerCuda::Diffusion::StageLaunchRequest request{};
        request.layout = prepared.executionDescriptor.layout;
        request.geometry = *geometry;
        request.fullFrame = frameSet.fullFrame;
        request.spectra =
            prepared.spectra[geometry->spectrumKeyIndex].spectra;
        request.execution = prepared.execution.execution;
        request.planes = &planes;
        request.stream = stream;
        const JuicerCuda::Diffusion::LaunchResult result =
            JuicerCuda::Diffusion::launch_stage(request);
        if (!result.ok()) {
            throw std::runtime_error(
                "camera diffusion launch status=" +
                std::to_string(result.code));
        }
        return planes;
    }

    SemanticPlanes<float> download_visual_carrier(
        const JuicerCuda::CameraFilmLinearExposurePlanes& carrier,
        int width,
        int height,
        cudaStream_t stream) {
        const std::size_t elements =
            static_cast<std::size_t>(width) * height;
        const std::size_t bytes = elements * sizeof(float);
        SemanticPlanes<float> storage;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            storage[channel].resize(elements);
            const float* source =
                channel == 0   ? carrier.redSensitive
                : channel == 1 ? carrier.greenSensitive
                               : carrier.blueSensitive;
            require_cuda(
                cudaMemcpyAsync(
                    storage[channel].data(),
                    source,
                    bytes,
                    cudaMemcpyDeviceToHost,
                    stream),
                "camera carrier download");
        }
        require_cuda(cudaStreamSynchronize(stream), "camera carrier synchronize");
        SemanticPlanes<float> visual;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            visual[channel] = storage_to_visual(
                storage[channel], width, height);
        }
        return visual;
    }

    void run_captured_route(
        Spektrafilm::ScanRoute route,
        const nlohmann::json& fixture,
        const std::vector<std::uint8_t>& binary,
        const ResolvedRow& row,
        const SemanticPlanes<float>& expectedSource,
        const SemanticPlanes<double>& expected,
        Results& results,
        std::uint64_t identity) {
        Spektrafilm::DiffusionFilterAuthoredControls cameraDiffusion;
        cameraDiffusion.active = true;
        cameraDiffusion.strength = 1.0e-12;
        cameraDiffusion.spatialScale = 1.0e-12;
        PreparedRouteInputs inputs = build_prepared_route_inputs(
            route,
            row.controls,
            cameraDiffusion,
            row.pixelSizeUm,
            row.width,
            row.height);
        const std::uint64_t expectedHash = parse_hash(
            fixture.at("descriptorInput").at("expectedRecipeHash"));
        if (!inputs.scatter_descriptor() ||
            inputs.scatter_descriptor()->recipeHash != expectedHash) {
            throw std::runtime_error("captured route descriptor identity mismatch");
        }
        CudaStream stream;
        const auto contextKey = current_context_key(0);
        auto request = inputs.request(row.width, row.height);
        std::string diagnostic;
        auto frame = JuicerProcess::root().prepare_cuda_frame(
            contextKey,
            inputs.submission_snapshot(contextKey, identity),
            request,
            {},
            stream.get(),
            diagnostic);
        if (!frame.active()) {
            throw std::runtime_error(diagnostic);
        }
        auto stagePlanes = run_camera_diffusion_from_source(
            frame,
            *inputs.diffusion_frame_set(),
            expectedSource,
            row.width,
            row.height,
            stream.get());
        JuicerCuda::CameraFilmLinearExposurePlanes carrier{
            stagePlanes.redSensitive,
            stagePlanes.greenSensitive,
            stagePlanes.blueSensitive,
            stagePlanes.rowStrideFloats};
        const SemanticPlanes<float> capturedSource = download_visual_carrier(
            carrier, row.width, row.height, stream.get());
        double capturedSourceMaximumDelta = 0.0;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            for (std::size_t index = 0;
                 index < capturedSource[channel].size();
                 ++index) {
                capturedSourceMaximumDelta = std::max(
                    capturedSourceMaximumDelta,
                    std::abs(
                        static_cast<double>(capturedSource[channel][index]) -
                        static_cast<double>(expectedSource[channel][index])));
            }
        }
        auto view = frame.scatter_halation_resources();
        if (view.carrierSource !=
            JuicerCuda::ScatterHalationCarrierSource::CameraDiffusionStagePlanes) {
            throw std::runtime_error("captured route carrier mode mismatch");
        }
        view.currentCarrier = carrier;
        const auto launch =
            JuicerCuda::launch_scatter_halation(view, stream.get());
        if (launch.status != cudaSuccess) {
            throw std::runtime_error(
                "captured route launch status=" +
                std::to_string(static_cast<int>(launch.status)));
        }
        CandidateOutput candidate;
        candidate.finalExposure = download_visual_carrier(
            carrier, row.width, row.height, stream.get());
        std::string comparisonDetail;
        const bool comparisonPassed = compare_operator_row(
            fixture,
            capturedSource,
            expected,
            candidate,
            comparisonDetail);
        std::string finishDiagnostic;
        const bool finished = frame.finish(stream.get(), finishDiagnostic);
        results.record(
            std::string("captured-carrier/") +
                Spektrafilm::scan_route_key(route),
            comparisonPassed && finished,
            "captured_e0_reused_sample_identically=1 seed_to_e0_max_delta=" +
                std::to_string(capturedSourceMaximumDelta) + " " +
                comparisonDetail + " " + finishDiagnostic);
        (void)binary;
    }

    struct AsymmetricHdrRgb {
        float red;
        float green;
        float blue;
    };
    static_assert(sizeof(AsymmetricHdrRgb) == 3 * sizeof(float));

    AsymmetricHdrRgb asymmetric_hdr_rgb(
        std::size_t index) {
        const float ramp = static_cast<float>((index * 37u) % 1021u) /
                           1020.0f;
        return {
            (index % 257u == 0u) ? 8.0f : 0.05f + 1.35f * ramp,
            (index % 383u == 0u) ? 4.0f : 0.02f + 0.95f * (1.0f - ramp),
            (index % 509u == 0u) ? 2.5f : -0.05f + 0.75f * ramp};
    }

    std::string asymmetric_hdr_source_checksum(std::size_t elements) {
        constexpr std::size_t kChunkElements = 4096;
        Sha256 checksum;
        std::vector<AsymmetricHdrRgb> chunk(
            std::min(elements, kChunkElements));
        for (std::size_t offset = 0; offset < elements;
             offset += kChunkElements) {
            const std::size_t count = std::min(
                elements - offset, kChunkElements);
            for (std::size_t index = 0; index < count; ++index) {
                chunk[index] = asymmetric_hdr_rgb(offset + index);
            }
            checksum.update(
                reinterpret_cast<const std::uint8_t*>(chunk.data()),
                count * sizeof(AsymmetricHdrRgb));
        }
        return checksum.finish();
    }

    void write_performance_report(
        const Arguments& arguments,
        const nlohmann::json& report) {
        std::filesystem::create_directories(
            arguments.performanceOutput.parent_path());
        std::ofstream output(arguments.performanceOutput);
        if (!output) {
            throw std::runtime_error("cannot write performance report");
        }
        output << report.dump(2) << '\n';
    }
} // namespace

namespace ScatterHalationValidation {
    void run_captured_carrier_rows(
        const Arguments& arguments,
        Results& results) {
        try {
            const nlohmann::json manifest = read_manifest(
                arguments.fixtureRoot / "manifest.json");
            const auto fixture = std::find_if(
                manifest.at("rows").begin(),
                manifest.at("rows").end(),
                [](const nlohmann::json& candidate) {
                    return candidate.at("id").get<std::string>() ==
                           "combined_still_strong_1920";
                });
            if (fixture == manifest.at("rows").end() ||
                !fixture->contains("descriptorInput")) {
                throw std::runtime_error("captured-carrier fixture unavailable");
            }
            const std::vector<std::uint8_t> binary = read_binary(
                arguments.fixtureRoot /
                manifest.at("binary").at("file").get<std::string>());
            const ResolvedRow row = resolve_row(fixture->at("descriptorInput"));
            const SemanticPlanes<float> source = read_planes<float>(
                fixture->at("source"), binary, "<f4", row.width, row.height);
            const SemanticPlanes<double> expected = read_planes<double>(
                fixture->at("expected"), binary, "<f8", row.width, row.height);
            run_captured_route(
                Spektrafilm::ScanRoute::NegativeDirectScan,
                *fixture,
                binary,
                row,
                source,
                expected,
                results,
                0x4101);
            run_captured_route(
                Spektrafilm::ScanRoute::NegativePrintScan,
                *fixture,
                binary,
                row,
                source,
                expected,
                results,
                0x4102);
        } catch (const std::exception& error) {
            results.record("captured-carrier/fatal", false, error.what());
        }
    }

    void run_performance_rows(
        const Arguments& arguments,
        Results& results) {
        try {
            require_cuda(cudaSetDevice(arguments.deviceIndex), "cudaSetDevice");
            const nlohmann::json manifest = read_manifest(
                arguments.fixtureRoot / "manifest.json");
            const auto fixture = std::find_if(
                manifest.at("rows").begin(),
                manifest.at("rows").end(),
                [](const nlohmann::json& candidate) {
                    return candidate.at("id").get<std::string>() ==
                           "combined_still_strong_1920";
                });
            if (fixture == manifest.at("rows").end()) {
                throw std::runtime_error("performance fixture unavailable");
            }
            const ResolvedRow fixtureRow = resolve_row(
                fixture->at("descriptorInput"));
            const bool active = arguments.performanceMode == "active";
            const ScatterHalationControls controls =
                active ? fixtureRow.controls : ScatterHalationControls{};
            const float pixelSizeUm = static_cast<float>(
                35000.0 / static_cast<double>(
                              std::max(arguments.performanceWidth,
                                       arguments.performanceHeight)));
            PreparedRouteInputs inputs = build_prepared_route_inputs(
                Spektrafilm::ScanRoute::NegativeDirectScan,
                controls,
                {},
                pixelSizeUm,
                arguments.performanceWidth,
                arguments.performanceHeight);
            CudaStream stream;
            const auto contextKey = current_context_key(arguments.deviceIndex);
            auto request = inputs.request(
                arguments.performanceWidth,
                arguments.performanceHeight);
            std::string diagnostic;
            auto frame = JuicerProcess::root().prepare_cuda_frame(
                contextKey,
                inputs.submission_snapshot(contextKey, 0x50455246),
                request,
                {},
                stream.get(),
                diagnostic);
            if (!frame.active()) {
                throw std::runtime_error(diagnostic);
            }
            const std::size_t elements =
                static_cast<std::size_t>(arguments.performanceWidth) *
                arguments.performanceHeight;
            const std::uint64_t planeBytes =
                static_cast<std::uint64_t>(elements) * sizeof(float);
            std::vector<float> samples;
            if (active) {
                const auto view = frame.scatter_halation_resources();
                if (!view.descriptor ||
                    view.carrierSource !=
                        JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes) {
                    throw std::runtime_error("performance prepared view unavailable");
                }
                std::vector<float> sourceRed(elements);
                std::vector<float> sourceGreen(elements);
                std::vector<float> sourceBlue(elements);
                for (std::size_t index = 0; index < elements; ++index) {
                    const AsymmetricHdrRgb source = asymmetric_hdr_rgb(index);
                    sourceRed[index] = source.red;
                    sourceGreen[index] = source.green;
                    sourceBlue[index] = source.blue;
                }
                const auto reset_source = [&] {
                    require_cuda(cudaMemcpyAsync(
                                     view.currentCarrier.redSensitive,
                                     sourceRed.data(),
                                     planeBytes,
                                     cudaMemcpyHostToDevice,
                                     stream.get()),
                                 "reset performance red source");
                    require_cuda(cudaMemcpyAsync(
                                     view.currentCarrier.greenSensitive,
                                     sourceGreen.data(),
                                     planeBytes,
                                     cudaMemcpyHostToDevice,
                                     stream.get()),
                                 "reset performance green source");
                    require_cuda(cudaMemcpyAsync(
                                     view.currentCarrier.blueSensitive,
                                     sourceBlue.data(),
                                     planeBytes,
                                     cudaMemcpyHostToDevice,
                                     stream.get()),
                                 "reset performance blue source");
                };
                for (int index = 0; index < arguments.warmupCount; ++index) {
                    reset_source();
                    const auto launch =
                        JuicerCuda::launch_scatter_halation(view, stream.get());
                    if (launch.status != cudaSuccess) {
                        throw std::runtime_error("performance warmup launch failed");
                    }
                }
                require_cuda(
                    cudaStreamSynchronize(stream.get()),
                    "performance warmup completion");
                cudaEvent_t start = nullptr;
                cudaEvent_t stop = nullptr;
                require_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
                require_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");
                for (int index = 0; index < arguments.sampleCount; ++index) {
                    reset_source();
                    require_cuda(
                        cudaEventRecord(start, stream.get()),
                        "cudaEventRecord(start)");
                    const auto launch =
                        JuicerCuda::launch_scatter_halation(view, stream.get());
                    if (launch.status != cudaSuccess) {
                        throw std::runtime_error("performance timed launch failed");
                    }
                    require_cuda(
                        cudaEventRecord(stop, stream.get()),
                        "cudaEventRecord(stop)");
                    require_cuda(
                        cudaEventSynchronize(stop),
                        "cudaEventSynchronize(stop)");
                    float elapsed = 0.0f;
                    require_cuda(
                        cudaEventElapsedTime(&elapsed, start, stop),
                        "cudaEventElapsedTime");
                    samples.push_back(elapsed);
                }
                require_cuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");
                require_cuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
            }
            if (arguments.holdMilliseconds > 0) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(arguments.holdMilliseconds));
            }
            std::vector<float> sorted = samples;
            std::sort(sorted.begin(), sorted.end());
            const auto percentile = [&](double fraction) {
                if (sorted.empty()) {
                    return 0.0f;
                }
                const std::size_t index = std::min(
                    sorted.size() - 1,
                    static_cast<std::size_t>(
                        std::ceil(fraction * sorted.size()) - 1.0));
                return sorted[index];
            };
            int driverVersion = 0;
            int runtimeVersion = 0;
            require_cuda(
                cudaDriverGetVersion(&driverVersion),
                "cudaDriverGetVersion");
            require_cuda(
                cudaRuntimeGetVersion(&runtimeVersion),
                "cudaRuntimeGetVersion");
            cudaDeviceProp properties{};
            require_cuda(
                cudaGetDeviceProperties(&properties, arguments.deviceIndex),
                "cudaGetDeviceProperties");
            nlohmann::json report{
                {"schema", "scatter-halation-performance.v1"},
                {"mode", arguments.performanceMode},
                {"deviceIndex", arguments.deviceIndex},
                {"deviceName", properties.name},
                {"driverVersion", driverVersion},
                {"runtimeVersion", runtimeVersion},
                {"width", arguments.performanceWidth},
                {"height", arguments.performanceHeight},
                {"pixelSizeUm", pixelSizeUm},
                {"warmupCount", arguments.warmupCount},
                {"sampleCount", active ? arguments.sampleCount : 0},
                {"samplesMs", samples},
                {"medianMs", percentile(0.5)},
                {"p95Ms", percentile(0.95)},
                {"descriptorRecipeHash",
                 inputs.scatter_descriptor()
                     ? inputs.scatter_descriptor()->recipeHash
                     : 0},
                {"sourceChecksum", asymmetric_hdr_source_checksum(elements)},
                {"sourceChecksumEncoding",
                 "Float32 index-major semantic RGB asymmetric-hdr-v1"},
                {"ledgerRecords", active ? 2 : 0},
                {"filterBytes", active ? 2 * planeBytes : 0},
                {"carrierBytes", active ? 3 * planeBytes : 0},
                {"admittedBytes", active ? 5 * planeBytes : 0}};
            write_performance_report(arguments, report);
            std::string finishDiagnostic;
            const bool finished = frame.finish(stream.get(), finishDiagnostic);
            results.record(
                "performance/" + arguments.performanceMode + "/" +
                    std::to_string(arguments.performanceWidth),
                finished,
                finishDiagnostic);
        } catch (const std::exception& error) {
            results.record("performance/fatal", false, error.what());
        }
    }
} // namespace ScatterHalationValidation
