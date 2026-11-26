#include "../route/route_fixture.h"
#include "integration_test.h"

#include "../ffi/prepared_product_fixture.h"

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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "Cuda/JuicerCudaFailure.h"

#include "Cuda/Diffusion/JuicerCudaDiffusion.h"
#include "Cuda/Film/JuicerCudaHighlightBoost.h"
#include "Cuda/Film/JuicerCudaScatterHalation.h"
#include "Cuda/JuicerCudaPayloads.h"
#include "ProcessRoot.h"
#include "RenderValues.h"
#include "Scanner.h"
#include "ScatterHalation.h"
#include "spatial_optics_fixture.h"
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
    using ScatterHalationValidation::ImageExtent;
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

    nlohmann::json advanced_manifest(const Arguments& arguments) {
        std::ifstream stream(arguments.fixtureRoot.parent_path() / "parity-fixtures/manifest.json");
        auto manifest = nlohmann::json::parse(stream);
        const auto& references = manifest.at("references");
        if (manifest.at("schema") != "spektrafilm-halation-feature-parity.v1" ||
            references.at("main").at("revision") != "3bb2c2d2801ff68b92019cf1dbcbb133d60832bc" ||
            references.at("experimental").at("revision") != "28bf883e1672e884307edc75852549376e13644e" ||
            references.at("main").at("original_binary_sha256") != references.at("experimental").at("original_binary_sha256")) {
            throw std::runtime_error("advanced halation oracle identity mismatch");
        }
        return manifest;
    }

    std::vector<std::string> parity_ids(const nlohmann::json& manifest, bool spatial) {
        std::vector<std::string> ids;
        for (const auto& row : manifest.at("rows")) {
            if (row.contains("pixel_size_um") == spatial) {
                ids.push_back(row.at("id").get<std::string>());
            }
        }
        return ids;
    }

    std::vector<std::uint8_t> load_parity_binary(const Arguments& arguments, const nlohmann::json& manifest) {
        const auto& payload = manifest.at("binary");
        auto binary = read_binary(arguments.fixtureRoot.parent_path() / "parity-fixtures" / payload.at("file").get<std::string>());
        if (binary.size() != payload.at("size").get<std::size_t>() ||
            sha256(binary.data(), binary.size()) != payload.at("sha256").get_ref<const std::string&>()) {
            throw std::runtime_error("advanced halation payload checksum mismatch");
        }
        return binary;
    }

    template <typename Sample, typename Stored = double>
    SemanticPlanes<Sample> read_parity_planes(
        const nlohmann::json& record,
        const std::vector<std::uint8_t>& binary,
        ImageExtent extent) {
        const auto count = static_cast<std::size_t>(extent.width) * extent.height;
        const auto offset = record.at("offset").get<std::size_t>();
        const auto size = record.at("size").get<std::size_t>();
        constexpr auto dtype = std::is_same_v<Stored, float> ? "<f4" : "<f8";
        if (record.at("dtype") != dtype || size != count * 3 * sizeof(Stored) ||
            offset > binary.size() || size > binary.size() - offset ||
            sha256(binary.data() + offset, size) != record.at("sha256").get_ref<const std::string&>()) {
            throw std::runtime_error("advanced halation plane checksum or extent mismatch");
        }
        SemanticPlanes<Sample> planes;
        for (std::size_t channel = 0; channel < planes.size(); ++channel) {
            planes[channel].resize(count);
            for (std::size_t index = 0; index < count; ++index) {
                Stored value{};
                std::memcpy(&value, binary.data() + offset + (index * 3 + channel) * sizeof(Stored), sizeof(Stored));
                planes[channel][index] = static_cast<Sample>(value);
            }
        }
        return planes;
    }

    FjPreparationHalation advanced_controls(const nlohmann::json& row) {
        const auto& settings = row.at("parameters");
        FjPreparationHalation result{};
        result.active = settings.at("active").get<bool>();
        result.scatter_amount = settings.at("scatter_amount").get<float>();
        result.scatter_spatial_scale = settings.at("scatter_spatial_scale").get<float>();
        result.halation_amount = settings.at("halation_amount").get<float>();
        result.halation_spatial_scale = settings.at("halation_spatial_scale").get<float>();
        result.reflection_source = 1;
        result.shape.present = 1;
        result.shape.bounce_count = settings.at("halation_n_bounces").get<std::int32_t>();
        result.shape.bounce_decay = settings.at("halation_bounce_decay").get<double>();
        result.shape.renormalize = settings.at("halation_renormalize").get<bool>();
        for (std::size_t channel = 0; channel < 3; ++channel) {
            result.reflection_strength[channel] = settings.at("halation_strength").at(channel).get<double>();
            result.first_sigma_um[channel] = settings.at("halation_first_sigma_um").at(channel).get<double>();
            result.shape.scatter_core_um[channel] = settings.at("scatter_core_um").at(channel).get<double>();
            result.shape.scatter_tail_um[channel] = settings.at("scatter_tail_um").at(channel).get<double>();
            result.shape.scatter_tail_weight[channel] = settings.at("scatter_tail_weight").at(channel).get<double>();
        }
        return result;
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
        std::array<float, 3> firstSigmaUm{};
        std::array<float, 3> primaryAmount{};
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
            row.firstSigmaUm[channel] = parse_float(
                profile.at("halationFirstSigmaUmFloat32Bits").at(channel));
            row.primaryAmount[channel] = parse_float(
                profile.at("halationPrimaryAmountFloat32Bits").at(channel));
        }
        if (!JuicerTestSupport::prepare_scatter_halation(
                row.controls, row.firstSigmaUm, row.primaryAmount, row.recipe, diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        row.pixelSizeUm = parse_float(input.at("pixelSizeUmFloat32Bits"));
        row.width = input.at("fullFrameWidth").get<int>();
        row.height = input.at("fullFrameHeight").get<int>();
        std::optional<ScatterHalationFrameDescriptor> descriptor;
        if (!JuicerTestSupport::prepare_halation_frame(
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
        ScatterHalationValidation::ImageExtent extent) {
        std::vector<T> storage(visual.size());
        for (int visualY = 0; visualY < extent.height; ++visualY) {
            const int storageY = extent.height - 1 - visualY;
            std::copy_n(
                visual.begin() +
                    static_cast<std::size_t>(visualY) * extent.width,
                extent.width,
                storage.begin() +
                    static_cast<std::size_t>(storageY) * extent.width);
        }
        return storage;
    }

    template <typename T>
    std::vector<T> storage_to_visual(
        const std::vector<T>& storage,
        ScatterHalationValidation::ImageExtent extent) {
        return visual_to_storage(storage, extent);
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

    class CudaTimingEvent final {
    public:
        CudaTimingEvent() {
            require_cuda(cudaEventCreate(&_event), "create benchmark timing event");
        }
        ~CudaTimingEvent() {
            cudaEventDestroy(_event);
        }
        CudaTimingEvent(const CudaTimingEvent&) = delete;
        CudaTimingEvent& operator=(const CudaTimingEvent&) = delete;
        [[nodiscard]] cudaEvent_t get() const noexcept {
            return _event;
        }

    private:
        cudaEvent_t _event = nullptr;
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
                visual_to_storage(source[channel], {row.width, row.height});
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
        const JuicerCuda::ScatterHalationLaunchResult launch = JuicerCuda::launch_scatter_halation(view, stream.get());
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
                    storage_to_visual(storage, {row.width, row.height});
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
                storage_to_visual(storage, {row.width, row.height});
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

    bool compare_large_gain(
        const SemanticPlanes<float>& source,
        const SemanticPlanes<double>& expected,
        const SemanticPlanes<float>& actual,
        const FjPreparationHalation& controls,
        std::string& detail) {
        if (controls.scatter_amount != 0.0f || controls.shape.renormalize != 0) {
            return false;
        }
        const auto ulps = [](float left, float right) {
            const auto a = std::bit_cast<std::uint32_t>(left);
            const auto b = std::bit_cast<std::uint32_t>(right);
            return std::max(a, b) - std::min(a, b);
        };
        constexpr std::array channels{'R', 'G', 'B'};
        std::ostringstream report;
        report << std::scientific;
        bool passed = true;
        for (std::size_t channel = 0; channel < channels.size(); ++channel) {
            const auto& input = source[channel];
            const auto& reference = expected[channel];
            const auto& output = actual[channel];
            if (input.empty() || reference.size() != input.size() ||
                output.size() != input.size() ||
                !std::all_of(input.begin(), input.end(), [&](float sample) {
                    return sample == input.front();
                })) {
                return false;
            }
            const double amount = controls.reflection_strength[channel] * controls.halation_amount;
            const float flatGain = static_cast<float>(input.front() * (1.0 + amount));
            const double peak = *std::max_element(reference.begin(), reference.end());
            const double scale = std::max(1.0, peak);
            double squaredError = 0.0;
            double maxRelative = 0.0;
            std::uint32_t maxReferenceUlps = 0;
            std::uint32_t maxGainUlps = 0;
            for (std::size_t i = 0; i < output.size(); ++i) {
                if (!std::isfinite(output[i]) || !std::isfinite(reference[i]) ||
                    output[i] < 0.0f || reference[i] < 0.0) {
                    return false;
                }
                maxReferenceUlps = std::max(maxReferenceUlps, ulps(output[i], static_cast<float>(reference[i])));
                maxGainUlps = std::max(maxGainUlps, ulps(output[i], flatGain));
                const double relative = std::abs(output[i] - reference[i]) / scale;
                squaredError += relative * relative;
                maxRelative = std::max(maxRelative, relative);
            }
            passed = passed && maxReferenceUlps <= 1 && maxGainUlps <= 1;
            report << ' ' << channels[channel] << "(peak=" << peak
                   << ",reference_ulps=" << maxReferenceUlps << ",flat_gain_ulps=" << maxGainUlps
                   << ",relative_max=" << maxRelative
                   << ",relative_rms=" << std::sqrt(squaredError / static_cast<double>(output.size())) << ')';
        }
        detail += report.str();
        return passed;
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

    struct DescriptorScale {
        double spatialScale;
        float pixelSizeUm;
    };

    ScatterHalationFrameDescriptor build_descriptor(DescriptorScale scale) {
        ScatterHalationRawControls raw;
        raw.active = true;
        raw.scatterAmount = 1.0;
        raw.scatterSpatialScale = scale.spatialScale;
        raw.halationAmount = 0.0;
        raw.halationSpatialScale = 1.0;
        ScatterHalationControls controls;
        std::string diagnostic;
        if (!Spektrafilm::build_scatter_halation_controls(
                raw, controls, diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        const std::array<float, 3> firstSigmaUm{65.0f, 65.0f, 65.0f};
        const std::array<float, 3> primaryAmount{0.08f, 0.02f, 0.0f};
        ScatterHalationOpticsRecipe recipe;
        if (!JuicerTestSupport::prepare_scatter_halation(
                controls, firstSigmaUm, primaryAmount, recipe, diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        std::optional<ScatterHalationFrameDescriptor> descriptor;
        if (!JuicerTestSupport::prepare_halation_frame(
                recipe, scale.pixelSizeUm, descriptor, diagnostic) ||
            !descriptor) {
            throw std::runtime_error(diagnostic);
        }
        return *descriptor;
    }

    void run_carrier_stride_rows(Results& results) {
        // Carrier layout must not alter the operator or leak poisoned padding
        // into the image, including complete and partial row batches and tiles.
        constexpr std::array extents{
            std::array{1, 33}, std::array{33, 1}, std::array{19, 7}, std::array{19, 8}, std::array{19, 9}, std::array{19, 15}, std::array{19, 16}, std::array{19, 17}, std::array{31, 31}, std::array{32, 32}, std::array{33, 33}, std::array{63, 65}, std::array{65, 63}, std::array{129, 67}};
        std::vector<std::string> ids;
        ids.reserve(extents.size());
        for (const auto extent : extents) {
            ids.push_back(std::to_string(extent[0]) + "x" +
                          std::to_string(extent[1]));
        }
        for (const auto extent : extents) {
            const std::string name = "operator/carrier-stride/" +
                                     std::to_string(extent[0]) + "x" + std::to_string(extent[1]);
            try {
                ResolvedRow row;
                row.width = extent[0];
                row.height = extent[1];
                row.descriptor = build_descriptor({2.0, 0.5f});
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
            {static_cast<double>(below), 1.0f});
        const auto exactDescriptor = build_descriptor({1.5, 1.0f});
        results.record(
            "reference/descriptor/green-core-dispatch-seam",
            belowDescriptor.channels[1].core.kind ==
                    ScatterHalationGaussianKind::FirReflect &&
                exactDescriptor.channels[1].core.kind ==
                    ScatterHalationGaussianKind::YvvReplicate,
            "pixel_size_um=1 green_core_um=2 retained_scale=nextafter(1.5,0)/1.5");

        const auto& nativeRows = manifest.at("nativeDoubleAcquisitionRows").get_ref<const nlohmann::json::array_t&>();
        std::set<std::string_view> narrowingRemaining{
            "native-double-narrows-onto-dispatch-seam",
            "native-double-below-seam-narrows-below",
        };
        bool nativePassed = true;
        std::ostringstream nativeDetail;
        for (const auto& row : nativeRows) {
            const auto& identifier = row.at("id").get_ref<const std::string&>();
            if (!narrowingRemaining.contains(identifier))
                throw std::runtime_error("unexpected or duplicate Double narrowing fixture: " + identifier);
            const std::string rawText = row.at("rawDoubleHex").get<std::string>();
            char* end = nullptr;
            const double raw = std::strtod(rawText.c_str(), &end);
            const float retained = static_cast<float>(raw);
            const bool retainedMatches = end && *end == '\0' &&
                                         std::bit_cast<std::uint32_t>(retained) ==
                                             parse_bits(row.at("retainedFloat32Bits"));
            const auto descriptor = build_descriptor({raw, 1.0f});
            const bool onto = identifier == "native-double-narrows-onto-dispatch-seam";
            const bool dispatchMatches = descriptor.channels[1].core.kind ==
                                         (onto ? ScatterHalationGaussianKind::YvvReplicate
                                               : ScatterHalationGaussianKind::FirReflect);
            nativePassed = nativePassed && retainedMatches && dispatchMatches;
            nativeDetail << identifier << '=' << row.at("retainedFloat32Bits").get<std::string>() << ' ';
            narrowingRemaining.erase(identifier);
        }
        if (!narrowingRemaining.empty())
            throw std::runtime_error("missing Double narrowing fixture comparisons");
        results.record(
            "reference/descriptor/native-double-one-time-narrowing",
            nativePassed,
            nativeDetail.str());

        constexpr std::array<int, 3> longEdgeRows{1920, 3840, 7680};
        bool formatRowsPassed = true;
        std::ostringstream formatDetail;
        for (int longEdgePixels : longEdgeRows) {
            const float pixelSize =
                static_cast<float>(35000.0 / static_cast<double>(longEdgePixels));
            const auto descriptor = build_descriptor({1.0, pixelSize});
            formatRowsPassed = formatRowsPassed && descriptor.recipeHash != 0;
            formatDetail << longEdgePixels << ':' << pixelSize << ' ';
        }
        results.record(
            "reference/descriptor/35mm-pixel-size-rows",
            formatRowsPassed,
            formatDetail.str());

        const auto extreme = build_descriptor(
            {2.0, std::numeric_limits<float>::denorm_min()});
        bool extremePassed = extreme.recipeHash != 0;
        for (const auto& channel : extreme.channels) {
            extremePassed = extremePassed &&
                            std::isfinite(channel.core.yvv[0]) &&
                            std::isfinite(channel.core.yvv[1]) &&
                            std::isfinite(channel.core.yvv[2]) &&
                            std::isfinite(channel.core.yvv[3]);
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
        constexpr std::array<float, static_cast<std::size_t>(kWidth) * kHeight> kSource{{0.03f, 0.11f, 0.29f, 0.47f, 0.83f, 0.07f, 0.19f, 0.31f, 0.61f, 0.97f, 0.13f, 0.23f, 0.41f, 0.73f, 1.09f, 0.17f, 0.37f, 0.59f, 0.89f, 1.31f}};
        constexpr std::array<float, 2 * kRadius + 1> kKernel{{0.00962005683f,
                                                              0.205423697f,
                                                              0.569912492f,
                                                              0.205423697f,
                                                              0.00962005683f}};
        constexpr std::array<float, static_cast<std::size_t>(kWidth) * kHeight> kExpected{{0.008215948f, 0.08396746f, 0.27832553f, 0.4252918f, 0.86518514f, 0.048741654f, 0.18746452f, 0.27515787f, 0.6005329f, 1.0301093f, 0.11236872f, 0.2057411f, 0.37625948f, 0.71667653f, 1.1366233f, 0.14732751f, 0.38531336f, 0.6033108f, 0.90200037f, 1.4113867f}};

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
            params.srcRowBytes =
                static_cast<std::size_t>(kWidth) * 3u * sizeof(float);
            params.dst = pixels.floats() + kElements * 3;
            params.dstRowBytes =
                static_cast<std::size_t>(kWidth) * 3u * sizeof(float);
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
            const auto& fixtures = manifest.at("rows").get_ref<const nlohmann::json::array_t&>();
            results.record(
                "reference/fixture/provenance-and-container",
                true,
                "commit=" + std::string(kReferenceCommit) +
                    " rows=" + std::to_string(fixtures.size()) +
                    " sha256=" + manifest.at("binary").at("sha256").get<std::string>());

            run_reference_descriptor_rows(manifest, results);
            run_carrier_stride_rows(results);

            std::set<std::string_view> operatorRemaining{
                "fir_center_impulse",
                "fir_corner_tiny",
                "fir_constant",
                "yvv_asymmetric_ramp",
                "dispatch_below_three",
                "dispatch_exact_three",
                "back_reflection_only",
                "back_reflection_identity_bounces_tiny",
                "combined_still_strong_1920",
                "combined_still_no_3840",
                "combined_cine_weak_7680_hdr",
                "combined_zero_channel",
                "extrapolative_amounts_downstream",
                "maximum_scatter_amount_downstream",
                "maximum_scatter_scale_downstream",
                "maximum_halation_amount_downstream",
                "maximum_halation_scale_downstream",
            };
            std::set<std::string_view> downstreamRemaining{
                "extrapolative_amounts_downstream",
                "maximum_scatter_amount_downstream",
                "maximum_scatter_scale_downstream",
                "maximum_halation_amount_downstream",
                "maximum_halation_scale_downstream",
            };
            constexpr std::string_view descriptorInputId = "combined_still_strong_1920";
            bool descriptorInputCompared = false;
            bool backReflectionCompared = false;
            for (const auto& fixture : fixtures) {
                const auto& identifier = fixture.at("id").get_ref<const std::string&>();
                if (!operatorRemaining.contains(identifier))
                    throw std::runtime_error("unexpected or duplicate operator fixture: " + identifier);
                const bool downstream = fixture.contains("downstreamExpected");
                const bool descriptorInput = fixture.contains("descriptorInput");
                if (downstream != downstreamRemaining.contains(identifier))
                    throw std::runtime_error("wrong downstream fixture membership: " + identifier);
                if (descriptorInput != (identifier == descriptorInputId))
                    throw std::runtime_error("wrong descriptor-input fixture membership: " + identifier);
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
                    const CandidateOutput candidate =
                        run_candidate(row, source, downstream);
                    std::string detail;
                    const bool operatorPassed = compare_operator_row(
                        fixture, source, expected, candidate, detail);
                    results.record(
                        "reference/operator/" + identifier,
                        operatorPassed,
                        fixture.at("category").get<std::string>() + " " + detail);
                    operatorRemaining.erase(identifier);

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
                        backReflectionCompared = true;
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
                        downstreamRemaining.erase(identifier);
                    }

                    if (descriptorInput) {
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
                        descriptorInputCompared = true;
                    }
                } catch (const std::exception& error) {
                    results.record(
                        "reference/operator/" + identifier,
                        false,
                        error.what());
                }
            }
            results.record(
                "reference/fixture/operator-completion",
                operatorRemaining.empty(),
                "missing comparisons=" + std::to_string(operatorRemaining.size()));
            results.record(
                "reference/fixture/downstream-completion",
                downstreamRemaining.empty(),
                "missing comparisons=" + std::to_string(downstreamRemaining.size()));
            results.record("reference/fixture/descriptor-input-completion", descriptorInputCompared, std::string(descriptorInputId));
            results.record("reference/fixture/back-reflection-schedule-completion", backReflectionCompared, "back_reflection_only");

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
                source[channel], {width, height});
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
                storage[channel], {width, height});
        }
        return visual;
    }

    std::string carrier_sha256(const SemanticPlanes<float>& planes) {
        Sha256 digest;
        for (const auto& plane : planes) {
            digest.update(
                reinterpret_cast<const std::uint8_t*>(plane.data()),
                plane.size() * sizeof(float));
        }
        return digest.finish();
    }

    void run_captured_route(
        JuicerTestSupport::RuntimeFixture& runtime,
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
            runtime,
            route,
            row.controls,
            cameraDiffusion,
            row.pixelSizeUm,
            {row.width, row.height});
        const std::uint64_t expectedHash = parse_hash(
            fixture.at("descriptorInput").at("expectedRecipeHash"));
        CudaStream stream;
        const auto contextKey = current_context_key(0);
        JuicerCuda::Failure diagnostic;
        auto frame = inputs.prepare_frame(
            contextKey,
            identity,
            {row.width, row.height},
            stream.get(),
            diagnostic,
            expectedHash);
        if (!frame.active()) {
            throw std::runtime_error(diagnostic.diagnostic);
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
        JuicerCuda::Failure finishDiagnostic;
        const bool finished = frame.finish(stream.get(), finishDiagnostic);
        const bool passed = comparisonPassed && finished;
        results.record(
            std::string("captured-carrier/") +
                Spektrafilm::scan_route_key(route),
            passed,
            "captured_e0_reused_sample_identically=1 seed_to_e0_max_delta=" +
                std::to_string(capturedSourceMaximumDelta) + " " +
                "post_diffusion_rgb_sha256=" + carrier_sha256(capturedSource) + " " +
                "post_spatial_rgb_sha256=" + carrier_sha256(candidate.finalExposure) + " " +
                comparisonDetail + " " + finishDiagnostic.diagnostic);
        (void)binary;
    }

    struct AsymmetricHdrRgb {
        float red;
        float green;
        float blue;
    };
    static_assert(sizeof(AsymmetricHdrRgb) == 3 * sizeof(float));

    enum class HighlightDensity : std::uint8_t {
        Spatial,
        Sparse,
        Dense
    };

    AsymmetricHdrRgb asymmetric_hdr_rgb(
        std::size_t index, HighlightDensity density) {
        const float ramp = static_cast<float>((index * 37u) % 1021u) /
                           1020.0f;
        const AsymmetricHdrRgb rgb{
            (index % 257u == 0u) ? 8.0f : 0.05f + 1.35f * ramp,
            (index % 383u == 0u) ? 4.0f : 0.02f + 0.95f * (1.0f - ramp),
            (index % 509u == 0u) ? 2.5f : -0.05f + 0.75f * ramp};
        if (density == HighlightDensity::Dense) {
            return {3.0f + 4.0f * rgb.red, 3.0f + 4.0f * rgb.green, 3.0f + 4.0f * std::max(0.0f, rgb.blue)};
        }
        if (density == HighlightDensity::Sparse) {
            return {rgb.red, rgb.green, std::max(0.0f, rgb.blue)};
        }
        return rgb;
    }

    std::string asymmetric_hdr_source_checksum(std::size_t elements, HighlightDensity density) {
        constexpr std::size_t kChunkElements = 4096;
        Sha256 checksum;
        std::vector<AsymmetricHdrRgb> chunk(
            std::min(elements, kChunkElements));
        for (std::size_t offset = 0; offset < elements;
             offset += kChunkElements) {
            const std::size_t count = std::min(
                elements - offset, kChunkElements);
            for (std::size_t index = 0; index < count; ++index) {
                chunk[index] = asymmetric_hdr_rgb(offset + index, density);
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
    void run_advanced_spatial_rows(const Arguments& arguments, Results& results) {
        const auto manifest = advanced_manifest(arguments);
        const auto ids = parity_ids(manifest, true);
        const auto binary = load_parity_binary(arguments, manifest);
        if (!arguments.runtimeOwner) {
            throw std::runtime_error("advanced halation runtime unavailable");
        }
        auto& runtime = *arguments.runtimeOwner;
        const auto context = current_context_key(0);
        const JuicerTestSupport::PreparationInput defaults;
        CudaStream stream;
        std::uint64_t identity = 0x4300;
        for (const auto& row : manifest.at("rows")) {
            if (!row.contains("pixel_size_um")) {
                continue;
            }
            const auto id = row.at("id").get<std::string>();
            try {
                const auto shape = row.at("shape").get<std::array<int, 3>>();
                if (shape[0] <= 0 || shape[1] <= 0 || shape[2] != 3) {
                    throw std::runtime_error("invalid advanced halation RGB extent");
                }
                const ImageExtent extent{shape[1], shape[0]};
                const auto source = read_parity_planes<float>(row.at("input"), binary, extent);
                auto controls = advanced_controls(row);
                const float pixelSize = row.at("pixel_size_um").get<float>();
                if (id == "wide-yvv-1024-above") {
                    JuicerTestSupport::PreparationInput params;
                    params.values.scatter_halation = controls;
                    auto state = JuicerTestSupport::prepare_product(runtime.borrow().assets, params);
                    FjPreparedStateView view{};
                    FjErrorBuffer error{};
                    if (state.view(view, error).category != FJ_STATUS_SUCCESS) {
                        throw std::runtime_error("wide radius fixture state unavailable");
                    }
                    const FjHalationFrameInput input{view.optics.scatter_halation, pixelSize};
                    FjScatterHalation unused{};
                    FjHalationFrameFailure failure{};
                    const auto status = fj_test_scatter_halation_frame_prepare(&input, &unused, &failure, &error);
                    const bool passed = status.category == FJ_STATUS_PREPARATION_FAILURE &&
                                        failure.reason == FJ_HALATION_FRAME_ADVANCED_RADIUS &&
                                        failure.field == FJ_HALATION_FRAME_FIELD_SCATTER_CORE &&
                                        failure.channel == 0 && failure.sigma_px > 1024.0 &&
                                        unused.recipe_hash == 0;
                    results.record(id, passed, "typed preparation rejection above the proposed radius");
                    continue;
                }
                bool passed = true;
                std::string details;
                const auto comparison = row.at("comparison").get<std::string>();
                for (const char* stage : {"post_reflection", "post_scatter"}) {
                    if (std::string_view(stage) == "post_scatter") {
                        if (controls.scatter_amount == 0.0f || controls.halation_amount == 0.0f) {
                            continue;
                        }
                        controls.halation_amount = 0.0f;
                    }
                    auto prepared = build_prepared_route_inputs(
                        runtime, Spektrafilm::ScanRoute::NegativeDirectScan, {}, {}, defaults.values.dir_couplers, pixelSize, extent, 1.0f, {}, &controls);
                    JuicerCuda::Failure failure;
                    auto frame = prepared.prepare_frame(context, ++identity, extent, stream.get(), failure);
                    if (!frame.active()) {
                        throw std::runtime_error(failure.diagnostic);
                    }
                    auto actual = source;
                    if (const auto* descriptor = prepared.scatter_descriptor()) {
                        ResolvedRow candidate{};
                        candidate.width = extent.width;
                        candidate.height = extent.height;
                        candidate.descriptor = *descriptor;
                        actual = run_candidate(candidate, source, false, 7).finalExposure;
                        const auto compact = run_candidate(candidate, source, false).finalExposure;
                        for (std::size_t channel = 0; channel < actual.size(); ++channel) {
                            for (std::size_t sample = 0; sample < actual[channel].size(); ++sample) {
                                if (std::bit_cast<std::uint32_t>(actual[channel][sample]) !=
                                    std::bit_cast<std::uint32_t>(compact[channel][sample])) {
                                    throw std::runtime_error("advanced halation compact/padded mismatch");
                                }
                            }
                        }
                    } else if (comparison != "identity") {
                        throw std::runtime_error("active advanced halation descriptor missing");
                    }
                    const auto expected = read_parity_planes<double>(row.at(stage), binary, extent);
                    if (comparison == "large-additive-gain") {
                        const bool comparisonPassed = compare_large_gain(source, expected, actual, controls, details);
                        passed = frame.finish(stream.get(), failure) && passed && comparisonPassed;
                        if (!failure.diagnostic.empty()) {
                            details += " " + failure.diagnostic;
                        }
                        continue;
                    }
                    std::array<ErrorStats, 3> stats{};
                    for (std::size_t channel = 0; channel < stats.size(); ++channel) {
                        const double peak = source_scale(source[channel]);
                        const double scale = comparison == "combined" ? std::max(1.0, peak) : 1.0;
                        stats[channel] = absolute_error(actual[channel], expected[channel], scale);
                        const auto& error = stats[channel];
                        const bool impulse = std::count_if(source[channel].begin(), source[channel].end(), [](float sample) {
                                                 return sample != 0.0f;
                                             }) == 1;
                        const double maximum = comparison == "identity" ? 0.0 : comparison == "fir" ? 2.0e-5
                                                                            : comparison == "yvv"   ? 2.0e-3
                                                                                                    : 3.0e-3;
                        const double mean = comparison == "combined" ? 3.0e-4 : comparison == "yvv" ? 2.0e-4
                                                                                                    : maximum;
                        passed = passed && error.finite && error.maximum <= maximum && error.mean <= mean &&
                                 (!impulse || error.l1 / (peak / scale) <= maximum);
                    }
                    details += " " + std::string(stage) + " " + stats_detail(stats, std::nullopt);
                    passed = frame.finish(stream.get(), failure) && passed;
                    if (!failure.diagnostic.empty()) {
                        details += " " + failure.diagnostic;
                    }
                }
                results.record(id, passed, details);
            } catch (const std::exception& error) {
                results.record(id, false, error.what());
            }
        }
    }

    void run_highlight_boost_rows(const Arguments& arguments, Results& results) {
        using JuicerCuda::HighlightBoostDisposition;
        const auto manifest = advanced_manifest(arguments);
        const auto binary = load_parity_binary(arguments, manifest);
        CudaStream stream;
        CudaAllocation workspaceStorage(sizeof(JuicerCuda::HighlightBoostWorkspace));
        auto* workspace = reinterpret_cast<JuicerCuda::HighlightBoostWorkspace*>(workspaceStorage.floats());

        for (const auto& row : manifest.at("rows")) {
            if (row.contains("pixel_size_um")) {
                continue;
            }
            const auto id = row.at("id").get<std::string>();
            try {
                const auto shape = row.at("shape").get<std::array<int, 3>>();
                if (shape[0] <= 0 || shape[1] <= 0 || shape[2] != 3) {
                    throw std::runtime_error("invalid highlight boost RGB extent");
                }
                const ImageExtent extent{shape[1], shape[0]};
                const auto source = read_parity_planes<float, float>(row.at("input"), binary, extent);
                const auto& parameters = row.at("parameters");
                const double boostEv = parameters.at("boost_ev").get<double>();
                const JuicerCuda::HighlightBoostDescriptor descriptor{
                    boostEv == 0.0 ? 0U : 1U,
                    0.184 * std::exp2(parameters.at("protect_ev").get<double>()),
                    std::pow(28.0, 1.0 - parameters.at("boost_range").get<double>()),
                    std::exp2(boostEv)};
                const float maximum = parse_float(row.at("maximum_f32_bits"));
                const auto comparison = row.at("comparison").get<std::string>();
                const auto expectedStatus = comparison == "nonfinite"          ? HighlightBoostDisposition::InvalidCarrier
                                            : comparison == "overflow"         ? HighlightBoostDisposition::PeakOverflow
                                            : maximum <= descriptor.protection ? HighlightBoostDisposition::Bypass
                                                                               : HighlightBoostDisposition::Applied;
                const auto width = static_cast<std::size_t>(extent.width);
                const auto stride = width + 7;
                const auto planeElements = stride * static_cast<std::size_t>(extent.height);
                const auto planeBytes = planeElements * sizeof(float);
                constexpr float padding = std::numeric_limits<float>::max();
                CudaAllocation planes(planeBytes * 3);
                const std::array channels{planes.floats(), planes.floats() + planeElements, planes.floats() + 2 * planeElements};
                const JuicerCuda::CameraFilmLinearExposurePlanes carrier{channels[0], channels[1], channels[2], stride};
                std::array<std::vector<float>, 3> original;
                for (std::size_t c = 0; c < channels.size(); ++c) {
                    original[c].resize(planeElements, padding);
                    for (std::size_t y = 0; y < static_cast<std::size_t>(extent.height); ++y) {
                        std::copy_n(source[c].data() + y * width, width, original[c].data() + y * stride);
                    }
                }
                SemanticPlanes<double> expected;
                if (comparison == "continuation") {
                    const auto values = row.at("decimal_continuation").get<std::array<std::string, 3>>();
                    for (std::size_t c = 0; c < channels.size(); ++c) {
                        expected[c].push_back(std::stod(values[c]));
                    }
                } else if (expectedStatus == HighlightBoostDisposition::Applied) {
                    expected = read_parity_planes<double>(row.at("pinned_output"), binary, extent);
                }

                bool passed = true;
                double maximumError = 0.0;
                auto firstOutput = original;
                for (int attempt = 0; attempt < 3; ++attempt) {
                    for (std::size_t c = 0; c < channels.size(); ++c) {
                        require_cuda(cudaMemcpy(channels[c], original[c].data(), planeBytes, cudaMemcpyHostToDevice), "boost input upload");
                    }
                    require_cuda(cudaMemsetAsync(workspace, 0xff, sizeof(*workspace), stream.get()), "boost status poison");
                    const auto launch = JuicerCuda::launch_highlight_boost(
                        descriptor, carrier, extent.width, extent.height, workspace, stream.get(), attempt == 0 ? 1 : JuicerCuda::kHighlightBoostMaxBlocks);
                    require_cuda(launch.status, "highlight boost kernels");
                    require_cuda(cudaStreamSynchronize(stream.get()), "highlight boost completion");
                    HighlightBoostDisposition status{};
                    require_cuda(cudaMemcpy(&status, &workspace->disposition, sizeof(status), cudaMemcpyDeviceToHost), "highlight boost status");
                    float observedMaximum = 0.0f;
                    require_cuda(cudaMemcpy(&observedMaximum, &workspace->maximum, sizeof(observedMaximum), cudaMemcpyDeviceToHost), "highlight boost maximum");
                    if (boostEv == 0.0) {
                        passed = passed && static_cast<std::uint8_t>(status) == UINT8_MAX &&
                                 std::bit_cast<std::uint32_t>(observedMaximum) == UINT32_MAX;
                    } else {
                        passed = passed && status == expectedStatus &&
                                 (expectedStatus == HighlightBoostDisposition::InvalidCarrier ||
                                  std::bit_cast<std::uint32_t>(observedMaximum) == std::bit_cast<std::uint32_t>(maximum));
                    }
                    for (std::size_t c = 0; c < channels.size(); ++c) {
                        std::vector<float> downloaded(planeElements);
                        require_cuda(cudaMemcpy(downloaded.data(), channels[c], planeBytes, cudaMemcpyDeviceToHost), "highlight boost output");
                        if (attempt == 0) {
                            firstOutput[c] = downloaded;
                        } else {
                            passed = passed && std::memcmp(downloaded.data(), firstOutput[c].data(), planeBytes) == 0;
                        }
                        std::vector<std::pair<float, float>> monotonic;
                        for (std::size_t i = 0; i < planeElements; ++i) {
                            const float raw = original[c][i];
                            const float sample = downloaded[i];
                            if (i % stride >= width || boostEv == 0.0 || expectedStatus != HighlightBoostDisposition::Applied ||
                                static_cast<double>(raw) <= descriptor.protection) {
                                passed = passed && std::bit_cast<std::uint32_t>(sample) == std::bit_cast<std::uint32_t>(raw);
                            } else {
                                const double reference = expected[c][i / stride * width + i % stride];
                                const float rounded = static_cast<float>(reference);
                                const auto a = std::bit_cast<std::uint32_t>(sample);
                                const auto b = std::bit_cast<std::uint32_t>(rounded);
                                const auto ulps = a >= b ? a - b : b - a;
                                const double difference = std::abs(static_cast<double>(sample) - reference);
                                maximumError = std::max(maximumError, difference);
                                const bool accurate = raw == maximum ? ulps <= 1 : ulps <= 4 || difference <= 2e-6 * std::abs(reference);
                                passed = passed && std::isfinite(sample) && accurate && sample >= raw;
                            }
                            if (i % stride < width && std::isfinite(raw)) {
                                monotonic.emplace_back(raw, sample);
                            }
                        }
                        if (expectedStatus == HighlightBoostDisposition::Applied) {
                            std::sort(monotonic.begin(), monotonic.end());
                            passed = passed && std::is_sorted(monotonic.begin(), monotonic.end(), [](const auto& a, const auto& b) {
                                         return a.second < b.second;
                                     });
                        }
                    }
                }
                std::ostringstream detail;
                detail << "expected_max=" << maximum << " max_error=" << maximumError
                       << " disposition=" << static_cast<std::uint32_t>(expectedStatus)
                       << " padded_stride=" << stride;
                results.record(id, passed, detail.str());
            } catch (const std::exception& error) {
                results.record(id, false, error.what());
            }
        }
    }

    void run_captured_carrier_rows(
        const Arguments& arguments,
        Results& results) {
        const std::vector<std::string> routeIds{
            Spektrafilm::scan_route_key(
                Spektrafilm::ScanRoute::NegativeDirectScan),
            Spektrafilm::scan_route_key(
                Spektrafilm::ScanRoute::NegativePrintScan)};
        try {
            if (!arguments.runtimeOwner) {
                throw std::runtime_error("captured carrier rows require a live process Runtime");
            }
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
                *arguments.runtimeOwner,
                Spektrafilm::ScanRoute::NegativeDirectScan,
                *fixture,
                binary,
                row,
                source,
                expected,
                results,
                0x4101);
            run_captured_route(
                *arguments.runtimeOwner,
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
            if (!arguments.runtimeOwner) {
                throw std::runtime_error("performance rows require a live process Runtime");
            }
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
            const auto& mode = arguments.performanceMode;
            const bool reflection = mode == "reflection-3" || mode == "reflection-5";
            const bool spatial = mode == "active" || reflection;
            const bool boost = mode == "boost-dense" || mode == "boost-sparse";
            const auto density = mode == "boost-dense" ? HighlightDensity::Dense
                                 : boost               ? HighlightDensity::Sparse
                                                       : HighlightDensity::Spatial;
            const ImageExtent extent{arguments.performanceWidth, arguments.performanceHeight};
            const float pixelSizeUm = static_cast<float>(35000.0 / static_cast<double>(std::max(extent.width, extent.height)));
            const ScatterHalationControls controls = mode == "active" ? fixtureRow.controls : ScatterHalationControls{};
            FjPreparationHalation reflectionControls{};
            if (reflection) {
                reflectionControls.active = 1;
                reflectionControls.halation_amount = 1.0f;
                reflectionControls.halation_spatial_scale = 1.0f;
                reflectionControls.reflection_source = 1;
                reflectionControls.shape.present = 1;
                reflectionControls.shape.bounce_count = mode == "reflection-5" ? 5 : 3;
                reflectionControls.shape.bounce_decay = 0.5;
                reflectionControls.shape.renormalize = 1;
                constexpr std::array strengths{0.6, 0.3, 0.1};
                constexpr std::array sigmaUm{64.0, 66.0, 68.0};
                std::ranges::copy(strengths, reflectionControls.reflection_strength);
                std::ranges::copy(sigmaUm, reflectionControls.first_sigma_um);
            }
            PreparedRouteInputs inputs = build_prepared_route_inputs(
                *arguments.runtimeOwner, Spektrafilm::ScanRoute::NegativeDirectScan, controls, {}, JuicerTestSupport::PreparationInput{}.values.dir_couplers, pixelSizeUm, extent, 1.0f, {}, reflection ? &reflectionControls : nullptr, boost ? std::optional{HighlightBoostSettings{4.0, 4.0, 0.3}} : std::nullopt);
            CudaStream stream;
            const auto contextKey = current_context_key(arguments.deviceIndex);
            std::uint64_t identity = 0x50455246;
            JuicerCuda::Failure diagnostic;
            auto prepare = [&] {
                auto frame = inputs.prepare_frame(contextKey, ++identity, extent, stream.get(), diagnostic);
                if (!frame.active()) {
                    throw std::runtime_error(diagnostic.diagnostic);
                }
                return frame;
            };
            auto finish = [&](JuicerProcess::Root::PreparedCudaFrame& frame) {
                if (!frame.finish(stream.get(), diagnostic)) {
                    throw std::runtime_error(diagnostic.diagnostic);
                }
            };
            auto hold_residency = [&] {
                if (arguments.holdMilliseconds > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(arguments.holdMilliseconds));
                }
            };
            const std::size_t elements = static_cast<std::size_t>(extent.width) * static_cast<std::size_t>(extent.height);
            const std::size_t planeBytes = elements * sizeof(float);
            std::vector<float> samples;
            std::vector<double> statusWaitMs;
            std::vector<double> preparationMs;
            float coldStageMs = 0.0f;
            if (spatial || boost) {
                const CudaTimingEvent start;
                const CudaTimingEvent stop;
                SemanticPlanes<float> source;
                for (auto& plane : source) {
                    plane.resize(elements);
                }
                for (std::size_t index = 0; index < elements; ++index) {
                    const auto sample = asymmetric_hdr_rgb(index, density);
                    source[0][index] = sample.red;
                    source[1][index] = sample.green;
                    source[2][index] = sample.blue;
                }
                const auto reset_source = [&](JuicerCuda::CameraFilmLinearExposurePlanes carrier) {
                    const std::array destinations{carrier.redSensitive, carrier.greenSensitive, carrier.blueSensitive};
                    for (std::size_t channel = 0; channel < source.size(); ++channel) {
                        require_cuda(cudaMemcpyAsync(destinations[channel], source[channel].data(), planeBytes, cudaMemcpyHostToDevice, stream.get()), "reset benchmark source");
                    }
                    require_cuda(cudaStreamSynchronize(stream.get()), "complete benchmark source reset");
                };
                const auto stage_duration = [&] {
                    require_cuda(cudaEventSynchronize(stop.get()), "wait for benchmark stage event");
                    float elapsed = 0.0f;
                    require_cuda(cudaEventElapsedTime(&elapsed, start.get(), stop.get()), "time benchmark stage");
                    return elapsed;
                };
                const auto collect_samples = [&](auto&& sample) {
                    coldStageMs = sample(false);
                    for (int index = 0; index < arguments.warmupCount; ++index) {
                        (void)sample(false);
                    }
                    for (int index = 0; index < arguments.sampleCount; ++index) {
                        samples.push_back(sample(true));
                    }
                };
                if (spatial) {
                    auto frame = prepare();
                    const auto view = frame.scatter_halation_resources();
                    if (!view.descriptor || view.carrierSource != JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes) {
                        throw std::runtime_error("performance prepared spatial view unavailable");
                    }
                    if (reflection) {
                        const auto count = static_cast<std::uint32_t>(reflectionControls.shape.bounce_count);
                        bool matched = view.descriptor->bounceCount == count && view.descriptor->scatterAmount == 0.0f;
                        for (const auto& channel : view.descriptor->channels) {
                            for (std::size_t bounce = 0; bounce < count; ++bounce) {
                                matched &= channel.bounce[bounce].kind == ScatterHalationGaussianKind::YvvReplicateFactored;
                            }
                        }
                        if (!matched) {
                            throw std::runtime_error("reflection comparison requires matching advanced YVV dispatch for every bounce");
                        }
                    }
                    collect_samples([&](bool) {
                        reset_source(view.currentCarrier);
                        require_cuda(cudaEventRecord(start.get(), stream.get()), "start spatial stage timing");
                        require_cuda(JuicerCuda::launch_scatter_halation(view, stream.get()).status, "launch spatial benchmark stage");
                        require_cuda(cudaEventRecord(stop.get(), stream.get()), "stop spatial stage timing");
                        return stage_duration();
                    });
                    hold_residency();
                    finish(frame);
                } else {
                    using Clock = std::chrono::steady_clock;
                    using Milliseconds = std::chrono::duration<double, std::milli>;
                    collect_samples([&](bool record) {
                        const auto preparationStart = Clock::now();
                        auto frame = prepare();
                        const auto preparationElapsed = Milliseconds(Clock::now() - preparationStart).count();
                        const auto carrier = frame.camera_film_linear_carrier();
                        if (frame.scatter_halation_resources().descriptor || carrier.rowStrideFloats != static_cast<std::size_t>(extent.width)) {
                            throw std::runtime_error("boost-only benchmark requires compact carrier without spatial scratch");
                        }
                        reset_source(carrier);
                        require_cuda(cudaEventRecord(start.get(), stream.get()), "start boost stage timing");
                        if (!frame.enqueue_highlight_boost(carrier, stream.get(), diagnostic)) {
                            throw std::runtime_error(diagnostic.diagnostic);
                        }
                        require_cuda(cudaEventRecord(stop.get(), stream.get()), "stop boost stage timing");
                        const auto waitStart = Clock::now();
                        if (!frame.check_highlight_boost(diagnostic)) {
                            throw std::runtime_error(diagnostic.diagnostic);
                        }
                        const auto waitElapsed = Milliseconds(Clock::now() - waitStart).count();
                        const float stageMs = stage_duration();
                        if (record && samples.size() + 1 == static_cast<std::size_t>(arguments.sampleCount)) {
                            hold_residency();
                        }
                        finish(frame);
                        if (record) {
                            statusWaitMs.push_back(waitElapsed);
                            preparationMs.push_back(preparationElapsed);
                        }
                        return stageMs;
                    });
                }
            } else {
                auto frame = prepare();
                hold_residency();
                finish(frame);
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
                        std::ceil(
                            fraction * static_cast<double>(sorted.size())) -
                        1.0));
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
                {"sampleCount", samples.size()},
                {"coldStageMs", coldStageMs},
                {"statusWaitMs", statusWaitMs},
                {"preparationMs", preparationMs},
                {"samplesMs", samples},
                {"medianMs", percentile(0.5)},
                {"p95Ms", percentile(0.95)},
                {"descriptorRecipeHash",
                 inputs.scatter_descriptor()
                     ? inputs.scatter_descriptor()->recipeHash
                     : 0},
                {"sourceChecksum", asymmetric_hdr_source_checksum(elements, density)},
                {"sourceChecksumEncoding", mode == "boost-dense" ? "Float32 index-major semantic RGB dense above-onset" : boost ? "Float32 index-major semantic RGB sparse above-onset"
                                                                                                                                : "Float32 index-major semantic RGB asymmetric-hdr-v1"},
                {"ledgerRecords", spatial || boost ? 2 : 0},
                {"filterBytes", spatial ? 2 * planeBytes : 0},
                {"carrierBytes", spatial || boost ? 3 * planeBytes : 0},
                {"boostWorkspaceBytes", boost ? sizeof(JuicerCuda::HighlightBoostWorkspace) : 0},
                {"boostPinnedStatusBytes", boost ? sizeof(JuicerCuda::HighlightBoostDisposition) : 0},
                {"boostStatusEvents", boost ? 1 : 0},
                {"admittedBytes", spatial ? 5 * planeBytes : boost ? 3 * planeBytes + sizeof(JuicerCuda::HighlightBoostWorkspace)
                                                                   : 0}};
            write_performance_report(arguments, report);
            results.record("performance/" + mode + "/" + std::to_string(extent.width), true, "");
        } catch (const std::exception& error) {
            results.record("performance/fatal", false, error.what());
        }
    }
} // namespace ScatterHalationValidation
