#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>

#include "Cuda/Film/JuicerCudaHighlightBoost.h"

namespace {
    using Json = nlohmann::json;
    using Disposition = JuicerCuda::HighlightBoostDisposition;

    void check(cudaError_t status, const char* operation) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
        }
    }
    struct Stream {
        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;
        cudaStream_t value{};
        Stream() {
            check(cudaStreamCreate(&value), "create stream");
        }
        ~Stream() {
            if (value) {
                cudaStreamSynchronize(value);
                cudaStreamDestroy(value);
            }
        }
    };
    struct Device {
        Device(const Device&) = delete;
        Device& operator=(const Device&) = delete;
        void* value{};
        explicit Device(std::size_t bytes) {
            check(cudaMalloc(&value, bytes), "allocate device memory");
        }
        ~Device() {
            if (value)
                cudaFree(value);
        }
    };
    Json read_json(const std::filesystem::path& path) {
        std::ifstream file(path);
        if (!file)
            throw std::runtime_error("cannot open " + path.string());
        return Json::parse(file);
    }
    std::vector<std::uint8_t> read_binary(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file)
            throw std::runtime_error("cannot open " + path.string());
        const auto size = file.tellg();
        if (size < 0)
            throw std::runtime_error("cannot size binary fixture");
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
        file.seekg(0);
        if (!file.read(reinterpret_cast<char*>(bytes.data()), size))
            throw std::runtime_error("cannot read binary fixture");
        return bytes;
    }
    template <class T>
    std::vector<T> plane(const Json& record, const std::vector<std::uint8_t>& binary, std::size_t count) {
        const auto offset = record.at("offset").get<std::size_t>();
        const auto bytes = count * sizeof(T);
        if (record.at("dtype") != (sizeof(T) == 4 ? "<f4" : "<f8") || record.at("size").get<std::size_t>() != bytes || offset > binary.size() || bytes > binary.size() - offset)
            throw std::runtime_error("invalid fixture plane bounds or dtype");
        std::vector<T> out(count);
        std::memcpy(out.data(), binary.data() + offset, bytes);
        return out;
    }
    std::size_t plane_elements(int width, int height, int padding) {
        if (width <= 0 || height <= 0 || padding < 0)
            throw std::runtime_error("invalid image extent");
        const auto w = static_cast<std::size_t>(width);
        const auto h = static_cast<std::size_t>(height);
        const auto pad = static_cast<std::size_t>(padding);
        constexpr auto limit = std::numeric_limits<std::size_t>::max();
        if (w > limit - pad || w + pad > limit / h || (w + pad) * h > limit / (3 * sizeof(double)))
            throw std::runtime_error("image extent exceeds addressable storage");
        return (w + pad) * h;
    }
    struct Row {
        std::string id;
        int width{}, height{};
        JuicerCuda::HighlightBoostDescriptor descriptor{};
        std::vector<float> input;
        std::vector<double> expected;
        float maximum{};
        Disposition status{};
        bool monotonic = true;
    };
    Disposition disposition(const std::string& value) {
        if (value == "Applied")
            return Disposition::Applied;
        if (value == "Bypass")
            return Disposition::Bypass;
        if (value == "InvalidCarrier")
            return Disposition::InvalidCarrier;
        if (value == "PeakOverflow")
            return Disposition::PeakOverflow;
        throw std::runtime_error("unknown disposition: " + value);
    }
    struct FixturePaths {
        std::filesystem::path root;
        std::filesystem::path supplement;
    };
    std::vector<Row> load_rows(const FixturePaths& paths) {
        const auto manifest = read_json(paths.root / "manifest.json");
        if (manifest.at("schema") != "spektrafilm-halation-feature-parity.v1")
            throw std::runtime_error("unexpected fixture schema");
        const auto binary = read_binary(paths.root / manifest.at("binary").at("file").get<std::string>());
        if (binary.size() != manifest.at("binary").at("size").get<std::size_t>())
            throw std::runtime_error("fixture size mismatch");
        std::vector<Row> rows;
        for (const auto& record : manifest.at("rows")) {
            if (record.contains("pixel_size_um"))
                continue;
            Row row;
            row.id = record.at("id").get<std::string>();
            const auto shape = record.at("shape").get<std::array<int, 3>>();
            if (shape[0] <= 0 || shape[1] <= 0 || shape[2] != 3)
                throw std::runtime_error("invalid fixture shape");
            row.width = shape[1];
            row.height = shape[0];
            const auto& p = record.at("parameters");
            const double ev = p.at("boost_ev").get<double>();
            row.descriptor = {ev == 0 ? 0U : 1U, 0.184 * std::exp2(p.at("protect_ev").get<double>()), std::pow(28.0, 1.0 - p.at("boost_range").get<double>()), std::exp2(ev)};
            const auto count = plane_elements(row.width, row.height, 0) * 3;
            row.input = plane<float>(record.at("input"), binary, count);
            row.maximum = std::bit_cast<float>(static_cast<std::uint32_t>(std::stoul(record.at("maximum_f32_bits").get<std::string>(), nullptr, 16)));
            const auto comparison = record.at("comparison").get<std::string>();
            row.status = comparison == "nonfinite" ? Disposition::InvalidCarrier : comparison == "overflow"               ? Disposition::PeakOverflow
                                                                               : row.maximum <= row.descriptor.protection ? Disposition::Bypass
                                                                                                                          : Disposition::Applied;
            if (comparison == "continuation") {
                for (const auto& value : record.at("decimal_continuation"))
                    row.expected.push_back(std::stod(value.get<std::string>()));
            } else if (row.status == Disposition::Applied)
                row.expected = plane<double>(record.at("pinned_output"), binary, count);
            rows.push_back(std::move(row));
        }
        const auto extra = read_json(paths.supplement);
        if (extra.at("schema") != "boost-precision-supplement.v1")
            throw std::runtime_error("unexpected supplement schema");
        for (const auto& record : extra.at("rows")) {
            Row row;
            row.id = record.at("id").get<std::string>();
            row.width = record.at("width").get<int>();
            row.height = record.at("height").get<int>();
            if (row.width <= 0 || row.height <= 0)
                throw std::runtime_error("invalid supplement shape");
            const auto& d = record.at("descriptor");
            row.descriptor = {d.at("recipe_hash").get<std::uint64_t>(), d.at("protection").get<double>(), d.at("sharpness").get<double>(), d.at("peak_gain").get<double>()};
            for (const auto& value : record.at("input_bits"))
                row.input.push_back(std::bit_cast<float>(value.get<std::uint32_t>()));
            for (const auto& value : record.at("expected"))
                row.expected.push_back(std::stod(value.get<std::string>()));
            const auto count = plane_elements(row.width, row.height, 0) * 3;
            if (row.input.size() != count || row.expected.size() != count)
                throw std::runtime_error("supplement extent mismatch");
            row.maximum = std::bit_cast<float>(record.at("maximum_bits").get<std::uint32_t>());
            row.status = disposition(record.at("status").get<std::string>());
            row.monotonic = record.at("monotonic").get<bool>();
            rows.push_back(std::move(row));
        }
        return rows;
    }
    struct Frame {
        int width, height;
        std::size_t stride, elements, bytes;
        Device planes, storage;
        JuicerCuda::CameraFilmLinearExposurePlanes carrier;
        JuicerCuda::HighlightBoostWorkspace* workspace;
        Frame(int w, int h, int padding) : width(w), height(h), stride(static_cast<std::size_t>(w) + padding), elements(plane_elements(w, h, padding)), bytes(elements * sizeof(float)), planes(bytes * 3), storage(sizeof(JuicerCuda::HighlightBoostWorkspace)), carrier{static_cast<float*>(planes.value), static_cast<float*>(planes.value) + elements, static_cast<float*>(planes.value) + 2 * elements, stride}, workspace(static_cast<JuicerCuda::HighlightBoostWorkspace*>(storage.value)) {}
        std::vector<float> layout(const std::vector<float>& input) const {
            std::vector<float> result(elements * 3, std::numeric_limits<float>::max());
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    for (int c = 0; c < 3; ++c)
                        result[static_cast<std::size_t>(c) * elements + static_cast<std::size_t>(y) * stride + x] = input[(static_cast<std::size_t>(y) * width + x) * 3 + c];
            return result;
        }
        void upload(const std::vector<float>& input, cudaStream_t stream) {
            check(cudaMemcpyAsync(planes.value, input.data(), bytes * 3, cudaMemcpyHostToDevice, stream), "upload input");
        }
        std::vector<float> download(cudaStream_t stream) {
            std::vector<float> output(elements * 3);
            check(cudaStreamSynchronize(stream), "download completion");
            check(cudaMemcpy(output.data(), planes.value, bytes * 3, cudaMemcpyDeviceToHost), "download output");
            return output;
        }
    };
    void launch(const Row& row, Frame& frame, cudaStream_t stream, int blocks) {
        const auto result = JuicerCuda::launch_highlight_boost(row.descriptor, frame.carrier, row.width, row.height, frame.workspace, stream, blocks);
        if (result.status != cudaSuccess)
            throw std::runtime_error("highlight launch stage " + std::to_string(static_cast<int>(result.stage)) + ": " + cudaGetErrorString(result.status));
    }
    std::uint32_t bits(float value) {
        return std::bit_cast<std::uint32_t>(value);
    }
    std::uint32_t ordered(float value) {
        const auto b = bits(value);
        return (b & 0x80000000U) ? ~b : b | 0x80000000U;
    }
    Json accuracy_row(const Row& row, int padding, std::vector<std::uint32_t>& compact) {
        Frame frame(row.width, row.height, padding);
        const auto original = frame.layout(row.input);
        Stream stream;
        std::vector<float> first;
        std::size_t failures = 0, samples = 0, finiteSamples = 0, nonfiniteCount = 0;
        double maxAbs = 0, maxRel = 0, squares = 0;
        std::uint32_t maxUlp = 0;
        Json worst = nullptr, firstFailure = nullptr;
        bool statusPass = true, maximumPass = true, repeatPass = true, layoutPass = true, exactPass = true, finitePass = true, monotonicPass = true, tolerancePass = true;
        for (int attempt = 0; attempt < 3; ++attempt) {
            frame.upload(original, stream.value);
            check(cudaMemsetAsync(frame.workspace, 0xff, sizeof(*frame.workspace), stream.value), "poison workspace");
            launch(row, frame, stream.value, attempt == 0 ? 1 : 4096);
            JuicerCuda::HighlightBoostWorkspace result{};
            check(cudaStreamSynchronize(stream.value), "highlight completion");
            check(cudaMemcpy(&result, frame.workspace, sizeof(result), cudaMemcpyDeviceToHost), "read workspace");
            auto output = frame.download(stream.value);
            if (row.descriptor.recipeHash == 0) {
                statusPass &= static_cast<std::uint8_t>(result.disposition) == UINT8_MAX;
                maximumPass &= bits(result.maximum) == UINT32_MAX;
            } else {
                statusPass &= result.disposition == row.status;
                maximumPass &= row.status == Disposition::InvalidCarrier || bits(result.maximum) == bits(row.maximum);
            }
            if (attempt == 0)
                first = output;
            else
                repeatPass &= std::memcmp(first.data(), output.data(), output.size() * sizeof(float)) == 0;
            for (int c = 0; c < 3; ++c) {
                std::vector<std::pair<float, float>> orderedSamples;
                for (int y = 0; y < row.height; ++y)
                    for (std::size_t x = 0; x < frame.stride; ++x) {
                        const auto i = static_cast<std::size_t>(c) * frame.elements + static_cast<std::size_t>(y) * frame.stride + x;
                        const auto raw = original[i], sample = output[i];
                        if (x < static_cast<std::size_t>(row.width) && std::isfinite(raw) && !std::isfinite(sample))
                            ++nonfiniteCount;
                        if (x >= static_cast<std::size_t>(row.width) || row.descriptor.recipeHash == 0 || row.status != Disposition::Applied || static_cast<double>(raw) <= row.descriptor.protection) {
                            const bool exact = bits(raw) == bits(sample);
                            exactPass &= exact;
                            if (!exact && firstFailure.is_null())
                                firstFailure = {{"attempt", attempt}, {"x", x}, {"y", y}, {"channel", c}, {"input_bits", bits(raw)}, {"actual_bits", bits(sample)}};
                        } else {
                            const auto logical = (static_cast<std::size_t>(y) * row.width + x) * 3 + c;
                            const double ref = row.expected.at(logical), diff = std::abs(static_cast<double>(sample) - ref);
                            const auto a = ordered(sample), b = ordered(static_cast<float>(ref));
                            const auto ulp = a > b ? a - b : b - a;
                            const bool finite = std::isfinite(sample);
                            finitePass &= finite && sample >= raw;
                            const bool accurate = finite && (raw == row.maximum ? ulp <= 1 : ulp <= 4 || diff <= 2e-6 * std::abs(ref));
                            tolerancePass &= accurate;
                            failures += !accurate;
                            ++samples;
                            if ((!accurate || sample < raw) && firstFailure.is_null())
                                firstFailure = {{"attempt", attempt}, {"x", x}, {"y", y}, {"channel", c}, {"input_bits", bits(raw)}, {"actual_bits", bits(sample)}, {"expected_bits", bits(static_cast<float>(ref))}};
                            if (finite) {
                                ++finiteSamples;
                                const double relative = ref == 0 ? (diff == 0 ? 0 : std::numeric_limits<double>::infinity()) : diff / std::abs(ref);
                                if (diff > maxAbs || worst.is_null()) {
                                    maxAbs = diff;
                                    worst = {{"x", x}, {"y", y}, {"channel", c}, {"input", raw}, {"actual", sample}, {"expected", ref}, {"ulp", ulp}};
                                }
                                maxRel = std::max(maxRel, relative);
                                maxUlp = std::max(maxUlp, ulp);
                                squares += diff * diff;
                            }
                        }
                        if (x < static_cast<std::size_t>(row.width) && std::isfinite(raw) && std::isfinite(sample))
                            orderedSamples.emplace_back(raw, sample);
                    }
                if (row.status == Disposition::Applied && row.monotonic) {
                    std::sort(orderedSamples.begin(), orderedSamples.end());
                    monotonicPass &= std::is_sorted(orderedSamples.begin(), orderedSamples.end(), [](const auto& a, const auto& b) {
                        return a.second < b.second;
                    });
                }
            }
        }
        std::size_t index = 0;
        for (int y = 0; y < row.height; ++y)
            for (int x = 0; x < row.width; ++x)
                for (int c = 0; c < 3; ++c) {
                    const auto value = bits(first[static_cast<std::size_t>(c) * frame.elements + static_cast<std::size_t>(y) * frame.stride + x]);
                    if (padding == 0)
                        compact.push_back(value);
                    else
                        layoutPass &= compact.at(index) == value;
                    ++index;
                }
        const bool pass = statusPass && maximumPass && repeatPass && layoutPass && exactPass && finitePass && monotonicPass && tolerancePass;
        return {{"id", row.id}, {"variant", "production"}, {"padding", padding}, {"pass", pass}, {"status_pass", statusPass}, {"maximum_pass", maximumPass}, {"repeat_pass", repeatPass}, {"layout_pass", layoutPass}, {"exact_pass", exactPass}, {"finite_pass", finitePass}, {"monotonic_pass", monotonicPass}, {"tolerance_pass", tolerancePass}, {"tolerance_failures", failures}, {"samples", samples}, {"finite_samples", finiteSamples}, {"nonfinite_count", nonfiniteCount}, {"first_failing_sample", firstFailure}, {"metrics_scope", "finite_samples_only"}, {"max_absolute", maxAbs}, {"max_relative", maxRel}, {"rms", finiteSamples ? std::sqrt(squares / static_cast<double>(finiteSamples)) : 0.0}, {"max_ulp", maxUlp}, {"worst_sample", worst}};
    }
} // namespace

int main(int argc, char** argv) try {
    std::string mode;
    std::filesystem::path root = "tests/scatter_halation/parity-fixtures", supplement, output;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (i + 1 >= argc)
            throw std::runtime_error("missing value for " + argument);
        const std::string value = argv[++i];
        if (argument == "--mode")
            mode = value;
        else if (argument == "--fixture-root")
            root = value;
        else if (argument == "--supplement")
            supplement = value;
        else if (argument == "--output")
            output = value;
        else
            throw std::runtime_error("unknown argument " + argument);
    }
    if (output.empty())
        throw std::runtime_error("--output is required");
    check(cudaSetDevice(0), "select device");
    Json report;
    bool productionPass = true;
    if (mode != "accuracy")
        throw std::runtime_error("--mode must be accuracy");
    if (supplement.empty())
        throw std::runtime_error("--supplement is required");
    report = {{"schema", "highlight-production-accuracy.v1"}, {"rows", Json::array()}};
    for (const auto& row : load_rows({.root = root, .supplement = supplement})) {
        std::vector<std::uint32_t> compact;
        for (const int padding : {0, 7}) {
            auto result = accuracy_row(row, padding, compact);
            productionPass &= result.at("pass").get<bool>();
            report["rows"].push_back(std::move(result));
        }
    }
    report["production_pass"] = productionPass;
    std::ofstream file(output);
    if (!file || !(file << report.dump(2) << '\n'))
        throw std::runtime_error("cannot write report");
    return productionPass ? 0 : 1;
} catch (const std::exception& error) {
    std::fprintf(stderr, "highlight precision: %s\n", error.what());
    return 1;
} catch (...) {
    std::fprintf(stderr, "highlight precision: unknown exception\n");
    return 1;
}
