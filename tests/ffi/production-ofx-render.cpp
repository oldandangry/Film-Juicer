#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "ofxCore.h"
#include "ofxImageEffect.h"
#include "ofxImageEffectExt.h"
#include "ofxParam.h"
#include "ofxParamExt.h"
#include "instance_host.h"
#include "juicer_cuda_api.h"
#include "production_images.h"


namespace {
    using Json = nlohmann::json;
    using ofx_test::InstanceHost;
    using ofx_test::InstanceValue;
    constexpr int kPitchPadding = 5;
    constexpr float kDestinationCanary = -8.0f;
    using ofx_test::alpha_preserved;
    using ofx_test::author_parameters;
    using ofx_test::canaries_preserved;
    using ofx_test::Case;
    using ofx_test::configure;
    using ofx_test::initial_source;
    constexpr std::array kOrdinaryCases{
        Case{"negative-direct", "negative-direct", "NegativeDirectScan", 0, 3},
        Case{"negative-print", "negative-print", "NegativePrintScan", 1, 4},
        Case{"positive-direct", "positive-direct", "PositiveDirectScan", 0, 3},
        Case{"positive-print", "positive-print", "PositivePrintScan", 0, 4}};
    constexpr std::array<std::string_view, 8> kCaseIds{
        "negative-direct/cold", "negative-direct/warm", "negative-print/cold", "negative-print/warm", "positive-direct/cold", "positive-direct/warm", "positive-print/cold", "positive-print/warm"};
    constexpr std::array kFilmV1Cases{
        Case{"negative-direct-quarter-size-half", "negative-direct", "NegativeDirectScan", 0, 4, 1208, 504},
        Case{"negative-print-proxy-sequence", "negative-print", "NegativePrintScan", 0, 4, 1208, 504},
        Case{"negative-direct-hd", "negative-direct", "NegativeDirectScan", 0, 4, 1920, 1080},
        Case{"negative-print-hd", "negative-print", "NegativePrintScan", 0, 4, 1920, 1080},
        Case{"negative-direct-uhd", "negative-direct", "NegativeDirectScan", 0, 4, 3840, 2160},
        Case{"negative-print-uhd", "negative-print", "NegativePrintScan", 0, 4, 3840, 2160},
        Case{"negative-direct-4832", "negative-direct", "NegativeDirectScan", 0, 4, 4832, 2016},
        Case{"negative-print-4832", "negative-print", "NegativePrintScan", 0, 4, 4832, 2016},
        Case{"negative-print-portrait", "negative-print", "NegativePrintScan", 0, 4, 5299, 7418},
        Case{"negative-direct-smaller-formats", "negative-direct", "NegativeDirectScan", 0, 4, 4832, 2016},
        Case{"negative-print-smaller-formats", "negative-print", "NegativePrintScan", 0, 4, 4832, 2016},
        Case{"negative-direct-larger-formats", "negative-direct", "NegativeDirectScan", 0, 4, 92, 46},
        Case{"negative-print-larger-formats", "negative-print", "NegativePrintScan", 0, 4, 92, 46},
        Case{"negative-direct-thumbnails", "negative-direct", "NegativeDirectScan", 0, 4, 92, 46},
        Case{"negative-print-thumbnails", "negative-print", "NegativePrintScan", 0, 4, 92, 46},
        Case{"negative-direct-8mm-proxy", "negative-direct", "NegativeDirectScan", 0, 4, 32, 16},
        Case{"negative-print-8mm-proxy", "negative-print", "NegativePrintScan", 0, 4, 32, 16},
        Case{"negative-direct-clipped-quarter", "negative-direct", "NegativeDirectScan", 0, 4, 1208, 2},
        Case{"negative-print-clipped-quarter", "negative-print", "NegativePrintScan", 0, 4, 1208, 2},
        Case{"negative-direct-time-endpoint", "negative-direct", "NegativeDirectScan", 0, 4, 1920, 1080},
        Case{"negative-print-time-endpoint", "negative-print", "NegativePrintScan", 0, 4, 1920, 1080},
        Case{"negative-direct-small-format-high-gain", "negative-direct", "NegativeDirectScan", 0, 4, 4832, 96},
        Case{"negative-print-small-format-high-gain", "negative-print", "NegativePrintScan", 0, 4, 4832, 96},
    };
    struct GrainGeometry {
        OfxRectD canonical{};
        OfxRectI bounds{};
        std::array<double, 2> scale{1.0, 1.0};
        std::array<double, 2> imageScale{1.0, 1.0};
    };
    constexpr GrainGeometry kEightMmProxy{{0, 0, 8192, 4096}, {0, 0, 32, 16}, {1.0 / 256, 1.0 / 256}};
    constexpr GrainGeometry kBelowMinimumSample{{0, 0, 16384, 8192}, {0, 0, 32, 16}, {1.0 / 512, 1.0 / 512}};
    constexpr GrainGeometry kClippedQuarter{{0, 4, 4832, 12}, {0, 0, 1208, 2}, {0.25, 0.125}};
    constexpr GrainGeometry kOffsetQuarter{{0, 4, 4832, 12}, {0, 1, 1208, 3}, {0.25, 0.25}};
    constexpr OfxRectD kShiftedRegion{1, 4, 3840, 8};
    constexpr GrainGeometry kShiftedQuarter{kShiftedRegion, {0, 1, 960, 2}, {0.25, 0.25}};
    constexpr GrainGeometry kShiftedClipped{kShiftedRegion, {0, 0, 960, 1}, {0.25, 0.125}};

    void set_grain_geometry(InstanceHost& host, OfxImageEffectHandle effect, const GrainGeometry& geometry) {
        const auto& [x1, y1, x2, y2] = geometry.bounds;
        const std::vector<InstanceValue> bounds{x1, y1, x2, y2};
        host.set_property(host.action_args(), kOfxImageEffectPropRenderWindow, bounds);
        host.set_property(host.action_args(), kOfxImageEffectPropRenderScale, {geometry.scale[0], geometry.scale[1]});
        for (const auto* clip : {"Source", "Output"}) {
            host.set_image_property(effect, clip, kOfxImagePropBounds, bounds);
            host.set_image_property(effect, clip, kOfxImagePropRegionOfDefinition, bounds);
            host.set_image_property(effect, clip, kOfxImageEffectPropRenderScale, {geometry.imageScale[0], geometry.imageScale[1]});
            host.set_clip_region(effect, clip, geometry.canonical);
        }
    }

    std::vector<float> flat_grain_source(const Case& test, float gray = 0.18F) {
        auto source = initial_source(test);
        const int pitch = test.width * test.components + kPitchPadding;
        for (int row = 0; row < test.height; ++row) {
            for (int column = 0; column < test.width; ++column) {
                const auto pixel = static_cast<std::size_t>(row) * pitch + static_cast<std::size_t>(column) * test.components;
                std::fill_n(source.data() + pixel, 3, gray);
            }
        }
        return source;
    }
    void require(bool condition, std::string_view message) {
        if (!condition)
            throw std::runtime_error(std::string(message));
    }
    void require_cuda(cudaError_t result, std::string_view operation) {
        if (result != cudaSuccess)
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
    class Module final {
    public:
        explicit Module(const std::filesystem::path& path, std::string_view error = "unable to load production OFX module") {
#ifdef _WIN32
            handle_ = LoadLibraryW(path.c_str());
#else
            handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
            require(handle_ != nullptr, error);
        }
        Module(const Module&) = delete;
        Module& operator=(const Module&) = delete;
        ~Module() {
            if (handle_)
                close_unchecked();
        }
        template <class T>
        T find_symbol(const char* name) const {
#ifdef _WIN32
            auto* address = GetProcAddress(handle_, name);
#else
            auto* address = dlsym(handle_, name);
#endif
            return reinterpret_cast<T>(address);
        }
        template <class T>
        T symbol(const char* name) const {
            const auto address = find_symbol<T>(name);
            require(address != nullptr, std::string("missing module export: ") + name);
            return address;
        }
        bool loaded_from(const std::filesystem::path& path) const {
#ifdef _WIN32
            std::array<wchar_t, 32768> filename{};
            const auto length = GetModuleFileNameW(handle_, filename.data(), static_cast<DWORD>(filename.size()));
            return length > 0 && length < filename.size() && std::filesystem::equivalent(path, filename.data());
#else
            Dl_info info{};
            return dladdr(symbol<void*>("OfxGetPlugin"), &info) != 0 && info.dli_fname &&
                   std::filesystem::equivalent(path, info.dli_fname);
#endif
        }
        void close() {
            require(handle_ != nullptr, "production module already closed");
#ifdef _WIN32
            const bool closed = FreeLibrary(handle_) != 0;
#else
            const bool closed = dlclose(handle_) == 0;
#endif
            if (closed)
                handle_ = nullptr;
            require(closed, "production module unload failed");
        }

    private:
        void close_unchecked() noexcept {
#ifdef _WIN32
            (void)FreeLibrary(handle_);
#else
            (void)dlclose(handle_);
#endif
            handle_ = nullptr;
        }
#ifdef _WIN32
        HMODULE handle_ = nullptr;
#else
        void* handle_ = nullptr;
#endif
    };
    // Discovery exercises the module on hosts without an installed CUDA driver.
    // Only the GPU paths resolve driver entry points.
    class CudaDriver final {
#ifdef _WIN32
        Module library_{"nvcuda.dll", "unable to load CUDA driver"};
#else
        Module library_{"libcuda.so.1", "unable to load CUDA driver"};
#endif

    public:
        const decltype(&cuInit) init = library_.symbol<decltype(&cuInit)>("cuInit");
        const decltype(&cuCtxGetCurrent) get_current = library_.symbol<decltype(&cuCtxGetCurrent)>("cuCtxGetCurrent");
        const decltype(&cuCtxSetCurrent) set_current = library_.symbol<decltype(&cuCtxSetCurrent)>("cuCtxSetCurrent");

        void check(CUresult result, std::string_view operation) const {
            if (result != CUDA_SUCCESS) {
                const char* name = nullptr;
                (void)error_name_(result, &name);
                throw std::runtime_error(std::string(operation) + ": " + (name ? name : "CUDA driver failure"));
            }
        }

    private:
        const decltype(&cuGetErrorName) error_name_ = library_.symbol<decltype(&cuGetErrorName)>("cuGetErrorName");
    };

    struct ContextRestore final {
        const CudaDriver& driver;
        CUcontext original = nullptr;
        bool restored = false;
        ContextRestore(const CudaDriver& cudaDriver, CUcontext value) noexcept : driver(cudaDriver), original(value) {}
        ~ContextRestore() {
            if (!restored)
                (void)driver.set_current(original);
        }
        void restore() {
            driver.check(driver.set_current(original), "restore original CUDA context");
            CUcontext current = nullptr;
            driver.check(driver.get_current(&current), "check restored CUDA context");
            restored = current == original;
            require(restored, "original CUDA context was not restored");
        }
    };
    struct DeviceBuffers final : ofx_test::ImageBuffers {
        ~DeviceBuffers() {
            if (stream)
                (void)cudaStreamSynchronize(stream);
            if (destination)
                (void)cudaFree(destination);
            if (source)
                (void)cudaFree(source);
            if (stream)
                (void)cudaStreamDestroy(stream);
        }
        void release_images() {
            require_cuda(cudaFree(destination), "free destination buffer");
            destination = nullptr;
            require_cuda(cudaFree(source), "free source buffer");
            source = nullptr;
        }
        void close() {
            require_cuda(cudaStreamSynchronize(stream), "complete all production stream work");
            require_cuda(cudaStreamDestroy(stream), "destroy production stream");
            stream = nullptr;
        }
    };
    class Lifecycle final {
    public:
        Lifecycle(OfxPlugin* plugin, InstanceHost& host) : plugin_(plugin), host_(host) {}
        ~Lifecycle() {
            for (auto* effect : effects_)
                if (effect)
                    (void)call(kOfxActionDestroyInstance, effect);
            if (loaded_)
                (void)call(kOfxActionUnload);
        }
        OfxStatus call(const char* action, const void* effect = nullptr, OfxPropertySetHandle in = nullptr) const {
            return plugin_->mainEntry(action, effect, in, nullptr);
        }
        OfxStatus load() {
            plugin_->setHost(host_.binding());
            const auto status = call(kOfxActionLoad);
            loaded_ = status == kOfxStatOK;
            return status;
        }
        void created(OfxImageEffectHandle effect) {
            try {
                effects_.push_back(effect);
            } catch (...) {
                if (call(kOfxActionDestroyInstance, effect) != kOfxStatOK)
                    std::fputs("secondary failure: unpublished production effect destruction failed\n", stderr);
                throw;
            }
        }
        OfxStatus destroy(OfxImageEffectHandle effect) {
            auto it = std::find(effects_.begin(), effects_.end(), effect);
            require(it != effects_.end() && *it, "effect lifecycle mismatch");
            *it = nullptr;
            return call(kOfxActionDestroyInstance, effect);
        }
        OfxStatus unload() {
            require(std::all_of(effects_.begin(), effects_.end(), [](auto* p) {
                        return p == nullptr;
                    }),
                    "module unload before effect destruction");
            loaded_ = false;
            return call(kOfxActionUnload);
        }

    private:
        OfxPlugin* plugin_;
        InstanceHost& host_;
        std::vector<OfxImageEffectHandle> effects_;
        bool loaded_ = false;
    };

    std::vector<float> active_pixels(const std::vector<float>& pitched, const Case& test) {
        const int pitch = test.width * test.components + kPitchPadding;
        std::vector<float> result;
        result.reserve(static_cast<std::size_t>(test.width) * static_cast<std::size_t>(test.height) *
                       static_cast<std::size_t>(test.components));
        for (int y = 0; y < test.height; ++y) {
            const auto begin = pitched.begin() + static_cast<std::ptrdiff_t>(y) * pitch;
            const auto rowSamples = static_cast<std::ptrdiff_t>(test.width) * test.components;
            result.insert(result.end(), begin, begin + rowSamples);
        }
        return result;
    }


    Json words_json(const std::vector<float>& values) {
        Json words = Json::array();
        for (const float value : values) {
            words.push_back(std::bit_cast<std::uint32_t>(value));
        }
        return words;
    }

    std::vector<float> words_floats(const Json& words) {
        std::vector<float> result;
        result.reserve(words.size());
        for (const auto& word : words) {
            result.push_back(std::bit_cast<float>(word.get<std::uint32_t>()));
        }
        return result;
    }

    std::array<std::vector<float>, kOrdinaryCases.size()> expected_pixels(const std::filesystem::path& path, const std::string& preset) {
        std::ifstream input(path);
        require(static_cast<bool>(input), "ordinary spectral fixture is unavailable");
        const Json fixture = Json::parse(input);
        require(path.filename() == "ordinary-" + preset + ".json" &&
                    fixture.at("schema") == "spectral-ordinary-v1" && fixture.at("preset") == preset,
                "ordinary spectral fixture identity");
        const std::string referencePreset = preset.ends_with("-release") ? preset.substr(0, preset.size() - 8) + "-debug" : preset;
        require(fixture.at("reference_preset") == referencePreset, "ordinary spectral fixture reference preset");
        const auto& rows = fixture.at("rows");
        require(rows.is_object() && rows.size() == kOrdinaryCases.size(), "ordinary spectral fixture membership");
        std::array<std::vector<float>, kOrdinaryCases.size()> result;
        for (std::size_t route = 0; route < kOrdinaryCases.size(); ++route) {
            const auto& test = kOrdinaryCases[route];
            const auto& row = rows.at(test.id);
            require(row.at("shape") == Json::array({test.height, test.width, test.components}) && row.at("route") == route &&
                        row.at("reconstruction") == test.reconstruction && row.at("grain") == 0 && row.at("glare") == 0 &&
                        row.at("frame_token") == test.time && row.at("frame_rate") == 24 &&
                        row.at("film") == (route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f") &&
                        row.at("paper") == (test.components == 4 ? "kodak_portra_endura" : ""),
                    "ordinary spectral fixture controls");
            const auto source = active_pixels(initial_source(test), test);
            require(row.at("source_bits") == words_json(source), "ordinary spectral smoke source changed");
            const auto& output = row.at("output_bits");
            require(output.is_array() && output.size() == source.size() && std::ranges::all_of(output, [](const Json& word) {
                        return word.is_number_unsigned() && word.get<std::uint64_t>() <= std::numeric_limits<std::uint32_t>::max();
                    }),
                    "ordinary spectral fixture output words");
            result[route] = words_floats(output);
        }
        return result;
    }

    bool compare_fixture(const std::vector<float>& actual, const std::vector<float>& expected, int components) {
        if (actual.size() != expected.size()) {
            return false;
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const double tolerance = 2.0e-4 + 3.0e-4 * std::abs(static_cast<double>(expected[i]));
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
                std::abs(static_cast<double>(actual[i]) - expected[i]) > tolerance ||
                (components == 4 && i % 4 == 3 &&
                 std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(expected[i]))) {
                return false;
            }
        }
        return true;
    }

    class StreamGate final {
    public:
        explicit StreamGate(cudaStream_t stream) : stream_(stream) {
            require_cuda(cudaLaunchHostFunc(stream, &wait, this), "gate supplied render stream");
        }

        ~StreamGate() {
            open();
            (void)cudaStreamSynchronize(stream_);
        }

        void open() {
            {
                std::lock_guard lock(mutex_);
                open_ = true;
            }
            wake_.notify_one();
        }

        bool timed_out() const {
            return timedOut_.load();
        }

    private:
        static void CUDART_CB wait(void* user) {
            auto& gate = *static_cast<StreamGate*>(user);
            std::unique_lock lock(gate.mutex_);
            gate.timedOut_ = !gate.wake_.wait_for(lock, std::chrono::seconds(12), [&] {
                return gate.open_;
            });
        }

        cudaStream_t stream_;
        std::mutex mutex_;
        std::condition_variable wake_;
        bool open_ = false;
        std::atomic_bool timedOut_{false};
    };

    struct PassImages {
        const std::vector<float>& source;
        const std::vector<float>* expected = nullptr;
        bool holdStream = false;
    };
    Json run_pass(const CudaDriver& driver, Lifecycle& lifecycle, InstanceHost& host, OfxImageEffectHandle effect, const Case& test, std::string_view pass, const PassImages& images, DeviceBuffers& device, std::vector<std::uint32_t>& coldWords, bool summary) {
        const auto& [source, expected, holdStream] = images;
        const auto bytes = source.size() * sizeof(float);
        std::vector<float> initial(source.size(), kDestinationCanary);
        std::vector<float> output(source.size(), kDestinationCanary);
        require_cuda(cudaMemcpyAsync(device.source, source.data(), bytes, cudaMemcpyHostToDevice, device.stream), "upload deterministic source");
        require_cuda(cudaMemcpyAsync(device.destination, initial.data(), bytes, cudaMemcpyHostToDevice, device.stream), "reset destination canaries");
        require_cuda(cudaStreamSynchronize(device.stream), "complete image upload");
        const auto sourceBefore = host.image_release_count(effect, "Source");
        const auto outputBefore = host.image_release_count(effect, "Output");
        host.clear_calls();
        CUcontext before = nullptr, after = nullptr;
        driver.check(driver.get_current(&before), "inspect context before render");
        const auto gate = holdStream ? std::make_unique<StreamGate>(device.stream) : nullptr;
        const OfxStatus status = lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args());
        driver.check(driver.get_current(&after), "inspect context after render");
        if (gate) {
            require(cudaStreamQuery(device.stream) == cudaErrorNotReady && !gate->timed_out(),
                    "OFX returned with the supplied stream still gated");
            gate->open();
        }
        Json messages = Json::array();
        for (const auto& call : host.calls())
            if (call.operation == "message") {
                Json values = Json::array();
                for (const auto& value : call.values) {
                    const auto* text = std::get_if<std::string>(&value);
                    require(text != nullptr, "unexpected host message value");
                    values.push_back(*text);
                }
                messages.push_back(Json::array({call.target, call.property, std::move(values), call.status}));
            }
        const auto sourceReleases = host.image_release_count(effect, "Source") - sourceBefore;
        const auto outputReleases = host.image_release_count(effect, "Output") - outputBefore;
        require(status == kOfxStatOK && messages.empty() && !host.callback_error(),
                "OFX render callback " + std::string(test.id) + " returned " + std::to_string(status) + ": " + messages.dump());
        require(sourceReleases == 1 && outputReleases == 1, "render did not release each acquired image once");
        require(before == after, "render changed current CUDA context");
        require_cuda(cudaStreamSynchronize(device.stream), "complete render stream");
        require_cuda(cudaMemcpy(output.data(), device.destination, bytes, cudaMemcpyDeviceToHost), "read rendered pixels");
        std::vector<float> sourceAfter(source.size());
        require_cuda(cudaMemcpy(sourceAfter.data(), device.source, bytes, cudaMemcpyDeviceToHost), "read source after render");
        require(sourceAfter == source, "render modified source pixels or padding");
        const bool padding = canaries_preserved(initial, output, test);
        const bool alpha = alpha_preserved(source, output, test);
        require(padding && alpha, "render changed alpha or destination padding");
        const auto activeInput = active_pixels(source, test);
        const auto activeOutput = active_pixels(output, test);
        const bool finite = std::all_of(activeOutput.begin(), activeOutput.end(), [](float value) {
            return std::isfinite(value);
        });
        require(finite, "rendered pixels are nonfinite");
        const Json fixtureMatch = expected ? Json(compare_fixture(activeOutput, *expected, test.components)) : Json(nullptr);
        require(!expected || fixtureMatch == true, "rendered " + std::string(test.id) + (holdStream ? " with its stream gated" : "") + " differs from accepted fixture");
        std::vector<std::uint32_t> outputWords;
        outputWords.reserve(activeOutput.size());
        for (float value : activeOutput)
            outputWords.push_back(std::bit_cast<std::uint32_t>(value));
        if (pass == "cold")
            coldWords = outputWords;
        else
            require(coldWords == outputWords, "warm output differs from cold output");
        Json row{{"case_id", std::string(test.id) + "/" + std::string(pass)}, {"route", test.routeName}, {"pass", std::string(pass)}, {"ofx_status", status}, {"messages", std::move(messages)}, {"input_shape", Json::array({test.height, test.width, test.components})}, {"output_shape", Json::array({test.height, test.width, test.components})}, {"accepted_fixture_match", fixtureMatch}, {"finite_output", finite}, {"alpha_preserved", alpha}, {"padding_preserved", padding}, {"source_preserved", true}, {"stream_completed", true}, {"readback_complete", true}, {"context_preserved", before == after}, {"context_restored", false}, {"source_image_releases", sourceReleases}, {"output_image_releases", outputReleases}, {"samples_compared", activeOutput.size()}};
        if (!summary) {
            row["input_f32_words"] = words_json(activeInput);
            row["output_f32_words"] = words_json(activeOutput);
        }
        return row;
    }

    OfxPlugin* check_discovery(const Module& module, const std::filesystem::path& path, const char* expected) {
        require(module.loaded_from(path), "discovery loaded path differs from declared module");
        const auto count = module.symbol<int (*)()>("OfxGetNumberOfPlugins");
        const auto get = module.symbol<OfxPlugin* (*)(int)>("OfxGetPlugin");
        require(count() == 1 && !get(-1) && !get(1) && !get(std::numeric_limits<int>::min()) &&
                    !get(std::numeric_limits<int>::max()),
                "discovery plugin count or index bounds");
        auto* plugin = get(0);
        require(plugin && plugin == get(0) && plugin->pluginIdentifier &&
                    std::string_view(plugin->pluginIdentifier) == expected && plugin->setHost && plugin->mainEntry,
                std::string("discovery identity or entry points: ") + expected);
        return plugin;
    }


    struct EditedRenderInputs {
        std::filesystem::path module;
        std::filesystem::path fixture;
        std::filesystem::path preflashFixtures;
        std::filesystem::path artifacts;
        std::string preset;
    };

    class ScratchBundle final {
    public:
        explicit ScratchBundle(const std::filesystem::path& artifacts)
            : directory_(artifacts / ("print-channel-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
            std::filesystem::create_directories(artifacts);
            require(std::filesystem::create_directory(directory_), "create isolated Print fixture bundle");
        }

        ~ScratchBundle() noexcept {
            try {
                std::error_code error;
                std::filesystem::remove_all(directory_, error);
            } catch (...) {
                std::fputs("could not remove isolated Print fixture bundle\n", stderr);
            }
        }

        const std::filesystem::path& directory() const {
            return directory_;
        }

    private:
        std::filesystem::path directory_;
    };

    Json captured_preflash(const EditedRenderInputs& input) {
        const auto manifest = Json::parse(std::ifstream(input.preflashFixtures / "manifest.json"));
        return Json::parse(std::ifstream(input.preflashFixtures /
                                         manifest.at("presets").at(input.preset).at("native").get<std::string>()));
    }

    constexpr const char* kViewingPaper = "portra_endura_viewing_d65";
    constexpr const char* kRetryFilm = "portra_400_base_plus_0125";

    std::filesystem::path bundle_with_print_edits(
        const EditedRenderInputs& input, const ScratchBundle& scratch, const Json& captured) {
        const auto product = input.module.parent_path().parent_path().parent_path();
        const auto bundle = scratch.directory() / product.filename();
        std::filesystem::copy(product, bundle, std::filesystem::copy_options::recursive | std::filesystem::copy_options::create_hard_links);
        const auto profiles = bundle / "Contents/Resources/profiles";
        const auto originalFilm = Json::parse(std::ifstream(profiles / "kodak_portra_400.json"));
        auto invalid = originalFilm;
        invalid.at("info").at("stock") = "preflash_power_overflow";
        invalid.at("info").at("name") = "Print preflash power overflow";
        const auto& cases = captured.at("cases");
        const auto failure = std::ranges::find_if(cases, [](const Json& row) {
            return row.at("id") == "preflash-power-overflow";
        });
        require(failure != cases.end() && failure->at("success") == false, "frozen preflash power failure missing");
        constexpr std::size_t sample = 40;
        auto& data = invalid.at("data");
        data.at("base_density").at(sample) = std::bit_cast<float>(failure->at("film_base_density").at(sample).get<std::uint32_t>());
        // Balance traverses developed dye density at this wavelength. Raw
        // preflash traverses the film base alone, which overflows the power.
        data.at("channel_density").at(sample) = {4096.0, 4096.0, 4096.0};
        std::ofstream output(profiles / "preflash_power_overflow.json");
        output << invalid;
        require(static_cast<bool>(output), "write isolated failing Print profile");

        auto retry = originalFilm;
        retry.at("info").at("stock") = kRetryFilm;
        auto& base = retry.at("data").at("base_density");
        require(base.size() == 81, "retry Film base must cover the fixed spectral axis");
        for (auto& value : base) {
            if (!value.is_null()) {
                value = value.get<double>() + 0.125;
            }
        }
        const auto write_profile = [&](const char* key, const Json& profile) {
            std::ofstream file(profiles / (std::string(key) + ".json"));
            file << profile;
            file.close();
            require(static_cast<bool>(file), "write isolated edited profile");
        };
        write_profile(kRetryFilm, retry);

        auto viewing = Json::parse(std::ifstream(profiles / "kodak_portra_endura.json"));
        auto& info = viewing.at("info");
        require(info.at("viewing_illuminant") == "D50", "original paper must be viewed under D50");
        info.at("viewing_illuminant") = "D65";
        // The loaded catalog retains each stock. Give the alternate specimen
        // a key and the same calibration; its paper samples are unchanged.
        info.at("stock") = kViewingPaper;
        write_profile(kViewingPaper, viewing);
        const auto calibrationPath = bundle / "Contents/Resources/filters/neutral_print_filters.json";
        auto calibration = Json::parse(std::ifstream(calibrationPath));
        auto& paper = calibration.at("kodak_portra_endura");
        for (auto& entry : paper) {
            if (entry.contains("kodak_portra_400")) {
                const auto neutral = entry.at("kodak_portra_400");
                entry[kRetryFilm] = neutral;
            }
        }
        const auto sameCalibration = paper;
        calibration[kViewingPaper] = sameCalibration;
        require(std::filesystem::remove(calibrationPath), "detach calibration hard link in scratch bundle");
        std::ofstream calibrationOutput(calibrationPath);
        calibrationOutput << calibration;
        calibrationOutput.close();
        require(static_cast<bool>(calibrationOutput), "write isolated calibration with matching entries");
        return bundle / input.module.lexically_relative(product);
    }

#if JUICER_DIAGNOSTICS_COMPILED
    void set_trace_environment(const std::filesystem::path& path) {
#ifdef _WIN32
        require(_putenv_s("JUICER_DIAGNOSTICS", "3") == 0 &&
                    _putenv_s("JUICER_DIAGNOSTICS_PATH", path.string().c_str()) == 0,
                "configure resource diagnostics");
#else
        require(setenv("JUICER_DIAGNOSTICS", "3", 1) == 0 &&
                    setenv("JUICER_DIAGNOSTICS_PATH", path.c_str(), 1) == 0,
                "configure resource diagnostics");
#endif
    }

    struct PrintReceipt {
        std::array<std::uint64_t, 4> keys{};
        std::array<std::uintptr_t, 2> slots{};
        std::array<std::uint32_t, 3> raw{};
        std::uint64_t chargedBytes = 0;
        std::size_t copies = 0;
    };

    constexpr std::string_view kPrintReceiptPrefix = "PHASE4C_BALANCE | event=print_resource_preparation";

    struct RenderReceipt {
        std::uint64_t snapshot = 0;
        std::uint64_t instance = 0;
        std::uint64_t status = 0;
        std::uint64_t scanner = 0;
        std::uint64_t paper = 0;
        bool rollbackFenced = false;
    };

    RenderReceipt read_render_receipt(const std::filesystem::path& path, std::uintmax_t offset = 0) {
        constexpr std::string_view prefix = "CUDALIFE | event=render_outcome ";
        constexpr std::string_view rollback = "CUDALIFE | event=frame_rollback ";
        std::ifstream trace(path, std::ios::binary);
        trace.seekg(static_cast<std::streamoff>(offset));
        std::string latest;
        std::vector<std::string> rollbacks;
        for (std::string line; std::getline(trace, line);) {
            if (line.starts_with(prefix)) {
                latest = line.substr(prefix.size());
            } else if (line.starts_with(rollback)) {
                rollbacks.push_back(std::move(line));
            }
        }
        require(!latest.empty(), "missing flushed native render outcome");
        std::istringstream fields(latest);
        const auto number = [&](std::string_view label) {
            std::string word;
            fields >> word;
            require(word.starts_with(label), "malformed native render outcome");
            return std::stoull(word.substr(label.size()));
        };
        RenderReceipt result{number("snapshot_id="), number("instance_token="), number("status="), number("scanner_hash="), number("paper_version=")};
        const auto fenced = std::string(rollback) + "snapshot_id=" + std::to_string(result.snapshot) +
                            " instance_token=" + std::to_string(result.instance) + " use_fenced=1";
        result.rollbackFenced = std::ranges::find(rollbacks, fenced) != rollbacks.end();
        return result;
    }

    PrintReceipt parse_print_receipt(const std::string& line) {
        const auto offset = line.find(" resident=");
        require(offset != std::string::npos, "missing Print residency receipt");
        std::string fields = line.substr(offset + 10);
        std::ranges::replace(fields, '/', ' ');
        std::istringstream input(fields);
        PrintReceipt receipt;
        std::string slotsLabel, chargeLabel, rawLabel;
        for (auto& key : receipt.keys) {
            input >> key;
        }
        input >> slotsLabel;
        require(slotsLabel.starts_with("slots="), "missing Print resident slots");
        receipt.slots[0] = std::stoull(slotsLabel.substr(6));
        input >> receipt.slots[1] >> chargeLabel >> rawLabel >> receipt.raw[1] >> receipt.raw[2];
        require(input && chargeLabel.starts_with("chargedBytes="), "missing Print allocation charge");
        receipt.chargedBytes = std::stoull(chargeLabel.substr(13));
        require(rawLabel.starts_with("raw="), "missing Print raw preflash");
        receipt.raw[0] = static_cast<std::uint32_t>(std::stoul(rawLabel.substr(4)));
        return receipt;
    }

    std::array<std::array<std::uint32_t, 81>, 2> read_resident_print_lights(const std::filesystem::path& path) {
        std::ifstream trace(path);
        std::string latest;
        for (std::string line; std::getline(trace, line);) {
            if (line.starts_with(kPrintReceiptPrefix)) {
                latest = line;
            }
        }
        const auto receipt = parse_print_receipt(latest);
        std::array<std::array<std::uint32_t, 81>, 2> lights{};
        for (std::size_t light = 0; light < lights.size(); ++light) {
            require(receipt.slots[light] != 0, "missing active Print light slot");
            require_cuda(cudaMemcpy(lights[light].data(), reinterpret_cast<const void*>(receipt.slots[light]), sizeof(lights[light]), cudaMemcpyDeviceToHost),
                         "read resident Print light words");
        }
        return lights;
    }

    void check_print_and_film_transfers(
        const std::filesystem::path& path, const std::vector<std::string_view>& printRenders, const Json& captured) {
        std::ifstream trace(path);
        require(static_cast<bool>(trace), "read production resource diagnostics");
        std::vector<PrintReceipt> print;
        std::vector<std::pair<std::uint64_t, std::size_t>> filmRefills;
        std::size_t printCopies = 0, curveCopies = 0;
        for (std::string line; std::getline(trace, line);) {
            if (line.starts_with("CUDA_UPLOAD | label=phase4B")) {
                ++printCopies;
            } else if (line.starts_with("CUDA_UPLOAD | label=focused film normalizedDens")) {
                ++curveCopies;
            } else if (line.starts_with("CUDA_UPLOAD | filmCurveHash=")) {
                filmRefills.emplace_back(std::stoull(line.substr(line.find('=') + 1)), std::exchange(curveCopies, 0));
            } else if (line.starts_with(kPrintReceiptPrefix)) {
                auto receipt = parse_print_receipt(line);
                receipt.copies = std::exchange(printCopies, 0);
                print.push_back(receipt);
            }
        }
        require(filmRefills.size() >= 3 &&
                    filmRefills[0].first != filmRefills[1].first && filmRefills[0].first == filmRefills[2].first &&
                    filmRefills[1].second == 6 && filmRefills[2].second == 6,
                "gated Portra thumbnail did not refill all normalized curve axes and samples after Ektar");
        require(print.size() == printRenders.size() && printCopies == 0, "unmatched Print preparations or copies");
        const auto receipt = [&](std::string_view operation) -> const PrintReceipt& {
            const auto found = std::ranges::find(printRenders, operation);
            require(found != printRenders.end(), "missing staged Print operation");
            return print[static_cast<std::size_t>(found - printRenders.begin())];
        };
        const auto& a = receipt("warm A");
        require(a.copies == 0 && a.slots[0] != 0 && a.slots[1] != 0 && a.chargedBytes != 0,
                "warm Print A did not retain both illuminants");
        for (const auto* edit : {"viewing only", "paper D50", "paper D65", "paper restored D50", "balance only", "active amount", "A after rejection"}) {
            const auto& result = receipt(edit);
            require(result.copies == 0 && result.keys == a.keys && result.slots == a.slots &&
                        result.chargedBytes == a.chargedBytes && result.raw == a.raw,
                    std::string(edit) + " changed Print uploads or residency");
        }
        const auto& main = receipt("main filter");
        require(main.copies == 1 && main.keys[0] == a.keys[0] && main.keys[1] == a.keys[1] &&
                    main.keys[2] != a.keys[2] && main.keys[3] == a.keys[3] && main.slots == a.slots &&
                    main.chargedBytes == a.chargedBytes,
                "main filter did not overwrite exactly the resident main illuminant");
        const auto& restored = receipt("restored main filter");
        require(restored.copies == 1 && restored.keys == a.keys && restored.slots == a.slots,
                "restoring the filter did not overwrite the main illuminant once");
        const auto& products = captured.at("products");
        const auto expected = std::ranges::find_if(products, [](const Json& row) {
            return row.at("route") == 1 && row.at("mode") == 1 && row.at("variant") == 1;
        });
        require(expected != products.end(), "missing frozen Print A raw expectation");
        const auto aWords = expected->at("raw").get<std::array<std::uint32_t, 3>>();
        require(a.raw == aWords, "warm Print A did not project the frozen raw words");
        const auto& retry = receipt("valid retry");
        require(retry.copies == 4 && retry.keys[0] != a.keys[0] &&
                    std::equal(retry.keys.begin() + 1, retry.keys.end(), a.keys.begin() + 1) &&
                    retry.slots == a.slots && retry.chargedBytes == a.chargedBytes,
                "valid post-rejection retry must refill the film lanes and retain both lights and the paper tables");
        const auto transmission = std::pow(10.0, -0.125);
        for (std::size_t channel = 0; channel < aWords.size(); ++channel) {
            const auto predicted = static_cast<double>(std::bit_cast<float>(aWords[channel])) * transmission;
            require(retry.raw[channel] != aWords[channel] &&
                        std::abs(std::bit_cast<float>(retry.raw[channel]) - predicted) <= 1e-8 + 0.01 * predicted,
                    "changed Film preflash must differ from A and match the accepted base-transmission prediction");
        }
        std::cout << "Print D50/D65/D50 copies=0/0/0, changed Film copies=" << retry.copies
                  << ", raw A=" << aWords[0] << '/' << aWords[1] << '/' << aWords[2]
                  << ", raw retry=" << retry.raw[0] << '/' << retry.raw[1] << '/' << retry.raw[2] << '\n';
        std::cout << "Film curve refill copies=" << filmRefills[2].second
                  << ", staged Print A retained four keys, both illuminant slots and byte arrays, and "
                  << a.chargedBytes << " charged bytes after Rust rejection and zero-copy edits\n";
    }
#endif

    int frame_and_print_edits(const EditedRenderInputs& input) {
        const auto expected = expected_pixels(input.fixture, input.preset);
        const ScratchBundle scratch(input.artifacts);
        const auto captured = captured_preflash(input);
        const auto path = bundle_with_print_edits(input, scratch, captured);
        const CudaDriver driver;
        driver.check(driver.init(0), "initialize CUDA driver");
        CUcontext original = nullptr;
        driver.check(driver.get_current(&original), "read CUDA context");
        ContextRestore restore(driver, original);
        require_cuda(cudaSetDevice(0), "select CUDA device");
#if JUICER_DIAGNOSTICS_COMPILED
        const auto trace = scratch.directory() / "resources.log";
        set_trace_environment(trace);
        std::vector<std::string_view> printRenders;
#endif
        Module module(path);
        InstanceHost host;
        Lifecycle lifecycle(check_discovery(module, path, "com.juicer.Juicer"), host);
        require(lifecycle.load() == kOfxStatOK, "load staged module for edited renders");
        require(lifecycle.call(kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK,
                "describe staged module for edited renders");
        DeviceBuffers device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create nonblocking render stream");
        for (std::size_t route = 0; route < kOrdinaryCases.size(); ++route) {
            const auto& test = kOrdinaryCases[route];
            auto* effect = host.create_effect();
            author_parameters(host, effect, test);
            host.set_value(effect, "GlareActive", {0});
            if (route >= 2) {
                host.set_value(effect, "FilmProfileKey", {std::string("fujifilm_provia_100f")});
            }
            require(lifecycle.call(kOfxActionCreateInstance, effect) == kOfxStatOK, "create edited-render effect");
            lifecycle.created(effect);
            const auto source = initial_source(test);
            const auto bytes = source.size() * sizeof(float);
            require_cuda(cudaMalloc(&device.source, bytes), "allocate source");
            require_cuda(cudaMalloc(&device.destination, bytes), "allocate destination");
            configure(host, effect, test, device);
            const auto edit = [&](const char* name, std::vector<InstanceValue> values) {
                host.set_value(effect, name, std::move(values));
                auto* args = host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, name});
                require(lifecycle.call(kOfxActionInstanceChanged, effect, args) == kOfxStatReplyDefault,
                        std::string("edit ") + name);
            };
            const auto render = [&](const std::vector<float>* expectation = nullptr, bool hold = false, [[maybe_unused]] std::string_view operation = {}) {
                std::vector<std::uint32_t> words;
                (void)run_pass(driver, lifecycle, host, effect, test, "cold", {source, expectation, hold}, device, words, true);
#if JUICER_DIAGNOSTICS_COMPILED
                if (route % 2 != 0) {
                    printRenders.push_back(operation);
                }
#endif
                return words;
            };
            const auto ordinary = render(&expected[route]);
            if (route == 0) {
                edit("FilmProfileKey", {std::string("kodak_ektar_100")});
                edit("ScanRoute", {std::string(test.route)});
                const auto ektar = render();
                require(ektar != ordinary, "Ektar must replace the warmed Portra curves");
                edit("FilmProfileKey", {std::string("kodak_portra_400")});
                edit("ScanRoute", {std::string(test.route)});
                edit("GrainModel", {std::string("film_v1")});
                edit("GrainActive", {1});
                host.set_clip_property(effect, "Source", kOfxImageClipPropThumbnail, {1});
                require(render(&expected[route], true) == ordinary, "queued Film thumbnail differs from the Portra fixture");

                unsigned queries = 0;
                const auto sourceReleases = host.image_release_count(effect, "Source");
                const auto outputReleases = host.image_release_count(effect, "Output");
                const std::vector<float> canaries(source.size(), kDestinationCanary);
                require_cuda(cudaMemcpyAsync(device.destination, canaries.data(), bytes, cudaMemcpyHostToDevice, device.stream),
                             "reset cancelled frame output");
                require_cuda(cudaStreamSynchronize(device.stream), "complete cancelled frame output reset");
#if JUICER_DIAGNOSTICS_COMPILED
                const auto beforeAbort = std::filesystem::file_size(trace);
#endif
                {
                    StreamGate gate(device.stream);
                    const auto prior = host.set_abort_callback([&]() {
                        return ++queries == 5 ? 1 : 0;
                    });
                    const auto status = lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args());
                    (void)host.set_abort_callback(prior);
                    require(status == kOfxStatOK && queries == 5, "post-enqueue abort did not reach the native completion query");
#if JUICER_DIAGNOSTICS_COMPILED
                    const auto outcome = read_render_receipt(trace, beforeAbort);
                    require(outcome.status == FJ_STATUS_CANCELLED && outcome.rollbackFenced,
                            "the cancelled invocation must report native cancellation and rollback with its use fenced");
                    std::cout << "Native cancellation snapshot=" << outcome.snapshot << " use_fenced=1\n";
#endif
                    require(cudaStreamQuery(device.stream) == cudaErrorNotReady && !gate.timed_out(),
                            "pending work disappeared before post-enqueue abort returned");
                    gate.open();
                    require_cuda(cudaStreamSynchronize(device.stream), "complete work submitted by the cancelled frame");
                }
                std::vector<float> cancelled(source.size());
                require_cuda(cudaMemcpy(cancelled.data(), device.destination, bytes, cudaMemcpyDeviceToHost), "read cancelled frame output");
                require(canaries_preserved(canaries, cancelled, test) && alpha_preserved(source, cancelled, test) &&
                            compare_fixture(active_pixels(cancelled, test), expected[route], test.components),
                        "cancelled frame did not enqueue the guarded Portra output behind the gate");
                require(host.image_release_count(effect, "Source") == sourceReleases + 1 &&
                            host.image_release_count(effect, "Output") == outputReleases + 1,
                        "cancelled render did not release both images once");
                require(render(&expected[route]) == ordinary, "render after queued abort did not recover");
            }
            if (route % 2 != 0) {
                edit("PrintPreflash", {0.05});
                const auto preflashed = render();
                require(preflashed != ordinary && render() == preflashed, "preflash did not change and then reuse output");
                if (route == 1) {
                    edit("Exposure", {0.5});
                    const auto a = render();
                    require(render(nullptr, false, "warm A") == a, "warm Print A changed pixels");
#if JUICER_DIAGNOSTICS_COMPILED
                    const auto lightsA = read_resident_print_lights(trace);
#endif
                    edit("OutputCctfEncoding", {0});
                    require(render(nullptr, false, "viewing only") != a, "viewing edit did not change encoding");
                    edit("OutputCctfEncoding", {1});
                    require(render() == a, "restoring viewing did not restore Print A");
                    const auto d50 = render(nullptr, false, "paper D50");
#if JUICER_DIAGNOSTICS_COMPILED
                    const auto d50Binding = read_render_receipt(trace);
#endif
                    edit("PrintProfileKey", {std::string(kViewingPaper)});
                    require(render(nullptr, false, "paper D65") != d50, "D65 viewing metadata did not change the guarded output");
#if JUICER_DIAGNOSTICS_COMPILED
                    const auto d65Binding = read_render_receipt(trace);
#endif
                    edit("PrintProfileKey", {std::string("kodak_portra_endura")});
                    require(render(nullptr, false, "paper restored D50") == d50, "restored D50 did not recover the guarded output");
#if JUICER_DIAGNOSTICS_COMPILED
                    const auto restoredBinding = read_render_receipt(trace);
                    require(d50Binding.scanner != d65Binding.scanner && d50Binding.scanner == restoredBinding.scanner &&
                                d50Binding.paper != d65Binding.paper && d50Binding.paper == restoredBinding.paper,
                            "paper viewing metadata did not change and restore the production scanner binding");
#endif
                    edit("PrintExposureCompensation", {0});
                    require(render(nullptr, false, "balance only") != a, "balance edit did not change normalization");
                    edit("PrintExposureCompensation", {1});
                    require(render() == a, "restoring balance did not restore Print A");
                    edit("PrintPreflash", {0.125});
                    require(render(nullptr, false, "active amount") != a, "active preflash amount did not change Print exposure");
                    edit("PrintPreflash", {0.05});
                    require(render() == a, "restoring preflash amount did not restore Print A");
                    edit("EnlargerY", {2.1234567890123});
                    require(render(nullptr, false, "main filter") != a, "main filter edit did not change Print A");
                    edit("EnlargerY", {0.0});
                    require(render(nullptr, false, "restored main filter") == a, "restoring main filter did not restore Print A");
                    edit("FilmProfileKey", {std::string("preflash_power_overflow")});
                    edit("ScanRoute", {std::string(test.route)});
                    const auto sourceReleases = host.image_release_count(effect, "Source");
                    const auto outputReleases = host.image_release_count(effect, "Output");
                    host.clear_calls();
                    require_cuda(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal), "observe rejected Print submissions");
                    const auto status = lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args());
                    cudaGraph_t graph = nullptr;
                    require_cuda(cudaStreamEndCapture(device.stream, &graph), "finish rejected Print submission observation");
                    std::size_t nodes = 0;
                    const auto nodeStatus = cudaGraphGetNodes(graph, nullptr, &nodes);
                    require_cuda(cudaGraphDestroy(graph), "destroy rejected Print submission graph");
                    require_cuda(nodeStatus, "read rejected Print submission graph");
                    require(status == kOfxStatErrFatal && nodes == 0, "Rust channel rejection submitted native work");
                    require(host.image_release_count(effect, "Source") == sourceReleases + 1 &&
                                host.image_release_count(effect, "Output") == outputReleases + 1,
                            "Rust channel rejection did not release both images once");
                    require(std::ranges::any_of(host.calls(), [](const auto& call) {
                                return call.operation == "message" && call.values.size() == 2 &&
                                       call.values[0] == InstanceValue{std::string("FilmJuicerRenderAdmission")} &&
                                       std::get<std::string>(call.values[1]).find("field=preflash_raw") != std::string::npos;
                            }),
                            "the failed stock must reach Rust preflash construction");
                    edit("FilmProfileKey", {std::string("kodak_portra_400")});
                    edit("ScanRoute", {std::string(test.route)});
                    require(render(nullptr, false, "A after rejection") == a, "rejecting the Rust channel changed warm A");
#if JUICER_DIAGNOSTICS_COMPILED
                    require(read_resident_print_lights(trace) == lightsA, "rejecting the Rust channel changed resident Print A light words");
#endif
                    edit("PrintPreflash", {0.125});
                    edit("FilmProfileKey", {std::string(kRetryFilm)});
                    edit("ScanRoute", {std::string(test.route)});
                    const auto retry = render(nullptr, false, "valid retry");
                    require(retry != a && render() == retry, "valid authored retry did not change and then reuse pixels");
                    edit("FilmProfileKey", {std::string("kodak_portra_400")});
                    edit("ScanRoute", {std::string(test.route)});
                    edit("PrintPreflash", {0.05});
                    edit("Exposure", {0.0});
                    require(render() == preflashed, "restoring exposure did not restore preflashed pixels");
                }
                edit("PrintProfileKey", {std::string("fujifilm_crystal_archive_typeii")});
                const auto otherPaper = render();
                require(otherPaper != preflashed && render() == otherPaper, "paper edit did not change and then reuse output");
                edit("PrintProfileKey", {std::string("kodak_portra_endura")});
                require(render() == preflashed, "paper A/B/A render did not restore preflash output");
                edit("PrintPreflash", {0.0});
                require(render(&expected[route]) == ordinary, "disabling preflash did not restore specimen A");
            }
            edit("ScannerLensBlurSigmaPx", {0.6});
            const auto fused = render();
            require(fused != ordinary, "fused DIR scan skipped the scanner post stage");
            // A post stage with no spatial DIR carrier uses the density planes.
            edit("CouplersActive", {0});
            const auto plane = render();
            require(render() == plane, "repeat of materialized scan was unstable");
            edit("ScannerLensBlurSigmaPx", {0.0});
            edit("CouplersActive", {1});
            require(render(&expected[route]) == ordinary, "restoring the source schedule changed output");
            require(lifecycle.destroy(effect) == kOfxStatOK, "destroy edited-render effect");
            device.release_images();
        }
        device.close();
        require(lifecycle.unload() == kOfxStatOK && !host.callback_error(), "unload edited-render module");
        module.close();
#if JUICER_DIAGNOSTICS_COMPILED
        check_print_and_film_transfers(trace, printRenders, captured);
#endif
        restore.restore();
        return 0;
    }

    int film_v1_witness(const std::filesystem::path& path) {
        require(path.filename() == "juicer.ofx", "expected standard module");
        const auto resources = path.parent_path().parent_path() / "Resources";
        require(std::filesystem::is_regular_file(resources / "grain/portra-400/manifest.json") &&
                    std::filesystem::is_regular_file(resources / "grain/portra-400/records.f32") &&
                    std::filesystem::is_regular_file(resources / "grain/portra-400/sums.f32"),
                "preview bundle grain bank missing");
        const CudaDriver driver;
        driver.check(driver.init(0), "initialize CUDA driver");
        CUcontext original = nullptr;
        driver.check(driver.get_current(&original), "read original CUDA context");
        ContextRestore restore(driver, original);
        require_cuda(cudaSetDevice(0), "select CUDA device");
        Module module(path);
        const auto count = module.symbol<int (*)()>("OfxGetNumberOfPlugins");
        const auto get = module.symbol<OfxPlugin* (*)(int)>("OfxGetPlugin");
        require(count() == 1 && get(0) && get(0)->pluginIdentifier &&
                    std::string_view(get(0)->pluginIdentifier) == "com.juicer.Juicer",
                "preview discovery identity");
        InstanceHost host;
        Lifecycle lifecycle(get(0), host);
        require(lifecycle.load() == kOfxStatOK, "preview load");
        require(lifecycle.call(kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK,
                "preview context description");
        DeviceBuffers device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create preview stream");
        Json routes = Json::array();
        bool allPassed = true;
        for (const auto& test : kFilmV1Cases) {
            auto* effect = host.create_effect();
            const auto amountDefault = host.value(effect, "FilmGrainAmountEv");
            const auto sizeDefault = host.value(effect, "FilmGrainTexelPitchUm");
            const auto colourDefault = host.value(effect, "FilmGrainChromaFraction");
            require(amountDefault == std::vector<InstanceValue>{0.0} &&
                        sizeDefault == std::vector<InstanceValue>{1.0} &&
                        colourDefault == std::vector<InstanceValue>{1.0},
                    "preview descriptor slider defaults");
            author_parameters(host, effect, test);
            const bool proxy = std::string_view(test.id) == "negative-direct-quarter-size-half" ||
                               std::string_view(test.id) == "negative-print-proxy-sequence";
            const bool smallerFormats = std::string_view(test.id).ends_with("-smaller-formats");
            const bool largerFormats = std::string_view(test.id).ends_with("-larger-formats");
            const bool thumbnails = std::string_view(test.id).ends_with("-thumbnails");
            const bool eightMmProxy = std::string_view(test.id).ends_with("-8mm-proxy");
            const bool clippedQuarter = std::string_view(test.id).ends_with("-clipped-quarter");
            const bool timeEndpoint = std::string_view(test.id).ends_with("-time-endpoint");
            const bool smallFormatHighGain = std::string_view(test.id).ends_with("-small-format-high-gain");
            if (proxy || smallerFormats || thumbnails || eightMmProxy || clippedQuarter || timeEndpoint || smallFormatHighGain)
                host.set_value(effect, "GlareActive", {0});
            host.set_value(effect, "CameraFilmFormatPreset", {2});
            host.set_value(effect, "CameraFilmFormatMm", {36.0});
            require(lifecycle.call(kOfxActionCreateInstance, effect) == kOfxStatOK, "preview create");
            lifecycle.created(effect);
            Case renderTest = test;
            auto source = smallFormatHighGain ? flat_grain_source(renderTest) : initial_source(renderTest);
            const auto bytes = source.size() * sizeof(float);
            require_cuda(cudaMalloc(&device.source, bytes), "allocate preview source");
            require_cuda(cudaMalloc(&device.destination, bytes), "allocate preview destination");
            configure(host, effect, test, device);
            std::string lastEdit = "initial grain-off state";
            auto change = [&](const char* name, std::vector<InstanceValue> value) {
                lastEdit = name;
                host.set_value(effect, name, std::move(value));
                auto* args = host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, name});
                require(lifecycle.call(kOfxActionInstanceChanged, effect, args) == kOfxStatReplyDefault,
                        "preview control change");
            };
            auto render = [&](double time) {
                host.set_property(host.action_args(), kOfxPropTime, {time});
                std::vector<std::uint32_t> words;
                try {
                    (void)run_pass(driver, lifecycle, host, effect, renderTest, "cold", {source, nullptr, {}}, device, words, true);
                } catch (const std::exception& error) {
                    throw std::runtime_error(lastEdit + " at frame " + std::to_string(time) + ": " + error.what());
                }
                return words;
            };
            const auto reject_footprint = [&](std::string_view expected) {
                host.clear_calls();
                const auto sourceBefore = host.image_release_count(effect, "Source");
                const auto outputBefore = host.image_release_count(effect, "Output");
                require(lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args()) == kOfxStatErrFatal,
                        "unsupported footprint was not rejected");
                require(host.image_release_count(effect, "Source") == sourceBefore + 1 &&
                            host.image_release_count(effect, "Output") == outputBefore + 1,
                        "unsupported footprint did not release each image once");
                require(std::ranges::any_of(host.calls(), [expected](const auto& call) {
                            return call.operation == "message" && call.values.size() == 2 &&
                                   call.values[0] == InstanceValue{std::string("FilmJuicerRenderAdmission")} &&
                                   std::get<std::string>(call.values[1]).find(expected) != std::string::npos;
                        }),
                        "unsupported footprint diagnostic omitted its kind, range or physical geometry");
            };
            if (smallFormatHighGain) {
                constexpr double time = 0.0;
                struct Request {
                    int width, height;
                    double mm, amount;
                };
                constexpr std::array requests{
                    Request{4832, 96, 8.0, 3.0},
                    Request{4832, 96, 11.66, 3.0},
                    Request{3840, 96, 8.0, 3.0},
                    Request{3840, 96, 11.66, 3.0},
                    Request{4832, 2016, 8.0, 3.0},
                    Request{4832, 2016, 11.66, 3.0},
                    Request{1920, 96, 8.0, 3.0},
                    Request{1920, 96, 11.66, 3.0},
                    Request{4832, 96, 8.0, 2.0},
                    Request{4832, 96, 36.0, 3.0},
                };
                auto full = [&](int width, int height) {
                    device.release_images();
                    renderTest.width = width;
                    renderTest.height = height;
                    source = flat_grain_source(renderTest);
                    const auto imageBytes = source.size() * sizeof(float);
                    require_cuda(cudaMalloc(&device.source, imageBytes), "allocate Full source");
                    require_cuda(cudaMalloc(&device.destination, imageBytes), "allocate Full destination");
                    configure(host, effect, renderTest, device);
                };
                change("GrainModel", {std::string("film_v1")});
                change("GrainActive", {1});
                const auto reference = render(time);
                Json rows = Json::array();
                std::set<std::tuple<int, int, double>> highGain;
                for (const auto& [width, height, mm, amount] : requests) {
                    if (renderTest.width != width || renderTest.height != height)
                        full(width, height);
                    change("GrainActive", {0});
                    change("CameraFilmFormatMm", {mm});
                    change("FilmGrainAmountEv", {amount});
                    change("FilmGrainTexelPitchUm", {0.5});
                    require(host.value(effect, "CameraFilmFormatPreset") == std::vector<InstanceValue>{0} &&
                                host.value(effect, "CameraFilmFormatMm") == std::vector<InstanceValue>{mm} &&
                                host.value(effect, "FilmGrainAmountEv") == std::vector<InstanceValue>{amount} &&
                                host.value(effect, "FilmGrainTexelPitchUm") == std::vector<InstanceValue>{0.5} &&
                                host.value(effect, "FilmGrainChromaFraction") == colourDefault,
                            "effective Film settings differ from the high-gain request");
                    const auto off = render(time);
                    change("GrainActive", {1});
                    const auto sampleWidth = static_cast<float>(static_cast<double>(static_cast<float>(mm)) * 2000.0 / width);
                    Json row{{"width", width}, {"height", height}, {"mm", mm}, {"amount_ev", amount}, {"size_um", 0.5}, {"colour", 1.0}, {"sample_width", sampleWidth}, {"grain_changes_output", false}};
                    try {
                        require(render(time) != off, "Film did not change matched grain-off pixels");
                        row["grain_changes_output"] = true;
                        row["ofx_status"] = kOfxStatOK;
                        row["finite_rgb"] = true;
                        row["alpha_padding_source_preserved"] = true;
                        row["released_both_images_once"] = true;
                        if (mm != 36.0 && width != 1920 && amount == 3.0)
                            require(highGain.emplace(width, height, mm).second, "repeated high-gain Full combination");
                    } catch (const std::exception& error) {
                        row["failure"] = error.what();
                    }
                    std::cout << test.id << ' ' << row.dump() << '\n';
                    rows.push_back(std::move(row));
                }
                full(test.width, test.height);
                change("CameraFilmFormatMm", {36.0});
                change("FilmGrainAmountEv", amountDefault);
                change("FilmGrainTexelPitchUm", sizeDefault);
                require(render(time) == reference, "restored 36 mm Film frame changed");
                const std::set expected{
                    std::tuple{3840, 96, 8.0},
                    std::tuple{3840, 96, 11.66},
                    std::tuple{4832, 96, 8.0},
                    std::tuple{4832, 96, 11.66},
                    std::tuple{4832, 2016, 8.0},
                    std::tuple{4832, 2016, 11.66},
                };
                const bool passed = highGain == expected && std::ranges::all_of(rows, [](const auto& row) {
                                        return row.contains("ofx_status") && !row.contains("failure");
                                    });
                allPassed = allPassed && passed;
                routes.push_back({{"case_id", test.id}, {"route", test.routeName}, {"renders", std::move(rows)}, {"paused_36mm_returned", true}});
                require(lifecycle.destroy(effect) == kOfxStatOK, "preview destroy");
                device.release_images();
                continue;
            }
            if (timeEndpoint) {
                constexpr double below29 = 0.29 * 100.0;
                static_assert(below29 < 29.0);
                const auto clean = render(29.0);
                change("GrainModel", {std::string("film_v1")});
                change("GrainActive", {1});
                const auto frame29 = render(below29);
                require(render(29.0) == frame29, "Film words differ just below and at frame 29");
                require(frame29 != clean, "Film at frame 29 did not change grain-off pixels");
                const auto frame38 = render(37.99999999);
                require(render(38.0) == frame38, "Film words differ just below and at frame 38");
                const auto frame0 = render(-1e-8);
                require(render(0.0) == frame0, "Film words differ across the negative-to-zero endpoint");
                require(frame29 != frame38 && frame29 != frame0 && frame38 != frame0, "distinct Film frame seeds did not change pixels");
                require(render(below29) == frame29, "Film did not return bit-exactly to the earlier paused frame");

                struct Diagnostic {
                    std::string_view id;
                    std::string_view text;
                };
                const auto reject = [&](double time, const Diagnostic& expected) {
                    host.set_property(host.action_args(), kOfxPropTime, {time});
                    const auto sourceBefore = host.image_release_count(effect, "Source");
                    const auto outputBefore = host.image_release_count(effect, "Output");
                    host.clear_calls();
                    require(lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args()) == kOfxStatFailed && !host.callback_error(),
                            "Film admission did not return kOfxStatFailed");
                    require(host.image_release_count(effect, "Source") == sourceBefore + 1 &&
                                host.image_release_count(effect, "Output") == outputBefore + 1,
                            "Film admission failure did not release both images exactly once");
                    int messages = 0;
                    for (const auto& call : host.calls()) {
                        if (call.operation != "message")
                            continue;
                        ++messages;
                        require(call.property == kOfxMessageError && call.values.size() == 2 &&
                                    std::get<std::string>(call.values[0]) == expected.id &&
                                    std::get<std::string>(call.values[1]).starts_with(expected.text),
                                "Film admission reported the wrong diagnostic");
                    }
                    require(messages == 1, "expected exactly one Film admission diagnostic");
                };
                for (const double limit : {-1'000'000.0, 1'000'000.0}) {
                    (void)render(limit);
                    reject(std::nextafter(limit, limit * 2.0),
                           {"FilmJuicerFilmGrainTime", "Film v1 frame time must be finite and within -1000000 to 1000000 frames, inclusive."});
                    require(render(29.0) == frame29, "Film did not recover after time rejection");
                }
                const OfxRectD canonical{0.0, 0.0, static_cast<double>(test.width), static_cast<double>(test.height)};
                auto mismatch = canonical;
                ++mismatch.x2;
                host.set_clip_region(effect, "Source", mismatch);
                reject(29.0, {"FilmJuicerFilmGrainGeometry", "Film v1 requires complete square-pixel source and output images"});
                host.set_clip_region(effect, "Source", canonical);
                require(render(29.0) == frame29, "Film did not recover after geometry rejection");
                routes.push_back({{"case_id", test.id}, {"route", test.routeName}, {"equal_endpoint_words_at", {29, 38, 0}}, {"film_changes_output", true}, {"distinct_seeds_change_output", true}, {"paused_repeat", true}, {"accepted_time_limits", {-1'000'000, 1'000'000}}, {"time_and_geometry_recovered", true}});
                require(lifecycle.destroy(effect) == kOfxStatOK, "preview destroy");
                device.release_images();
                continue;
            }
            if (eightMmProxy || clippedQuarter) {
                source = flat_grain_source(test);
                change("CouplersActive", {0});
                change("GrainModel", {std::string("film_v1")});
                change("FilmGrainAmountEv", {0.0});
                change("FilmGrainChromaFraction", {1.0});
                change("FilmGrainTexelPitchUm", {eightMmProxy ? 4.0 : 1.0});
                if (eightMmProxy) {
                    change("CameraFilmFormatPreset", {0});
                    change("CameraFilmFormatMm", {8.0});
                }
                set_grain_geometry(host, effect, eightMmProxy ? kEightMmProxy : kClippedQuarter);
                change("GrainActive", {0});
                const auto clean = render(0.0);
                change("GrainActive", {1});
                const auto paused = render(0.0);
                require(paused != clean && render(0.0) == paused, "expected active grain and an exact paused repeat");
                if (eightMmProxy) {
                    set_grain_geometry(host, effect, kBelowMinimumSample);
                    reject_footprint("Film v1 unsupported sample footprint (film_long_edge_mm=8, grain_size_um=4, footprint_texels=0.1220703125, supported_texels=0.244140625..=240000) (rendered=32x16, action_scale=");
                    set_grain_geometry(host, effect, kEightMmProxy);
                    require(render(0.0) == paused, "minimum-width proxy did not recover bit-exactly");
                }
                if (clippedQuarter) {
                    set_grain_geometry(host, effect, kOffsetQuarter);
                    change("GrainActive", {0});
                    require(render(0.0) == clean, "clipped and square Quarter clean RGB words differ");
                    change("GrainActive", {1});
                    require(render(0.0) == paused, "clipped and square Quarter grained RGB words differ");

                    device.release_images();
                    renderTest.width = 960;
                    renderTest.height = 1;
                    source = flat_grain_source(renderTest);
                    const auto shiftedBytes = source.size() * sizeof(float);
                    require_cuda(cudaMalloc(&device.source, shiftedBytes), "allocate shifted source");
                    require_cuda(cudaMalloc(&device.destination, shiftedBytes), "allocate shifted destination");
                    configure(host, effect, renderTest, device);
                    set_grain_geometry(host, effect, kShiftedQuarter);
                    change("GrainActive", {0});
                    const auto shiftedClean = render(0.0);
                    change("GrainActive", {1});
                    const auto shiftedFilm = render(0.0);
                    // This pixel straddles the first 128-texel placement seam.
                    require(!std::equal(shiftedFilm.begin() + 12, shiftedFilm.begin() + 15, shiftedClean.begin() + 12),
                            "shifted seam interior did not receive Film grain");
                    set_grain_geometry(host, effect, kShiftedClipped);
                    require(render(0.0) == shiftedFilm, "equivalent shifted Film delivered words differ");
                    change("GrainActive", {0});
                    require(render(0.0) == shiftedClean, "equivalent shifted clean delivered words differ");
                    set_grain_geometry(host, effect, kShiftedQuarter);
                    change("GrainActive", {1});
                    require(render(0.0) == shiftedFilm, "Film did not return to the first shifted encoding");
                    std::cout << test.id << " equivalent shifted delivered words; active seam interior\n";
                }
                Json row{{"case_id", test.id}, {"route", test.routeName}, {"paused_repeat", true}};
                if (eightMmProxy)
                    row["sample_footprint_recovered"] = true;
                routes.push_back(std::move(row));
                require(lifecycle.destroy(effect) == kOfxStatOK, "preview destroy");
                device.release_images();
                continue;
            }
            if (proxy || smallerFormats || largerFormats || thumbnails) {
                auto resize = [&](int width, int height) {
                    device.release_images();
                    renderTest.width = width;
                    renderTest.height = height;
                    source = initial_source(renderTest);
                    const auto imageBytes = source.size() * sizeof(float);
                    require_cuda(cudaMalloc(&device.source, imageBytes), "allocate proxy source");
                    require_cuda(cudaMalloc(&device.destination, imageBytes), "allocate proxy destination");
                    configure(host, effect, renderTest, device);
                };
                auto geometry = [&](int divisor) {
                    resize(4832 / divisor, 2016 / divisor);
                    const double scale = 1.0 / divisor;
                    host.set_property(host.action_args(), kOfxImageEffectPropRenderScale, {scale, scale});
                    for (const auto* clip : {"Source", "Output"}) {
                        host.set_image_property(effect, clip, kOfxImageEffectPropRenderScale, {scale, scale});
                        host.set_clip_region(effect, clip, OfxRectD{0.0, 0.0, 4832.0, 2016.0});
                    }
                };
                change("GrainModel", {std::string("film_v1")});
                if (thumbnails) {
                    constexpr double time = 158.0;
                    Json rows = Json::array();
                    Json flags = Json::array();
                    Json mismatches = Json::array();
                    auto source_flag = [&](std::vector<InstanceValue> values) {
                        host.set_clip_property(effect, "Source", kOfxImageClipPropThumbnail, std::move(values));
                    };
                    auto output_flag = [&](std::vector<InstanceValue> values) {
                        host.set_clip_property(effect, "Output", kOfxImageClipPropThumbnail, std::move(values));
                    };
                    const auto reject = [&](OfxStatus expected, std::string_view id, std::string_view text) {
                        host.set_property(host.action_args(), kOfxPropTime, {time});
                        const auto sourceBefore = host.image_release_count(effect, "Source");
                        const auto outputBefore = host.image_release_count(effect, "Output");
                        host.clear_calls();
                        const auto status = lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args());
                        require(status == expected && !host.callback_error(),
                                std::string(id) + ": unexpected render status " + std::to_string(status));
                        require(host.image_release_count(effect, "Source") == sourceBefore + 1 &&
                                    host.image_release_count(effect, "Output") == outputBefore + 1,
                                "rejected thumbnail witness render did not release both images exactly once");
                        int messages = 0;
                        for (const auto& call : host.calls()) {
                            if (call.operation != "message")
                                continue;
                            ++messages;
                            require(call.property == kOfxMessageError && call.values.size() == 2 &&
                                        std::get<std::string>(call.values[0]) == id &&
                                        std::get<std::string>(call.values[1]).starts_with(text),
                                    "unexpected thumbnail witness rejection diagnostic");
                        }
                        require(messages == 1, "expected one thumbnail witness rejection diagnostic");
                    };
                    for (const auto& [width, height] : std::array{std::pair{92, 46}, std::pair{151, 63}, std::pair{100, 50}}) {
                        const std::vector<InstanceValue> scale{width / 4832.0, height / 2016.0};
                        auto resized = [&] {
                            resize(width, height);
                            host.set_property(host.action_args(), kOfxImageEffectPropRenderScale, scale);
                            for (const auto* clip : {"Source", "Output"})
                                host.set_clip_region(effect, clip, OfxRectD{0.0, 0.0, 4832.0, 2016.0});
                        };
                        resized();
                        source_flag({});
                        output_flag({});
                        change("GrainActive", {0});
                        change("FilmDustAmount", {0.0});
                        change("FilmScratchAmount", {0.0});
                        const auto clean = render(time);
                        change("FilmDustAmount", {10.0});
                        change("FilmScratchAmount", {10.0});
                        // Use Current at the physical image scale as the processed reference.
                        change("GrainModel", {std::string("current")});
                        for (const auto* clip : {"Source", "Output"})
                            host.set_image_property(effect, clip, kOfxImageEffectPropRenderScale, scale);
                        const auto artifacts = render(time);
                        require(artifacts != clean, "dust and scratches did not change the processed reference");
                        change("GrainModel", {std::string("film_v1")});
                        for (const auto* clip : {"Source", "Output"})
                            host.set_image_property(effect, clip, kOfxImageEffectPropRenderScale, {1.0, 1.0});
                        require(render(time) == artifacts, "unit image scales changed grain-off artifact geometry");
                        change("GrainActive", {1});
                        const auto grained = render(time);
                        require(grained != artifacts, "absent thumbnail flag suppressed Film");
                        if (width == 92) {
                            struct FlagCase {
                                const char* id;
                                std::vector<InstanceValue> source;
                                std::vector<InstanceValue> output;
                                bool thumbnail = false;
                            };
                            // Empty storage erases the property. Only one or two
                            // scalar ones without any explicit veto suppress Film.
                            const std::array flagCases{
                                FlagCase{"absent-absent", {}, {}},
                                FlagCase{"source-one", {1}, {}, true},
                                FlagCase{"output-one", {}, {1}, true},
                                FlagCase{"both-one", {1}, {1}, true},
                                FlagCase{"source-zero", {0}, {}},
                                FlagCase{"output-zero", {}, {0}},
                                FlagCase{"both-zero", {0}, {0}},
                                FlagCase{"source-one-output-zero", {1}, {0}},
                                FlagCase{"source-zero-output-one", {0}, {1}},
                                FlagCase{"source-two", {2}, {}},
                                FlagCase{"output-two", {}, {2}},
                                FlagCase{"source-two-output-one", {2}, {1}},
                                FlagCase{"source-one-output-two", {1}, {2}},
                                FlagCase{"source-vector", {1, 0}, {}},
                                FlagCase{"output-vector", {}, {1, 0}},
                                FlagCase{"source-vector-output-one", {1, 0}, {1}},
                                FlagCase{"source-one-output-vector", {1}, {1, 0}},
                                FlagCase{"source-string", {std::string("1")}, {}},
                                FlagCase{"output-string", {}, {std::string("1")}},
                                FlagCase{"source-string-output-one", {std::string("1")}, {1}},
                                FlagCase{"source-one-output-string", {1}, {std::string("1")}},
                            };
                            for (const auto& entry : flagCases) {
                                source_flag(entry.source);
                                output_flag(entry.output);
                                require(render(time) == (entry.thumbnail ? artifacts : grained),
                                        std::string("wrong thumbnail flag pixels: ") + entry.id);
                                flags.push_back(Json::array({entry.id, entry.thumbnail ? "artifact-reference" : "timeline-film"}));
                            }

                            source_flag({1});
                            output_flag({1});
                            struct FlagFault {
                                const char* operation;
                                OfxStatus status;
                            };
                            for (const auto& [operation, status] : std::array{
                                     FlagFault{"dimension", kOfxStatFailed},
                                     FlagFault{"dimension", kOfxStatErrUnsupported},
                                     FlagFault{"get_int", kOfxStatErrUnknown},
                                     FlagFault{"get_int", kOfxStatErrUnsupported},
                                 }) {
                                const auto property = std::string(operation) + "/" + kOfxImageClipPropThumbnail;
                                auto check_fault = [&](std::string_view clip) {
                                    const auto words = render(time);
                                    const auto hits = std::count_if(host.calls().begin(), host.calls().end(), [&](const auto& call) {
                                        return call.target == clip && call.property == kOfxImageClipPropThumbnail &&
                                               call.operation == operation && call.status == status;
                                    });
                                    require(hits == 1, "thumbnail property fault did not reach its intended clip and operation");
                                    require(words == grained, "thumbnail property fault suppressed Film");
                                    flags.push_back(Json::array({clip, operation, status, "timeline-film"}));
                                };
                                host.fail_next(property, status);
                                check_fault("Source");
                                host.fail_on_nth(2, property, status);
                                check_fault("Output");
                            }
                            require(render(time) == artifacts, "valid flags did not recover after the property faults");

                            // Invalid but still captured as enabled Film; geometry
                            // and stock are unchanged so they cannot mask rejection.
                            change("FilmGrainAmountEv", {4.0});
                            const auto reject_film_controls = [&] {
                                reject(kOfxStatErrFatal, "FilmJuicerRenderAdmission", "Film v1 requires Kodak Portra 400, a supported camera film format, a negative route and valid Film grain controls");
                            };
                            reject_film_controls();
                            source_flag({});
                            output_flag({});
                            reject_film_controls();
                            change("FilmGrainAmountEv", amountDefault);
                            require(render(time) == grained, "restored Film controls did not recover timeline words");
                            source_flag({1});
                            require(render(time) == artifacts, "restored Film controls did not recover thumbnail words");
                            source_flag({});
                            change("FilmGrainAmountEv", {1.0});
                            require(render(time) != grained, "valid Film amount edit did not change delivered pixels");
                            change("FilmGrainAmountEv", amountDefault);
                            require(render(time) == grained, "timeline did not return to the original Film amount words");
                        }
                        source_flag({1});
                        output_flag({});
                        require(render(time) == artifacts, "Source thumbnail did not match grain-off effects");
                        source_flag({});
                        geometry(8);
                        const auto timeline = render(time);
                        resized();
                        output_flag({1});
                        require(render(time) == artifacts, "Output thumbnail did not match grain-off effects");
                        geometry(8);
                        output_flag({});
                        require(render(time) == timeline, "timeline Film changed after a flagged thumbnail");
                        rows.push_back({{"width", width}, {"height", height}, {"artifacts_change_pixels", true}, {"film_differs_from_artifacts", true}, {"processed_thumbnail_matches", true}, {"timeline_returned", true}});
                    }
                    for (const auto& [rasterWidth, mappedWidth] : std::array{std::pair{100, 101}, std::pair{101, 100}}) {
                        resize(rasterWidth, 50);
                        for (const auto* clip : {"Source", "Output"})
                            host.set_clip_region(effect, clip, OfxRectD{0.0, 0.0, 4832.0, 2016.0});
                        host.set_property(host.action_args(), kOfxImageEffectPropRenderScale, {mappedWidth / 4832.0, 50.0 / 2016.0});
                        reject(kOfxStatFailed, "FilmJuicerFilmGrainGeometry", "Film v1 requires complete square-pixel source and output images");
                        mismatches.push_back(Json::array({rasterWidth, mappedWidth, kOfxStatFailed}));
                    }
                    geometry(8);
                    change("GrainActive", {0});
                    const auto off = render(time);
                    change("GrainModel", {std::string("current")});
                    change("GrainActive", {1});
                    const auto current = render(time);
                    require(current != off, "Current grain did not change grain-off effects pixels");
                    for (const auto* clip : {"Source", "Output"})
                        host.set_image_property(effect, clip, kOfxImageEffectPropRenderScale, {1.0, 1.0});
                    const auto currentUnit = render(time);
                    require(currentUnit != current, "Current did not use its source image scale");
                    source_flag({1});
                    output_flag({1});
                    require(render(time) == currentUnit, "thumbnail flags changed pre-resized Current grain or effects");
                    geometry(8);
                    require(render(time) == current, "Current did not recover when the source image scale returned");
                    routes.push_back({{"case_id", test.id}, {"route", test.routeName}, {"renders", std::move(rows)}, {"flag_raster", {92, 46}}, {"flag_checks", std::move(flags)}, {"property_faults_recovered", true}, {"film_controls_rejected", {{"thumbnail", kOfxStatErrFatal}, {"timeline", kOfxStatErrFatal}, {"diagnostic", "FilmJuicerRenderAdmission"}, {"recovered", true}}}, {"film_amount_edit_changed_words_and_returned", true}, {"one_pixel_mismatches", std::move(mismatches)}, {"current_image_scale_preserved", true}, {"images_released_once", true}});
                    require(lifecycle.destroy(effect) == kOfxStatOK, "preview destroy");
                    device.release_images();
                    continue;
                }
                if (smallerFormats || largerFormats) {
                    Json rows = Json::array();
                    auto select_format = [&](int preset, double mm) {
                        if (preset == 0)
                            change("CameraFilmFormatMm", {mm});
                        else
                            change("CameraFilmFormatPreset", {preset});
                        require(host.value(effect, "CameraFilmFormatPreset") == std::vector<InstanceValue>{preset} &&
                                    host.value(effect, "CameraFilmFormatMm") == std::vector<InstanceValue>{mm},
                                "effective film format differs from the host edit");
                    };
                    const double frameTime = largerFormats ? 158.0 : 37.0;
                    auto film_frame = [&](double mm, double size) {
                        change("FilmGrainTexelPitchUm", {size});
                        lastEdit += " for " + std::to_string(mm) + " mm at Size " + std::to_string(size);
                        auto frame = render(frameTime);
                        const auto footprint = (mm * 1000.0) / std::max(renderTest.width, renderTest.height) / size;
                        rows.push_back({{"mm", mm}, {"size_um", size}, {"width", renderTest.width}, {"height", renderTest.height}, {"expected_footprint", footprint}, {"ofx_status", kOfxStatOK}});
                        return frame;
                    };
                    change("GrainActive", {1});
                    if (largerFormats) {
                        // Resolve project-open geometry as recorded at time 158.
                        constexpr double scaleX = 92.0 / 4832.0;
                        constexpr double scaleY = 46.0 / 2016.0;
                        host.set_property(host.action_args(), kOfxImageEffectPropRenderScale, {scaleX, scaleY});
                        for (const auto* clip : {"Source", "Output"}) {
                            host.set_image_property(effect, clip, kOfxImageEffectPropRenderScale, {1.0, 1.0});
                            host.set_clip_region(effect, clip, OfxRectD{0.0, 0.0, 4832.0, 2016.0});
                        }
                        for (const auto& [preset, mm] : std::array{std::pair{2, 36.0}, std::pair{5, 120.0}}) {
                            select_format(preset, mm);
                            for (double size : {0.5, 1.0, 4.0})
                                (void)film_frame(mm, size);
                        }
                        const auto thumbnail = render(158.0);
                        host.set_image_property(effect, "Output", kOfxImageEffectPropRenderScale, {scaleX, scaleY});
                        host.clear_calls();
                        require(lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args()) == kOfxStatFailed,
                                "thumbnail with inconsistent image scales was not rejected");
                        host.set_image_property(effect, "Output", kOfxImageEffectPropRenderScale, {1.0, 1.0});
                        require(render(158.0) == thumbnail, "thumbnail did not recover after scale rejection");

                        geometry(1);
                        for (const auto& [preset, mm] : std::array{std::pair{9, 37.72}, std::pair{0, 48.5}, std::pair{3, 56.0}, std::pair{4, 70.0}, std::pair{10, 70.4}, std::pair{5, 120.0}}) {
                            select_format(preset, mm);
                            (void)film_frame(mm, 1.0);
                        }
                        change("CouplersActive", {0});
                        change("GlareActive", {0});
                        const auto native = film_frame(120.0, 0.5);
                        set_grain_geometry(host, effect, {{0, 0, 9664, 4032}, {0, 0, 4832, 2016}, {0.5, 0.5}});
                        require(render(frameTime) != native, "equal rendered pitch reused native Full grain on the proxy");
                        set_grain_geometry(host, effect, {{0, 0, 4832, 2016}, {0, 0, 4832, 2016}, {1.0, 1.0}});
                        require(render(frameTime) == native, "120 mm native Full did not return bit-exactly after the proxy");
                        std::cout << test.id << " 120 mm equal-pitch Full/proxy/Full passed\n";
                        change("CouplersActive", {1});
                        change("GlareActive", {1});
                        (void)film_frame(120.0, 4.0);
                        geometry(2);
                        (void)film_frame(120.0, 0.5);
                        geometry(8);
                        (void)film_frame(120.0, 0.5);
                        routes.push_back({{"case_id", test.id}, {"route", test.routeName}, {"renders", std::move(rows)}, {"thumbnail_recovered", true}});
                        require(lifecycle.destroy(effect) == kOfxStatOK, "preview destroy");
                        device.release_images();
                        continue;
                    }
                    select_format(2, 36.0);
                    const auto reference = film_frame(36.0, 1.0);
                    select_format(6, 11.66);
                    require(film_frame(11.66, 1.0) != reference, "the format edit did not change the delivered image");
                    select_format(2, 36.0);
                    require(film_frame(36.0, 1.0) == reference, "36 mm output changed after paused format edits");
                    for (const auto& [preset, mm] : std::array{std::pair{2, 36.0}, std::pair{0, 35.0}, std::pair{8, 24.89}, std::pair{6, 11.66}}) {
                        select_format(preset, mm);
                        for (const double size : {0.5, 1.0, 4.0})
                            if (size != 1.0 || (preset != 2 && preset != 6))
                                (void)film_frame(mm, size);
                    }
                    select_format(0, 8.0);
                    const auto smaller = film_frame(8.0, 4.0);
                    change("GrainActive", {0});
                    const auto off = render(37.0);
                    require(smaller != off, "8 mm Film v1 did not change delivered pixels");
                    change("GrainModel", {std::string("current")});
                    change("GrainActive", {1});
                    require(render(37.0) != off, "8 mm Current did not change delivered pixels");

                    resize(8192, 16);
                    change("GrainModel", {std::string("film_v1")});
                    select_format(0, 8.0);
                    change("FilmGrainTexelPitchUm", {4.0});
                    reject_footprint("Film v1 unsupported rendered footprint (film_long_edge_mm=8, grain_size_um=4, footprint_texels=0.244140625, supported_texels=0.25..=240000) (rendered=8192x16, action_scale=");

                    geometry(2);
                    select_format(6, 11.66);
                    (void)film_frame(11.66, 4.0);
                    routes.push_back({{"case_id", test.id}, {"route", test.routeName}, {"renders", std::move(rows)}, {"paused_36mm_returned", true}, {"current_and_off_at_8mm", true}, {"footprint_recovered", true}});
                    require(lifecycle.destroy(effect) == kOfxStatOK, "preview destroy");
                    device.release_images();
                    continue;
                }
                for (int divisor : {1, 2, 4, 8}) {
                    geometry(divisor);
                    change("GrainActive", {0});
                    const auto off = render(37.0);
                    change("GrainActive", {1});
                    for (double size : {0.5, 1.0, 4.0}) {
                        change("FilmGrainTexelPitchUm", {size});
                        require(render(37.0) != off, "proxy Film v1 did not change delivered pixels");
                    }
                }
                change("FilmGrainTexelPitchUm", {0.5});
                geometry(1);
                const auto full = render(37.0);
                geometry(4);
                const auto paused = render(37.0);
                require(render(38.0) != paused, "wide proxy grain did not advance on a static source");
                require(render(37.0) == paused, "wide proxy did not return bit-exactly to the paused frame");
                geometry(1);
                require(render(37.0) == full, "Full output changed after Quarter");
                host.set_property(host.action_args(), kOfxImageEffectPropRenderScale, {0.5, 0.5});
                host.clear_calls();
                require(lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args()) == kOfxStatFailed,
                        "inconsistent action scale was not rejected");
                const auto& calls = host.calls();
                require(std::any_of(calls.begin(), calls.end(), [](const auto& call) {
                            return call.operation == "message" && !call.values.empty() &&
                                   call.values.front() == InstanceValue{std::string("FilmJuicerFilmGrainGeometry")};
                        }),
                        "inconsistent action scale did not report geometry error");
                host.set_property(host.action_args(), kOfxImageEffectPropRenderScale, {1.0, 1.0});
                require(render(37.0) == full, "valid frame did not recover after geometry error");
                routes.push_back({{"case_id", test.id}, {"route", test.routeName}, {"proxy_scales", {1.0, 0.5, 0.25, 0.125}}, {"sizes_um", {0.5, 1.0, 4.0}}, {"wide_proxy_advanced_and_returned", true}, {"full_returned", true}, {"geometry_recovered", true}});
                require(lifecycle.destroy(effect) == kOfxStatOK, "preview destroy");
                device.release_images();
                continue;
            }
            const auto off = render(37.0);
            change("GrainActive", {1});
            if (test.height > test.width) {
                change("GrainModel", {std::string("film_v1")});
                const auto film = render(37.0);
                require(film != off, "portrait Film v1 did not change delivered pixels");
                change("FilmGrainTexelPitchUm", {4.0});
                const auto maxSize = render(37.0);
                require(maxSize != film && maxSize != off, "portrait maximum Size did not change delivered pixels");
                routes.push_back({{"case_id", test.id}, {"route", test.routeName}, {"width", test.width}, {"height", test.height}, {"film_changes_output", true}, {"max_size_changes_output", true}});
                require(lifecycle.destroy(effect) == kOfxStatOK, "preview destroy");
                device.release_images();
                continue;
            }
            const auto current = render(37.0);
            require(current != off, "Current grain did not change delivered pixels");
            change("GrainDebugView", {1});
            const auto diagnostic = render(37.0);
            require(diagnostic != current, "Current debug view did not change delivered pixels");
            change("GrainModel", {std::string("film_v1")});
            const auto film = render(37.0);
            require(film != off, "Film v1 did not change delivered pixels");
            change("FilmGrainChromaFraction", {1.0});
            require(render(37.0) == film, "explicit Colour 1 changed default Film output");
            for (const auto& [name, values, reset] : std::array{
                     std::tuple{"FilmGrainAmountEv", std::array{-3.0, 1.25, 3.0}, amountDefault},
                     std::tuple{"FilmGrainTexelPitchUm", std::array{0.5, 1.5, 4.0}, sizeDefault},
                     std::tuple{"FilmGrainChromaFraction", std::array{0.0, 0.5, 0.2406350076198578}, colourDefault},
                 }) {
                for (double value : values) {
                    change(name, {value});
                    lastEdit += "=" + std::to_string(value);
                    const auto edited = render(37.0);
                    require(edited != off && edited != film, "Film slider did not change the delivered image");
                }
                change(name, reset);
                require(render(37.0) == film, "descriptor-default reset did not restore Film output");
            }
            change("FilmGrainAmountEv", {3.0});
            for (double size : {0.5, 4.0}) {
                change("FilmGrainTexelPitchUm", {size});
                for (double colour : {0.0, 1.0}) {
                    change("FilmGrainChromaFraction", {colour});
                    require(render(37.5) != off, "combined Film slider extrema produced grain-off output");
                }
            }
            change("FilmGrainAmountEv", amountDefault);
            change("FilmGrainTexelPitchUm", sizeDefault);
            change("FilmGrainChromaFraction", colourDefault);
            require(render(37.0) == film, "paused Film frame changed");
            const auto later = render(38.0);
            require(later != film, "Film frame seed did not change");
            require(render(37.0) == film, "out-of-order Film frame changed");
            const auto fractional = render(37.5);
            require(fractional != film && fractional != later, "fractional Film frame did not change");
            change("GrainModel", {std::string("current")});
            require(render(37.0) == diagnostic, "saved Current debug view changed after Film");
            change("GrainDebugView", {0});
            require(render(37.0) == current, "Current output changed after Film");
            change("GrainActive", {0});
            require(render(37.0) == off, "grain-off output changed after model edits");
            routes.push_back({{"case_id", test.id}, {"route", test.routeName}, {"width", test.width}, {"height", test.height}, {"film_changes_output", true}, {"paused_repeat", true}, {"out_of_order_repeat", true}, {"fractional_distinct", true}, {"current_changes_output", true}, {"off_restored", true}});
            require(lifecycle.destroy(effect) == kOfxStatOK, "preview destroy");
            device.release_images();
        }
        device.close();
        require(lifecycle.unload() == kOfxStatOK && !host.callback_error(), "preview unload");
        module.close();
        restore.restore();
        std::set<std::string> compared;
        for (const auto& row : routes)
            require(compared.insert(row.at("case_id").get<std::string>()).second, "duplicate Film v1 witness case");
        require(compared.size() == 23, "expected 23 completed Film v1 witness cases");
        std::cout << "CTEST_FULL_OUTPUT\n";
        std::cout << Json{{"schema", "film-v1-witness-v1"}, {"routes", routes}, {"context_restored", true}}.dump() << '\n';
        return allPassed && std::cout ? 0 : 1;
    }


} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 7 && std::string_view(argv[1]) == "--frame-and-print-edits") {
            return frame_and_print_edits({argv[2], argv[3], argv[4], argv[5], argv[6]});
        }
        if (argc >= 3 && std::string_view(argv[1]) == "--film-v1-witness") {
            require(argc == 3, "preview witness arguments");
            return film_v1_witness(argv[2]);
        }
        const bool summary = argc == 6 && std::string_view(argv[5]) == "--summary";
        require(argc == 5 || summary, "usage: JuicerProductionPreparedRenderProbe module Resources ordinary-${preset}.json preset [--summary]");
        const std::filesystem::path modulePath(argv[1]);
        const auto resources = modulePath.parent_path().parent_path() / "Resources";
        require(std::filesystem::is_directory(resources), "module Resources directory unavailable");
        require(std::filesystem::path(argv[2]).lexically_normal() == resources.lexically_normal(), "ordinary smoke requires the staged module Resources");
        const auto expected = expected_pixels(argv[3], argv[4]);
        const CudaDriver driver;
        driver.check(driver.init(0), "initialize CUDA driver");
        CUcontext original = nullptr;
        driver.check(driver.get_current(&original), "read original CUDA context");
        ContextRestore restore(driver, original);
        require_cuda(cudaSetDevice(0), "select CUDA device");
        Module module(modulePath);
        using Count = int (*)();
        using Get = OfxPlugin* (*)(int);
        const auto count = module.symbol<Count>("OfxGetNumberOfPlugins");
        const auto get = module.symbol<Get>("OfxGetPlugin");
        require(count() == 1 && get(-1) == nullptr && get(1) == nullptr, "production discovery membership");
        auto* plugin = get(0);
        require(plugin && plugin->pluginApi && std::string_view(plugin->pluginApi) == kOfxImageEffectPluginApi &&
                    plugin->pluginIdentifier && std::string_view(plugin->pluginIdentifier) == "com.juicer.Juicer" &&
                    plugin->apiVersion == 1 && plugin->pluginVersionMajor == 1 && plugin->pluginVersionMinor == 0 &&
                    plugin->setHost && plugin->mainEntry && get(0) == plugin,
                "production descriptor contract");
        InstanceHost host;
        Lifecycle lifecycle(plugin, host);
        const OfxStatus load = lifecycle.load();
        require(load == kOfxStatOK, "production module load failed");
        require(lifecycle.call(kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK,
                "production context description failed");
        DeviceBuffers device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create production stream");
        Json rows = Json::array(), destroyStatuses = Json::array();
        int effectsCreated = 0, effectsDestroyed = 0;
        for (std::size_t index = 0; index < kOrdinaryCases.size(); ++index) {
            const auto& test = kOrdinaryCases[index];
            auto* effect = host.create_effect();
            author_parameters(host, effect, test);
            host.set_value(effect, "GlareActive", {0});
            if (index >= 2) {
                host.set_value(effect, "FilmProfileKey", {std::string("fujifilm_provia_100f")});
            }
            const auto create = lifecycle.call(kOfxActionCreateInstance, effect);
            require(create == kOfxStatOK, "production instance creation failed");
            lifecycle.created(effect);
            ++effectsCreated;
            if (index == 0) {
                require(host.value(effect, "GrainModel") == std::vector<InstanceValue>{std::string("current")}, "grain model default");
                require(host.value(effect, "FilmGrainAmountEv") == std::vector<InstanceValue>{0.0} &&
                            host.value(effect, "FilmGrainTexelPitchUm") == std::vector<InstanceValue>{1.0} &&
                            host.value(effect, "FilmGrainChromaFraction") == std::vector<InstanceValue>{1.0},
                        "Film v1 control defaults");
                host.set_value(effect, "GrainDebugView", {3});
                host.set_value(effect, "GrainModel", {std::string("film_v1")});
                auto* args = host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "GrainModel"});
                require(lifecycle.call(kOfxActionInstanceChanged, effect, args) == kOfxStatReplyDefault, "switch to Film v1");
                host.set_value(effect, "GrainModel", {std::string("current")});
                args = host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, "GrainModel"});
                require(lifecycle.call(kOfxActionInstanceChanged, effect, args) == kOfxStatReplyDefault, "switch to Current");
                require(host.value(effect, "GrainDebugView") == std::vector<InstanceValue>{3}, "Current debug view retained");
                host.set_value(effect, "GrainDebugView", {0});
            }
            const auto source = initial_source(test);
            const auto bytes = source.size() * sizeof(float);
            require_cuda(cudaMalloc(&device.source, bytes), "allocate source buffer");
            require_cuda(cudaMalloc(&device.destination, bytes), "allocate destination buffer");
            configure(host, effect, test, device);
            std::vector<std::uint32_t> coldWords;
            for (const auto* pass : {"cold", "warm"}) {
                rows.push_back(run_pass(driver, lifecycle, host, effect, test, pass, {.source = source, .expected = &expected[index]}, device, coldWords, summary));
            }
            const auto destroy = lifecycle.destroy(effect);
            destroyStatuses.push_back(destroy);
            require(destroy == kOfxStatOK, "production instance destruction failed");
            ++effectsDestroyed;
            device.release_images();
        }
        device.close();
        const OfxStatus unload = lifecycle.unload();
        require(unload == kOfxStatOK && !host.callback_error(), "production module unload action failed");
        module.close();
        restore.restore();
        require(restore.restored && effectsCreated == effectsDestroyed, "production render lifecycle did not complete");
        require(rows.size() == kCaseIds.size(), "incomplete production prepared-render comparisons");
        for (std::size_t index = 0; index < kCaseIds.size(); ++index)
            require(rows.at(index).at("case_id").get_ref<const std::string&>() == kCaseIds[index], "unexpected production prepared-render case");
        for (auto& row : rows)
            row["context_restored"] = restore.restored;
        Json report{{"rows", std::move(rows)}, {"lifecycle", {{"module_load_status", load}, {"effects_created", effectsCreated}, {"effects_destroyed", effectsDestroyed}, {"destroy_statuses", std::move(destroyStatuses)}, {"module_unload_status", unload}, {"context_restored", restore.restored}}}};
        const auto output = report.dump();
        if (std::fwrite(output.data(), 1, output.size(), stdout) != output.size() || std::fputc('\n', stdout) == EOF || std::fflush(stdout) != 0)
            return 1;
        return 0;
    } catch (const std::exception& error) {
        (void)std::fputs(error.what(), stderr);
        (void)std::fputc('\n', stderr);
        return 1;
    } catch (...) {
        (void)std::fputs("unknown production OFX render failure\n", stderr);
        return 1;
    }
}
