#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <span>
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
#include "production_images.h"
#include "case_results.h"

#if defined(JUICER_GRAIN_MOTION_REPLAY)
#include <optional>
#ifndef _WIN32
#include <cerrno>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "motion-replay.h"
#include "Cuda/JuicerCudaFilmGrainStage.h"
#include "juicer_ofx_api.h"
#include "film_oracle.h"
#endif

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
    constexpr std::array<std::string_view, 4> kCaseIds{
        "negative-direct/cold", "negative-direct/warm", "negative-print/cold", "negative-print/warm"};
    constexpr std::array kGrainPreviewCases{
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
    };
    struct GrainGeometry {
        OfxRectD canonical{};
        OfxRectI bounds{};
        std::array<double, 2> scale{1.0, 1.0};
        std::array<double, 2> imageScale{1.0, 1.0};
    };
    constexpr GrainGeometry kEightMmProxy{{0, 0, 8192, 4096}, {0, 0, 32, 16}, {1.0 / 256, 1.0 / 256}};
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
    constexpr std::array<Case, 8> kHalationCases{{{"negative-direct-ae-off", "negative-direct", "NegativeDirectScan", 0, 4, 1920, 1080},
                                                  {"negative-direct-ae-on", "negative-direct", "NegativeDirectScan", 0, 4, 1920, 1080},
                                                  {"negative-print-ae-off", "negative-print", "NegativePrintScan", 0, 4, 1920, 1080},
                                                  {"negative-print-ae-on", "negative-print", "NegativePrintScan", 0, 4, 1920, 1080},
                                                  {"positive-direct-ae-off", "positive-direct", "NegativeDirectScan", 0, 4, 1920, 1080},
                                                  {"positive-direct-ae-on", "positive-direct", "NegativeDirectScan", 0, 4, 1920, 1080},
                                                  {"positive-print-ae-off", "positive-print", "NegativePrintScan", 0, 4, 1920, 1080},
                                                  {"positive-print-ae-on", "positive-print", "NegativePrintScan", 0, 4, 1920, 1080}}};
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

    void write_pixels(const std::filesystem::path& path, const std::vector<float>& pixels) {
        static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);
        require(!std::filesystem::exists(path), "capture already exists");
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size() * sizeof(float)));
        stream.close();
        require(static_cast<bool>(stream), "FP32 capture write failed");
    }

    Json parameter_values(InstanceHost& host, OfxImageEffectHandle effect, const std::vector<std::string>& names) {
        Json result = Json::object();
        for (const auto& name : names) {
            Json values = Json::array();
            for (const auto& value : host.value(effect, name)) {
                std::visit([&](const auto& item) {
                    using T = std::decay_t<decltype(item)>;
                    if constexpr (std::is_same_v<T, void*>)
                        throw std::runtime_error("unexpected pointer parameter");
                    else
                        values.push_back(item);
                },
                           value);
            }
            if (!values.empty())
                result[name] = std::move(values);
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

    std::array<std::vector<float>, 2> expected_pixels(const std::filesystem::path& path, const std::string& preset) {
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
        std::array<std::vector<float>, 2> result;
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
            if (route < result.size()) {
                result[route] = words_floats(output);
            }
        }
        return result;
    }

    bool compare_fixture(const std::vector<float>& actual, const std::vector<float>& expected) {
        if (actual.size() != expected.size()) {
            return false;
        }
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const float tolerance = 2.0e-4f + 3.0e-4f * std::abs(expected[i]);
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) || std::abs(actual[i] - expected[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }

    struct PassImages {
        const std::vector<float>& source;
        const std::vector<float>* expected = nullptr;
        std::filesystem::path capture;
    };
    Json run_pass(const CudaDriver& driver, Lifecycle& lifecycle, InstanceHost& host, OfxImageEffectHandle effect, const Case& test, std::string_view pass, const PassImages& images, DeviceBuffers& device, std::vector<std::uint32_t>& coldWords, bool summary) {
        const auto& [source, expected, capture] = images;
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
        const OfxStatus status = lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args());
        driver.check(driver.get_current(&after), "inspect context after render");
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
        const Json fixtureMatch = expected ? Json(compare_fixture(activeOutput, *expected)) : Json(nullptr);
        require(!expected || fixtureMatch == true, "rendered pixels differ from accepted fixture");
        std::vector<std::uint32_t> outputWords;
        outputWords.reserve(activeOutput.size());
        for (float value : activeOutput)
            outputWords.push_back(std::bit_cast<std::uint32_t>(value));
        if (pass == "cold")
            coldWords = outputWords;
        else
            require(coldWords == outputWords, "warm output differs from cold output");
        Json row{{"case_id", std::string(test.id) + "/" + std::string(pass)}, {"route", test.routeName}, {"pass", std::string(pass)}, {"ofx_status", status}, {"messages", std::move(messages)}, {"input_shape", Json::array({test.height, test.width, test.components})}, {"output_shape", Json::array({test.height, test.width, test.components})}, {"accepted_fixture_match", fixtureMatch}, {"finite_output", finite}, {"alpha_preserved", alpha}, {"padding_preserved", padding}, {"source_preserved", true}, {"stream_completed", true}, {"readback_complete", true}, {"context_preserved", before == after}, {"context_restored", false}, {"source_image_releases", sourceReleases}, {"output_image_releases", outputReleases}, {"samples_compared", activeOutput.size()}};
        if (!capture.empty()) {
            row["cold_warm_identical"] = true;
            if (pass == "cold")
                write_pixels(capture / (std::string(test.id) + ".f32"), activeOutput);
        }
        if (!summary) {
            row["input_f32_words"] = words_json(activeInput);
            row["output_f32_words"] = words_json(activeOutput);
        }
        return row;
    }

    const OfxPlugin* check_discovery(const Module& module, const std::filesystem::path& path, const char* expected) {
        require(module.loaded_from(path), "discovery loaded path differs from declared module");
        const auto count = module.symbol<int (*)()>("OfxGetNumberOfPlugins");
        const auto get = module.symbol<OfxPlugin* (*)(int)>("OfxGetPlugin");
        require(count() == 1 && !get(-1) && !get(1) && !get(std::numeric_limits<int>::min()) &&
                    !get(std::numeric_limits<int>::max()),
                "discovery plugin count or index bounds");
        const auto* plugin = get(0);
        require(plugin && plugin == get(0) && plugin->pluginIdentifier &&
                    std::string_view(plugin->pluginIdentifier) == expected && plugin->setHost && plugin->mainEntry,
                std::string("discovery identity or entry points: ") + expected);
        return plugin;
    }

    int scanner_capacity(const std::filesystem::path& path, const std::filesystem::path& output) {
        constexpr Case test{"capacity", "negative-direct", "NegativeDirectScan", 0, 3, 12800, 96, 0, 0.0};
        constexpr GrainGeometry geometry{{0, 0, test.width, test.height}, {0, 0, test.width, test.height}};
        require(path.is_absolute() && path.filename() == "juicer-grain-preview.ofx" &&
                    std::filesystem::is_directory(output) && std::filesystem::is_empty(output),
                "capacity requires the absolute Preview module and an empty output directory");
        const auto resources = std::filesystem::canonical(path.parent_path().parent_path() / "Resources");
        require(std::filesystem::is_regular_file(resources / "grain/portra-400/manifest.json"), "capacity grain bank unavailable");
        const CudaDriver driver;
        driver.check(driver.init(0), "initialize CUDA driver");
        CUcontext original = nullptr;
        driver.check(driver.get_current(&original), "read initial CUDA context");
        ContextRestore restore(driver, original);
        require_cuda(cudaSetDevice(0), "select capacity CUDA device");
        Module module(path);
        auto* plugin = const_cast<OfxPlugin*>(check_discovery(module, path, "com.juicer.Juicer.GrainPreview"));
        using TokenSetter = void (*)(std::uint64_t, std::uint64_t, std::uint64_t);
        require(!module.find_symbol<TokenSetter>("fj_preview_replay_set_tokens") &&
                    !module.find_symbol<TokenSetter>("fj_preview_benchmark_set_tokens"),
                "capacity requires natural product seeds");
        InstanceHost host;
        DeviceBuffers device;
        Lifecycle lifecycle(plugin, host);
        require(lifecycle.load() == kOfxStatOK, "capacity Load failed");
        require(lifecycle.call(kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK,
                "capacity DescribeInContext failed");
        auto* effect = host.create_effect();
        author_parameters(host, effect, test);
        host.set_value(effect, "GlareActive", {0});
        host.set_value(effect, "CameraFilmFormatPreset", {2});
        host.set_value(effect, "CameraFilmFormatMm", {36.0});
        require(lifecycle.call(kOfxActionCreateInstance, effect) == kOfxStatOK, "capacity CreateInstance failed");
        lifecycle.created(effect);

        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create capacity CUDA stream");
        const auto source = flat_grain_source(test);
        const auto bytes = source.size() * sizeof(float);
        require_cuda(cudaMalloc(&device.source, bytes), "allocate capacity source");
        require_cuda(cudaMalloc(&device.destination, bytes), "allocate capacity output");
        configure(host, effect, test, device);
        host.set_effect_property(effect, kOfxImageEffectPropFrameRate, {24.0});
        set_grain_geometry(host, effect, geometry);
        std::vector<std::string> controls{"InputCctfDecoding", "HalationActive", "GlareActive"};
        const auto change = [&](const char* name, const std::vector<InstanceValue>& value) {
            host.set_value(effect, name, value);
            require(lifecycle.call(kOfxActionInstanceChanged, effect, host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, name})) == kOfxStatReplyDefault &&
                        host.value(effect, name) == value,
                    std::string("capacity control readback differs: ") + name);
            controls.emplace_back(name);
        };
        change("CouplersActive", {0});
        change("GrainModel", {std::string("film_v1")});
        change("FilmGrainAmountEv", {3.0});
        change("FilmGrainChromaFraction", {1.0});
        change("FilmGrainTexelPitchUm", {0.5});
        change("CameraFilmFormatPreset", {0});
        change("CameraFilmFormatMm", {8.0});
        change("InputColorSpace", {0});
        change("OutputCctfEncoding", {1});
        change("OutputGamutCompression", {0});
        change("GrainActive", {1});

        const auto activeControls = parameter_values(host, effect, controls);
        write_pixels(output / "source-pitched.f32", source);
        Json rows = Json::array();
        const auto render = [&](std::string_view phase, double time) {
            host.set_property(host.action_args(), kOfxPropTime, {time});
            std::vector<std::uint32_t> result;
            auto row = run_pass(driver, lifecycle, host, effect, test, "cold", {source, nullptr, {}}, device, result, true);
            std::vector<float> pixels;
            pixels.reserve(result.size());
            for (const auto word : result) {
                const auto pixel = std::bit_cast<float>(word);
                require(pixel != kDestinationCanary, "capacity output has an unwritten RGB word");
                pixels.push_back(pixel);
            }
            const auto filename = std::string(phase) + ".f32";
            write_pixels(output / filename, pixels);
            row["pass"] = phase;
            row["time_bits"] = std::bit_cast<std::uint64_t>(time);
            row["output_file"] = filename;
            rows.push_back(std::move(row));
            return result;
        };
        const auto measured = render("measured", 0.0);
        const auto reporter = render("reporting", 1.0);
        require(reporter != measured, "capacity Film v1 temporal frames match");
        require(lifecycle.destroy(effect) == kOfxStatOK, "capacity DestroyInstance failed");
        device.release_images();
        device.close();
        require(lifecycle.unload() == kOfxStatOK && !host.callback_error(), "capacity Unload failed");
        module.close();
        restore.restore();
        const Json report{{"schema", "juicer-scanner-capacity-driver-v1"},
                          {"module", path.string()},
                          {"resources", resources.string()},
                          {"controls", activeControls},
                          {"frame_rate", 24.0},
                          {"row_bytes", (test.width * test.components + kPitchPadding) * sizeof(float)},
                          {"bounds", {0, 0, test.width, test.height}},
                          {"context_restored", restore.restored},
                          {"unloaded", true},
                          {"rows", rows}};
        std::ofstream receipt(output / "receipt.json");
        receipt << report.dump(2) << '\n';
        receipt.close();
        require(static_cast<bool>(receipt), "capacity receipt write failed");
        return 0;
    }

    struct SpectralReplayPaths {
        std::filesystem::path module;
        std::filesystem::path output;
        std::string_view variant;
        std::string_view grain;
        std::string_view route;
        std::filesystem::path draw;
    };

    int spectral_seed_pilot(const SpectralReplayPaths& paths) {
        const auto& [path, output, variant, grain, route, draw] = paths;
        const bool preview = variant == "preview";
        const bool filmV1 = grain == "film-v1";
        const bool print = route == "negative-print";
        require((preview || variant == "ordinary") && (grain == "current" || (preview && filmV1)) &&
                    (print || route == "negative-direct"),
                "invalid spectral replay variant, grain model or route");
        require(path.is_absolute() && std::filesystem::is_directory(output) && std::filesystem::is_empty(output),
                "spectral replay requires an absolute module and an empty output directory");
        constexpr int width = 64;
        constexpr int height = 32;
        const char* routeName = print ? "negative-print" : "negative-direct";
        const Case test{routeName, routeName, print ? "NegativePrintScan" : "NegativeDirectScan", 0, 4, width, height, 0, 0.0};
        const GrainGeometry geometry{{0, 0, width, height}, {0, 0, width, height}};
        const auto resources = std::filesystem::canonical(path.parent_path().parent_path() / "Resources");
        const CudaDriver driver;
        driver.check(driver.init(0), "initialize replay CUDA driver");
        CUcontext original = nullptr;
        driver.check(driver.get_current(&original), "read initial replay context");
        ContextRestore restore(driver, original);
        require_cuda(cudaSetDevice(0), "select replay CUDA device");
        Module module(path);
        const char* identifier = preview ? "com.juicer.Juicer.GrainPreview" : "com.juicer.Juicer";
        auto* plugin = const_cast<OfxPlugin*>(check_discovery(module, path, identifier));
        using TokenSetter = void (*)(std::uint64_t, std::uint64_t, std::uint64_t);
        const auto setTokens = module.find_symbol<TokenSetter>("fj_spectral_replay_set_tokens");
        require(draw.empty() == (setTokens == nullptr), "spectral replay module has the wrong token export surface");
        require(!module.find_symbol<TokenSetter>("fj_preview_replay_set_tokens") &&
                    !module.find_symbol<TokenSetter>("fj_preview_benchmark_set_tokens"),
                "unexpected historical replay token export");
        Json tokens = nullptr;
        if (setTokens) {
            std::ifstream input(draw);
            require(static_cast<bool>(input), "natural spectral replay draw unavailable");
            const auto capture = Json::parse(input);
            require(capture.at("variant").get_ref<const std::string&>() == variant &&
                        capture.at("route").get_ref<const std::string&>() == route &&
                        capture.at("grain").get_ref<const std::string&>() == grain,
                    "natural spectral replay draw is bound to another workload");
            tokens = capture.at("tokens");
            require(tokens.is_array() && tokens.size() == 3 &&
                        std::all_of(tokens.begin(), tokens.end(), [](const Json& value) {
                            return value.is_number_unsigned();
                        }),
                    "natural spectral replay draw requires three unsigned tokens");
            setTokens(tokens.at(0).get<std::uint64_t>(), tokens.at(1).get<std::uint64_t>(), tokens.at(2).get<std::uint64_t>());
        }

        InstanceHost host;
        DeviceBuffers device;
        Lifecycle lifecycle(plugin, host);
        require(lifecycle.load() == kOfxStatOK, "spectral replay Load failed");
        require(lifecycle.call(kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK,
                "spectral replay DescribeInContext failed");
        std::set<std::string, std::less<>> declared;
        for (const auto& call : host.calls()) {
            if (call.operation == "param_define")
                declared.insert(call.property);
        }
        std::vector<std::string> names;
        for (const auto& definition : ofx_test::kInstanceParameters) {
            if (definition.kind != ofx_test::InstanceParamKind::PushButton && declared.contains(definition.name))
                names.emplace_back(definition.name);
        }
        for (const char* name : {"GrainModel", "FilmGrainAmountEv", "FilmGrainTexelPitchUm", "FilmGrainChromaFraction"})
            if (declared.contains(name))
                names.emplace_back(name);
        auto* effect = host.create_effect();
        author_parameters(host, effect, test);
        host.set_value(effect, "GlareActive", {0});
        host.set_value(effect, "CouplersActive", {0});
        host.set_value(effect, "CameraFilmFormatPreset", {2});
        host.set_value(effect, "CameraFilmFormatMm", {36.0});
        require(lifecycle.call(kOfxActionCreateInstance, effect) == kOfxStatOK, "spectral replay CreateInstance failed");
        lifecycle.created(effect);
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create spectral replay stream");
        const auto source = flat_grain_source(test);
        const auto bytes = source.size() * sizeof(float);
        require_cuda(cudaMalloc(&device.source, bytes), "allocate spectral replay source");
        require_cuda(cudaMalloc(&device.destination, bytes), "allocate spectral replay destination");
        configure(host, effect, test, device);
        host.set_effect_property(effect, kOfxImageEffectPropFrameRate, {24.0});
        set_grain_geometry(host, effect, geometry);
        const auto change = [&](const char* name, const std::vector<InstanceValue>& value) {
            host.set_value(effect, name, value);
            require(lifecycle.call(kOfxActionInstanceChanged, effect, host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, name})) ==
                            kOfxStatReplyDefault &&
                        host.value(effect, name) == value,
                    std::string("spectral replay control readback differs: ") + name);
        };
        if (preview)
            change("GrainModel", {std::string(filmV1 ? "film_v1" : "current")});
        if (filmV1) {
            change("FilmGrainAmountEv", {0.0});
            change("FilmGrainTexelPitchUm", {4.0});
            change("FilmGrainChromaFraction", {1.0});
            change("CameraFilmFormatPreset", {0});
            change("CameraFilmFormatMm", {36.0});
        }
        write_pixels(output / "source-pitched.f32", source);
        Json rows = Json::array();
        const auto render = [&](std::string_view phase, double time) {
            host.set_property(host.action_args(), kOfxPropTime, {time});
            const auto controls = parameter_values(host, effect, names);
            std::vector<std::uint32_t> words;
            auto row = run_pass(driver, lifecycle, host, effect, test, "cold", {source, nullptr, {}}, device, words, true);
            for (const auto& call : host.calls()) {
                if (call.operation == "param_get" || call.operation == "param_get_time") {
                    require(call.status == kOfxStatOK && controls.contains(call.target) &&
                                call.values == host.value(effect, call.target),
                            "spectral replay OFX control readback differs");
                }
            }
            std::vector<float> pixels;
            pixels.reserve(words.size());
            for (const auto word : words) {
                const auto pixel = std::bit_cast<float>(word);
                require(pixel != kDestinationCanary, "spectral replay has an unwritten output word");
                pixels.push_back(pixel);
            }
            const auto filename = std::string(phase) + ".f32";
            write_pixels(output / filename, pixels);
            row["pass"] = phase;
            row["time_bits"] = std::bit_cast<std::uint64_t>(time);
            row["output_file"] = filename;
            row["controls"] = controls;
            rows.push_back(std::move(row));
            return words;
        };
        const auto clean = render("grain-off", 0.0);
        change("GrainActive", {1});
        const auto integral = render("integral", 0.0);
        const auto fractional = render("fractional", 0.5);
        require(clean != integral && integral != fractional, "spectral replay grain or fractional time did not change output");
        (void)render("reporting", 1.0);
        require(lifecycle.destroy(effect) == kOfxStatOK, "spectral replay DestroyInstance failed");
        device.release_images();
        device.close();
        require(lifecycle.unload() == kOfxStatOK && !host.callback_error(), "spectral replay Unload failed");
        module.close();
        restore.restore();
        const Json receipt{{"schema", "juicer-ofx-spectral-seed-replay-v1"},
                           {"module", path.string()},
                           {"resources", resources.string()},
                           {"variant", variant},
                           {"grain", grain},
                           {"route", route},
                           {"plugin_id", identifier},
                           {"replay_export", setTokens != nullptr},
                           {"replay_tokens", tokens},
                           {"components", test.components},
                           {"canonical_rod", {0, 0, width, height}},
                           {"bounds", {0, 0, width, height}},
                           {"row_bytes", (width * test.components + kPitchPadding) * sizeof(float)},
                           {"context_restored", restore.restored},
                           {"grain_changes_output", true},
                           {"fraction_changes_output", true},
                           {"rows", rows}};
        std::ofstream stream(output / "receipt.json");
        stream << receipt.dump(2) << '\n';
        stream.close();
        require(static_cast<bool>(stream), "spectral replay receipt write failed");
        return 0;
    }

    int grain_preview_discovery(const std::filesystem::path& previewPath, const std::filesystem::path& ordinaryPath) {
        const bool paired = !ordinaryPath.empty();
        const char* caller = paired ? "grain-preview-paired-discovery" : "grain-preview-discovery";
        const char* id = paired ? "ordinary-and-preview" : "preview-module";
        JuicerTests::CaseLog cases;
        cases.membership(caller, {id});
        {
            JuicerTests::CaseEvent event(caller, id);
            require(previewPath.is_absolute() && previewPath.filename() == "juicer-grain-preview.ofx", "preview discovery module path");
            Module preview(previewPath);
            if (paired) {
                require(ordinaryPath.is_absolute() && ordinaryPath.filename() == "juicer.ofx", "ordinary discovery module path");
                Module ordinary(ordinaryPath);
                const auto* preview_plugin = check_discovery(preview, previewPath, "com.juicer.Juicer.GrainPreview");
                require(preview_plugin != check_discovery(ordinary, ordinaryPath, "com.juicer.Juicer"), "paired discovery descriptors alias");
                ordinary.close();
            } else {
                check_discovery(preview, previewPath, "com.juicer.Juicer.GrainPreview");
            }
            preview.close();
            event.passed();
        }
        cases.finish();
        return 0;
    }

    int grain_preview_witness(const std::filesystem::path& path) {
        JuicerTests::CaseLog cases;
        std::vector<std::string> ids;
        ids.reserve(kGrainPreviewCases.size());
        for (const auto& test : kGrainPreviewCases)
            ids.emplace_back(test.id);
        cases.membership("grain-preview-render", ids);
        require(path.filename() == "juicer-grain-preview.ofx", "expected preview module");
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
                    std::string_view(get(0)->pluginIdentifier) == "com.juicer.Juicer.GrainPreview",
                "preview discovery identity");
        InstanceHost host;
        Lifecycle lifecycle(get(0), host);
        require(lifecycle.load() == kOfxStatOK, "preview load");
        require(lifecycle.call(kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK,
                "preview context description");
        DeviceBuffers device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create preview stream");
        Json routes = Json::array();
        for (const auto& test : kGrainPreviewCases) {
            JuicerTests::CaseEvent event("grain-preview-render", test.id);
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
            if (proxy || smallerFormats || thumbnails || eightMmProxy || clippedQuarter || timeEndpoint)
                host.set_value(effect, "GlareActive", {0});
            host.set_value(effect, "CameraFilmFormatPreset", {2});
            host.set_value(effect, "CameraFilmFormatMm", {36.0});
            require(lifecycle.call(kOfxActionCreateInstance, effect) == kOfxStatOK, "preview create");
            lifecycle.created(effect);
            Case renderTest = test;
            auto source = initial_source(renderTest);
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
                event.passed();
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
                routes.push_back({{"case_id", test.id}, {"route", test.routeName}, {"paused_repeat", true}});
                require(lifecycle.destroy(effect) == kOfxStatOK, "preview destroy");
                device.release_images();
                event.passed();
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
                        // The ordinary image-scale path supplies an independent
                        // processed reference for the same physical artifact grid.
                        for (const auto* clip : {"Source", "Output"})
                            host.set_image_property(effect, clip, kOfxImageEffectPropRenderScale, scale);
                        const auto artifacts = render(time);
                        require(artifacts != clean, "dust and scratches did not change the processed reference");
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
                    require(render(time) == current, "unit image scales changed Current grain or effects");
                    source_flag({1});
                    output_flag({1});
                    require(render(time) == current, "thumbnail flags changed pre-resized Current grain or effects");
                    routes.push_back({{"case_id", test.id}, {"route", test.routeName}, {"renders", std::move(rows)}, {"flag_raster", {92, 46}}, {"flag_checks", std::move(flags)}, {"property_faults_recovered", true}, {"film_controls_rejected", {{"thumbnail", kOfxStatErrFatal}, {"timeline", kOfxStatErrFatal}, {"diagnostic", "FilmJuicerRenderAdmission"}, {"recovered", true}}}, {"film_amount_edit_changed_words_and_returned", true}, {"one_pixel_mismatches", std::move(mismatches)}, {"current_pre_resized_preserved", true}, {"images_released_once", true}});
                    require(lifecycle.destroy(effect) == kOfxStatOK, "preview destroy");
                    device.release_images();
                    event.passed();
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
                        event.passed();
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
                    host.clear_calls();
                    const auto sourceBefore = host.image_release_count(effect, "Source");
                    const auto outputBefore = host.image_release_count(effect, "Output");
                    require(lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args()) == kOfxStatErrFatal,
                            "unsupported footprint was not rejected");
                    require(host.image_release_count(effect, "Source") == sourceBefore + 1 &&
                                host.image_release_count(effect, "Output") == outputBefore + 1,
                            "unsupported footprint did not release both images");
                    require(std::any_of(host.calls().begin(), host.calls().end(), [](const auto& call) {
                                if (call.operation != "message" || call.values.size() != 2 ||
                                    call.values[0] != InstanceValue{std::string("FilmJuicerRenderAdmission")})
                                    return false;
                                const auto& message = std::get<std::string>(call.values[1]);
                                return message.find("film_long_edge_mm=8") != std::string::npos &&
                                       message.find("grain_size_um=4") != std::string::npos &&
                                       message.find("footprint_texels=0.244") != std::string::npos &&
                                       message.find("rendered=8192x16") != std::string::npos;
                            }),
                            "unsupported footprint diagnostic omitted physical geometry");

                    geometry(2);
                    select_format(6, 11.66);
                    (void)film_frame(11.66, 4.0);
                    routes.push_back({{"case_id", test.id}, {"route", test.routeName}, {"renders", std::move(rows)}, {"paused_36mm_returned", true}, {"current_and_off_at_8mm", true}, {"footprint_recovered", true}});
                    require(lifecycle.destroy(effect) == kOfxStatOK, "preview destroy");
                    device.release_images();
                    event.passed();
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
                event.passed();
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
                event.passed();
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
            event.passed();
        }
        device.close();
        require(lifecycle.unload() == kOfxStatOK && !host.callback_error(), "preview unload");
        module.close();
        restore.restore();
        cases.finish();
        // Keep the verified subcases of both routes in the staged CTest receipt.
        std::cout << "CTEST_FULL_OUTPUT\n";
        std::cout << Json{{"schema", "grain-preview-witness-v1"}, {"routes", routes}, {"context_restored", true}}.dump() << '\n';
        return std::cout ? 0 : 1;
    }

    const char* replay_type(ofx_test::InstanceParamKind kind) {
        using Kind = ofx_test::InstanceParamKind;
        switch (kind) {
            case Kind::Double:
                return kOfxParamTypeDouble;
            case Kind::Boolean:
                return kOfxParamTypeBoolean;
            case Kind::Choice:
                return kOfxParamTypeChoice;
            case Kind::StringChoice:
                return kOfxParamTypeStrChoice;
            case Kind::Integer:
                return kOfxParamTypeInteger;
            case Kind::Double2D:
                return kOfxParamTypeDouble2D;
            case Kind::Double3D:
                return kOfxParamTypeDouble3D;
            case Kind::PushButton:
                return kOfxParamTypePushButton;
        }
        throw std::runtime_error("unknown replay parameter kind");
    }

    bool advanced_default(std::string_view name) {
        constexpr std::array names{"HalationScatterCoreSigmaUm", "HalationScatterTailScaleUm", "HalationScatterTailWeight", "HalationReflectionSource", "HalationStrengthRgb", "HalationFirstBounceSigmaUm", "HalationBounceCount", "HalationBounceDecay", "HalationRenormalize", "HalationHighlightBoostEv", "HalationHighlightProtectionEv", "HalationHighlightBoostRange"};
        return std::find(names.begin(), names.end(), name) != names.end();
    }

    std::vector<InstanceValue> replay_values(const Json& parameter, ofx_test::InstanceParamKind kind) {
        using Kind = ofx_test::InstanceParamKind;
        const auto& values = parameter.at("values");
        const std::size_t count = kind == Kind::Double3D ? 3 : kind == Kind::Double2D ? 2
                                                                                      : 1;
        require(values.is_array() && values.size() == count && kind != Kind::PushButton, "invalid replay parameter arity");
        std::vector<InstanceValue> result;
        for (std::size_t index = 0; index < values.size(); ++index) {
            const auto& value = values[index];
            if (kind == Kind::StringChoice) {
                require(value.is_string(), "invalid replay string parameter");
                result.emplace_back(value.get<std::string>());
            } else if (kind == Kind::Double || kind == Kind::Double2D || kind == Kind::Double3D) {
                const auto& bits = parameter.at("f64_bits");
                require(bits.is_array() && bits.size() == count && bits[index].is_string(), "missing exact replay double bits");
                const auto text = bits[index].get<std::string>();
                require(text.size() == 16 && text.find_first_not_of("0123456789abcdefABCDEF") == std::string::npos, "invalid replay double bits");
                const double number = std::bit_cast<double>(static_cast<std::uint64_t>(std::stoull(text, nullptr, 16)));
                result.emplace_back(number);
            } else {
                require(value.is_number_integer(), "invalid replay integer parameter");
                result.emplace_back(value.get<int>());
            }
        }
        return result;
    }

    template <class T, std::size_t N>
    std::array<T, N> replay_array(const Json& value) {
        require(value.is_array() && value.size() == N, "invalid replay array");
        return value.get<std::array<T, N>>();
    }

    template <class T, std::size_t N>
    std::vector<InstanceValue> replay_properties(const Json& value) {
        const auto array = replay_array<T, N>(value);
        return std::vector<InstanceValue>(array.begin(), array.end());
    }

    struct ReplayImage {
        std::array<int, 4> bounds{};
        int rowBytes = 0;
        int components = 0;
        std::size_t rowSize = 0;
        std::size_t storageSize = 0;
        std::size_t originOffset = 0;
    };

    ReplayImage replay_image(InstanceHost& host, OfxImageEffectHandle effect, const char* clip, const Json& image) {
        ReplayImage result;
        result.bounds = replay_array<int, 4>(image.at("bounds"));
        const auto width = static_cast<std::int64_t>(result.bounds[2]) - result.bounds[0];
        const auto height = static_cast<std::int64_t>(result.bounds[3]) - result.bounds[1];
        require(width > 0 && height > 0, "invalid replay image bounds");
        result.rowBytes = image.at("row_bytes").get<int>();
        const auto components = image.at("components").get<std::string>();
        result.components = components == kOfxImageComponentRGB ? 3 : components == kOfxImageComponentRGBA ? 4
                                                                                                           : 0;
        require(result.components != 0 && image.at("depth") == kOfxBitDepthFloat, "unsupported replay image format");
        const auto pitch = result.rowBytes < 0 ? -static_cast<std::int64_t>(result.rowBytes) : result.rowBytes;
        const auto rowSize = width * result.components * static_cast<std::int64_t>(sizeof(float));
        require(pitch >= rowSize && pitch <= std::numeric_limits<int>::max() &&
                    static_cast<std::uint64_t>(height) <= std::numeric_limits<std::size_t>::max() / static_cast<std::uint64_t>(pitch),
                "invalid replay image pitch");
        result.rowSize = static_cast<std::size_t>(rowSize);
        result.storageSize = static_cast<std::size_t>(height * pitch);
        result.originOffset = result.rowBytes < 0 ? static_cast<std::size_t>((height - 1) * pitch) : 0;
        host.set_image_property(effect, clip, kOfxImagePropBounds, replay_properties<int, 4>(image.at("bounds")));
        host.set_image_property(effect, clip, kOfxImagePropRegionOfDefinition, replay_properties<int, 4>(image.at("rod")));
        host.set_image_property(effect, clip, kOfxImagePropRowBytes, {result.rowBytes});
        host.set_image_property(effect, clip, kOfxImageEffectPropComponents, {components});
        host.set_image_property(effect, clip, kOfxImageEffectPropPixelDepth, {image.at("depth").get<std::string>()});
        host.set_image_property(effect, clip, kOfxImageEffectPropPreMultiplication, {image.at("premultiplication").get<std::string>()});
        host.set_image_property(effect, clip, kOfxImagePropPixelAspectRatio, {image.at("pixel_aspect").get<double>()});
        host.set_image_property(effect, clip, kOfxImageEffectPropRenderScale, replay_properties<double, 2>(image.at("scale")));
        host.set_image_property(effect, clip, kOfxImagePropField, image.at("field").is_null() ? std::vector<InstanceValue>{} : std::vector<InstanceValue>{image.at("field").get<std::string>()});
        host.set_image_property(effect, clip, kOfxImagePropUniqueIdentifier, image.at("unique_id").is_null() ? std::vector<InstanceValue>{} : std::vector<InstanceValue>{image.at("unique_id").get<std::string>()});
        return result;
    }

    std::vector<unsigned char> read_replay_source(const std::filesystem::path& path, const ReplayImage& image) {
        const auto height = static_cast<std::size_t>(image.bounds[3] - image.bounds[1]);
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        require(static_cast<bool>(input) && input.tellg() == static_cast<std::streamoff>(height * image.rowSize), "replay source size mismatch");
        input.seekg(0);
        std::vector<unsigned char> storage(image.storageSize, 0);
        for (std::size_t y = 0; y < height; ++y) {
            const auto offset = static_cast<std::ptrdiff_t>(image.originOffset) + static_cast<std::ptrdiff_t>(y) * image.rowBytes;
            input.read(reinterpret_cast<char*>(storage.data() + offset), static_cast<std::streamsize>(image.rowSize));
        }
        require(static_cast<bool>(input), "replay source read failed");
        return storage;
    }

    void write_replay_output(const std::filesystem::path& path, const ReplayImage& image, const std::vector<unsigned char>& storage) {
        require(!std::filesystem::exists(path), "replay output already exists");
        std::ofstream output(path, std::ios::binary);
        const auto height = static_cast<std::size_t>(image.bounds[3] - image.bounds[1]);
        for (std::size_t y = 0; y < height; ++y) {
            const auto offset = static_cast<std::ptrdiff_t>(image.originOffset) + static_cast<std::ptrdiff_t>(y) * image.rowBytes;
            output.write(reinterpret_cast<const char*>(storage.data() + offset), static_cast<std::streamsize>(image.rowSize));
        }
        output.close();
        require(static_cast<bool>(output), "replay output write failed");
    }

    struct ReplayPaths {
        std::filesystem::path module;
        std::filesystem::path capture;
        std::filesystem::path output;
        std::filesystem::path parameters;
    };

    int frame_replay(const ReplayPaths& paths) {
        require(std::filesystem::is_directory(paths.output) && std::filesystem::is_empty(paths.output), "replay output directory must exist and be empty");
        std::ifstream input(paths.capture / "capture.json");
        require(static_cast<bool>(input), "replay capture manifest unavailable");
        const Json capture = Json::parse(input);
        require(capture.at("schema") == "juicer-ofx-capture-v1" && capture.at("complete") == true &&
                    capture.at("status") == kOfxStatOK,
                "unsupported or incomplete replay capture");
        require(std::filesystem::is_directory(paths.module.parent_path().parent_path() / "Resources"), "module Resources directory unavailable");
        const auto sourceName = capture.at("source_file").get<std::string>();
        require(std::filesystem::path(sourceName).filename() == std::filesystem::path(sourceName), "invalid replay source filename");

        const CudaDriver driver;
        driver.check(driver.init(0), "initialize CUDA driver");
        CUcontext original = nullptr;
        driver.check(driver.get_current(&original), "read original CUDA context");
        ContextRestore restore(driver, original);
        require_cuda(cudaSetDevice(0), "select CUDA device");
        Module module(paths.module);
        using Count = int (*)();
        using Get = OfxPlugin* (*)(int);
        const auto count = module.symbol<Count>("OfxGetNumberOfPlugins");
        const auto get = module.symbol<Get>("OfxGetPlugin");
        require(count() == 1 && get(0) != nullptr, "replay plugin discovery failed");
        auto* plugin = get(0);
        require(plugin->setHost && plugin->mainEntry && plugin->pluginIdentifier &&
                    std::string_view(plugin->pluginIdentifier) ==
                        (std::filesystem::path(paths.module).filename() == "juicer-grain-preview.ofx" ? "com.juicer.Juicer.GrainPreview" : "com.juicer.Juicer"),
                "unexpected replay plugin");
        InstanceHost host;
        Lifecycle lifecycle(plugin, host);
        require(lifecycle.load() == kOfxStatOK, "replay module load failed");
        require(lifecycle.call(kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK,
                "replay context description failed");
        std::set<std::string> declared;
        for (const auto& call : host.calls()) {
            if (call.operation == "param_define" && std::any_of(ofx_test::kInstanceParameters.begin(), ofx_test::kInstanceParameters.end(), [&](const auto& definition) {
                    return definition.name == call.property;
                }))
                declared.insert(call.property);
        }
        auto* effect = host.create_effect();
        Json parameters = capture.at("parameters");
        require(parameters.is_object(), "replay parameters must be an object");
        Json overrides = Json::object();
        Json originalValues = Json::object();
        Json ignoredOverrides = Json::array();
        if (!paths.parameters.empty()) {
            std::ifstream overrideInput(paths.parameters);
            require(static_cast<bool>(overrideInput), "replay parameter overrides unavailable");
            overrides = Json::parse(overrideInput);
            require(overrides.is_object() && !overrides.empty(), "replay parameter overrides must be a nonempty object");
            for (auto it = overrides.begin(); it != overrides.end(); ++it) {
                const auto& name = it.key();
                require(parameters.contains(name) && (declared.contains(name) || advanced_default(name)), "override parameter is absent from capture or module");
                if (!declared.contains(name))
                    ignoredOverrides.push_back(name);
                const auto* const definition = std::find_if(ofx_test::kInstanceParameters.data(), ofx_test::kInstanceParameters.data() + ofx_test::kInstanceParameters.size(), [&](const auto& item) {
                    return item.name == name;
                });
                require(definition != ofx_test::kInstanceParameters.data() + ofx_test::kInstanceParameters.size() &&
                            it.value().at("type") == replay_type(definition->kind) &&
                            it.value().at("type") == parameters.at(name).at("type") &&
                            it.value().at("status") == kOfxStatOK,
                        "invalid replay parameter override type or status");
                (void)replay_values(it.value(), definition->kind);
                originalValues[name] = parameters.at(name);
                parameters[name] = it.value();
            }
        }
        std::set<std::string> supplied;
        Json defaults = Json::array();
        for (const auto& definition : ofx_test::kInstanceParameters) {
            const std::string name(definition.name);
            if (!declared.contains(name))
                continue;
            if (!parameters.contains(name)) {
                if (advanced_default(name))
                    defaults.push_back(name);
                continue;
            }
            const auto& parameter = parameters.at(name);
            require(parameter.at("type") == replay_type(definition.kind), "invalid captured parameter type");
            if (parameter.at("status") != kOfxStatOK)
                continue;
            host.set_value(effect, name, replay_values(parameter, definition.kind));
            supplied.insert(name);
        }
        const auto& action = capture.at("action");
        auto* args = host.action_args();
        host.set_property(args, kOfxPropTime, {action.at("time").get<double>()});
        host.set_property(args, kOfxImageEffectPropRenderWindow, replay_properties<int, 4>(action.at("render_window")));
        host.set_property(args, kOfxImageEffectPropRenderScale, replay_properties<double, 2>(action.at("render_scale")));
        host.set_property(args, kOfxImageEffectPropFieldToRender, action.at("field").is_null() ? std::vector<InstanceValue>{} : std::vector<InstanceValue>{action.at("field").get<std::string>()});
        host.set_property(args, kOfxImageEffectPropCudaEnabled, {1});
        host.set_effect_property(effect, kOfxImageEffectPropFrameRate, action.at("frame_rate").is_null() ? std::vector<InstanceValue>{} : std::vector<InstanceValue>{action.at("frame_rate").get<double>()});
        if (!capture.at("source_canonical_rod").is_null()) {
            const auto rod = replay_array<double, 4>(capture.at("source_canonical_rod"));
            host.set_clip_region(effect, "Source", OfxRectD{rod[0], rod[1], rod[2], rod[3]});
        }
        const auto sourceImage = replay_image(host, effect, "Source", capture.at("source"));
        const auto outputImage = replay_image(host, effect, "Output", capture.at("output"));
        const auto source = read_replay_source(paths.capture / sourceName, sourceImage);
        DeviceBuffers device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create replay stream");
        require_cuda(cudaMalloc(&device.source, sourceImage.storageSize), "allocate replay source");
        require_cuda(cudaMalloc(&device.destination, outputImage.storageSize), "allocate replay output");
        const bool streamPresent = action.at("cuda_stream_present").get<bool>();
        host.set_property(args, kOfxImageEffectPropCudaStream, streamPresent ? std::vector<InstanceValue>{static_cast<void*>(device.stream)} : std::vector<InstanceValue>{});
        host.set_image_property(effect, "Source", kOfxImagePropData, {static_cast<void*>(reinterpret_cast<unsigned char*>(device.source) + sourceImage.originOffset)});
        host.set_image_property(effect, "Output", kOfxImagePropData, {static_cast<void*>(reinterpret_cast<unsigned char*>(device.destination) + outputImage.originOffset)});
        host.clear_calls();
        require(lifecycle.call(kOfxActionCreateInstance, effect) == kOfxStatOK, "replay instance creation failed");
        lifecycle.created(effect);
        Json rows = Json::array();
        for (const auto* pass : {"cold", "warm"}) {
            require_cuda(cudaMemcpyAsync(device.source, source.data(), source.size(), cudaMemcpyHostToDevice, device.stream), "upload replay source");
            require_cuda(cudaMemsetAsync(device.destination, 0, outputImage.storageSize, device.stream), "clear replay output");
            require_cuda(cudaStreamSynchronize(device.stream), "complete replay upload");
            const auto sourceBefore = host.image_release_count(effect, "Source");
            const auto outputBefore = host.image_release_count(effect, "Output");
            CUcontext before = nullptr, after = nullptr;
            driver.check(driver.get_current(&before), "inspect context before replay");
            const auto status = lifecycle.call(kOfxImageEffectActionRender, effect, args);
            driver.check(driver.get_current(&after), "inspect context after replay");
            require(status == kOfxStatOK && !host.callback_error(), "replay render failed");
            require(before == after, "replay changed current CUDA context");
            const auto sourceReleases = host.image_release_count(effect, "Source") - sourceBefore;
            const auto outputReleases = host.image_release_count(effect, "Output") - outputBefore;
            require(sourceReleases == 1 && outputReleases == 1, "replay did not release each image once");
            require_cuda(cudaStreamSynchronize(streamPresent ? device.stream : nullptr), "complete replay render");
            std::vector<unsigned char> output(outputImage.storageSize);
            require_cuda(cudaMemcpy(output.data(), device.destination, output.size(), cudaMemcpyDeviceToHost), "read replay output");
            std::vector<unsigned char> sourceAfter(source.size());
            require_cuda(cudaMemcpy(sourceAfter.data(), device.source, sourceAfter.size(), cudaMemcpyDeviceToHost), "read replay source");
            require(sourceAfter == source, "replay modified source image or padding");
            write_replay_output(paths.output / (std::string(pass) + ".f32"), outputImage, output);
            rows.push_back({{"pass", pass}, {"status", status}, {"output_file", std::string(pass) + ".f32"}, {"source_preserved", true}, {"context_preserved", true}, {"source_releases", sourceReleases}, {"output_releases", outputReleases}, {"stream_completed", true}});
        }
        Json reads = Json::array();
        for (const auto& call : host.calls()) {
            require(call.operation != "clip_get_region" || !capture.at("source_canonical_rod").is_null(),
                    "uncaptured required source canonical region");
            if (call.operation != "param_get" && call.operation != "param_get_time")
                continue;
            require(call.status == kOfxStatOK && (supplied.contains(call.target) || advanced_default(call.target)),
                    std::string("uncaptured required replay parameter: ") + call.target);
            if (supplied.contains(call.target)) {
                const auto* const definition = std::find_if(ofx_test::kInstanceParameters.data(), ofx_test::kInstanceParameters.data() + ofx_test::kInstanceParameters.size(), [&](const auto& item) {
                    return item.name == call.target;
                });
                require(definition != ofx_test::kInstanceParameters.data() + ofx_test::kInstanceParameters.size() &&
                            call.values == replay_values(parameters.at(call.target), definition->kind),
                        std::string("replay getter changed from captured snapshot: ") + call.target);
            }
            reads.push_back(call.target);
        }
        const auto resolved = parameter_values(host, effect, std::vector<std::string>(declared.begin(), declared.end()));
        require(lifecycle.destroy(effect) == kOfxStatOK, "replay instance destruction failed");
        device.release_images();
        device.close();
        require(lifecycle.unload() == kOfxStatOK, "replay module unload failed");
        module.close();
        restore.restore();
        Json report{{"schema", "juicer-ofx-replay-v1"}, {"rows", std::move(rows)}, {"reads", std::move(reads)}, {"advanced_defaults", std::move(defaults)}, {"parameters", resolved}, {"context_restored", restore.restored}};
        if (!paths.parameters.empty())
            report["parameter_overrides"] = {{"path", paths.parameters.string()}, {"values", std::move(overrides)}, {"original_values", std::move(originalValues)}, {"ignored_absent_advanced", std::move(ignoredOverrides)}};
        std::ofstream manifest(paths.output / "replay.json");
        manifest << report.dump(2) << '\n';
        manifest.close();
        require(static_cast<bool>(manifest), "replay report write failed");
        return 0;
    }
#if defined(JUICER_GRAIN_MOTION_REPLAY)
    using JuicerTests::GrainMotion::MotionDensity;
    using JuicerTests::GrainMotion::MotionDensityScope;
    using JuicerTests::GrainMotion::MotionPass;
    using JuicerTests::GrainMotion::MotionSeedPair;
    using JuicerTests::GrainMotion::project_motion_density;
#include "film-ofx-apertures.inc"
#ifndef _WIN32
#include "motion-replay.inc"
#endif
#endif
} // namespace

#if defined(JUICER_SPECTRAL_REPLAY_CAPTURE)
#if defined(_WIN32)
#define FJ_REPLAY_CAPTURE_EXPORT __declspec(dllexport)
#else
#define FJ_REPLAY_CAPTURE_EXPORT __attribute__((used, retain, visibility("default")))
#endif
// Untimed scanner replay calls this in the host driver while the producer is
// stopped at the selected density-plane launch. The staging is driver-owned.
extern "C" FJ_REPLAY_CAPTURE_EXPORT const std::uint32_t* fj_observe_replay_density(
    const float* cyan, const float* magenta, const float* yellow, std::size_t count, cudaStream_t stream, CUcontext expectedContext) noexcept {
    constexpr std::size_t samples = 64 * 32;
    thread_local std::array<std::uint32_t, samples * 3 + 1> words{};
    CUcontext context = nullptr;
    if (!cyan || !magenta || !yellow || !stream || count != samples ||
        cuCtxGetCurrent(&context) != CUDA_SUCCESS || context != expectedContext) {
        words[0] = static_cast<std::uint32_t>(cudaErrorInvalidValue);
        return words.data();
    }
    auto status = cudaStreamSynchronize(stream);
    const std::array planes{cyan, magenta, yellow};
    for (std::size_t lane = 0; lane < planes.size() && status == cudaSuccess; ++lane)
        status = cudaMemcpy(words.data() + 1 + lane * samples, planes[lane], samples * sizeof(float), cudaMemcpyDeviceToHost);
    words[0] = static_cast<std::uint32_t>(status);
    return words.data();
}
#undef FJ_REPLAY_CAPTURE_EXPORT
#endif

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::string_view(argv[1]) == "--spectral-seed-pilot") {
            require(argc == 7 || argc == 8,
                    "expected --spectral-seed-pilot module output-directory variant grain route [natural-draw]");
            return spectral_seed_pilot({argv[2], argv[3], argv[4], argv[5], argv[6], argc == 8 ? argv[7] : ""});
        }
        if (argc >= 2 && std::string_view(argv[1]) == "--scanner-capacity") {
            require(argc == 4, "expected --scanner-capacity preview-module output-directory");
            return scanner_capacity(argv[2], argv[3]);
        }
#if defined(JUICER_GRAIN_MOTION_REPLAY)
        if (argc >= 2 && std::string_view(argv[1]) == "--grain-preview-apertures") {
            if (JuicerTests::list_requested(argc, argv)) {
                for (const auto& witness : kOfxApertures)
                    JuicerTests::list_case(kApertureCaller, witness.id);
                return std::fflush(stdout) == 0 ? 0 : 1;
            }
            require(argc == 2, "aperture witness arguments");
            return grain_preview_apertures();
        }
#endif
#if defined(JUICER_GRAIN_MOTION_REPLAY) && !defined(_WIN32)
        if (argc == 3 && !std::string_view(argv[1]).starts_with("--"))
            return grain_motion_replay(argv[1], argv[2]);
#endif
#if defined(JUICER_GRAIN_MOTION_REPLAY) && defined(_WIN32)
        require(false, "expected --grain-preview-apertures");
#endif
        if (argc >= 3 && std::string_view(argv[1]) == "--grain-preview-witness") {
            if (JuicerTests::list_requested(argc, argv)) {
                for (const auto& test : kGrainPreviewCases)
                    JuicerTests::list_case("grain-preview-render", test.id);
                return std::fflush(stdout) == 0 ? 0 : 1;
            }
            require(argc == 3, "preview witness arguments");
            return grain_preview_witness(argv[2]);
        }
        if (argc >= 2 && (std::string_view(argv[1]) == "--grain-preview-discovery" ||
                          std::string_view(argv[1]) == "--grain-preview-paired-discovery")) {
            const bool paired = std::string_view(argv[1]) == "--grain-preview-paired-discovery";
            if (JuicerTests::list_requested(argc, argv)) {
                JuicerTests::list_case(paired ? "grain-preview-paired-discovery" : "grain-preview-discovery",
                                       paired ? "ordinary-and-preview" : "preview-module");
                return std::fflush(stdout) == 0 ? 0 : 1;
            }
            require(argc == 4, "preview discovery arguments");
            return grain_preview_discovery(argv[2], paired ? std::filesystem::path(argv[3]) : std::filesystem::path{});
        }
        if ((argc == 5 || argc == 7) && std::string_view(argv[1]) == "--frame-replay") {
            require(argc == 5 || std::string_view(argv[5]) == "--parameters", "invalid frame replay option");
            return frame_replay({.module = argv[2], .capture = argv[3], .output = argv[4], .parameters = argc == 7 ? argv[6] : ""});
        }
        if (JuicerTests::list_requested(argc, argv)) {
            for (const auto id : kCaseIds)
                JuicerTests::list_case("production-prepared-render", id);
            return std::fflush(stdout) == 0 ? 0 : 1;
        }
        const bool capture = argc == 4 && std::string_view(argv[1]) == "--halation-capture";
        const bool summary = capture || (argc == 6 && std::string_view(argv[5]) == "--summary");
        require(argc == 5 || summary, "usage: JuicerProductionPreparedRenderProbe module Resources ordinary-${preset}.json preset [--summary]; or --halation-capture module output-directory");
        const auto selected = capture ? std::span<const Case>(kHalationCases)
                                      : std::span<const Case>(kOrdinaryCases).first(2);
        const std::string caller = capture ? "production-halation-capture" : "production-prepared-render";
        JuicerTests::CaseLog cases;
        std::vector<std::string> intended;
        for (const auto& test : selected)
            for (const auto* pass : {"cold", "warm"})
                intended.emplace_back(std::string(test.id) + "/" + pass);
        cases.membership(caller, intended);
        const std::filesystem::path modulePath(argv[capture ? 2 : 1]);
        const std::filesystem::path captureDirectory = capture ? std::filesystem::path(argv[3]) : std::filesystem::path{};
        if (capture)
            require(std::filesystem::is_directory(captureDirectory) && std::filesystem::is_empty(captureDirectory), "capture directory must exist and be empty");
        const auto resources = modulePath.parent_path().parent_path() / "Resources";
        require(std::filesystem::is_directory(resources), "module Resources directory unavailable");
        std::array<std::vector<float>, 2> expected;
        if (!capture) {
            require(std::filesystem::path(argv[2]).lexically_normal() == resources.lexically_normal(), "ordinary smoke requires the staged module Resources");
            expected = expected_pixels(argv[3], argv[4]);
        }
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
                    plugin->pluginIdentifier && std::string_view(plugin->pluginIdentifier) == (modulePath.filename() == "juicer-grain-preview.ofx" ? "com.juicer.Juicer.GrainPreview" : "com.juicer.Juicer") &&
                    plugin->apiVersion == 1 && plugin->pluginVersionMajor == 1 && plugin->pluginVersionMinor == 0 &&
                    plugin->setHost && plugin->mainEntry && get(0) == plugin,
                "production descriptor contract");
        InstanceHost host;
        Lifecycle lifecycle(plugin, host);
        const OfxStatus load = lifecycle.load();
        require(load == kOfxStatOK, "production module load failed");
        require(lifecycle.call(kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK,
                "production context description failed");
        std::vector<std::string> parameterNames;
        for (const auto& call : host.calls())
            if (call.operation == "param_define" && std::any_of(ofx_test::kInstanceParameters.begin(), ofx_test::kInstanceParameters.end(), [&](const auto& definition) {
                    return definition.name == call.property;
                }))
                parameterNames.push_back(call.property);
        Json authoredParameters = Json::object(), parameters = Json::object();
        DeviceBuffers device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create production stream");
        Json rows = Json::array(), destroyStatuses = Json::array();
        Json deviceInfo;
        if (capture) {
            cudaDeviceProp properties{};
            int driverVersion = 0;
            require_cuda(cudaGetDeviceProperties(&properties, 0), "read CUDA device");
            require_cuda(cudaDriverGetVersion(&driverVersion), "read CUDA driver version");
            deviceInfo = {{"ordinal", 0}, {"name", properties.name}, {"driver_version", driverVersion}};
        }
        int effectsCreated = 0, effectsDestroyed = 0;
        for (std::size_t index = 0; index < selected.size(); ++index) {
            const auto& test = selected[index];
            auto* effect = host.create_effect();
            author_parameters(host, effect, test);
            if (!capture) {
                host.set_value(effect, "GlareActive", {0});
            }
            if (capture) {
                host.set_value(effect, "HalationActive", {1});
                host.set_value(effect, "InputColorSpace", {1});
                host.set_value(effect, "CameraAutoExposure", {static_cast<int>(index % 2)});
                host.set_value(effect, "FilmProfileKey", {std::string(index < 4 ? "kodak_portra_400" : "fujifilm_provia_100f")});
                authoredParameters[test.id] = parameter_values(host, effect, parameterNames);
            }
            const auto create = lifecycle.call(kOfxActionCreateInstance, effect);
            require(create == kOfxStatOK, "production instance creation failed");
            lifecycle.created(effect);
            ++effectsCreated;
            if (index == 0 && modulePath.filename() == "juicer-grain-preview.ofx") {
                require(host.value(effect, "GrainModel") == std::vector<InstanceValue>{std::string("current")}, "preview model default");
                require(host.value(effect, "FilmGrainAmountEv") == std::vector<InstanceValue>{0.0} &&
                            host.value(effect, "FilmGrainTexelPitchUm") == std::vector<InstanceValue>{1.0} &&
                            host.value(effect, "FilmGrainChromaFraction") == std::vector<InstanceValue>{1.0},
                        "preview slider defaults");
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
            if (capture && index == 0)
                write_pixels(captureDirectory / "input.f32", active_pixels(source, test));
            const auto bytes = source.size() * sizeof(float);
            require_cuda(cudaMalloc(&device.source, bytes), "allocate source buffer");
            require_cuda(cudaMalloc(&device.destination, bytes), "allocate destination buffer");
            configure(host, effect, test, device);
            std::vector<std::uint32_t> coldWords;
            for (const auto* pass : {"cold", "warm"}) {
                const std::string id = std::string(test.id) + "/" + pass;
                cases.event(caller, id, "start");
                rows.push_back(run_pass(driver, lifecycle, host, effect, test, pass, {.source = source, .expected = capture ? nullptr : &expected[index], .capture = captureDirectory}, device, coldWords, summary));
            }
            if (capture)
                parameters[test.id] = parameter_values(host, effect, parameterNames);
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
        for (auto& row : rows)
            row["context_restored"] = restore.restored;
        Json report{{"rows", std::move(rows)}, {"lifecycle", {{"module_load_status", load}, {"effects_created", effectsCreated}, {"effects_destroyed", effectsDestroyed}, {"destroy_statuses", std::move(destroyStatuses)}, {"module_unload_status", unload}, {"context_restored", restore.restored}}}};
        if (capture) {
            report["schema"] = "ofx-halation-capture-v1";
            report["authored_parameters"] = std::move(authoredParameters);
            report["parameters"] = std::move(parameters);
            report["geometry"] = {{"bounds", {0, 0, 1920, 1080}}, {"rod", {0, 0, 1920, 1080}}, {"render_scale", {1.0, 1.0}}, {"pixel_aspect", 1.0}, {"pitch_padding_floats", kPitchPadding}, {"time", 37.0}};
            report["device"] = std::move(deviceInfo);
            std::ofstream manifest(captureDirectory / "capture.json");
            manifest << report.dump(2) << '\n';
            manifest.close();
            require(static_cast<bool>(manifest), "capture manifest write failed");
        }
        const auto output = report.dump();
        if (std::fwrite(output.data(), 1, output.size(), stdout) != output.size() || std::fputc('\n', stdout) == EOF || std::fflush(stdout) != 0)
            return 1;
        for (const auto& row : report.at("rows"))
            cases.event(caller, row.at("case_id").get<std::string>(), "terminal", "passed");
        cases.finish();
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
