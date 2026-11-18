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
    constexpr std::array<Case, 2> kCases{{{"negative-direct", "negative-direct", "NegativeDirectScan", 0, 3},
                                          {"negative-print", "negative-print", "NegativePrintScan", 1, 4}}};
    constexpr std::array<std::string_view, 4> kCaseIds{
        "negative-direct/cold", "negative-direct/warm", "negative-print/cold", "negative-print/warm"};
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
    void require_driver(CUresult result, std::string_view operation) {
        if (result != CUDA_SUCCESS) {
            const char* name = nullptr;
            (void)cuGetErrorName(result, &name);
            throw std::runtime_error(std::string(operation) + ": " + (name ? name : "CUDA driver failure"));
        }
    }
    class Module final {
    public:
        explicit Module(const std::filesystem::path& path) {
#ifdef _WIN32
            handle_ = LoadLibraryW(path.c_str());
#else
            handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
            require(handle_ != nullptr, "unable to load production OFX module");
        }
        Module(const Module&) = delete;
        Module& operator=(const Module&) = delete;
        ~Module() {
            if (handle_)
                close_unchecked();
        }
        template <class T>
        T symbol(const char* name) const {
#ifdef _WIN32
            auto* address = GetProcAddress(handle_, name);
#else
            auto* address = dlsym(handle_, name);
#endif
            require(address != nullptr, std::string("missing OFX module export: ") + name);
            return reinterpret_cast<T>(address);
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
    struct ContextRestore final {
        CUcontext original = nullptr;
        bool restored = false;
        explicit ContextRestore(CUcontext value) noexcept : original(value) {}
        ~ContextRestore() {
            if (!restored)
                (void)cuCtxSetCurrent(original);
        }
        void restore() {
            require_driver(cuCtxSetCurrent(original), "restore original CUDA context");
            CUcontext current = nullptr;
            require_driver(cuCtxGetCurrent(&current), "check restored CUDA context");
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

    std::array<std::vector<float>, 2> expected_pixels(const std::filesystem::path& fixture,
                                                      const Json& updates) {
        std::ifstream input(fixture);
        require(static_cast<bool>(input), "processor fixture is unavailable");
        require(updates.at("schema") == 1 && updates.at("policy") == "print-erff-0.2.15-2026-10-20",
                "unsupported print illuminant expectation policy");
        constexpr std::array<std::string_view, 7> fixtureCases{
            "negative-direct", "negative-print", "positive-direct", "positive-print", "combined-print", "glare-plus-zero", "glare-minus-zero"};
        std::array<std::vector<float>, 2> result;
        for (std::size_t index = 0; index < fixtureCases.size(); ++index) {
            std::string name;
            std::size_t count = 0;
            input >> name >> count;
            require(static_cast<bool>(input) && name == fixtureCases[index] && count > 0,
                    "processor fixture membership or ordering changed");
            std::vector<float> values(count);
            for (float& value : values) {
                input >> value;
            }
            require(static_cast<bool>(input), "processor fixture is truncated");
            if (index < result.size()) {
                Json words = words_json(values);
                const auto logicalPath = "ofx/fixtures/exposure/" + fixture.filename().string() + "." + name;
                const auto& fixtures = updates.at("fixtures");
                if (fixtures.contains(logicalPath)) {
                    const auto& patch = fixtures.at(logicalPath);
                    require(patch.is_array() && patch.size() % 2 == 0,
                            "incomplete print illuminant expectation supplement");
                    for (std::size_t i = 0; i < patch.size(); i += 2) {
                        require(patch[i].at("op") == "test" && patch[i + 1].at("op") == "replace" &&
                                    patch[i].at("path") == patch[i + 1].at("path"),
                                "unbounded print illuminant expectation supplement");
                    }
                    words = words.patch(patch);
                }
                result[index] = words_floats(words);
            }
        }
        std::string trailing;
        require(!(input >> trailing), "processor fixture has unexpected trailing rows");
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
    Json run_pass(Lifecycle& lifecycle, InstanceHost& host, OfxImageEffectHandle effect, const Case& test, std::string_view pass, const PassImages& images, DeviceBuffers& device, std::vector<std::uint32_t>& coldWords, bool summary) {
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
        require_driver(cuCtxGetCurrent(&before), "inspect context before render");
        const OfxStatus status = lifecycle.call(kOfxImageEffectActionRender, effect, host.action_args());
        require_driver(cuCtxGetCurrent(&after), "inspect context after render");
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
        require(status == kOfxStatOK && messages.empty() && !host.callback_error(), "OFX render callback failed or emitted a host message");
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

    int grain_preview_discovery(const std::filesystem::path& previewPath, const std::filesystem::path& regularPath) {
        JuicerTests::CaseLog cases;
        cases.membership("grain-preview-discovery", {"both-modules"});
        {
            JuicerTests::CaseEvent event("grain-preview-discovery", "both-modules");
            require(previewPath.filename() == "juicer-grain-preview.ofx" && regularPath.filename() == "juicer.ofx",
                    "discovery module paths");
            Module preview(previewPath);
            Module regular(regularPath);
            for (const auto& [module, expected] : std::array{
                     std::pair{&preview, "com.juicer.Juicer.GrainPreview"},
                     std::pair{&regular, "com.juicer.Juicer"},
                 }) {
                const auto count = module->symbol<int (*)()>("OfxGetNumberOfPlugins");
                const auto get = module->symbol<OfxPlugin* (*)(int)>("OfxGetPlugin");
                require(count() == 1 && get(-1) == nullptr && get(1) == nullptr && get(0) &&
                            get(0)->pluginIdentifier && std::string_view(get(0)->pluginIdentifier) == expected &&
                            get(0)->setHost && get(0)->mainEntry,
                        "side-by-side discovery identity");
            }
            regular.close();
            preview.close();
            event.passed();
        }
        cases.finish();
        return 0;
    }

    int grain_preview_witness(const std::filesystem::path& path) {
        JuicerTests::CaseLog cases;
        cases.membership("grain-preview-render", {"negative-direct", "negative-print"});
        require(path.filename() == "juicer-grain-preview.ofx", "expected preview module");
        const auto resources = path.parent_path().parent_path() / "Resources";
        require(std::filesystem::is_regular_file(resources / "grain/portra-400/manifest.json") &&
                    std::filesystem::is_regular_file(resources / "grain/portra-400/records.f32") &&
                    std::filesystem::is_regular_file(resources / "grain/portra-400/sums.f32"),
                "preview bundle grain bank missing");
        require_driver(cuInit(0), "initialize CUDA driver");
        CUcontext original = nullptr;
        require_driver(cuCtxGetCurrent(&original), "read original CUDA context");
        ContextRestore restore(original);
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
        for (const Case test : std::array{
                 Case{"negative-direct", "negative-direct", "NegativeDirectScan", 0, 4, 1920, 1080},
                 Case{"negative-print", "negative-print", "NegativePrintScan", 0, 4, 1920, 1080},
             }) {
            JuicerTests::CaseEvent event("grain-preview-render", test.id);
            auto* effect = host.create_effect();
            author_parameters(host, effect, test);
            host.set_value(effect, "CameraFilmFormatPreset", {2});
            host.set_value(effect, "CameraFilmFormatMm", {36.0});
            require(lifecycle.call(kOfxActionCreateInstance, effect) == kOfxStatOK, "preview create");
            lifecycle.created(effect);
            const auto source = initial_source(test);
            const auto bytes = source.size() * sizeof(float);
            require_cuda(cudaMalloc(&device.source, bytes), "allocate preview source");
            require_cuda(cudaMalloc(&device.destination, bytes), "allocate preview destination");
            configure(host, effect, test, device);
            auto change = [&](const char* name, std::vector<InstanceValue> value) {
                host.set_value(effect, name, std::move(value));
                auto* args = host.change_args({kOfxChangeUserEdited, kOfxTypeParameter, name});
                require(lifecycle.call(kOfxActionInstanceChanged, effect, args) == kOfxStatReplyDefault,
                        "preview control change");
            };
            auto render = [&](double time) {
                host.set_property(host.action_args(), kOfxPropTime, {time});
                std::vector<std::uint32_t> words;
                (void)run_pass(lifecycle, host, effect, test, "cold", {source, nullptr, {}}, device, words, true);
                return words;
            };
            const auto off = render(37.0);
            change("GrainModel", {std::string("film_v1")});
            change("GrainActive", {1});
            const auto film = render(37.0);
            require(film != off, "Film v1 did not change delivered pixels");
            require(render(37.0) == film, "paused Film frame changed");
            const auto later = render(38.0);
            require(later != film, "Film frame seed did not change");
            require(render(37.0) == film, "out-of-order Film frame changed");
            const auto fractional = render(37.5);
            require(fractional != film && fractional != later, "fractional Film frame did not change");
            change("GrainModel", {std::string("current")});
            const auto current = render(37.0);
            require(current != off, "Current grain did not change delivered pixels");
            change("GrainActive", {0});
            require(render(37.0) == off, "grain-off output changed after model edits");
            routes.push_back({{"route", test.routeName}, {"film_changes_output", true}, {"paused_repeat", true}, {"out_of_order_repeat", true}, {"fractional_distinct", true}, {"current_changes_output", true}, {"off_restored", true}});
            require(lifecycle.destroy(effect) == kOfxStatOK, "preview destroy");
            device.release_images();
            event.passed();
        }
        device.close();
        require(lifecycle.unload() == kOfxStatOK && !host.callback_error(), "preview unload");
        module.close();
        restore.restore();
        cases.finish();
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

        require_driver(cuInit(0), "initialize CUDA driver");
        CUcontext original = nullptr;
        require_driver(cuCtxGetCurrent(&original), "read original CUDA context");
        ContextRestore restore(original);
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
            require_driver(cuCtxGetCurrent(&before), "inspect context before replay");
            const auto status = lifecycle.call(kOfxImageEffectActionRender, effect, args);
            require_driver(cuCtxGetCurrent(&after), "inspect context after replay");
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
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 3 && std::string_view(argv[1]) == "--grain-preview-witness") {
            if (JuicerTests::list_requested(argc, argv)) {
                JuicerTests::list_case("grain-preview-render", "negative-direct");
                JuicerTests::list_case("grain-preview-render", "negative-print");
                return std::fflush(stdout) == 0 ? 0 : 1;
            }
            require(argc == 3, "preview witness arguments");
            return grain_preview_witness(argv[2]);
        }
        if (argc >= 4 && std::string_view(argv[1]) == "--grain-preview-discovery") {
            if (JuicerTests::list_requested(argc, argv)) {
                JuicerTests::list_case("grain-preview-discovery", "both-modules");
                return std::fflush(stdout) == 0 ? 0 : 1;
            }
            require(argc == 4, "preview discovery arguments");
            return grain_preview_discovery(argv[2], argv[3]);
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
        const bool summary = capture || (argc == 7 && std::string_view(argv[6]) == "--summary");
        require(argc == 6 || summary, "usage: JuicerProductionPreparedRenderProbe module Resources processor-${preset}.dat preset product-updates.json [--summary]; or --halation-capture module output-directory");
        const auto selected = capture ? std::span<const Case>(kHalationCases) : std::span<const Case>(kCases);
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
        require(std::filesystem::is_directory(modulePath.parent_path().parent_path() / "Resources"), "module Resources directory unavailable");
        std::array<std::vector<float>, 2> expected;
        if (!capture) {
            const std::filesystem::path resources(argv[2]), fixture(argv[3]), updates(argv[5]);
            const std::string preset(argv[4]);
            require(fixture.filename().string() == "processor-" + preset + ".dat", "processor fixture does not match preset");
            require(std::filesystem::is_directory(resources), "reference Resources directory unavailable");
            std::ifstream updatesInput(updates);
            require(static_cast<bool>(updatesInput), "print expectation updates unavailable");
            expected = expected_pixels(fixture, Json::parse(updatesInput));
        }
        require_driver(cuInit(0), "initialize CUDA driver");
        CUcontext original = nullptr;
        require_driver(cuCtxGetCurrent(&original), "read original CUDA context");
        ContextRestore restore(original);
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
                            host.value(effect, "FilmGrainChromaFraction") == std::vector<InstanceValue>{0.2406350076198578},
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
                rows.push_back(run_pass(lifecycle, host, effect, test, pass, {.source = source, .expected = capture ? nullptr : &expected[index], .capture = captureDirectory}, device, coldWords, summary));
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
