#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
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
#include "instance_host.h"
#include "case_results.h"

namespace {
    using Json = nlohmann::json;
    using ofx_test::InstanceHost;
    using ofx_test::InstanceValue;
    constexpr int kWidth = 7;
    constexpr int kHeight = 5;
    constexpr int kPitchPadding = 5;
    constexpr float kSourceCanary = -7.0f;
    constexpr float kDestinationCanary = -8.0f;
    struct Case {
        const char* id;
        const char* routeName;
        const char* route;
        int reconstruction;
        int components;
    };
    constexpr std::array<Case, 2> kCases{{{"negative-direct", "negative-direct", "NegativeDirectScan", 0, 3},
                                          {"negative-print", "negative-print", "NegativePrintScan", 1, 4}}};
    constexpr std::array<std::string_view, 4> kCaseIds{
        "negative-direct/cold", "negative-direct/warm", "negative-print/cold", "negative-print/warm"};
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
    struct DeviceBuffers final {
        cudaStream_t stream = nullptr;
        float* source = nullptr;
        float* destination = nullptr;
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
    std::vector<float> initial_source(const Case& test) {
        const int pitch = kWidth * test.components + kPitchPadding;
        std::vector<float> source(static_cast<std::size_t>(pitch) * static_cast<std::size_t>(kHeight), kSourceCanary);
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const auto pixel = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) +
                                   static_cast<std::size_t>(x) * static_cast<std::size_t>(test.components);
                for (int c = 0; c < test.components; ++c) {
                    source[pixel + static_cast<std::size_t>(c)] =
                        c == 3 ? 0.125f + static_cast<float>((x + y) % 4) * 0.25f
                               : static_cast<float>((x + 3 * y + 7 * c) % 19) / 19.0f;
                }
            }
        }
        return source;
    }

    std::vector<float> active_pixels(const std::vector<float>& pitched, const Case& test) {
        const int pitch = kWidth * test.components + kPitchPadding;
        std::vector<float> result;
        result.reserve(static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight) *
                       static_cast<std::size_t>(test.components));
        for (int y = 0; y < kHeight; ++y) {
            const auto begin = pitched.begin() + static_cast<std::ptrdiff_t>(y) * pitch;
            const auto rowSamples = static_cast<std::ptrdiff_t>(kWidth) * test.components;
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

    bool canaries_preserved(const std::vector<float>& initial, const std::vector<float>& actual, const Case& test) {
        const int pitch = kWidth * test.components + kPitchPadding;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = kWidth * test.components; x < pitch; ++x) {
                const auto index = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) +
                                   static_cast<std::size_t>(x);
                if (std::bit_cast<std::uint32_t>(actual[index]) != std::bit_cast<std::uint32_t>(initial[index])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool alpha_preserved(const std::vector<float>& source, const std::vector<float>& output, const Case& test) {
        if (test.components != 4) {
            return true;
        }
        const int pitch = kWidth * test.components + kPitchPadding;
        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                const auto index = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) +
                                   static_cast<std::size_t>(x) * static_cast<std::size_t>(test.components) + 3u;
                if (std::bit_cast<std::uint32_t>(source[index]) != std::bit_cast<std::uint32_t>(output[index])) {
                    return false;
                }
            }
        }
        return true;
    }

    struct ParameterValues {
        std::vector<InstanceValue> uiDefault;
        std::vector<InstanceValue> authored;
    };
    void author(InstanceHost& host, OfxImageEffectHandle effect, const char* name, ParameterValues values) {
        require(host.value(effect, name) == values.uiDefault, std::string("unexpected UI default: ") + name);
        host.set_value(effect, name, std::move(values.authored));
    }
    void author_parameters(InstanceHost& host, OfxImageEffectHandle effect, const Case& test) {
        author(host, effect, "Exposure", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "CameraMeteringMethod", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "CameraAutoExposure", {.uiDefault = {1}, .authored = {0}});
        author(host, effect, "CameraFilmFormatPreset", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "CameraFilmFormatMm", {.uiDefault = {35.0}, .authored = {35.0}});
        author(host, effect, "FilmProfileKey", {.uiDefault = {std::string("kodak_portra_400")}, .authored = {std::string("kodak_portra_400")}});
        author(host, effect, "SpectralUpsampling", {.uiDefault = {0}, .authored = {test.reconstruction}});
        author(host, effect, "InputColorSpace", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "InputCctfDecoding", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "InputCompression", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "Hanatos2025AdaptationWindow", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "Hanatos2025AdaptationSurface", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "PrintProfileKey", {.uiDefault = {std::string("kodak_portra_endura")}, .authored = {std::string("kodak_portra_endura")}});
        author(host, effect, "ScanRoute", {.uiDefault = {std::string("NegativePrintScan")}, .authored = {std::string(test.route)}});
        author(host, effect, "PrintExposure", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintPreflash", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "PrintExposureCompensation", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "EnlargerY", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "EnlargerM", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "EnlargerC", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "FilmGammaFactor", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintGammaFactor", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CouplersActive", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "CouplersAmount", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CouplersInhibitionSameLayer", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CouplersInhibitionInterlayer", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CouplersDiffusionSizeUm", {.uiDefault = {20.0}, .authored = {20.0}});
        author(host, effect, "CouplersLangmuirDonorKRgb", {.uiDefault = {1.0, 1.0, 1.0}, .authored = {1.0, 1.0, 1.0}});
        author(host, effect, "CouplersLangmuirReceiverKRgb", {.uiDefault = {1.0, 1.0, 1.0}, .authored = {1.0, 1.0, 1.0}});
        author(host, effect, "CouplersDiffusionTailUm", {.uiDefault = {200.0}, .authored = {200.0}});
        author(host, effect, "CouplersDiffusionTailWeight", {.uiDefault = {0.03}, .authored = {0.03}});
        author(host, effect, "CouplersGammaUseStock", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "CouplersGammaSameLayerRgb", {.uiDefault = {0.341, 0.324, 0.273}, .authored = {0.341, 0.324, 0.273}});
        author(host, effect, "CouplersGammaInterlayerRToGb", {.uiDefault = {0.355, 0.305}, .authored = {0.355, 0.305}});
        author(host, effect, "CouplersGammaInterlayerGToRb", {.uiDefault = {0.154, 0.358}, .authored = {0.154, 0.358}});
        author(host, effect, "CouplersGammaInterlayerBToRg", {.uiDefault = {0.171, 0.225}, .authored = {0.171, 0.225}});
        author(host, effect, "ScannerLensBlurSigmaPx", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "ScannerUnsharpMask", {.uiDefault = {0.7, 0.7}, .authored = {0.0, 0.0}});
        author(host, effect, "ScannerBlackCorrection", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "ScannerWhiteCorrection", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "ScannerBlackLevel", {.uiDefault = {0.01}, .authored = {0.01}});
        author(host, effect, "ScannerWhiteLevel", {.uiDefault = {0.98}, .authored = {0.98}});
        author(host, effect, "ScannerLutResolution", {.uiDefault = {17}, .authored = {17}});
        author(host, effect, "HalationActive", {.uiDefault = {1}, .authored = {0}});
        author(host, effect, "HalationScatterAmount", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "HalationScatterSpatialScale", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "HalationAmount", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "HalationSpatialScale", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "GrainActive", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "GrainSublayersActive", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "GrainPreset", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "GrainAmplitude", {.uiDefault = {-1.20}, .authored = {0.0}});
        author(host, effect, "GrainBlur", {.uiDefault = {0.50}, .authored = {0.65}});
        author(host, effect, "GrainSharpness", {.uiDefault = {0.5}, .authored = {0.5}});
        author(host, effect, "GrainChroma", {.uiDefault = {0.3}, .authored = {0.0}});
        author(host, effect, "GrainTexture", {.uiDefault = {0.55}, .authored = {0.55}});
        author(host, effect, "GrainParticleAreaUm2", {.uiDefault = {0.25}, .authored = {0.2}});
        author(host, effect, "GrainParticleScaleMaster", {.uiDefault = {1.48}, .authored = {1.48}});
        author(host, effect, "GrainParticleScaleLayersMaster", {.uiDefault = {1.922}, .authored = {1.922}});
        author(host, effect, "GrainDensityMinMaster", {.uiDefault = {0.08}, .authored = {0.08}});
        author(host, effect, "GrainUniformityMaster", {.uiDefault = {0.97}, .authored = {0.97}});
        author(host, effect, "GrainBlurDyeCloudsUm", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "GrainSizeMixWeight", {.uiDefault = {0.226}, .authored = {0.0}});
        author(host, effect, "GrainSizeMixWeightMid", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "GrainSizeMixScale", {.uiDefault = {19.0}, .authored = {1.0}});
        author(host, effect, "GrainMicroStructure", {.uiDefault = {60.0, 170.0}, .authored = {0.2, 30.0}});
        author(host, effect, "GrainDebugView", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "GrainParticleScale", {.uiDefault = {1.48, 1.48, 1.48}, .authored = {0.8, 1.0, 2.0}});
        author(host, effect, "GrainParticleScaleLayers", {.uiDefault = {1.922, 1.922, 1.922}, .authored = {2.5, 1.0, 0.5}});
        author(host, effect, "GrainDensityMin", {.uiDefault = {0.08, 0.08, 0.08}, .authored = {0.07, 0.08, 0.12}});
        author(host, effect, "GrainUniformity", {.uiDefault = {0.97, 0.97, 0.97}, .authored = {0.97, 0.97, 0.99}});
        author(host, effect, "GrainClumpTemporalMix", {.uiDefault = {0.30}, .authored = {0.0}});
        author(host, effect, "GrainClumpMorphPeriodSec", {.uiDefault = {8.0}, .authored = {8.0}});
        author(host, effect, "GateWeaveAmount", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "FilmDustAmount", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "GateDustAmount", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "FilmScratchAmount", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "GateScratchAmount", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "CameraDiffusionEnabled", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "CameraDiffusionFamily", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "CameraDiffusionStrength", {.uiDefault = {0.5}, .authored = {0.5}});
        author(host, effect, "CameraDiffusionSpatialScale", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CameraDiffusionHaloWarmth", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "CameraDiffusionCoreIntensity", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CameraDiffusionCoreSize", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CameraDiffusionHaloIntensity", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CameraDiffusionHaloSize", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CameraDiffusionBloomIntensity", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "CameraDiffusionBloomSize", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionEnabled", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "PrintDiffusionFamily", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "PrintDiffusionStrength", {.uiDefault = {0.5}, .authored = {0.5}});
        author(host, effect, "PrintDiffusionSpatialScale", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionHaloWarmth", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "PrintDiffusionCoreIntensity", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionCoreSize", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionHaloIntensity", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionHaloSize", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionBloomIntensity", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "PrintDiffusionBloomSize", {.uiDefault = {1.0}, .authored = {1.0}});
        author(host, effect, "GlareActive", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "GlarePercent", {.uiDefault = {0.03}, .authored = {0.03}});
        author(host, effect, "GlareRoughness", {.uiDefault = {0.7}, .authored = {0.7}});
        author(host, effect, "GlareBlurSigmaPx", {.uiDefault = {0.5}, .authored = {0.5}});
        author(host, effect, "PrintShadowCompensationFactor", {.uiDefault = {0.0}, .authored = {0.0}});
        author(host, effect, "PrintShadowCompensationDensity", {.uiDefault = {1.2}, .authored = {1.2}});
        author(host, effect, "PrintShadowCompensationTransition", {.uiDefault = {0.3}, .authored = {0.3}});
        author(host, effect, "ReferenceIlluminant", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "EnlargerIlluminant", {.uiDefault = {3}, .authored = {3}});
        author(host, effect, "OutputColorSpace", {.uiDefault = {0}, .authored = {0}});
        author(host, effect, "OutputCctfEncoding", {.uiDefault = {1}, .authored = {1}});
        author(host, effect, "OutputGamutCompression", {.uiDefault = {1}, .authored = {1}});
    }
    void configure(InstanceHost& host, OfxImageEffectHandle effect, const Case& test, const DeviceBuffers& device) {
        host.set_effect_property(effect, kOfxImageEffectPropFrameRate, {24.0});
        auto* args = host.action_args();
        host.set_property(args, kOfxPropTime, {37.0});
        host.set_property(args, kOfxImageEffectPropRenderScale, {1.0, 1.0});
        host.set_property(args, kOfxImageEffectPropRenderWindow, {0, 0, kWidth, kHeight});
        host.set_property(args, kOfxImageEffectPropCudaEnabled, {1});
        host.set_property(args, kOfxImageEffectPropCudaStream, {static_cast<void*>(device.stream)});
        host.set_property(args, kOfxImageEffectPropFieldToRender, {std::string(kOfxImageFieldNone)});
        const int rowBytes = (kWidth * test.components + kPitchPadding) * static_cast<int>(sizeof(float));
        for (const auto* clip : {"Source", "Output"}) {
            const bool source = std::string_view(clip) == "Source";
            auto set = [&](const char* key, std::vector<InstanceValue> values) {
                host.set_image_property(effect, clip, key, std::move(values));
            };
            set(kOfxImagePropRowBytes, {rowBytes});
            set(kOfxImagePropPixelAspectRatio, {1.0});
            set(kOfxImageEffectPropComponents, {std::string(test.components == 3 ? kOfxImageComponentRGB : kOfxImageComponentRGBA)});
            set(kOfxImageEffectPropPixelDepth, {std::string(kOfxBitDepthFloat)});
            set(kOfxImageEffectPropPreMultiplication, {std::string(kOfxImageUnPreMultiplied)});
            set(kOfxImagePropBounds, {0, 0, kWidth, kHeight});
            set(kOfxImagePropRegionOfDefinition, {0, 0, kWidth, kHeight});
            set(kOfxImagePropField, {std::string(kOfxImageFieldNone)});
            set(kOfxImagePropUniqueIdentifier, {std::string(clip)});
            set(kOfxImageEffectPropRenderScale, {1.0, 1.0});
            set(kOfxImagePropData, {static_cast<void*>(source ? device.source : device.destination)});
            host.set_clip_region(effect, clip, OfxRectD{0.0, 0.0, static_cast<double>(kWidth), static_cast<double>(kHeight)});
        }
    }
    struct PassImages {
        const std::vector<float>& source;
        const std::vector<float>& expected;
    };
    Json run_pass(Lifecycle& lifecycle, InstanceHost& host, OfxImageEffectHandle effect, const Case& test, std::string_view pass, PassImages images, DeviceBuffers& device, std::vector<std::uint32_t>& coldWords, bool summary) {
        const auto& [source, expected] = images;
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
        const bool padding = canaries_preserved(initial, output, test);
        const bool alpha = alpha_preserved(source, output, test);
        require(padding && alpha, "render changed alpha or destination padding");
        const auto activeInput = active_pixels(source, test);
        const auto activeOutput = active_pixels(output, test);
        const bool finite = std::all_of(activeOutput.begin(), activeOutput.end(), [](float value) {
            return std::isfinite(value);
        });
        const bool fixtureMatch = compare_fixture(activeOutput, expected);
        require(finite && fixtureMatch, "rendered pixels differ from accepted fixture");
        std::vector<std::uint32_t> outputWords;
        outputWords.reserve(activeOutput.size());
        for (float value : activeOutput)
            outputWords.push_back(std::bit_cast<std::uint32_t>(value));
        if (pass == "cold")
            coldWords = outputWords;
        else
            require(coldWords == outputWords, "warm output differs from cold output");
        Json row{{"case_id", std::string(test.id) + "/" + std::string(pass)}, {"route", test.routeName}, {"pass", std::string(pass)}, {"ofx_status", status}, {"messages", std::move(messages)}, {"input_shape", Json::array({kHeight, kWidth, test.components})}, {"output_shape", Json::array({kHeight, kWidth, test.components})}, {"accepted_fixture_match", fixtureMatch}, {"finite_output", finite}, {"alpha_preserved", alpha}, {"padding_preserved", padding}, {"stream_completed", true}, {"readback_complete", true}, {"context_preserved", before == after}, {"context_restored", false}, {"source_image_releases", sourceReleases}, {"output_image_releases", outputReleases}, {"samples_compared", activeOutput.size()}};
        if (!summary) {
            row["input_f32_words"] = words_json(activeInput);
            row["output_f32_words"] = words_json(activeOutput);
        }
        return row;
    }
} // namespace

int main(int argc, char** argv) {
    try {
        if (JuicerTests::list_requested(argc, argv)) {
            for (const auto id : kCaseIds)
                JuicerTests::list_case("production-prepared-render", id);
            return std::fflush(stdout) == 0 ? 0 : 1;
        }
        const bool summary = argc == 7 && std::string_view(argv[6]) == "--summary";
        require(argc == 6 || summary, "usage: JuicerProductionPreparedRenderProbe module Resources processor-${preset}.dat preset product-updates.json [--summary]");
        JuicerTests::CaseLog cases;
        std::vector<std::string> intended;
        intended.reserve(kCaseIds.size());
        for (const auto id : kCaseIds)
            intended.emplace_back(id);
        cases.membership("production-prepared-render", intended);
        const std::filesystem::path modulePath(argv[1]), resources(argv[2]), fixture(argv[3]), updates(argv[5]);
        const std::string preset(argv[4]);
        require(fixture.filename().string() == "processor-" + preset + ".dat", "processor fixture does not match preset");
        require(std::filesystem::is_directory(resources), "reference Resources directory unavailable");
        require(std::filesystem::is_directory(modulePath.parent_path().parent_path() / "Resources"), "module Resources directory unavailable");
        std::ifstream updatesInput(updates);
        require(static_cast<bool>(updatesInput), "print expectation updates unavailable");
        const auto expected = expected_pixels(fixture, Json::parse(updatesInput));
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
        for (std::size_t index = 0; index < kCases.size(); ++index) {
            const auto& test = kCases[index];
            auto* effect = host.create_effect();
            author_parameters(host, effect, test);
            const auto create = lifecycle.call(kOfxActionCreateInstance, effect);
            require(create == kOfxStatOK, "production instance creation failed");
            lifecycle.created(effect);
            ++effectsCreated;
            const auto source = initial_source(test);
            const auto bytes = source.size() * sizeof(float);
            require_cuda(cudaMalloc(&device.source, bytes), "allocate source buffer");
            require_cuda(cudaMalloc(&device.destination, bytes), "allocate destination buffer");
            configure(host, effect, test, device);
            std::vector<std::uint32_t> coldWords;
            for (const auto* pass : {"cold", "warm"}) {
                const std::string id = std::string(test.id) + "/" + pass;
                cases.event("production-prepared-render", id, "start");
                rows.push_back(run_pass(lifecycle, host, effect, test, pass, {.source = source, .expected = expected[index]}, device, coldWords, summary));
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
        for (auto& row : rows)
            row["context_restored"] = restore.restored;
        const Json report{{"rows", std::move(rows)}, {"lifecycle", {{"module_load_status", load}, {"effects_created", effectsCreated}, {"effects_destroyed", effectsDestroyed}, {"destroy_statuses", std::move(destroyStatuses)}, {"module_unload_status", unload}, {"context_restored", restore.restored}}}};
        const auto output = report.dump();
        if (std::fwrite(output.data(), 1, output.size(), stdout) != output.size() || std::fputc('\n', stdout) == EOF || std::fflush(stdout) != 0)
            return 1;
        for (const auto& row : report.at("rows"))
            cases.event("production-prepared-render", row.at("case_id").get<std::string>(), "terminal", "passed");
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
