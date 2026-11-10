#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>


#include "instance_host.h"
#include "Hash.h"
#include "juicer_ofx_api.h"
#include "../ffi/case_results.h"
#include "../ffi/juicer_test_api.h"
#include "../ffi/print_illuminant_fixture.h"
#include "../ffi/preparation_fixture.h"
#include "../ffi/capture_identity.h"

extern "C" std::uint64_t fj_test_ofx_clock_conversion(std::int64_t, std::int64_t);
extern "C" void fj_test_ofx_clock_token(std::uint64_t);
extern "C" std::uint64_t fj_test_ofx_seed(std::uint64_t, std::uint64_t);
extern "C" int fj_test_ofx_instance_identity(OfxImageEffectHandle, std::uint64_t*, std::uint64_t*, std::uintptr_t*);
extern "C" FjStatus fj_test_ofx_instance_observe(OfxImageEffectHandle, FjInstanceObservation**, FjErrorBuffer*);
extern "C" FjStatus fj_test_ofx_instance_admit(OfxImageEffectHandle, FjPreparedState**, FjErrorBuffer*);

namespace {
    void require(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    void success(FjStatus status, const char* message) {
        require(status.category == FJ_STATUS_SUCCESS, message);
    }
    OfxStatus action(OfxPlugin* plugin, const char* name, const void* effect = nullptr, OfxPropertySetHandle args = nullptr) {
        return plugin->mainEntry(name, effect, args, nullptr);
    }
    void change(OfxPlugin* plugin, ofx_test::InstanceHost& host, OfxImageEffectHandle effect, const char* name, const char* reason, double time = 0.0) {
        auto* args = host.change_args({reason, kOfxTypeParameter, name}, time);
        require(action(plugin, kOfxActionInstanceChanged, effect, args) == kOfxStatReplyDefault, "instance change");
    }
    struct Observation {
        FjInstanceObservation* owner = nullptr;
        FjInstanceObservationView view{};
        explicit Observation(OfxImageEffectHandle effect) {
            success(fj_test_ofx_instance_observe(effect, &owner, nullptr), "observe instance");
            require(owner != nullptr, "missing observation");
            success(fj_test_instance_observation_view(owner, &view, nullptr), "observation view");
        }
        ~Observation() {
            if (owner)
                (void)fj_test_instance_observation_release(owner, nullptr);
        }
        Observation(const Observation&) = delete;
        Observation& operator=(const Observation&) = delete;
    };
    struct Admission {
        FjPreparedState* owner = nullptr;
        FjPreparedStateView view{};
        explicit Admission(OfxImageEffectHandle effect) {
            success(fj_test_ofx_instance_admit(effect, &owner, nullptr), "admit capture");
            require(owner != nullptr, "missing admitted state");
            success(fj_test_prepared_state_view(owner, &view, nullptr), "prepared view");
        }
        ~Admission() {
            if (owner)
                (void)fj_test_prepared_state_release(owner, nullptr);
        }
        Admission(const Admission&) = delete;
        Admission& operator=(const Admission&) = delete;
    };
    void words(std::ostream& output, const FjVisualGrainParams& g, const FjPreparedStateView& state) {
        const auto word = [&](float value) {
            output << ' ' << std::bit_cast<std::uint32_t>(value);
        };
        word(g.particle_area_um2);
        for (float value : g.particle_scale_cmy)
            word(value);
        for (float value : g.particle_scale_layers)
            word(value);
        for (float value : g.particle_density_min_cmy)
            word(value);
        for (float value : g.uniformity_cmy)
            word(value);
        word(g.correlation_sigma_px);
        word(g.dye_cloud_blur_um);
        for (float value : g.micro_structure)
            word(value);
        output << ' ' << g.sublayer_count;
        for (float value : {g.amplitude, g.chroma_mix, g.chroma_shared_weight, g.chroma_independent_weight, g.coarse_weight, g.mid_weight, g.size_mix_scale, g.clump_temporal_mix, g.clump_morph_period_sec})
            word(value);
        output << ' ' << g.debug_view << ' ' << state.grain.hash << ' ' << state.grain.layers_hash << ' ' << state.recipe_hash << '\n';
    }
    std::vector<std::string> grain_capture_ids() {
        std::vector<std::string> ids;
        for (const char* route : {"direct", "print"})
            for (int preset = 0; preset < 3; ++preset)
                for (int policy = 0; policy < 3; ++policy)
                    for (int sample = 0; sample < 11; ++sample)
                        ids.push_back(std::string(route) + "/preset/" + std::to_string(preset) + "/policy/" + std::to_string(policy) + "/sample/" + std::to_string(sample));
        return ids;
    }
    // NativeHost::configure_params and the exposure/format/gamma resets immediately
    // preceding AdapterTrace's frozen grain witnesses define these authored inputs.
    void grain_fixture_inputs(ofx_test::InstanceHost& host, OfxImageEffectHandle effect) {
        for (const auto& [name, value] : std::to_array<std::pair<const char*, double>>({{"Exposure", 0.5}, {"CameraFilmFormatMm", 35.0}, {"FilmGammaFactor", 1.0}, {"PrintGammaFactor", 1.0}, {"CouplersAmount", 1.0}, {"CouplersInhibitionSameLayer", 1.0}, {"CouplersInhibitionInterlayer", 1.0}, {"CouplersDiffusionSizeUm", 20.0}, {"CouplersDiffusionTailUm", 200.0}, {"CouplersDiffusionTailWeight", 0.03}, {"ScannerLensBlurSigmaPx", 0.0}, {"ScannerBlackLevel", 0.01}, {"ScannerWhiteLevel", 0.98}, {"PrintExposure", 1.0}, {"PrintPreflash", 0.0}, {"EnlargerY", 0.0}, {"EnlargerM", 0.0}, {"EnlargerC", 0.0}, {"HalationScatterAmount", 1.0}, {"HalationScatterSpatialScale", 1.0}, {"HalationAmount", 1.0}, {"HalationSpatialScale", 1.0}, {"FilmDustAmount", 0.0}, {"GateDustAmount", 0.0}, {"FilmScratchAmount", 0.0}, {"GateScratchAmount", 0.0}, {"GateWeaveAmount", 0.0}, {"GlarePercent", 0.03}, {"GlareRoughness", 0.7}, {"GlareBlurSigmaPx", 0.5}, {"PrintShadowCompensationFactor", 0.0}, {"PrintShadowCompensationDensity", 1.2}, {"PrintShadowCompensationTransition", 0.3}}))
            host.set_value(effect, name, {value});
        for (const auto& [name, value] : std::to_array<std::pair<const char*, int>>({{"CameraAutoExposure", 0}, {"CameraFilmFormatPreset", 0}, {"CameraMeteringMethod", 0}, {"SpectralUpsampling", 0}, {"ReferenceIlluminant", 0}, {"EnlargerIlluminant", 3}, {"InputColorSpace", 0}, {"InputCctfDecoding", 0}, {"InputCompression", 1}, {"Hanatos2025AdaptationWindow", 1}, {"Hanatos2025AdaptationSurface", 0}, {"OutputColorSpace", 0}, {"OutputCctfEncoding", 1}, {"OutputGamutCompression", 1}, {"CouplersActive", 0}, {"CouplersGammaUseStock", 1}, {"ScannerBlackCorrection", 0}, {"ScannerWhiteCorrection", 0}, {"ScannerLutResolution", 17}, {"PrintExposureCompensation", 1}, {"HalationActive", 0}, {"GlareActive", 1}}))
            host.set_value(effect, name, {value});
        host.set_value(effect, "FilmProfileKey", {std::string("kodak_portra_400")});
        host.set_value(effect, "PrintProfileKey", {std::string("kodak_portra_endura")});
        host.set_value(effect, "ScanRoute", {std::string("NegativePrintScan")});
        host.set_value(effect, "ScannerUnsharpMask", {0.0, 0.0});
        host.set_value(effect, "CouplersLangmuirDonorKRgb", {1.0, 1.0, 1.0});
        host.set_value(effect, "CouplersLangmuirReceiverKRgb", {1.0, 1.0, 1.0});
        host.set_value(effect, "CouplersGammaSameLayerRgb", {0.341, 0.324, 0.273});
        host.set_value(effect, "CouplersGammaInterlayerRToGb", {0.355, 0.305});
        host.set_value(effect, "CouplersGammaInterlayerGToRb", {0.154, 0.358});
        host.set_value(effect, "CouplersGammaInterlayerBToRg", {0.171, 0.225});
        for (const char* prefix : {"Camera", "Print"}) {
            const std::string stage(prefix);
            host.set_value(effect, stage + "DiffusionEnabled", {0});
            host.set_value(effect, stage + "DiffusionFamily", {1});
            for (const auto& [suffix, value] : std::to_array<std::pair<const char*, double>>({{"Strength", 0.5}, {"SpatialScale", 1.0}, {"HaloWarmth", 0.0}, {"CoreIntensity", 1.0}, {"CoreSize", 1.0}, {"HaloIntensity", 1.0}, {"HaloSize", 1.0}, {"BloomIntensity", 1.0}, {"BloomSize", 1.0}}))
                host.set_value(effect, stage + "Diffusion" + suffix, {value});
        }
    }
    void grain_capture(OfxPlugin* plugin, ofx_test::InstanceHost& host, OfxImageEffectHandle effect, std::ostream& output, std::vector<std::unique_ptr<JuicerTests::CaseEvent>>& events) {
        const auto ids = grain_capture_ids();
        JuicerTests::Cohort rows("grain_capture", ids);
        const std::array<double, 3> preset_ev{-1.396, -1.20, -0.57};
        const std::array<double, 3> preset_chroma{0.0, 0.3, 0.5};
        const double threshold = static_cast<double>(0.999f);
        const std::array<double, 11> amounts{-3.0, 0.0, 3.0, -1.25, 1.234567890123, -9.0, 9.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), std::nextafter(1.234567890123, 2.0)};
        const std::array<double, 11> chromas{0.0, 0.375, 1.0, std::nextafter(threshold, 0.0), std::nextafter(threshold, 1.0), static_cast<double>(std::nextafter(0.999f, 0.0f)), static_cast<double>(std::nextafter(0.999f, 1.0f)), std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -1.0, 2.0};
        std::size_t row = 0;
        for (int print : {0, 1}) {
            host.set_value(effect, "ScanRoute", {std::string(print ? "NegativePrintScan" : "NegativeDirectScan")});
            for (int preset = 0; preset < 3; ++preset) {
                host.set_value(effect, "GrainPreset", {preset});
                change(plugin, host, effect, "GrainPreset", kOfxChangeUserEdited);
                for (int policy = 0; policy < 3; ++policy)
                    for (std::size_t sample = 0; sample < amounts.size(); ++sample) {
                        events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_capture", ids.at(row++)));
                        host.set_value(effect, "GrainActive", {int(policy != 0)});
                        host.set_value(effect, "GrainSublayersActive", {int(policy != 1)});
                        host.set_value(effect, "GrainAmplitude", {amounts[sample]});
                        host.set_value(effect, "GrainChroma", {chromas[sample]});
                        host.clear_calls();
                        change(plugin, host, effect, "GrainActive", kOfxChangeTime);
                        Observation observed(effect);
                        const auto& g = observed.view.captured.grain;
                        const double ev = std::isfinite(amounts[sample]) ? std::clamp(amounts[sample], -3.0, 3.0) : preset_ev[preset];
                        const float chroma = static_cast<float>(std::isfinite(chromas[sample]) ? std::clamp(chromas[sample], 0.0, 1.0) : preset_chroma[preset]);
                        require(observed.view.pending_kind == 4 && g.amplitude == static_cast<float>(std::exp2(ev)) && g.chroma_mix == chroma && g.chroma_shared_weight == std::sqrt(std::max(0.0f, 1.0f - chroma)) && g.chroma_independent_weight == std::sqrt(std::max(0.0f, chroma)), "grain native primitive comparison");
                        require(std::none_of(host.calls().begin(), host.calls().end(), [](const auto& call) {
                                    return call.operation == "param_get_time";
                                }),
                                "capture used at-time getter");
                        constexpr std::array getters{"GrainActive", "GrainPreset", "GrainSublayersActive", "GrainAmplitude", "GrainBlur", "GrainSharpness", "GrainChroma", "GrainTexture", "GrainParticleAreaUm2", "GrainSizeMixScale", "GrainSizeMixWeight", "GrainSizeMixWeightMid", "GrainBlurDyeCloudsUm", "GrainParticleScale", "GrainParticleScaleLayers", "GrainDensityMin", "GrainUniformity", "GrainClumpTemporalMix", "GrainClumpMorphPeriodSec", "GrainMicroStructure", "GrainDebugView"};
                        const auto& calls = host.calls();
                        const auto ordered = std::search(calls.begin(), calls.end(), getters.begin(), getters.end(), [](const auto& call, const char* name) {
                            return call.operation == "param_get" && call.target == name;
                        });
                        require(ordered != calls.end(), "grain current getter order");
                        Admission admitted(effect);
                        output << "GRAIN_CAPTURE " << print << ' ' << preset << ' ' << policy << ' ' << sample << ' ' << std::bit_cast<std::uint64_t>(amounts[sample]) << ' ' << std::bit_cast<std::uint64_t>(chromas[sample]);
                        words(output, g, admitted.view);
                    }
            }
        }
        require(row == ids.size(), "grain capture membership");
    }
    void compare_grain_fixture(const std::filesystem::path& manifest_path, std::string_view preset, const std::string& actual_words) {
        std::ifstream manifest_stream(manifest_path);
        require(bool(manifest_stream), "grain manifest open");
        const auto manifest = nlohmann::json::parse(manifest_stream);
        const auto expected_path = manifest_path.parent_path() / manifest.at("presets").at(std::string(preset)).at("ofx").get<std::string>();
        std::ifstream expected(expected_path);
        require(bool(expected), "grain fixture open");
        std::vector<std::string> original;
        std::vector<std::pair<std::size_t, std::size_t>> positions;
        std::size_t row = 0;
        for (std::string line; std::getline(expected, line); ++row) {
            std::istringstream words(line);
            std::size_t token = 0;
            for (std::string word; words >> word; ++token) {
                original.push_back(word);
                positions.emplace_back(row, token);
            }
        }
        const auto selected = PrintIlluminantFixtures::apply(expected_path, original);
        require(selected.size() == positions.size(), "grain overlay token count");
        std::istringstream actual(actual_words);
        std::string word;
        for (std::size_t i = 0; i < selected.size(); ++i) {
            const auto expected_word = selected.at(i).get<std::string>();
            if (!(actual >> word)) {
                std::ostringstream diagnostic;
                diagnostic << "grain fixture missing row=" << positions[i].first << " token=" << positions[i].second << " expected=" << expected_word;
                throw std::runtime_error(diagnostic.str());
            }
            if (word != expected_word) {
                std::ostringstream diagnostic;
                diagnostic << "grain fixture mismatch row=" << positions[i].first << " token=" << positions[i].second << " expected=" << expected_word << " actual=" << word;
                throw std::runtime_error(diagnostic.str());
            }
        }
        require(!(actual >> word), "extra grain fixture word");
    }
    std::vector<std::string> grain_edit_ids() {
        std::vector<std::string> ids;
        for (const char* route : {"direct", "print"}) {
            for (int choice : {-8, 0, 1, 2, 8}) {
                ids.push_back(std::string(route) + "/choice/" + std::to_string(choice));
                ids.push_back(std::string(route) + "/fallback/" + std::to_string(choice));
            }
            for (const char* suffix : {"master-ratios", "manual-custom", "linked-sharpness-texture", "reset-advanced"})
                ids.push_back(std::string(route) + "/" + suffix);
            for (int time = 0; time < 4; ++time)
                ids.push_back(std::string(route) + "/time/" + std::to_string(time));
        }
        return ids;
    }
    void grain_edit(OfxPlugin* plugin, ofx_test::InstanceHost& host, OfxImageEffectHandle effect, std::ostream& output, std::vector<std::unique_ptr<JuicerTests::CaseEvent>>& events) {
        const auto ids = grain_edit_ids();
        JuicerTests::Cohort rows("grain_edit", ids);
        std::size_t row = 0;
        for (int print : {0, 1}) {
            host.set_value(effect, "ScanRoute", {std::string(print ? "NegativePrintScan" : "NegativeDirectScan")});
            host.set_value(effect, "GrainActive", {1});
            host.set_value(effect, "GrainSublayersActive", {1});
            const auto record = [&](const std::string& label) {
                Observation observed(effect);
                require(observed.view.pending_kind == 4, "grain edit capture valid");
                Admission admitted(effect);
                output << "GRAIN_EDIT_CAPTURE " << print << ' ' << label;
                words(output, observed.view.captured.grain, admitted.view);
                Observation first(effect);
                Admission repeated(effect);
                Observation after(effect);
                require(after.view.build == first.view.build && after.view.backing_identity == first.view.backing_identity, "grain publication reuse");
            };
            for (int choice : {-8, 0, 1, 2, 8}) {
                {
                    events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", ids.at(row++)));
                    host.set_value(effect, "GrainPreset", {choice});
                    change(plugin, host, effect, "GrainPreset", kOfxChangeUserEdited);
                    record("preset-" + std::to_string(choice));
                }
                {
                    events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", ids.at(row++)));
                    change(plugin, host, effect, "GrainPreset", kOfxChangeTime);
                    change(plugin, host, effect, "GrainPreset", kOfxChangePluginEdited);
                    const double nan = std::numeric_limits<double>::quiet_NaN();
                    for (const char* name : {"GrainAmplitude", "GrainBlur", "GrainSharpness", "GrainChroma", "GrainTexture", "GrainParticleAreaUm2", "GrainSizeMixScale", "GrainSizeMixWeight", "GrainSizeMixWeightMid", "GrainBlurDyeCloudsUm", "GrainClumpTemporalMix", "GrainClumpMorphPeriodSec"})
                        host.set_value(effect, name, {nan});
                    for (const char* name : {"GrainParticleScale", "GrainParticleScaleLayers", "GrainDensityMin", "GrainUniformity"})
                        host.set_value(effect, name, {nan, nan, nan});
                    host.set_value(effect, "GrainMicroStructure", {nan, nan});
                    host.set_value(effect, "GrainDebugView", {choice});
                    change(plugin, host, effect, "GrainActive", kOfxChangeTime);
                    Observation observed(effect);
                    for (float value : observed.view.captured.grain.particle_scale_cmy)
                        require(value == 1.48f, "grain triplet fallback");
                    record("fallback-" + std::to_string(choice));
                }
            }
            {
                events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", ids.at(row++)));
                host.set_value(effect, "GrainPreset", {1});
                change(plugin, host, effect, "GrainPreset", kOfxChangeUserEdited);
                host.set_value(effect, "GrainParticleScale", {0.5, 1.0, 1.5});
                host.set_value(effect, "GrainParticleScaleMaster", {1.25});
                change(plugin, host, effect, "GrainParticleScaleMaster", kOfxChangeUserEdited);
                record("master-ratios");
            }
            {
                events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", ids.at(row++)));
                host.set_value(effect, "GrainParticleScale", {2.0, 3.0, 4.0});
                change(plugin, host, effect, "GrainParticleScale", kOfxChangeUserEdited);
                record("manual-custom");
            }
            {
                events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", ids.at(row++)));
                host.set_value(effect, "GrainSharpness", {0.25});
                change(plugin, host, effect, "GrainSharpness", kOfxChangeUserEdited);
                host.set_value(effect, "GrainTexture", {0.75});
                change(plugin, host, effect, "GrainTexture", kOfxChangeUserEdited);
                const auto lerp = [](double a, double b, double t) {
                    return a + (b - a) * t;
                };
                const auto blur = std::get<double>(host.value(effect, "GrainBlurDyeCloudsUm").at(0));
                const auto weight = std::get<double>(host.value(effect, "GrainSizeMixWeight").at(0));
                const auto micro = host.value(effect, "GrainMicroStructure");
                require(std::bit_cast<std::uint64_t>(blur) == std::bit_cast<std::uint64_t>(lerp(1.40, 0.60, 0.25)) &&
                            std::bit_cast<std::uint64_t>(weight) == std::bit_cast<std::uint64_t>(lerp(0.072, 0.38, 0.75)) &&
                            std::bit_cast<std::uint64_t>(std::get<double>(micro.at(0))) == std::bit_cast<std::uint64_t>(lerp(50.0, 70.0, 0.75)) &&
                            std::bit_cast<std::uint64_t>(std::get<double>(micro.at(1))) == std::bit_cast<std::uint64_t>(lerp(140.0, 200.0, 0.75)),
                        "native lerp identity");
                record("linked-sharpness-texture");
            }
            {
                events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", ids.at(row++)));
                change(plugin, host, effect, "GrainResetAdvanced", kOfxChangeUserEdited);
                record("reset-advanced");
            }
            host.set_value(effect, "GrainActive", {0});
            // NativeParam::value_at uses the unkeyed 0 EV before its first key at t=10.
            host.set_value(effect, "Exposure", {0.0});
            for (double time : {2.0, 0.0, 2.0, 2.0}) {
                events.push_back(std::make_unique<JuicerTests::CaseEvent>("grain_edit", ids.at(row++)));
                change(plugin, host, effect, "GrainActive", kOfxChangeTime, time);
                record(time == 0.0 ? "disabled-backward" : "disabled-repeat");
            }
        }
        require(row == ids.size(), "grain edit membership");
    }

    class IdentityReference final {
    public:
        IdentityReference(std::string preset, std::filesystem::path path)
            : preset_(std::move(preset)), path_(std::move(path)) {
            require(preset_ == "windows-clang-debug" || preset_ == "windows-clang-release" ||
                        preset_ == "linux-debug" || preset_ == "linux-release",
                    "identity reference preset");
            {
                std::ifstream input(path_, std::ios::binary);
                require(static_cast<bool>(input), "identity reference unavailable");
                Json record;
                input >> record;
                input >> std::ws;
                require(input.eof(), "identity reference trailing data");
                expected_ = decode(record);
                auto altered = record;
                altered[3][0][0] = "altered-label";
                bool rejected = false;
                try {
                    (void)decode(altered);
                } catch (const std::exception&) {
                    rejected = true;
                }
                require(rejected, "identity reference altered label accepted");
                altered = record;
                std::swap(altered[3][0], altered[3][1]);
                rejected = false;
                try {
                    (void)decode(altered);
                } catch (const std::exception&) {
                    rejected = true;
                }
                require(rejected, "identity reference reordered rows accepted");
                require(!matches(expected_.front(), expected_.front() ^ UINT64_C(1)), "identity reference altered value accepted");
            }
        }
        void observe(std::string_view label, std::uint64_t value) {
            const auto sites = membership();
            require(actual_.size() < sites.size(), "extra identity reference observation");
            const auto& expectedSite = sites[actual_.size()];
            require(expectedSite.first == label, "identity reference call order");
            {
                require(actual_.size() < expected_.size() && matches(expected_[actual_.size()], value),
                        "captured identity differs from frozen native reference");
                if (actual_.empty())
                    require(!matches(expected_[0] ^ UINT64_C(1), value), "candidate accepted altered identity reference");
            }
            actual_.push_back(value);
        }
        void finish() const {
            require(actual_.size() == membership().size(), "missing identity reference observation");
            require(actual_.size() == expected_.size(), "identity reference unconsumed rows");
        }

    private:
        using Json = nlohmann::json;
        using Site = std::pair<std::string, std::size_t>;
        static bool matches(std::uint64_t expected, std::uint64_t actual) noexcept {
            return expected == actual;
        }
        static std::vector<Site> membership() {
            constexpr std::array<std::pair<std::string_view, std::size_t>, 17> groups{{{"ordinary/profiles", 4}, {"ordinary/exposure", 4}, {"ordinary/identity", 3}, {"profiles/route", 4}, {"profiles/empty", 1}, {"gamma/print-direct-ignored", 1}, {"optics/diffusion-nan", 1}, {"optics/signed-zero", 1}, {"print/finite", 1}, {"print/glare-inactive", 1}, {"effects/all-off", 1}, {"effects/weave-double", 1}, {"identity/native-default", 1}, {"identity/mode", 3}, {"identity/effects-off", 1}, {"identity/signed-zero", 1}, {"identity/changed-exposure", 1}}};
            std::vector<Site> result;
            result.reserve(30);
            for (const auto& [label, count] : groups)
                for (std::size_t occurrence = 0; occurrence < count; ++occurrence)
                    result.emplace_back(label, occurrence);
            require(result.size() == 30, "identity reference declared membership");
            return result;
        }
        std::vector<std::uint64_t> decode(const Json& record) const {
            const auto presets = Json::array({"linux-debug", "linux-release", "windows-clang-debug", "windows-clang-release"});
            require(record.is_array() && record.size() == 4 &&
                        ((record[0] == "film-juicer-capture-identity-v1" && record[2] == preset_) ||
                         (record[0] == "film-juicer-capture-identity-v2" && record[2] == presets &&
                          std::find(presets.begin(), presets.end(), preset_) != presets.end())) &&
                        record[1] == "d4bc2135d02ed88c91403040c4429a4fdb802a2b" && record[3].is_array(),
                    "identity reference schema or provenance");
            const auto sites = membership();
            require(record[3].size() == sites.size(), "identity reference membership count");
            std::vector<std::uint64_t> result;
            result.reserve(sites.size());
            for (std::size_t i = 0; i < sites.size(); ++i) {
                const auto& row = record[3][i];
                require(row.is_array() && row.size() == 3 && row[0] == sites[i].first &&
                            row[1].is_number_unsigned() && row[1].get<std::uint64_t>() == sites[i].second &&
                            row[2].is_number_unsigned(),
                        "identity reference row or order");
                result.push_back(row[2].get<std::uint64_t>());
            }
            return result;
        }
        std::string preset_;
        std::filesystem::path path_;
        std::vector<std::uint64_t> expected_;
        std::vector<std::uint64_t> actual_;
    };

    void identity_comparison(OfxImageEffectHandle effect, IdentityReference& reference, std::string_view label) {
        Observation observed(effect);
        require(observed.view.pending_kind == 4, "identity capture valid");
        const JuicerTestSupport::PreparationInput input(observed.view.captured);
        require(JuicerTestSupport::hash_input_identity(input) == observed.view.input_identity, "captured input identity differs from independent record oracle");
        reference.observe(label, observed.view.input_identity);
    }

    void ordinary_cases(OfxPlugin* plugin, ofx_test::InstanceHost& host, OfxImageEffectHandle effect, IdentityReference& reference) {
        const auto set_double = [&](const char* name, double value) {
            host.set_value(effect, name, {value});
        };
        const auto set_int = [&](const char* name, int value) {
            host.set_value(effect, name, {value});
        };
        const auto capture = [&] {
            change(plugin, host, effect, "Exposure", kOfxChangeTime);
        };
        {
            JuicerTests::CaseEvent test("ofx.capture", "camera");
            set_double("CameraFilmFormatMm", 36.0);
            set_double("Exposure", std::numeric_limits<double>::infinity());
            capture();
            Observation observed(effect);
            require(observed.view.pending_kind == 4 && observed.view.captured.camera_film_format_long_edge_mm == 36.0f && observed.view.captured.camera_exposure_compensation_ev == 0.0, "camera capture");
            test.passed();
        }
        {
            JuicerTests::CaseEvent test("ofx.capture", "profiles");
            host.set_value(effect, "FilmProfileKey", {std::string()});
            host.set_value(effect, "PrintProfileKey", {std::string()});
            for (int positive : {0, 1}) {
                host.set_value(effect, "FilmProfileKey", {std::string(positive ? "fujifilm_provia_100f" : "kodak_portra_400")});
                for (int print : {0, 1}) {
                    host.set_value(effect, "ScanRoute", {std::string(print ? "NegativePrintScan" : "NegativeDirectScan")});
                    capture();
                    Observation observed(effect);
                    require(observed.view.pending_kind == 4 && observed.view.captured.scan_route == unsigned(2 * positive + print), "polarity preserves route selection");
                    identity_comparison(effect, reference, "ordinary/profiles");
                }
            }
            host.set_value(effect, "FilmProfileKey", {std::string("kodak_portra_400")});
            test.passed();
        }
        {
            JuicerTests::CaseEvent test("ofx.capture", "gamma");
            host.set_value(effect, "ScanRoute", {std::string("NegativeDirectScan")});
            set_double("FilmGammaFactor", 1.25);
            set_double("PrintGammaFactor", std::numeric_limits<double>::quiet_NaN());
            capture();
            {
                Observation observed(effect);
                require(observed.view.pending_kind == 4 && observed.view.captured.film_gamma_factor == 1.25f, "direct gamma ignores print gamma");
            }
            host.set_value(effect, "ScanRoute", {std::string("NegativePrintScan")});
            capture();
            {
                Observation observed(effect);
                require(observed.view.pending_kind == 5, "print gamma rejected");
            }
            set_double("PrintGammaFactor", 1.0);
            capture();
            test.passed();
        }
        {
            JuicerTests::CaseEvent test("ofx.capture", "optics");
            set_int("HalationActive", 0);
            set_double("HalationScatterAmount", std::numeric_limits<double>::quiet_NaN());
            host.clear_calls();
            capture();
            require(std::none_of(host.calls().begin(), host.calls().end(), [](const auto& call) {
                        return call.operation == "param_get" && call.target == "HalationScatterAmount";
                    }),
                    "disabled halation getter");
            set_int("HalationActive", 1);
            set_double("HalationScatterAmount", 1.0);
            capture();
            Observation observed(effect);
            require(observed.view.pending_kind == 4 && observed.view.captured.scatter_halation.active, "halation capture");
            set_int("HalationActive", 0);
            test.passed();
        }
        {
            JuicerTests::CaseEvent test("ofx.capture", "print");
            set_double("PrintExposure", std::numeric_limits<double>::quiet_NaN());
            set_double("EnlargerY", std::numeric_limits<double>::infinity());
            capture();
            Observation observed(effect);
            require(observed.view.pending_kind == 4 && observed.view.captured.print_exposure == 1.0 && observed.view.captured.print_ui_ymc_cc[0] == 0.0, "print fallback");
            test.passed();
        }
        {
            JuicerTests::CaseEvent test("ofx.capture", "dir");
            set_int("CouplersActive", 0);
            set_double("CouplersAmount", std::numeric_limits<double>::quiet_NaN());
            host.clear_calls();
            capture();
            require(std::none_of(host.calls().begin(), host.calls().end(), [](const auto& call) {
                        return call.operation == "param_get" && call.target == "CouplersAmount";
                    }),
                    "disabled DIR getter");
            set_int("CouplersActive", 1);
            set_double("CouplersAmount", 0.0);
            capture();
            Observation observed(effect);
            require(observed.view.pending_kind == 4 && observed.view.captured.dir_couplers.amount == 0.0f, "zero DIR amount");
            test.passed();
        }
        {
            JuicerTests::CaseEvent test("ofx.capture", "scanner");
            set_double("ScannerBlackLevel", -1.0);
            set_double("ScannerWhiteLevel", std::numeric_limits<double>::quiet_NaN());
            capture();
            Observation observed(effect);
            require(observed.view.pending_kind == 4 && observed.view.captured.scanner_black_level == 0.0 && observed.view.captured.scanner_white_level == 0.98, "scanner fallback");
            test.passed();
        }
        {
            JuicerTests::CaseEvent test("ofx.capture", "effects");
            set_double("FilmDustAmount", 20.0);
            set_double("GateWeaveAmount", -1.0);
            capture();
            Observation observed(effect);
            require(observed.view.pending_kind == 4 && observed.view.captured.film_dust_amount == 10.0f && observed.view.captured.gate_weave_amount == 0.0, "effects capture");
            test.passed();
        }
        {
            JuicerTests::CaseEvent test("ofx.capture", "identity");
            for (double exposure : {-0.0, 0.0, -1.25, 1.234567890123}) {
                set_double("Exposure", exposure);
                capture();
                identity_comparison(effect, reference, "ordinary/exposure");
            }
            host.set_value(effect, "ScanRoute", {std::string("NegativeDirectScan")});
            set_int("GrainActive", 0);
            set_int("SpectralUpsampling", 1);
            capture();
            const auto identity = [&] {
                Observation observed(effect);
                identity_comparison(effect, reference, "ordinary/identity");
                return observed.view.input_identity;
            };
            const auto baseline = identity();
            set_double("GrainAmplitude", 2.0);
            set_int("Hanatos2025AdaptationWindow", 0);
            set_int("InputCompression", 0);
            host.set_value(effect, "PrintProfileKey", {std::string("ignored-print-key")});
            capture();
            require(identity() == baseline, "direct and inactive inputs must not change identity");
            set_double("Exposure", 0.75);
            capture();
            require(identity() != baseline, "exposure changes identity");
            host.set_value(effect, "PrintProfileKey", {std::string()});
            for (std::int64_t counter : {0LL, 1LL, 2LL, 3LL, 1234567LL}) {
                require(fj_test_ofx_clock_conversion(counter, 10000000) == static_cast<std::uint64_t>(counter * 100), "10 MHz clock conversion");
                require(fj_test_ofx_clock_conversion(counter, 24000000) == static_cast<std::uint64_t>(counter * 41 + counter * 2 / 3), "24 MHz clock conversion");
                constexpr std::int64_t frequency = 3579545;
                const auto native = counter / frequency * 1000000000 + (counter % frequency) * 1000000000 / frequency;
                require(fj_test_ofx_clock_conversion(counter, frequency) == static_cast<std::uint64_t>(native), "general clock conversion");
            }
            test.passed();
        }
    }

    template <typename F>
    void row(const JuicerTests::Cohort& cohort, const char* id, F&& body) {
        auto event = cohort.case_event(id);
        body();
        event.passed();
    }
    void branch_matrix(OfxPlugin* plugin, ofx_test::InstanceHost& host, OfxImageEffectHandle effect, IdentityReference& reference) {
        const auto real = [&](const char* n, double v) {
            host.set_value(effect, n, {v});
        };
        const auto integer = [&](const char* n, int v) {
            host.set_value(effect, n, {v});
        };
        const auto text = [&](const char* n, const char* v) {
            host.set_value(effect, n, {std::string(v)});
        };
        const auto capture = [&] {
            change(plugin, host, effect, "Exposure", kOfxChangeTime);
        };
        const auto status = [&] {
            Observation o(effect);
            return o.view.pending_kind;
        };
        const auto read = [&](const char* name) {
            const auto& calls = host.calls();
            const auto start = std::find_if(calls.rbegin(), calls.rend(), [](const auto& call) {
                return call.operation == "param_get" && call.target == "HalationActive";
            });
            require(start != calls.rend(), "snapshot getter prefix");
            return std::any_of(start.base() - 1, calls.end(), [&](const auto& call) {
                return call.operation == "param_get" && call.target == name;
            });
        };
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();
        {
            JuicerTests::Cohort rows("capture.camera", {"format-min", "format-max", "format-nonfinite", "format-below", "format-above", "exposure-nonfinite"});
            row(rows, "format-min", [&] {
                real("CameraFilmFormatMm", 8);
                capture();
                Observation o(effect);
                require(o.view.pending_kind == 4 && o.view.captured.camera_film_format_long_edge_mm == 8.0f, "format minimum");
            });
            row(rows, "format-max", [&] {
                real("CameraFilmFormatMm", 120);
                capture();
                require(status() == 4, "format maximum");
            });
            row(rows, "format-nonfinite", [&] {
                for (double v : {nan, inf, -inf}) {
                    real("CameraFilmFormatMm", v);
                    capture();
                    require(status() == 5, "nonfinite format");
                }
            });
            row(rows, "format-below", [&] {
                real("CameraFilmFormatMm", std::nextafter(8.0, 0.0));
                capture();
                require(status() == 5, "format below");
            });
            row(rows, "format-above", [&] {
                real("CameraFilmFormatMm", std::nextafter(120.0, inf));
                capture();
                require(status() == 5, "format above");
            });
            row(rows, "exposure-nonfinite", [&] {
                real("CameraFilmFormatMm", 35);
                for (double v : {nan, inf, -inf}) {
                    real("Exposure", v);
                    capture();
                    Observation o(effect);
                    require(o.view.pending_kind == 4 && o.view.captured.camera_exposure_compensation_ev == 0.0, "exposure fallback");
                }
            });
        }
        {
            JuicerTests::Cohort rows("capture.profiles", {"negative-direct", "negative-print", "positive-direct", "positive-print", "empty", "unknown", "copied-string"});
            for (int positive : {0, 1})
                for (int print : {0, 1}) {
                    const char* id = positive ? (print ? "positive-print" : "positive-direct") : (print ? "negative-print" : "negative-direct");
                    row(rows, id, [&] {
                        text("FilmProfileKey", positive ? "fujifilm_provia_100f" : "kodak_portra_400");
                        text("ScanRoute", print ? "NegativePrintScan" : "NegativeDirectScan");
                        capture();
                        Observation o(effect);
                        require(o.view.pending_kind == 4 && o.view.captured.scan_route == unsigned(2 * positive + print), "route resolution");
                        identity_comparison(effect, reference, "profiles/route");
                    });
                }
            row(rows, "empty", [&] {
                text("FilmProfileKey", "");
                text("PrintProfileKey", "");
                text("ScanRoute", "");
                capture();
                Observation o(effect);
                require(o.view.pending_kind == 4 && o.view.captured.scan_route == 1, "empty profile defaults");
                identity_comparison(effect, reference, "profiles/empty");
            });
            row(rows, "unknown", [&] {
                text("FilmProfileKey", "unknown");
                text("ScanRoute", "unknown");
                capture();
                Observation o(effect);
                require(o.view.pending_kind == 4 && o.view.captured.scan_route == 1, "unknown profile fallback");
            });
            row(rows, "copied-string", [&] {
                host.reuse_string_getter_storage(true);
                text("FilmProfileKey", "fujifilm_provia_100f");
                text("PrintProfileKey", "kodak_portra_endura");
                text("ScanRoute", "NegativePrintScan");
                capture();
                Observation o(effect);
                const auto& p = o.view.captured;
                require(host.string_getter_reuses() > 0 && o.view.pending_kind == 4 &&
                            std::string_view(p.film_profile_key.data, p.film_profile_key.count) == "fujifilm_provia_100f" &&
                            std::string_view(p.print_profile_key.data, p.print_profile_key.count) == "kodak_portra_endura" &&
                            p.scan_route == 3,
                        "copied profile keys survive reused getter storage");
                Admission admitted(effect);
                require(admitted.view.route == 3, "copied profile keys reach preparation");
                host.reuse_string_getter_storage(false);
            });
            text("FilmProfileKey", "kodak_portra_400");
            text("ScanRoute", "NegativePrintScan");
        }
        {
            JuicerTests::Cohort rows("capture.gamma", {"film-bounds", "film-nonfinite", "print-direct-ignored", "print-route-invalid"});
            row(rows, "film-bounds", [&] {
                for (double v : {0.05, 4.0}) {
                    real("FilmGammaFactor", v);
                    capture();
                    require(status() == 4, "film gamma boundary");
                }
            });
            row(rows, "film-nonfinite", [&] {
                for (double v : {nan, inf, -inf}) {
                    real("FilmGammaFactor", v);
                    capture();
                    require(status() == 5, "film gamma nonfinite");
                }
                real("FilmGammaFactor", 1);
            });
            row(rows, "print-direct-ignored", [&] {
                text("ScanRoute", "NegativeDirectScan");
                real("PrintGammaFactor", nan);
                capture();
                require(status() == 4, "unused print gamma");
                identity_comparison(effect, reference, "gamma/print-direct-ignored");
            });
            row(rows, "print-route-invalid", [&] {
                text("ScanRoute", "NegativePrintScan");
                real("PrintGammaFactor", 1);
                capture();
                Admission held(effect);
                const auto held_hash = held.view.recipe_hash;
                real("PrintGammaFactor", nan);
                capture();
                require(status() == 5, "print gamma admission");
                FjPreparedState* blocked = nullptr;
                require(fj_test_ofx_instance_admit(effect, &blocked, nullptr).category != FJ_STATUS_SUCCESS && blocked == nullptr,
                        "invalid current controls block old publication");
                require(held.view.recipe_hash == held_hash, "already admitted publication stays alive");
                real("PrintGammaFactor", 1);
            });
        }
        {
            JuicerTests::Cohort rows("capture.optics", {"disabled-skips", "active-bounds", "diffusion-nan", "signed-zero"});
            row(rows, "disabled-skips", [&] {
                integer("HalationActive", 0);
                host.clear_calls();
                capture();
                require(status() == 4 && !read("HalationScatterAmount"), "disabled scatter getter");
            });
            row(rows, "active-bounds", [&] {
                integer("HalationActive", 1);
                for (double v : {0.0, 2.0}) {
                    real("HalationScatterAmount", v);
                    capture();
                    require(status() == 4, "scatter bounds");
                }
                real("HalationScatterAmount", std::nextafter(2.0, inf));
                capture();
                require(status() == 5, "scatter invalid");
                integer("HalationActive", 0);
            });
            row(rows, "diffusion-nan", [&] {
                real("CameraDiffusionStrength", nan);
                capture();
                require(status() == 4, "authored diffusion retains nan");
                identity_comparison(effect, reference, "optics/diffusion-nan");
            });
            row(rows, "signed-zero", [&] {
                real("CameraDiffusionStrength", -0.0);
                capture();
                Observation a(effect);
                auto first = a.view.input_identity;
                real("CameraDiffusionStrength", 0.0);
                capture();
                Observation b(effect);
                require(first == b.view.input_identity, "diffusion canonical zero");
                identity_comparison(effect, reference, "optics/signed-zero");
            });
        }
        {
            JuicerTests::Cohort rows("capture.print", {"finite", "nonfinite", "clamp", "glare-inactive"});
            row(rows, "finite", [&] {
                real("PrintExposure", 1.25);
                real("EnlargerY", -0.0);
                capture();
                Observation o(effect);
                require(o.view.pending_kind == 4 && o.view.captured.print_exposure == 1.25, "print finite");
                identity_comparison(effect, reference, "print/finite");
            });
            row(rows, "nonfinite", [&] {
                real("PrintExposure", nan);
                real("PrintPreflash", inf);
                capture();
                Observation o(effect);
                require(o.view.pending_kind == 4 && o.view.captured.print_exposure == 1.0 && o.view.captured.print_preflash_exposure == 0.0, "print defaults");
            });
            row(rows, "clamp", [&] {
                real("PrintShadowCompensationFactor", 4);
                real("PrintShadowCompensationDensity", -1);
                real("PrintShadowCompensationTransition", 3);
                capture();
                Observation o(effect);
                require(o.view.captured.print_shadow_compensation_factor == 1 && o.view.captured.print_shadow_compensation_density == 0 && o.view.captured.print_shadow_compensation_transition == 2, "print shadow clamp");
            });
            row(rows, "glare-inactive", [&] {
                integer("GlareActive", 0);
                real("GlarePercent", 0.25);
                capture();
                Observation a(effect);
                auto first = a.view.input_identity;
                real("GlarePercent", 0.5);
                capture();
                Observation b(effect);
                require(first != b.view.input_identity, "inactive glare retained in identity");
                identity_comparison(effect, reference, "print/glare-inactive");
            });
        }
        {
            JuicerTests::Cohort rows("capture.dir", {"inactive", "zero", "custom-gamma", "negative-donor", "positive-receiver", "tail-skips", "range-failure"});
            row(rows, "inactive", [&] {
                integer("CouplersActive", 0);
                host.clear_calls();
                capture();
                require(status() == 4 && !read("CouplersAmount"), "inactive DIR");
            });
            row(rows, "zero", [&] {
                integer("CouplersActive", 1);
                real("CouplersAmount", 0);
                host.clear_calls();
                capture();
                require(status() == 4 && !read("CouplersInhibitionSameLayer"), "zero DIR");
            });
            row(rows, "custom-gamma", [&] {
                real("CouplersAmount", 1);
                integer("CouplersGammaUseStock", 0);
                host.set_value(effect, "CouplersGammaSameLayerRgb", {0.5, 0.75, 1.0});
                capture();
                Observation o(effect);
                require(o.view.pending_kind == 4 && o.view.captured.dir_couplers.gamma_same_layer_rgb[0] == 0.5f, "custom gamma");
            });
            row(rows, "negative-donor", [&] {
                text("FilmProfileKey", "kodak_portra_400");
                host.set_value(effect, "CouplersLangmuirReceiverKRgb", {nan, nan, nan});
                host.clear_calls();
                capture();
                require(status() == 4 && read("CouplersLangmuirDonorKRgb") && !read("CouplersLangmuirReceiverKRgb"), "negative donor");
                host.set_value(effect, "CouplersLangmuirReceiverKRgb", {1.0, 1.0, 1.0});
            });
            row(rows, "positive-receiver", [&] {
                text("FilmProfileKey", "fujifilm_provia_100f");
                host.set_value(effect, "CouplersLangmuirDonorKRgb", {nan, nan, nan});
                host.clear_calls();
                capture();
                require(status() == 4 && read("CouplersLangmuirReceiverKRgb") && !read("CouplersLangmuirDonorKRgb"), "positive receiver");
                host.set_value(effect, "CouplersLangmuirDonorKRgb", {1.0, 1.0, 1.0});
                text("FilmProfileKey", "kodak_portra_400");
            });
            row(rows, "tail-skips", [&] {
                real("CouplersDiffusionSizeUm", 0);
                host.clear_calls();
                capture();
                require(status() == 4 && !read("CouplersDiffusionTailWeight") && !read("CouplersDiffusionTailUm"), "zero core skips tail");
                real("CouplersDiffusionSizeUm", 20);
                real("CouplersDiffusionTailWeight", 0);
                host.clear_calls();
                capture();
                require(status() == 4 && read("CouplersDiffusionTailWeight") && !read("CouplersDiffusionTailUm"), "zero weight skips tail");
            });
            row(rows, "range-failure", [&] {
                real("CouplersAmount", 3);
                capture();
                require(status() == 5, "DIR amount rejection");
                real("CouplersAmount", 0);
            });
        }
        {
            JuicerTests::Cohort rows("capture.scanner", {"clamp", "nonfinite", "recovery"});
            row(rows, "clamp", [&] {
                real("ScannerBlackLevel", -1);
                real("ScannerWhiteLevel", 2);
                capture();
                Observation o(effect);
                require(o.view.captured.scanner_black_level == 0 && o.view.captured.scanner_white_level == 1, "scanner clamps");
            });
            row(rows, "nonfinite", [&] {
                real("ScannerBlackLevel", nan);
                real("ScannerWhiteLevel", inf);
                capture();
                Observation o(effect);
                require(o.view.captured.scanner_black_level == 0.01 && o.view.captured.scanner_white_level == 0.98, "scanner defaults");
            });
            row(rows, "recovery", [&] {
                real("ScannerBlackLevel", 0.25);
                real("ScannerWhiteLevel", 0.75);
                capture();
                Observation o(effect);
                require(o.view.pending_kind == 4 && o.view.captured.scanner_black_level == 0.25 && o.view.captured.scanner_white_level == 0.75, "scanner recovery");
            });
        }
        {
            JuicerTests::Cohort rows("capture.effects", {"all-off", "clamp", "signed-zero", "weave-double"});
            row(rows, "all-off", [&] {
                for (const char* n : {"FilmDustAmount", "GateDustAmount", "FilmScratchAmount", "GateScratchAmount", "GateWeaveAmount"})
                    real(n, 0);
                capture();
                identity_comparison(effect, reference, "effects/all-off");
            });
            row(rows, "clamp", [&] {
                real("FilmDustAmount", 11);
                real("GateScratchAmount", -1);
                capture();
                Observation o(effect);
                require(o.view.captured.film_dust_amount == 10 && o.view.captured.gate_scratch_amount == 0, "effects clamps");
            });
            row(rows, "signed-zero", [&] {
                real("FilmDustAmount", -0.0);
                capture();
                Observation a(effect);
                auto first = a.view.input_identity;
                real("FilmDustAmount", 0.0);
                capture();
                Observation b(effect);
                require(first == b.view.input_identity, "effects zero canonicalization");
            });
            row(rows, "weave-double", [&] {
                real("GateWeaveAmount", 0.5);
                capture();
                Observation a(effect);
                auto first = a.view.input_identity;
                real("GateWeaveAmount", std::nextafter(0.5, 1.0));
                capture();
                Observation b(effect);
                require(first != b.view.input_identity, "weave exact double identity");
                identity_comparison(effect, reference, "effects/weave-double");
            });
        }
        {
            JuicerTests::Cohort rows("capture.identity", {"native-default", "print-only", "mode-0", "mode-1", "mode-2", "grain-inactive", "effects-off", "signed-zero", "changed-exposure"});
            row(rows, "native-default", [&] {
                auto* fresh = host.create_effect();
                require(action(plugin, kOfxActionCreateInstance, fresh) == kOfxStatOK, "default instance create");
                change(plugin, host, fresh, "Exposure", kOfxChangeTime);
                identity_comparison(fresh, reference, "identity/native-default");
                require(action(plugin, kOfxActionDestroyInstance, fresh) == kOfxStatOK, "default instance destroy");
            });
            row(rows, "print-only", [&] {
                text("ScanRoute", "NegativeDirectScan");
                capture();
                Observation a(effect);
                auto first = a.view.input_identity;
                text("PrintProfileKey", "ignored");
                capture();
                Observation b(effect);
                require(first == b.view.input_identity, "direct print key ignored");
                text("ScanRoute", "NegativePrintScan");
                capture();
                Observation print_with_key(effect);
                const auto print_identity = print_with_key.view.input_identity;
                text("PrintProfileKey", "");
                capture();
                Observation print_default(effect);
                require(print_identity != print_default.view.input_identity, "print key contributes on print route");
                text("ScanRoute", "NegativeDirectScan");
            });
            for (int mode : {0, 1, 2}) {
                const char* id = mode == 0 ? "mode-0" : mode == 1 ? "mode-1"
                                                                  : "mode-2";
                row(rows, id, [&] {
                    integer("SpectralUpsampling", mode);
                    integer("InputCompression", 0);
                    capture();
                    Observation a(effect);
                    auto first = a.view.input_identity;
                    integer("InputCompression", 1);
                    capture();
                    Observation b(effect);
                    require((first != b.view.input_identity) == (mode != 1), "compression mode identity");
                    integer("Hanatos2025AdaptationWindow", 0);
                    capture();
                    Observation c(effect);
                    const auto before_window = c.view.input_identity;
                    integer("Hanatos2025AdaptationWindow", 1);
                    capture();
                    Observation d(effect);
                    require((before_window != d.view.input_identity) == (mode == 0), "adaptation mode identity");
                    identity_comparison(effect, reference, "identity/mode");
                });
            }
            row(rows, "grain-inactive", [&] {
                integer("GrainActive", 0);
                real("GrainAmplitude", -1);
                capture();
                Observation a(effect);
                auto first = a.view.input_identity;
                real("GrainAmplitude", 2);
                capture();
                Observation b(effect);
                require(first == b.view.input_identity, "inactive grain identity");
            });
            row(rows, "effects-off", [&] {
                for (const char* n : {"FilmDustAmount", "GateDustAmount", "FilmScratchAmount", "GateScratchAmount", "GateWeaveAmount"})
                    real(n, 0);
                capture();
                identity_comparison(effect, reference, "identity/effects-off");
            });
            row(rows, "signed-zero", [&] {
                real("Exposure", -0.0);
                capture();
                Observation a(effect);
                auto first = a.view.input_identity;
                real("Exposure", 0.0);
                capture();
                Observation b(effect);
                require(first != b.view.input_identity, "exposure retains zero sign");
                integer("GrainActive", 1);
                real("GrainAmplitude", -0.0);
                real("GrainChroma", -0.0);
                capture();
                Observation grain(effect);
                const auto& controls = grain.view.captured.grain;
                const float native_chroma = static_cast<float>(-0.0);
                const float native_shared = std::sqrt(std::max(0.0f, 1.0f - native_chroma));
                const float native_independent = std::sqrt(std::max(0.0f, native_chroma));
                require(std::bit_cast<std::uint32_t>(controls.amplitude) == std::bit_cast<std::uint32_t>(static_cast<float>(std::exp2(-0.0))) &&
                            std::bit_cast<std::uint32_t>(controls.chroma_mix) == std::bit_cast<std::uint32_t>(native_chroma) &&
                            std::bit_cast<std::uint32_t>(controls.chroma_shared_weight) == std::bit_cast<std::uint32_t>(native_shared) &&
                            std::bit_cast<std::uint32_t>(controls.chroma_independent_weight) == std::bit_cast<std::uint32_t>(native_independent),
                        "negative zero grain primitive words");
                identity_comparison(effect, reference, "identity/signed-zero");
                integer("GrainActive", 0);
            });
            row(rows, "changed-exposure", [&] {
                real("Exposure", 0);
                capture();
                Observation a(effect);
                auto first = a.view.input_identity;
                real("Exposure", 0.5);
                capture();
                Observation b(effect);
                require(first != b.view.input_identity, "exposure identity");
                identity_comparison(effect, reference, "identity/changed-exposure");
            });
        }
    }

    std::uint64_t native_seed(std::uint64_t clock, std::uint64_t address) {
        const std::array<std::uint64_t, 2> fields{clock, address};
        const auto hash = Hash::hash_bytes(fields.data(), sizeof(fields));
        return hash == 0 ? 1 : hash;
    }
    struct Identity {
        std::uint64_t seed, token;
        std::uintptr_t address;
    };
    Identity instance_identity(OfxImageEffectHandle effect) {
        Identity result{};
        require(fj_test_ofx_instance_identity(effect, &result.seed, &result.token, &result.address) == 1, "instance identity observation");
        return result;
    }
    void identity_lifetime(OfxPlugin* plugin, ofx_test::InstanceHost& host) {
        JuicerTests::Cohort rows("capture.identity.lifetime", {"fixed-seed", "two-owners", "callback-stability", "reload"});
        row(rows, "fixed-seed", [&] {
            for (const auto& pair : std::array<std::array<std::uint64_t, 2>, 4>{{{0, 0}, {1, 2}, {0x0123456789abcdefULL, 0xfedcba9876543210ULL}, {~0ULL, 1}}})
                require(fj_test_ofx_seed(pair[0], pair[1]) == native_seed(pair[0], pair[1]), "native fixed seed");
        });
        constexpr std::uint64_t clock_a = 0x0123456789abcdefULL;
        constexpr std::uint64_t clock_b = 0xfedcba9876543210ULL;
        auto* first = host.create_effect();
        auto* second = host.create_effect();
        fj_test_ofx_clock_token(clock_a);
        require(action(plugin, kOfxActionCreateInstance, first) == kOfxStatOK, "first identity owner create");
        fj_test_ofx_clock_token(clock_b);
        require(action(plugin, kOfxActionCreateInstance, second) == kOfxStatOK, "second identity owner create");
        const auto a = instance_identity(first);
        const auto b = instance_identity(second);
        row(rows, "two-owners", [&] {
            require(a.address != 0 && b.address != 0 && a.address != b.address, "simultaneous owner addresses");
            require(a.seed == native_seed(clock_a, a.address) && a.token == a.seed, "first owner identity");
            require(b.seed == native_seed(clock_b, b.address) && b.token == b.seed, "second owner identity");
        });
        row(rows, "callback-stability", [&] {
            change(plugin, host, first, "Exposure", kOfxChangeTime);
            change(plugin, host, second, "Exposure", kOfxChangeTime);
            const auto after_a = instance_identity(first);
            const auto after_b = instance_identity(second);
            require(after_a.seed == a.seed && after_a.token == a.token && after_a.address == a.address, "first identity stable");
            require(after_b.seed == b.seed && after_b.token == b.token && after_b.address == b.address, "second identity stable");
        });
        require(action(plugin, kOfxActionDestroyInstance, first) == kOfxStatOK, "first identity owner destroy");
        require(action(plugin, kOfxActionDestroyInstance, second) == kOfxStatOK, "second identity owner destroy");
        row(rows, "reload", [&] {
            require(action(plugin, kOfxActionUnload) == kOfxStatOK, "identity unload");
            require(action(plugin, kOfxActionLoad) == kOfxStatOK, "identity reload");
            auto* fresh = host.create_effect();
            constexpr std::uint64_t reload_clock = 0x55aa55aa55aa55aaULL;
            fj_test_ofx_clock_token(reload_clock);
            require(action(plugin, kOfxActionCreateInstance, fresh) == kOfxStatOK, "reload owner create");
            const auto identity = instance_identity(fresh);
            require(identity.address != 0 && identity.seed == native_seed(reload_clock, identity.address) && identity.token == identity.seed, "reload owner identity");
            require(action(plugin, kOfxActionDestroyInstance, fresh) == kOfxStatOK, "reload owner destroy");
        });
    }

} // namespace

int main(int argc, char** argv) try {
    constexpr std::array cases{"camera", "profiles", "gamma", "optics", "print", "grain_capture", "grain_edit", "dir", "scanner", "effects", "identity"};
    if (JuicerTests::list_requested(argc, argv)) {
        for (const char* name : cases)
            JuicerTests::list_case("ofx.capture", name);
        return 0;
    }
    require(argc == 5 && std::string_view(argv[3]) == "--identity-reference", "usage: capture_test <grain manifest> <preset> --identity-reference path");
    IdentityReference reference(argv[2], argv[4]);
    JuicerTests::CaseLog log;
    ofx_test::InstanceHost host;
    auto* plugin = OfxGetPlugin(0);
    require(plugin != nullptr, "candidate plugin");
    plugin->setHost(host.binding());
    require(action(plugin, kOfxActionLoad) == kOfxStatOK, "plugin load");
    require(action(plugin, kOfxActionDescribe, host.descriptor_handle()) == kOfxStatOK, "plugin describe");
    require(action(plugin, kOfxImageEffectActionDescribeInContext, host.descriptor_handle(), host.context_args()) == kOfxStatOK, "plugin context");
    auto* effect = host.create_effect();
    require(action(plugin, kOfxActionCreateInstance, effect) == kOfxStatOK, "create instance");
    {
        JuicerTests::Cohort rows("capture.initial", {"uninitialized"});
        row(rows, "uninitialized", [&] {
            Observation initial(effect);
            require(initial.view.pending_kind == 0, "create leaves pending controls uninitialized");
        });
    }
    ordinary_cases(plugin, host, effect, reference);
    branch_matrix(plugin, host, effect, reference);
    require(action(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "destroy instance");
    effect = host.create_effect();
    require(action(plugin, kOfxActionCreateInstance, effect) == kOfxStatOK, "create grain instance");
    grain_fixture_inputs(host, effect);
    std::ostringstream grain_words;
    std::vector<std::unique_ptr<JuicerTests::CaseEvent>> events;
    events.reserve(234);
    JuicerTests::CaseEvent capture_case("ofx.capture", "grain_capture");
    grain_capture(plugin, host, effect, grain_words, events);
    JuicerTests::CaseEvent edit_case("ofx.capture", "grain_edit");
    grain_edit(plugin, host, effect, grain_words, events);
    require(events.size() == 234, "grain row membership");
    compare_grain_fixture(argv[1], argv[2], grain_words.str());
    for (auto& event : events)
        event->passed();
    capture_case.passed();
    edit_case.passed();
    require(action(plugin, kOfxActionDestroyInstance, effect) == kOfxStatOK, "destroy grain instance");
    identity_lifetime(plugin, host);
    require(action(plugin, kOfxActionUnload) == kOfxStatOK, "plugin unload");
    reference.finish();
    log.finish();
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "capture fixture: %s\n", error.what());
    return 1;
}
