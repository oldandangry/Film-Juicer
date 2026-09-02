#include "ScatterHalationValidation.h"

#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <cuda.h>
#include <cuda_runtime.h>

#include "SpectralProcessing.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "ProfileAssets.h"
#include "ScatterHalation.h"
#include "Cuda/Film/JuicerCudaScatterHalation.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "nlohmann/json.hpp"

namespace {

    using ScatterHalationValidation::Arguments;
    using ScatterHalationValidation::Results;

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
            } else if (option == "--fixture-root") {
                arguments.fixtureRoot = value;
            } else if (option == "--resource-root") {
                arguments.resourceRoot = value;
            } else if (option == "--scratch-root") {
                arguments.scratchRoot = value;
            } else if (option == "--performance-output") {
                arguments.performanceOutput = value;
            } else if (option == "--performance-mode") {
                arguments.performanceMode = value;
            } else if (option == "--device-index") {
                arguments.deviceIndex = std::stoi(value);
            } else if (option == "--width") {
                arguments.performanceWidth = std::stoi(value);
            } else if (option == "--height") {
                arguments.performanceHeight = std::stoi(value);
            } else if (option == "--warmup-count") {
                arguments.warmupCount = std::stoi(value);
            } else if (option == "--sample-count") {
                arguments.sampleCount = std::stoi(value);
            } else if (option == "--hold-milliseconds") {
                arguments.holdMilliseconds = std::stoi(value);
            } else {
                throw std::runtime_error("unknown option " + std::string(option));
            }
        }
        const bool knownGroup =
            arguments.caseGroup == "host-gate-1" ||
            arguments.caseGroup == "prepared-gate-2" ||
            arguments.caseGroup == "focused-cuda-gate-3" ||
            arguments.caseGroup == "unpublished-routes" ||
            arguments.caseGroup == "captured-carrier" ||
            arguments.caseGroup == "zero-work" ||
            arguments.caseGroup == "lifecycle" ||
            arguments.caseGroup == "performance" ||
            arguments.caseGroup == "all-agent";
        const bool fixtureRequired =
            arguments.caseGroup == "focused-cuda-gate-3" ||
            arguments.caseGroup == "captured-carrier" ||
            arguments.caseGroup == "performance" ||
            arguments.caseGroup == "all-agent";
        const bool performanceArgumentsValid =
            arguments.caseGroup != "performance" ||
            ((arguments.performanceMode == "active" ||
              arguments.performanceMode == "inactive-baseline") &&
             arguments.deviceIndex >= 0 &&
             arguments.performanceWidth > 0 &&
             arguments.performanceHeight > 0 &&
             arguments.warmupCount >= 0 && arguments.sampleCount > 0 &&
             arguments.holdMilliseconds >= 0 &&
             !arguments.performanceOutput.empty());
        if (!knownGroup ||
            arguments.resourceRoot.empty() ||
            arguments.scratchRoot.empty() ||
            (fixtureRequired && arguments.fixtureRoot.empty()) ||
            !performanceArgumentsValid) {
            throw std::runtime_error(
                "expected --case-group host-gate-1|prepared-gate-2|"
                "focused-cuda-gate-3|unpublished-routes|captured-carrier|"
                "zero-work|lifecycle|performance|all-agent "
                "[--fixture-root PATH] "
                "--resource-root PATH --scratch-root PATH");
        }
        return arguments;
    }

    template <typename T>
    std::uint32_t float_bits(T value) {
        return std::bit_cast<std::uint32_t>(static_cast<float>(value));
    }

    bool positive_zero(float value) {
        return float_bits(value) == 0u;
    }

    bool controls_equal(
        const ScatterHalationControls& left,
        const ScatterHalationControls& right) {
        return left.active == right.active &&
               float_bits(left.scatterAmount) == float_bits(right.scatterAmount) &&
               float_bits(left.scatterSpatialScale) ==
                   float_bits(right.scatterSpatialScale) &&
               float_bits(left.halationAmount) == float_bits(right.halationAmount) &&
               float_bits(left.halationSpatialScale) ==
                   float_bits(right.halationSpatialScale);
    }

    ScatterHalationControls convert_or_throw(const ScatterHalationRawControls& raw) {
        ScatterHalationControls controls;
        std::string diagnostic;
        if (!Spektrafilm::build_scatter_halation_controls(raw, controls, diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        return controls;
    }

    Profiles::ProfileDigest digest(
        std::array<float, 3> sigma,
        std::array<float, 3> strength) {
        Profiles::ProfileDigest result;
        result.halationFirstSigmaUm = sigma;
        result.halationPrimaryAmount = strength;
        return result;
    }

    ScatterHalationOpticsRecipe resolve_or_throw(
        const ScatterHalationControls& controls,
        const Profiles::ProfileDigest& profile) {
        ScatterHalationOpticsRecipe recipe;
        std::string diagnostic;
        if (!Spektrafilm::resolve_scatter_halation_recipe(
                controls,
                profile,
                recipe,
                diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        return recipe;
    }

    std::string active_recipe_detail(
        const ScatterHalationControls& controls,
        const Profiles::ProfileDigest& profile,
        const ScatterHalationOpticsRecipe& recipe) {
        std::ostringstream stream;
        stream << std::hex << std::setfill('0')
               << "control_bits=[0x" << std::setw(8) << float_bits(controls.scatterAmount)
               << ",0x" << std::setw(8) << float_bits(controls.scatterSpatialScale)
               << ",0x" << std::setw(8) << float_bits(controls.halationAmount)
               << ",0x" << std::setw(8) << float_bits(controls.halationSpatialScale)
               << "] profile_sigma_bits=[0x" << std::setw(8)
               << float_bits(profile.halationFirstSigmaUm[0]) << ",0x" << std::setw(8)
               << float_bits(profile.halationFirstSigmaUm[1]) << ",0x" << std::setw(8)
               << float_bits(profile.halationFirstSigmaUm[2])
               << "] profile_strength_bits=[0x" << std::setw(8)
               << float_bits(profile.halationPrimaryAmount[0]) << ",0x" << std::setw(8)
               << float_bits(profile.halationPrimaryAmount[1]) << ",0x" << std::setw(8)
               << float_bits(profile.halationPrimaryAmount[2]) << "] recipe_hash=0x"
               << std::setw(16) << recipe.hash;
        return stream.str();
    }

    void run_control_rows(Results& results) {
        const auto check_success = [&](std::string name, double value) {
            ScatterHalationRawControls raw;
            raw.active = true;
            raw.scatterAmount = value;
            raw.scatterSpatialScale = value;
            raw.halationAmount = value;
            raw.halationSpatialScale = value;
            ScatterHalationControls controls;
            std::string diagnostic;
            const bool success = Spektrafilm::build_scatter_halation_controls(
                raw,
                controls,
                diagnostic);
            const std::uint32_t expectedBits = value == 0.0 ? 0u : float_bits(value);
            results.record(
                std::move(name),
                success && controls.active &&
                    float_bits(controls.scatterAmount) == expectedBits &&
                    float_bits(controls.scatterSpatialScale) == expectedBits &&
                    float_bits(controls.halationAmount) == expectedBits &&
                    float_bits(controls.halationSpatialScale) == expectedBits,
                diagnostic);
        };
        check_success("controls/default-one", 1.0);
        check_success("controls/exact-zero", 0.0);
        check_success("controls/above-one-extrapolation", 1.25);
        check_success("controls/exact-two", 2.0);
        check_success("controls/smallest-positive-float", std::numeric_limits<float>::denorm_min());
        check_success("controls/negative-zero", -0.0);

        const auto check_failure = [&](std::string name, double value) {
            ScatterHalationRawControls raw;
            raw.active = true;
            raw.scatterAmount = value;
            ScatterHalationControls controls;
            std::string diagnostic;
            const bool success = Spektrafilm::build_scatter_halation_controls(
                raw,
                controls,
                diagnostic);
            results.record(
                std::move(name),
                !success && !diagnostic.empty(),
                diagnostic);
        };
        check_failure("controls/nan-rejected", std::numeric_limits<double>::quiet_NaN());
        check_failure("controls/positive-infinity-rejected", std::numeric_limits<double>::infinity());
        check_failure("controls/negative-infinity-rejected", -std::numeric_limits<double>::infinity());
        check_failure("controls/below-zero-rejected", -0.01);
        check_failure("controls/above-two-nextafter-rejected", std::nextafter(2.0, 3.0));

        ScatterHalationRawControls inactive;
        inactive.scatterAmount = std::numeric_limits<double>::quiet_NaN();
        inactive.scatterSpatialScale = -1.0;
        inactive.halationAmount = std::numeric_limits<double>::infinity();
        inactive.halationSpatialScale = std::nextafter(2.0, 3.0);
        ScatterHalationControls inactiveControls;
        std::string inactiveDiagnostic;
        const bool inactiveSuccess = Spektrafilm::build_scatter_halation_controls(
            inactive,
            inactiveControls,
            inactiveDiagnostic);
        results.record(
            "controls/inactive-ignores-subordinates",
            inactiveSuccess && !inactiveControls.active &&
                positive_zero(inactiveControls.scatterAmount) &&
                positive_zero(inactiveControls.scatterSpatialScale) &&
                positive_zero(inactiveControls.halationAmount) &&
                positive_zero(inactiveControls.halationSpatialScale),
            inactiveDiagnostic);

        const float retained = 1.0f;
        const double retainedDouble = static_cast<double>(retained);
        const double sameFloat = std::nextafter(
            retainedDouble,
            static_cast<double>(std::nextafter(retained, 2.0f)));
        const ScatterHalationControls first = convert_or_throw(
            ScatterHalationRawControls{true, retainedDouble, 1.0, 1.0, 1.0});
        const ScatterHalationControls second = convert_or_throw(
            ScatterHalationRawControls{true, sameFloat, 1.0, 1.0, 1.0});
        const ScatterHalationControls distinct = convert_or_throw(
            ScatterHalationRawControls{
                true,
                static_cast<double>(std::nextafter(retained, 2.0f)),
                1.0,
                1.0,
                1.0});
        const Profiles::ProfileDigest profile = digest(
            {65.0f, 65.0f, 65.0f},
            {0.08f, 0.02f, 0.0f});
        const auto firstRecipe = resolve_or_throw(first, profile);
        const auto secondRecipe = resolve_or_throw(second, profile);
        const auto distinctRecipe = resolve_or_throw(distinct, profile);
        results.record(
            "controls/same-float32-same-identity",
            controls_equal(first, second) && firstRecipe.hash == secondRecipe.hash,
            "canonical Float32 controls and recipe hashes must match");
        results.record(
            "controls/distinct-float32-distinct-identity",
            !controls_equal(first, distinct) && firstRecipe.hash != distinctRecipe.hash,
            "distinct retained Float32 controls must change recipe identity");
    }

    void run_recipe_rows(Results& results) {
        const ScatterHalationControls active = convert_or_throw(
            ScatterHalationRawControls{true, 1.0, 1.0, 1.0, 1.0});
        struct Mapping {
            const char* name;
            float sigma;
            std::array<float, 3> strength;
        };
        constexpr std::array<Mapping, 6> mappings{{
            {"recipe/still-strong", 65.0f, {0.015f, 0.005f, 0.0f}},
            {"recipe/still-weak", 65.0f, {0.08f, 0.02f, 0.0f}},
            {"recipe/still-no", 65.0f, {0.3f, 0.1f, 0.015f}},
            {"recipe/cine-strong", 50.0f, {0.015f, 0.005f, 0.0f}},
            {"recipe/cine-weak", 50.0f, {0.08f, 0.02f, 0.0f}},
            {"recipe/cine-no", 50.0f, {0.3f, 0.1f, 0.015f}},
        }};
        for (const Mapping& mapping : mappings) {
            const auto profile = digest(
                {mapping.sigma, mapping.sigma, mapping.sigma},
                mapping.strength);
            const auto recipe = resolve_or_throw(active, profile);
            bool passed = recipe.scatterActive && recipe.backReflectionActive &&
                          recipe.hash != 0 &&
                          recipe.halationFirstSigmaUm == profile.halationFirstSigmaUm;
            for (std::size_t channel = 0; channel < 3; ++channel) {
                passed = passed &&
                         float_bits(recipe.totalStrength[channel]) ==
                             float_bits(mapping.strength[channel]);
            }
            results.record(
                mapping.name,
                passed,
                active_recipe_detail(active, profile, recipe));
        }

        const ScatterHalationControls inactive;
        const auto inactiveRecipe = resolve_or_throw(
            inactive,
            digest({65.0f, 65.0f, 65.0f}, {0.08f, 0.02f, 0.0f}));
        results.record(
            "recipe/complete-identity-hash-zero",
            !inactiveRecipe.scatterActive && !inactiveRecipe.backReflectionActive &&
                inactiveRecipe.hash == 0,
            "complete no-work identity must reserve hash zero");

        const auto scatterOnlyControls = convert_or_throw(
            ScatterHalationRawControls{true, 1.0, 1.0, 0.0, 1.0});
        const auto scatterOnlyProfile =
            digest({65.0f, 65.0f, 65.0f}, {0.08f, 0.02f, 0.0f});
        const auto scatterOnly =
            resolve_or_throw(scatterOnlyControls, scatterOnlyProfile);
        results.record(
            "recipe/scatter-only",
            scatterOnly.scatterActive && !scatterOnly.backReflectionActive &&
                scatterOnly.hash != 0,
            active_recipe_detail(scatterOnlyControls, scatterOnlyProfile, scatterOnly));

        const auto reflectionOnlyControls = convert_or_throw(
            ScatterHalationRawControls{true, 0.0, 1.0, 1.0, 1.0});
        const auto reflectionOnlyProfile =
            digest({65.0f, 65.0f, 65.0f}, {0.08f, 0.02f, 0.0f});
        const auto reflectionOnly =
            resolve_or_throw(reflectionOnlyControls, reflectionOnlyProfile);
        results.record(
            "recipe/back-reflection-only",
            !reflectionOnly.scatterActive && reflectionOnly.backReflectionActive &&
                reflectionOnly.hash != 0,
            active_recipe_detail(
                reflectionOnlyControls,
                reflectionOnlyProfile,
                reflectionOnly));

        const auto crossChannelProfile =
            digest({0.0f, 65.0f, 0.0f}, {0.08f, 0.0f, 0.0f});
        const auto crossChannel = resolve_or_throw(
            active,
            crossChannelProfile);
        results.record(
            "recipe/cross-channel-activation",
            crossChannel.backReflectionActive && crossChannel.totalStrength[0] > 0.0f &&
                crossChannel.halationFirstSigmaUm[1] > 0.0f && crossChannel.hash != 0,
            active_recipe_detail(active, crossChannelProfile, crossChannel));

        results.record(
            "recipe/active-folded-zero-remap-reserved",
            scatterOnly.hash != 0 && reflectionOnly.hash != 0 && crossChannel.hash != 0,
            "every active focused recipe must avoid the reserved zero identity");
    }

    nlohmann::json load_json(const std::filesystem::path& path) {
        std::ifstream stream(path);
        if (!stream) {
            throw std::runtime_error("failed to open JSON fixture " + path.string());
        }
        nlohmann::json value;
        stream >> value;
        return value;
    }

    void write_json(const std::filesystem::path& path, const nlohmann::json& value) {
        std::ofstream stream(path, std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("failed to write JSON fixture " + path.string());
        }
        stream << value.dump(4) << '\n';
        if (!stream) {
            throw std::runtime_error("failed to finish JSON fixture " + path.string());
        }
    }

    std::shared_ptr<const Profiles::ValidatedFilmProfile> load_scratch_profile(
        const std::filesystem::path& path,
        const std::string& key,
        std::string& diagnostic) {
        Spektrafilm::ProfileCatalogEntry entry;
        entry.key = key;
        entry.label = key;
        entry.sourcePath = path.string();

        Spektrafilm::ProfileCatalog catalog;
        catalog.valid = true;
        catalog.filmProfiles.push_back(std::move(entry));

        Profiles::ProfileAssetStore store;
        return store.load_film_profile_by_key(catalog, key, &diagnostic);
    }

    void run_profile_rows(const Arguments& arguments, Results& results) {
        const std::filesystem::path source =
            arguments.resourceRoot / "profiles" / "kodak_portra_400.json";
        const nlohmann::json completeProfile = load_json(source);
        const std::filesystem::path profileRoot = arguments.scratchRoot / "profiles";
        std::filesystem::create_directories(profileRoot);

        struct Mapping {
            const char* use;
            const char* antihalation;
            float sigma;
            std::array<float, 3> strength;
        };
        constexpr std::array<Mapping, 6> mappings{{
            {"still", "strong", 65.0f, {0.015f, 0.005f, 0.0f}},
            {"still", "weak", 65.0f, {0.08f, 0.02f, 0.0f}},
            {"still", "no", 65.0f, {0.3f, 0.1f, 0.015f}},
            {"cine", "strong", 50.0f, {0.015f, 0.005f, 0.0f}},
            {"cine", "weak", 50.0f, {0.08f, 0.02f, 0.0f}},
            {"cine", "no", 50.0f, {0.3f, 0.1f, 0.015f}},
        }};
        for (const Mapping& mapping : mappings) {
            const std::string key =
                std::string("gate1_") + mapping.use + "_" + mapping.antihalation;
            nlohmann::json profile = completeProfile;
            profile["info"]["stock"] = key;
            profile["info"]["name"] = key;
            profile["info"]["use"] = mapping.use;
            profile["info"]["antihalation"] = mapping.antihalation;
            const std::filesystem::path path = profileRoot / (key + ".json");
            write_json(path, profile);

            std::string diagnostic;
            const auto loaded = load_scratch_profile(path, key, diagnostic);
            bool passed = loaded != nullptr;
            if (loaded) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    passed = passed &&
                             float_bits(loaded->digest.halationFirstSigmaUm[channel]) ==
                                 float_bits(mapping.sigma) &&
                             float_bits(loaded->digest.halationPrimaryAmount[channel]) ==
                                 float_bits(mapping.strength[channel]);
                }
            }
            results.record(
                "profile/" + std::string(mapping.use) + "-" + mapping.antihalation,
                passed,
                diagnostic);
        }

        nlohmann::json defaulted = completeProfile;
        const std::string defaultedKey = "gate1_missing_metadata_defaults";
        defaulted["info"]["stock"] = defaultedKey;
        defaulted["info"]["name"] = defaultedKey;
        defaulted["info"].erase("use");
        defaulted["info"].erase("antihalation");
        const std::filesystem::path defaultedPath =
            profileRoot / (defaultedKey + ".json");
        write_json(defaultedPath, defaulted);
        std::string diagnostic;
        const auto defaultedProfile =
            load_scratch_profile(defaultedPath, defaultedKey, diagnostic);
        results.record(
            "profile/missing-metadata-defaults-still-weak",
            defaultedProfile &&
                defaultedProfile->info.use == Profiles::ProfileUse::Still &&
                defaultedProfile->info.antihalation == Profiles::ProfileAntihalation::Weak &&
                defaultedProfile->digest.halationFirstSigmaUm ==
                    std::array<float, 3>{{65.0f, 65.0f, 65.0f}} &&
                defaultedProfile->digest.halationPrimaryAmount ==
                    std::array<float, 3>{{0.08f, 0.02f, 0.0f}},
            diagnostic);

        const auto check_unsupported = [&](const char* field, const char* value) {
            const std::string key = std::string("gate1_unsupported_") + field;
            nlohmann::json profile = completeProfile;
            profile["info"]["stock"] = key;
            profile["info"]["name"] = key;
            profile["info"][field] = value;
            const std::filesystem::path path = profileRoot / (key + ".json");
            write_json(path, profile);
            std::string failure;
            const auto loaded = load_scratch_profile(path, key, failure);
            results.record(
                "profile/unsupported-" + std::string(field) + "-rejected",
                !loaded && !failure.empty(),
                failure);
        };
        check_unsupported("use", "unsupported");
        check_unsupported("antihalation", "unsupported");
    }

    void run_descriptor_rows(Results& results) {
        const ScatterHalationControls active = convert_or_throw(
            ScatterHalationRawControls{true, 1.0, 1.0, 1.0, 1.0});
        const auto recipe = resolve_or_throw(
            active,
            digest({65.0f, 65.0f, 65.0f}, {0.08f, 0.02f, 0.0f}));
        std::optional<ScatterHalationFrameDescriptor> descriptor;
        std::string diagnostic;
        const bool success = Spektrafilm::build_scatter_halation_frame_descriptor(
            recipe,
            1.0f,
            descriptor,
            diagnostic);
        const bool scheduleKinds = success && descriptor &&
                                   descriptor->channels[0].core.kind ==
                                       ScatterHalationGaussianKind::FirReflect &&
                                   descriptor->channels[0].tail[0].kind ==
                                       ScatterHalationGaussianKind::YvvReplicate &&
                                   descriptor->channels[0].bounce[0].kind ==
                                       ScatterHalationGaussianKind::YvvReplicate;
        results.record(
            "descriptor/fir-yvv-dispatch",
            scheduleKinds,
            diagnostic);
        results.record(
            "descriptor/recipe-hash-binding",
            success && descriptor && descriptor->recipeHash == recipe.hash &&
                descriptor->recipeHash != 0,
            diagnostic);

        ScatterHalationOpticsRecipe identity;
        descriptor.emplace();
        diagnostic.clear();
        const bool absent = Spektrafilm::build_scatter_halation_frame_descriptor(
            identity,
            std::numeric_limits<float>::quiet_NaN(),
            descriptor,
            diagnostic);
        results.record(
            "descriptor/hash-zero-absence-before-pixel-validation",
            absent && !descriptor,
            diagnostic);

        results.record(
            "constants/retained-float-bits",
            float_bits(Spektrafilm::kScatterHalationTailWeights[0]) == float_bits(0.78f) &&
                float_bits(Spektrafilm::kScatterHalationTailWeights[1]) == float_bits(0.65f) &&
                float_bits(Spektrafilm::kScatterHalationTailWeights[2]) == float_bits(0.67f) &&
                float_bits(Spektrafilm::kScatterHalationExponentialAmplitudes[0]) ==
                    float_bits(0.1633f) &&
                float_bits(Spektrafilm::kScatterHalationExponentialAmplitudes[1]) ==
                    float_bits(0.6496f) &&
                float_bits(Spektrafilm::kScatterHalationExponentialAmplitudes[2]) ==
                    float_bits(0.1870f) &&
                float_bits(Spektrafilm::kScatterHalationBounceWeights[0]) ==
                    float_bits(4.0f / 7.0f) &&
                float_bits(Spektrafilm::kScatterHalationBounceWeights[1]) ==
                    float_bits(2.0f / 7.0f) &&
                float_bits(Spektrafilm::kScatterHalationBounceWeights[2]) ==
                    float_bits(1.0f / 7.0f),
            "fixed CUDA-visible constants retain frozen Float32 bit patterns");
    }

    ParamSnapshot direct_snapshot() {
        ParamSnapshot snapshot;
        snapshot.scanRoute = Spektrafilm::ScanRoute::NegativeDirectScan;
        return snapshot;
    }

    void seed_valid_pending(
        InstanceState& state,
        const ParamSnapshot& snapshot,
        std::optional<std::uint64_t> fullHash = std::nullopt) {
        std::lock_guard<std::mutex> lock(state.pending.m);
        state.pending.value = PendingParamsState::Valid{
            snapshot,
            fullHash.value_or(hash_params(snapshot))};
    }

    void seed_invalid_pending(InstanceState& state, std::string diagnostic) {
        std::lock_guard<std::mutex> lock(state.pending.m);
        state.pending.value =
            PendingParamsState::InvalidSnapshotControls{std::move(diagnostic)};
    }

    void run_build_product_rows(Results& results) {
        ScannerOutputRecipe scannerRecipe;
        scannerRecipe.route = Spektrafilm::ScanRoute::NegativePrintScan;
        scannerRecipe.glareActive = true;
        scannerRecipe.glarePercent = 0.1f;
        scannerRecipe.glareRoughness = 0.2f;
        scannerRecipe.glareBlurSigmaPx = 0.7f;
        scannerRecipe.lensBlurSigmaPx = 0.7f;
        scannerRecipe.unsharpSigmaPx = 0.7f;
        scannerRecipe.unsharpAmount = 0.7f;
        Scanner::ScannerPostEffectsDescriptor scannerDescriptor;
        std::string scannerDiagnostic;
        const bool scannerBuilt = Scanner::build_scanner_post_effects_descriptor(
            scannerRecipe,
            scannerDescriptor,
            scannerDiagnostic);
        results.record(
            "scanner-post/descriptor/truncate-three-radii",
            scannerBuilt && scannerDescriptor.glareBlurRadius == 2 &&
                scannerDescriptor.lensBlurRadius == 2 &&
                scannerDescriptor.unsharpRadius == 2,
            scannerDiagnostic);

        const ParamSnapshot direct = direct_snapshot();
        FocusedRenderStateBuildProduct directProduct;
        std::string directDiagnostic;
        const bool directBuilt = build_direct_render_state_product(
            direct,
            directProduct,
            directDiagnostic);
        results.record(
            "build-product/direct-complete",
            directBuilt && directProduct.recipe.directStructuralReady &&
                !directProduct.recipe.printStructuralReady &&
                directProduct.payload.uploadCoreHash != 0 &&
                directProduct.payload.scannerHash != 0,
            directDiagnostic);

        ParamSnapshot print;
        FocusedRenderStateBuildProduct printProduct;
        std::string printDiagnostic;
        const bool printBuilt = build_print_render_state_product(
            print,
            printProduct,
            printDiagnostic);
        results.record(
            "build-product/print-complete",
            printBuilt && printProduct.recipe.printStructuralReady &&
                !printProduct.recipe.directStructuralReady &&
                printProduct.payload.uploadCoreHash != 0 &&
                printProduct.payload.scannerHash != 0,
            printDiagnostic);

        ParamSnapshot requested = direct;
        requested.scatterHalationControls = convert_or_throw(
            ScatterHalationRawControls{true, 1.0, 1.0, 1.0, 1.0});
        FocusedRenderStateBuildProduct requestedProduct;
        std::string requestedDiagnostic;
        const bool requestedBuilt = build_direct_render_state_product(
            requested,
            requestedProduct,
            requestedDiagnostic);
        results.record(
            "build-product/requested-halation-complete-unpublished",
            requestedBuilt && requestedProduct.recipe.directStructuralReady &&
                requestedProduct.recipe.spatialOptics.scatterHalation.hash != 0,
            requestedDiagnostic);

        InstanceState publicationState;
        seed_valid_pending(publicationState, requested);
        const PendingRenderAdmissionResult admitted =
            admit_pending_render_state(publicationState);
        const ScatterHalationControls& defaultControls =
            requested.scatterHalationControls;
        results.record(
            "publication/default-active-one-admitted",
            defaultControls.active &&
                float_bits(defaultControls.scatterAmount) == float_bits(1.0f) &&
                float_bits(defaultControls.scatterSpatialScale) == float_bits(1.0f) &&
                float_bits(defaultControls.halationAmount) == float_bits(1.0f) &&
                float_bits(defaultControls.halationSpatialScale) == float_bits(1.0f) &&
                admitted.status == PendingRenderAdmissionStatus::AdmittedDirect &&
                admitted.directState && !admitted.printState &&
                admitted.directState->recipe.spatialOptics.scatterHalation.hash ==
                    requestedProduct.recipe.spatialOptics.scatterHalation.hash,
            admitted.diagnostic);
    }

    void run_pending_admission_rows(Results& results) {
        InstanceState uninitialized;
        const PendingRenderAdmissionResult uninitializedResult =
            admit_pending_render_state(uninitialized);
        results.record(
            "pending/uninitialized-needs-snapshot",
            uninitializedResult.status ==
                    PendingRenderAdmissionStatus::NeedsSnapshotAcquisition &&
                !uninitializedResult.directState && !uninitializedResult.printState,
            uninitializedResult.diagnostic);

        InstanceState invalidBeforeLoad;
        seed_invalid_pending(invalidBeforeLoad, "InvalidSnapshotControls before-load");
        const PendingRenderAdmissionResult invalidBefore =
            admit_pending_render_state(invalidBeforeLoad);
        results.record(
            "pending/invalid-before-load",
            invalidBefore.status ==
                    PendingRenderAdmissionStatus::InvalidSnapshotControls &&
                invalidBefore.diagnostic == "InvalidSnapshotControls before-load" &&
                invalidBeforeLoad.buildCounterNext.load(std::memory_order_relaxed) == 0,
            invalidBefore.diagnostic);

        const ParamSnapshot direct = direct_snapshot();
        InstanceState directState;
        seed_valid_pending(directState, direct);
        const PendingRenderAdmissionResult directResult =
            admit_pending_render_state(directState);
        results.record(
            "pending/valid-direct-capture",
            directResult.status == PendingRenderAdmissionStatus::AdmittedDirect &&
                directResult.directState && !directResult.printState &&
                directResult.snapshot.scanRoute ==
                    Spektrafilm::ScanRoute::NegativeDirectScan,
            directResult.diagnostic);

        ParamSnapshot print;
        InstanceState printState;
        seed_valid_pending(printState, print);
        const PendingRenderAdmissionResult printResult =
            admit_pending_render_state(printState);
        results.record(
            "pending/valid-print-capture",
            printResult.status == PendingRenderAdmissionStatus::AdmittedPrint &&
                printResult.printState && !printResult.directState &&
                Spektrafilm::scan_route_is_print(printResult.snapshot.scanRoute),
            printResult.diagnostic);

        InstanceState zeroHashState;
        seed_valid_pending(zeroHashState, direct, 0);
        const PendingRenderAdmissionResult zeroHash =
            admit_pending_render_state(zeroHashState);
        results.record(
            "pending/valid-hash-zero",
            zeroHash.status == PendingRenderAdmissionStatus::AdmittedDirect &&
                zeroHash.directState &&
                zeroHashState.lastHash.load(std::memory_order_acquire) == 0,
            zeroHash.diagnostic);

        InstanceState sameHashState;
        seed_valid_pending(sameHashState, direct);
        const PendingRenderAdmissionResult first =
            admit_pending_render_state(sameHashState);
        const std::uint64_t firstCounter =
            first.directState ? first.directState->buildCounter : 0;
        const PendingRenderAdmissionResult readyReuse =
            admit_pending_render_state(sameHashState);
        results.record(
            "pending/same-hash-ready-reuse",
            firstCounter != 0 && readyReuse.directState &&
                readyReuse.directState->buildCounter == firstCounter,
            readyReuse.diagnostic);

        JuicerAtomic::store_shared_ptr(
            &sameHashState.activeDirectState,
            std::shared_ptr<const DirectRenderState>{});
        const PendingRenderAdmissionResult notReadyRebuild =
            admit_pending_render_state(sameHashState);
        results.record(
            "pending/same-hash-not-ready-rebuild",
            notReadyRebuild.status == PendingRenderAdmissionStatus::AdmittedDirect &&
                notReadyRebuild.directState &&
                notReadyRebuild.directState->buildCounter > firstCounter,
            notReadyRebuild.diagnostic);

        InstanceState invalidAfterState;
        seed_valid_pending(invalidAfterState, direct);
        const PendingRenderAdmissionResult beforeInvalid =
            admit_pending_render_state(invalidAfterState);
        const std::uint64_t retainedCounter =
            beforeInvalid.directState ? beforeInvalid.directState->buildCounter : 0;
        seed_invalid_pending(
            invalidAfterState,
            "InvalidSnapshotControls after-linearization");
        const PendingRenderAdmissionResult invalidAfter =
            admit_pending_render_state(invalidAfterState);
        const auto retainedState =
            JuicerAtomic::load_shared_ptr(&invalidAfterState.activeDirectState);
        results.record(
            "pending/invalid-after-linearization-retains-publication",
            invalidAfter.status ==
                    PendingRenderAdmissionStatus::InvalidSnapshotControls &&
                retainedState && retainedState->buildCounter == retainedCounter,
            invalidAfter.diagnostic);

        seed_valid_pending(invalidAfterState, direct);
        const PendingRenderAdmissionResult equalValid =
            admit_pending_render_state(invalidAfterState);
        results.record(
            "pending/invalid-to-equal-valid-reuse",
            equalValid.status == PendingRenderAdmissionStatus::AdmittedDirect &&
                equalValid.directState &&
                equalValid.directState->buildCounter == retainedCounter,
            equalValid.diagnostic);

        InstanceState retryState;
        ParamSnapshot edited = direct;
        edited.cameraFilmFormatLongEdgeMm = 36.0f;
        seed_valid_pending(retryState, direct);
        PendingRenderAdmissionResult retryResult;
        std::atomic<bool> admissionStarted{false};
        {
            std::unique_lock<std::mutex> holdRebuild(retryState.rebuildMutex);
            std::thread admission([&] {
                admissionStarted.store(true, std::memory_order_release);
                retryResult = admit_pending_render_state(retryState);
            });
            while (!admissionStarted.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            seed_valid_pending(retryState, edited);
            holdRebuild.unlock();
            admission.join();
        }
        results.record(
            "pending/edit-during-rebuild-retries-current",
            retryResult.status == PendingRenderAdmissionStatus::AdmittedDirect &&
                retryResult.directState &&
                retryResult.snapshot.cameraFilmFormatLongEdgeMm == 36.0f &&
                retryState.lastHash.load(std::memory_order_acquire) ==
                    hash_params(edited) &&
                retryState.buildCounterNext.load(std::memory_order_relaxed) >= 2,
            retryResult.diagnostic);
    }

    void run_camera_hash_rows(Results& results) {
        ParamSnapshot minimum;
        minimum.cameraFilmFormatLongEdgeMm = 8.0f;
        ParamSnapshot standard = minimum;
        standard.cameraFilmFormatLongEdgeMm = 35.0f;
        ParamSnapshot maximum = minimum;
        maximum.cameraFilmFormatLongEdgeMm = 120.0f;
        results.record(
            "camera-format/canonical-bounds-and-default-hash",
            hash_params(minimum) != hash_params(standard) &&
                hash_params(standard) != hash_params(maximum) &&
                hash_params(minimum) != hash_params(maximum),
            "8, 35, and 120 mm exact-bit lanes must remain distinct");

        ParamSnapshot same = standard;
        ParamSnapshot distinct = standard;
        distinct.cameraFilmFormatLongEdgeMm = std::nextafter(35.0f, 36.0f);
        results.record(
            "camera-format/same-vs-distinct-float32-hash",
            hash_params(standard) == hash_params(same) &&
                hash_params(standard) != hash_params(distinct),
            "snapshot hashing consumes the canonical Float32 exactly");

        InstanceState state;
        {
            std::lock_guard<std::mutex> lock(state.pending.m);
            state.pending.value = PendingParamsState::InvalidSnapshotControls{
                "InvalidSnapshotControls field=camera_film_format_long_edge_mm"};
        }
        const PendingRenderAdmissionResult invalid = admit_pending_render_state(state);
        results.record(
            "camera-format/invalid-alternative-admission",
            invalid.status == PendingRenderAdmissionStatus::InvalidSnapshotControls &&
                !invalid.diagnostic.empty(),
            invalid.diagnostic);

        InstanceState publicationState;
        seed_valid_pending(publicationState, standard);
        const PendingRenderAdmissionResult first =
            admit_pending_render_state(publicationState);
        const std::uint64_t firstCounter =
            first.printState ? first.printState->buildCounter : 0;
        seed_valid_pending(publicationState, same);
        const PendingRenderAdmissionResult sameResult =
            admit_pending_render_state(publicationState);
        const std::uint64_t sameCounter =
            sameResult.printState ? sameResult.printState->buildCounter : 0;
        seed_valid_pending(publicationState, distinct);
        const PendingRenderAdmissionResult distinctResult =
            admit_pending_render_state(publicationState);
        results.record(
            "camera-format/same-float32-reuse-distinct-rebuild",
            firstCounter != 0 && sameCounter == firstCounter &&
                distinctResult.status == PendingRenderAdmissionStatus::AdmittedPrint &&
                distinctResult.printState &&
                distinctResult.printState->buildCounter > firstCounter,
            distinctResult.diagnostic);
    }

    struct PreparedInputs {
        FocusedRenderStateBuildProduct product;
        Scanner::ScannerSpectralLutDescriptor scannerDescriptor{};
        std::optional<ScatterHalationFrameDescriptor> scatterDescriptor;
        std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusionFrameSet;

        JuicerProcess::Root::CudaFramePreparationRequest request(
            int width,
            int height) const {
            JuicerProcess::Root::CudaFramePreparationRequest value{};
            value.recipe = &product.recipe;
            value.exposureTables = &product.payload.exposureTables;
            value.spdSInv = product.payload.spdSInv.data();
            value.filmRawConfig = &product.payload.filmRawConfig;
            value.scannerTables = &product.payload.scannerTables;
            value.scannerColor = &product.payload.scannerColor;
            value.scannerLutDescriptor = &scannerDescriptor;
            value.diffusionFrameSetDescriptor =
                diffusionFrameSet ? &*diffusionFrameSet : nullptr;
            value.scatterHalationDescriptor =
                scatterDescriptor ? &*scatterDescriptor : nullptr;
            value.requestedWidth = width;
            value.requestedHeight = height;
            return value;
        }
    };

    bool initialize_route_inputs(
        Spektrafilm::ScanRoute route,
        const ScatterHalationControls& scatterHalationControls,
        const Spektrafilm::DiffusionFilterAuthoredControls& cameraDiffusion,
        float pixelSizeUm,
        int width,
        int height,
        PreparedInputs& out,
        std::string& diagnostic) {
        out = PreparedInputs{};
        ParamSnapshot snapshot;
        snapshot.scanRoute = route;
        snapshot.scatterHalationControls = scatterHalationControls;
        snapshot.cameraDiffusion = cameraDiffusion;
        snapshot.gateWeaveAmount = 0.0;
        if (Spektrafilm::scan_route_metadata(route).capturePolarity ==
            Spektrafilm::ProfilePolarity::Positive) {
            snapshot.filmProfileKey = "fujifilm_provia_100f";
        }
        const bool printRoute = Spektrafilm::scan_route_is_print(route);
        const bool productBuilt =
            printRoute
                ? build_print_render_state_product(
                      snapshot,
                      out.product,
                      diagnostic)
                : build_direct_render_state_product(
                      snapshot,
                      out.product,
                      diagnostic);
        if (!productBuilt) {
            return false;
        }
        if (!Spektrafilm::build_scatter_halation_frame_descriptor(
                out.product.recipe.spatialOptics.scatterHalation,
                pixelSizeUm,
                out.scatterDescriptor,
                diagnostic)) {
            return false;
        }
        const Scanner::DirectScannerSpectralLutDescriptorInput directInput{
            &out.product.recipe.profileRoute,
            &out.product.recipe.densityBounds,
            &out.product.recipe.scannerOutput};
        const Scanner::PrintScannerSpectralLutDescriptorInput printInput{
            &out.product.recipe.profileRoute,
            &out.product.recipe.densityBounds,
            &out.product.recipe.scannerOutput};
        const bool scannerBuilt =
            printRoute
                ? Scanner::build_print_scanner_spectral_lut_descriptor(
                      printInput,
                      out.scannerDescriptor,
                      diagnostic)
                : Scanner::build_direct_scanner_spectral_lut_descriptor(
                      directInput,
                      out.scannerDescriptor,
                      diagnostic);
        if (!scannerBuilt) {
            return false;
        }
        if (cameraDiffusion.active &&
            !Spektrafilm::build_diffusion_frame_set_descriptor(
                out.product.recipe.spatialOptics,
                route,
                static_cast<double>(pixelSizeUm),
                Spektrafilm::DiffusionFrameDomain{0, 0, width, height},
                out.diffusionFrameSet,
                diagnostic)) {
            return false;
        }
        return !cameraDiffusion.active || out.diffusionFrameSet.has_value();
    }

    bool initialize_prepared_inputs(
        bool printRoute,
        bool cameraDiffusion,
        bool scatterHalation,
        int width,
        int height,
        PreparedInputs& out,
        std::string& diagnostic) {
        ScatterHalationControls controls{};
        if (scatterHalation) {
            controls = convert_or_throw(
                ScatterHalationRawControls{true, 1.0, 1.0, 1.0, 1.0});
        }
        Spektrafilm::DiffusionFilterAuthoredControls cameraControls{};
        cameraControls.active = cameraDiffusion;
        if (!initialize_route_inputs(
                printRoute
                    ? Spektrafilm::ScanRoute::NegativePrintScan
                    : Spektrafilm::ScanRoute::NegativeDirectScan,
                controls,
                cameraControls,
                10.0f,
                width,
                height,
                out,
                diagnostic)) {
            return false;
        }
        return out.scatterDescriptor.has_value() == scatterHalation;
    }

    class CudaPreparedFixture final {
    public:
        bool initialize(std::string& diagnostic) {
            diagnostic.clear();
            cudaError_t runtimeStatus = cudaSetDevice(0);
            if (runtimeStatus == cudaSuccess) {
                runtimeStatus = cudaFree(nullptr);
            }
            if (runtimeStatus != cudaSuccess) {
                diagnostic = std::string("CUDA runtime initialization failed: ") +
                             cudaGetErrorString(runtimeStatus);
                return false;
            }
            CUcontext context = nullptr;
            const CUresult contextStatus = cuCtxGetCurrent(&context);
            if (contextStatus != CUDA_SUCCESS || !context) {
                diagnostic = "cuCtxGetCurrent did not return the active primary context";
                return false;
            }
            _contextKey.deviceId = 0;
            _contextKey.contextOpaque = context;
            runtimeStatus = cudaStreamCreateWithFlags(
                &_stream,
                cudaStreamNonBlocking);
            if (runtimeStatus != cudaSuccess || !_stream) {
                diagnostic = std::string("cudaStreamCreateWithFlags failed: ") +
                             cudaGetErrorString(runtimeStatus);
                return false;
            }
            _instanceToken =
                0x48414c4154494f4eull + s_nextInstance.fetch_add(1);
            return true;
        }

        ~CudaPreparedFixture() {
            if (_stream) {
                (void)cudaStreamDestroy(_stream);
            }
        }

        const JuicerCuda::ResourceManager::DeviceContextKey& context_key() const {
            return _contextKey;
        }

        void* stream_opaque() const {
            return _stream;
        }

        bool synchronize(std::string& diagnostic) const {
            const cudaError_t status = cudaStreamSynchronize(_stream);
            if (status == cudaSuccess) {
                return true;
            }
            diagnostic = std::string("cudaStreamSynchronize failed: ") +
                         cudaGetErrorString(status);
            return false;
        }

        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot(
            const PreparedInputs& inputs) {
            JuicerCuda::ResourceManager::SubmissionSnapshot value{};
            value.instanceToken.value = _instanceToken;
            value.frameToken.value = _nextIdentity;
            value.snapshotId = _nextIdentity++;
            value.deviceContextKey = _contextKey;
            value.contextEpoch = 1;
            value.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                inputs.product.payload.uploadCoreHash,
                inputs.product.recipe.dirCouplers.hash,
                inputs.product.payload.scannerHash,
                0);
            return value;
        }

    private:
        inline static std::atomic<std::uint64_t> s_nextInstance{1};
        JuicerCuda::ResourceManager::DeviceContextKey _contextKey{};
        cudaStream_t _stream = nullptr;
        std::uint64_t _instanceToken = 0;
        std::uint64_t _nextIdentity = 1;
    };

    bool contains_text(const std::string& text, std::string_view expected) {
        return text.find(expected) != std::string::npos;
    }

    std::string prepared_shape_detail(
        int width,
        int height,
        bool dedicated) {
        const std::size_t planeBytes = static_cast<std::size_t>(width) *
                                       static_cast<std::size_t>(height) *
                                       sizeof(float);
        const std::size_t filterBytes = 2u * planeBytes;
        const std::size_t carrierBytes = dedicated ? 3u * planeBytes : 0u;
        return "plane_bytes=" + std::to_string(planeBytes) +
               " filter_bytes=" + std::to_string(filterBytes) +
               " carrier_bytes=" + std::to_string(carrierBytes) +
               " requested_bytes=" +
               std::to_string(filterBytes + carrierBytes) +
               " physical_allocation_count=" +
               std::to_string(dedicated ? 2 : 1);
    }

    void run_prepared_gate_2_rows(Results& results) {
        constexpr int kWidth = 32;
        constexpr int kHeight = 24;
        const std::uintptr_t planeBytes =
            static_cast<std::uintptr_t>(kWidth) * kHeight * sizeof(float);

        CudaPreparedFixture fixture;
        std::string diagnostic;
        if (!fixture.initialize(diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        JuicerProcess::Root& root = JuicerProcess::root();

        PreparedInputs zeroInputs;
        if (!initialize_prepared_inputs(
                true,
                false,
                false,
                kWidth,
                kHeight,
                zeroInputs,
                diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        auto zeroRequest = zeroInputs.request(kWidth, kHeight);
        auto zeroFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(zeroInputs),
            zeroRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const auto zeroView = zeroFrame.scatter_halation_resources();
        results.record(
            "prepared/zero-no-descriptor",
            zeroFrame.active() && zeroView.descriptor == nullptr,
            diagnostic);
        std::string terminalDiagnostic;
        const bool zeroFinished =
            zeroFrame.finish(fixture.stream_opaque(), terminalDiagnostic) &&
            fixture.synchronize(terminalDiagnostic);
        results.record(
            "prepared/finish-cleanup-zero",
            zeroFinished &&
                zeroFrame.scatter_halation_resources().descriptor == nullptr,
            terminalDiagnostic);

        PreparedInputs dedicatedInputs;
        if (!initialize_prepared_inputs(
                false,
                false,
                true,
                kWidth,
                kHeight,
                dedicatedInputs,
                diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        auto dedicatedRequest = dedicatedInputs.request(kWidth, kHeight);
        auto dedicatedFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            dedicatedRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const auto dedicatedView = dedicatedFrame.scatter_halation_resources();
        const bool dedicatedOffsets =
            dedicatedView.descriptor && dedicatedView.filterTemp &&
            dedicatedView.weightedAccumulation &&
            dedicatedView.currentCarrier.redSensitive &&
            dedicatedView.currentCarrier.greenSensitive &&
            dedicatedView.currentCarrier.blueSensitive &&
            reinterpret_cast<std::uintptr_t>(
                dedicatedView.weightedAccumulation) -
                    reinterpret_cast<std::uintptr_t>(
                        dedicatedView.filterTemp) ==
                planeBytes &&
            reinterpret_cast<std::uintptr_t>(
                dedicatedView.currentCarrier.greenSensitive) -
                    reinterpret_cast<std::uintptr_t>(
                        dedicatedView.currentCarrier.redSensitive) ==
                planeBytes &&
            reinterpret_cast<std::uintptr_t>(
                dedicatedView.currentCarrier.blueSensitive) -
                    reinterpret_cast<std::uintptr_t>(
                        dedicatedView.currentCarrier.greenSensitive) ==
                planeBytes;
        results.record(
            "prepared/dedicated-carrier",
            dedicatedFrame.active() &&
                dedicatedView.descriptor->recipeHash ==
                    dedicatedInputs.product.recipe.spatialOptics.scatterHalation.hash &&
                dedicatedView.carrierSource ==
                    JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes &&
                dedicatedView.fullFrameWidth == kWidth &&
                dedicatedView.fullFrameHeight == kHeight && dedicatedOffsets &&
                dedicatedView.currentCarrier.rowStrideFloats == kWidth &&
                dedicatedView.filterTemp !=
                    dedicatedView.currentCarrier.redSensitive,
            prepared_shape_detail(kWidth, kHeight, true));
        terminalDiagnostic.clear();
        const bool dedicatedFinished =
            dedicatedFrame.finish(fixture.stream_opaque(), terminalDiagnostic) &&
            fixture.synchronize(terminalDiagnostic);
        results.record(
            "prepared/finish-cleanup",
            dedicatedFinished &&
                dedicatedFrame.scatter_halation_resources().descriptor == nullptr,
            terminalDiagnostic);

        PreparedInputs cameraInputs;
        if (!initialize_prepared_inputs(
                false,
                true,
                true,
                kWidth,
                kHeight,
                cameraInputs,
                diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        auto cameraRequest = cameraInputs.request(kWidth, kHeight);
        auto cameraFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(cameraInputs),
            cameraRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const auto cameraView = cameraFrame.scatter_halation_resources();
        const auto diffusionView = cameraFrame.diffusion_resources();
        results.record(
            "prepared/camera-diffusion-carrier",
            cameraFrame.active() && cameraView.descriptor &&
                cameraView.carrierSource ==
                    JuicerCuda::ScatterHalationCarrierSource::CameraDiffusionStagePlanes &&
                !cameraView.currentCarrier.redSensitive &&
                !cameraView.currentCarrier.greenSensitive &&
                !cameraView.currentCarrier.blueSensitive &&
                cameraView.currentCarrier.rowStrideFloats == 0 &&
                cameraView.filterTemp && cameraView.weightedAccumulation &&
                diffusionView.active &&
                diffusionView.executionDescriptor.contextEpoch != 0,
            prepared_shape_detail(kWidth, kHeight, false));
        terminalDiagnostic.clear();
        const bool cameraFinished =
            cameraFrame.finish(fixture.stream_opaque(), terminalDiagnostic) &&
            fixture.synchronize(terminalDiagnostic);
        results.record(
            "prepared/camera-finish-cleanup",
            cameraFinished &&
                cameraFrame.scatter_halation_resources().descriptor == nullptr &&
                !cameraFrame.diffusion_resources().active,
            terminalDiagnostic);

        auto cameraOverlapFirst = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(cameraInputs),
            cameraRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const std::string cameraOverlapFirstDiagnostic = diagnostic;
        auto cameraOverlapSecond = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(cameraInputs),
            cameraRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const std::string cameraOverlapSecondDiagnostic = diagnostic;
        const auto cameraOverlapFirstView =
            cameraOverlapFirst.scatter_halation_resources();
        const auto cameraDiffusionFirst = cameraOverlapFirst.diffusion_resources();
        const bool cameraLeaseAdmissionRejected =
            cameraOverlapFirst.active() && cameraOverlapFirstView.descriptor &&
            cameraDiffusionFirst.active && !cameraOverlapSecond.active() &&
            cameraOverlapSecond.scatter_halation_resources().descriptor == nullptr &&
            contains_text(
                cameraOverlapSecondDiagnostic,
                "field=serialized_host_lease");
        results.record(
            "prepared/camera-overlap-lease-admission",
            cameraLeaseAdmissionRejected,
            "first=" + cameraOverlapFirstDiagnostic +
                " second=" + cameraOverlapSecondDiagnostic);
        terminalDiagnostic.clear();
        const bool cameraOverlapFinished =
            cameraOverlapFirst.finish(
                fixture.stream_opaque(),
                terminalDiagnostic) &&
            fixture.synchronize(terminalDiagnostic);
        results.record(
            "prepared/camera-overlap-finish-cleanup",
            cameraOverlapFinished,
            terminalDiagnostic);

        auto absentRequest = dedicatedInputs.request(kWidth, kHeight);
        absentRequest.scatterHalationDescriptor = nullptr;
        auto absentFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            absentRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        results.record(
            "prepared/nonzero-recipe-absent-descriptor",
            !absentFrame.active() &&
                contains_text(diagnostic, "MissingRequiredResource") &&
                contains_text(diagnostic, "field=descriptor") &&
                absentFrame.scatter_halation_resources().descriptor == nullptr,
            diagnostic);

        auto presentForZeroRequest = zeroInputs.request(kWidth, kHeight);
        presentForZeroRequest.scatterHalationDescriptor =
            &*dedicatedInputs.scatterDescriptor;
        auto presentForZeroFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(zeroInputs),
            presentForZeroRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        results.record(
            "prepared/zero-recipe-present-descriptor",
            !presentForZeroFrame.active() &&
                contains_text(diagnostic, "field=recipe_hash") &&
                contains_text(diagnostic, "expected=0") &&
                presentForZeroFrame.scatter_halation_resources().descriptor ==
                    nullptr,
            diagnostic);

        auto mismatched = *dedicatedInputs.scatterDescriptor;
        ++mismatched.recipeHash;
        auto mismatchRequest = dedicatedInputs.request(kWidth, kHeight);
        mismatchRequest.scatterHalationDescriptor = &mismatched;
        auto mismatchFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            mismatchRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        results.record(
            "prepared/hash-mismatch",
            !mismatchFrame.active() &&
                contains_text(diagnostic, "field=recipe_hash") &&
                mismatchFrame.scatter_halation_resources().descriptor == nullptr,
            diagnostic);

        auto zeroExtentRequest = dedicatedInputs.request(0, kHeight);
        auto zeroExtentFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            zeroExtentRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        results.record(
            "prepared/zero-extent",
            !zeroExtentFrame.active() && contains_text(diagnostic, "field=extent"),
            diagnostic);
        auto negativeExtentRequest = dedicatedInputs.request(-1, kHeight);
        auto negativeExtentFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            negativeExtentRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        results.record(
            "prepared/negative-extent",
            !negativeExtentFrame.active() && contains_text(diagnostic, "field=extent"),
            diagnostic);

        auto overflowRequest = dedicatedInputs.request(
            std::numeric_limits<int>::max(),
            std::numeric_limits<int>::max());
        auto overflowFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            overflowRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        results.record(
            "prepared/overflow",
            !overflowFrame.active() &&
                contains_text(diagnostic, "ExactAdmissionFailure") &&
                contains_text(diagnostic, "failed_fact=filter_bytes") &&
                contains_text(diagnostic, "bytes_admitted_before_failure=0") &&
                contains_text(diagnostic, "physical_allocation_count=2"),
            diagnostic);

        auto incompatibleFrameSet = *cameraInputs.diffusionFrameSet;
        incompatibleFrameSet.camera->stage =
            Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear;
        auto incompatibleRequest = cameraInputs.request(kWidth, kHeight);
        incompatibleRequest.diffusionFrameSetDescriptor = &incompatibleFrameSet;
        auto incompatibleFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(cameraInputs),
            incompatibleRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        results.record(
            "prepared/incompatible-carrier-admission",
            !incompatibleFrame.active() &&
                contains_text(diagnostic, "ExactAdmissionFailure") &&
                contains_text(diagnostic, "failed_fact=camera_carrier_stage") &&
                contains_text(diagnostic, "bytes_admitted_before_failure=0"),
            diagnostic);

        auto overlapFirst = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            dedicatedRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        auto overlapSecond = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            dedicatedRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const auto overlapFirstView = overlapFirst.scatter_halation_resources();
        const auto overlapSecondView = overlapSecond.scatter_halation_resources();
        const bool overlapNonalias =
            overlapFirst.active() && overlapSecond.active() &&
            overlapFirstView.filterTemp != overlapSecondView.filterTemp &&
            overlapFirstView.weightedAccumulation !=
                overlapSecondView.weightedAccumulation &&
            overlapFirstView.currentCarrier.redSensitive !=
                overlapSecondView.currentCarrier.redSensitive &&
            overlapFirstView.currentCarrier.greenSensitive !=
                overlapSecondView.currentCarrier.greenSensitive &&
            overlapFirstView.currentCarrier.blueSensitive !=
                overlapSecondView.currentCarrier.blueSensitive;
        results.record(
            "prepared/overlap-nonalias",
            overlapNonalias,
            "two active frames retain distinct two-record mutable block sets");
        terminalDiagnostic.clear();
        const bool overlapFinished =
            overlapFirst.finish(fixture.stream_opaque(), terminalDiagnostic) &&
            overlapSecond.finish(fixture.stream_opaque(), terminalDiagnostic) &&
            fixture.synchronize(terminalDiagnostic);
        results.record(
            "prepared/overlap-finish-cleanup",
            overlapFinished,
            terminalDiagnostic);

        auto abortFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            dedicatedRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const bool abortWasActive = abortFrame.active();
        abortFrame.abort();
        terminalDiagnostic.clear();
        results.record(
            "prepared/abort-cleanup",
            abortWasActive && !abortFrame.active() &&
                abortFrame.scatter_halation_resources().descriptor == nullptr &&
                fixture.synchronize(terminalDiagnostic),
            terminalDiagnostic);

        auto dependentRequest = dedicatedInputs.request(kWidth, kHeight);
        auto dependentFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            dependentRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const std::string namedDependentFailure =
            "NamedDependentFailure stage=post_preparation_validation";
        const bool dependentWasActive = dependentFrame.active();
        dependentFrame.abort();
        results.record(
            "prepared/dependent-failure-cleanup",
            dependentWasActive && !dependentFrame.active() &&
                dependentFrame.scatter_halation_resources().descriptor == nullptr,
            namedDependentFailure);

        terminalDiagnostic.clear();
        const bool idleRetired = root.retire_idle_context(
            fixture.context_key().deviceId,
            fixture.context_key().contextOpaque,
            terminalDiagnostic);
        results.record(
            "prepared/idle-context-retire",
            idleRetired,
            terminalDiagnostic);

        auto resetFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            dedicatedRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        terminalDiagnostic.clear();
        const bool resetReady =
            resetFrame.active() &&
            resetFrame.finish(fixture.stream_opaque(), terminalDiagnostic) &&
            fixture.synchronize(terminalDiagnostic);
        const bool resetRetired =
            resetReady &&
            root.retire_reset_context(
                fixture.context_key().deviceId,
                fixture.context_key().contextOpaque,
                terminalDiagnostic);
        results.record(
            "prepared/reset-context-retire",
            resetRetired,
            terminalDiagnostic);

        root.shutdown();
        std::vector<JuicerCuda::ResourceManager::DeviceContextKey> liveContexts;
        JuicerCuda::ResourceManager::registry_snapshot_context_keys(liveContexts);
        results.record(
            "prepared/root-shutdown-last",
            liveContexts.empty(),
            "ordered Root shutdown leaves no registered CUDA context");
    }

    bool upload_and_launch_dedicated(
        const JuicerCuda::ScatterHalationPreparedView& view,
        cudaStream_t stream,
        std::string& diagnostic) {
        diagnostic.clear();
        if (!view.descriptor ||
            view.carrierSource !=
                JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes ||
            view.fullFrameWidth <= 0 || view.fullFrameHeight <= 0) {
            diagnostic = "dedicated prepared view unavailable";
            return false;
        }
        const std::size_t elements =
            static_cast<std::size_t>(view.fullFrameWidth) *
            static_cast<std::size_t>(view.fullFrameHeight);
        const std::size_t bytes = elements * sizeof(float);
        std::vector<float> carrier(elements);
        for (std::size_t index = 0; index < elements; ++index) {
            carrier[index] = static_cast<float>(
                0.01 + 0.97 * static_cast<double>((index * 37u) % 101u) /
                           100.0);
        }
        for (float* plane : {view.currentCarrier.redSensitive,
                             view.currentCarrier.greenSensitive,
                             view.currentCarrier.blueSensitive}) {
            const cudaError_t copyStatus = cudaMemcpyAsync(
                plane,
                carrier.data(),
                bytes,
                cudaMemcpyHostToDevice,
                stream);
            if (copyStatus != cudaSuccess) {
                diagnostic = std::string("carrier upload failed: ") +
                             cudaGetErrorString(copyStatus);
                return false;
            }
        }
        const JuicerCuda::ScatterHalationLaunchResult launch =
            JuicerCuda::launch_scatter_halation(view, stream);
        if (launch.status != cudaSuccess) {
            diagnostic = "launcher status=" +
                         std::to_string(static_cast<int>(launch.status));
            return false;
        }
        std::vector<float> output(elements);
        const cudaError_t copyStatus = cudaMemcpyAsync(
            output.data(),
            view.currentCarrier.redSensitive,
            bytes,
            cudaMemcpyDeviceToHost,
            stream);
        if (copyStatus != cudaSuccess) {
            diagnostic = std::string("carrier download failed: ") +
                         cudaGetErrorString(copyStatus);
            return false;
        }
        const cudaError_t syncStatus = cudaStreamSynchronize(stream);
        if (syncStatus != cudaSuccess) {
            diagnostic = std::string("carrier completion failed: ") +
                         cudaGetErrorString(syncStatus);
            return false;
        }
        return std::all_of(output.begin(), output.end(), [](float value) {
            return std::isfinite(value);
        });
    }

    void run_unpublished_route_rows_impl(Results& results) {
        constexpr int kWidth = 16;
        constexpr int kHeight = 12;
        CudaPreparedFixture fixture;
        std::string diagnostic;
        if (!fixture.initialize(diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        const ScatterHalationControls controls = convert_or_throw(
            ScatterHalationRawControls{true, 1.0, 1.0, 1.0, 1.0});
        const Spektrafilm::DiffusionFilterAuthoredControls noDiffusion{};
        for (const Spektrafilm::ScanRouteMetadata& metadata :
             Spektrafilm::kScanRouteMatrix) {
            PreparedInputs inputs;
            diagnostic.clear();
            const bool initialized = initialize_route_inputs(
                metadata.route,
                controls,
                noDiffusion,
                10.0f,
                kWidth,
                kHeight,
                inputs,
                diagnostic);
            bool passed = initialized && inputs.scatterDescriptor.has_value() &&
                          inputs.product.recipe.profileRoute.scanRoute ==
                              metadata.route;
            if (passed) {
                auto request = inputs.request(kWidth, kHeight);
                auto frame = JuicerProcess::root().prepare_cuda_frame(
                    fixture.context_key(),
                    fixture.snapshot(inputs),
                    request,
                    {},
                    fixture.stream_opaque(),
                    diagnostic);
                const auto view = frame.scatter_halation_resources();
                passed = frame.active() && view.descriptor &&
                         view.descriptor->recipeHash ==
                             inputs.product.recipe.spatialOptics.scatterHalation.hash &&
                         upload_and_launch_dedicated(
                             view,
                             static_cast<cudaStream_t>(fixture.stream_opaque()),
                             diagnostic);
                std::string finishDiagnostic;
                passed = frame.finish(
                             fixture.stream_opaque(),
                             finishDiagnostic) &&
                         passed;
                if (!finishDiagnostic.empty()) {
                    diagnostic = finishDiagnostic;
                }
            }
            results.record(
                std::string("gate4/unpublished-route/") + metadata.key,
                passed,
                diagnostic);
        }
    }

    void run_zero_work_rows_impl(Results& results) {
        constexpr int kWidth = 16;
        constexpr int kHeight = 12;
        CudaPreparedFixture fixture;
        std::string diagnostic;
        if (!fixture.initialize(diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        const ScatterHalationControls zeroControls = convert_or_throw(
            ScatterHalationRawControls{true, 0.0, 1.0, 0.0, 1.0});
        const Spektrafilm::DiffusionFilterAuthoredControls noDiffusion{};
        for (const Spektrafilm::ScanRoute route :
             {Spektrafilm::ScanRoute::NegativeDirectScan,
              Spektrafilm::ScanRoute::NegativePrintScan}) {
            PreparedInputs inputs;
            diagnostic.clear();
            const bool initialized = initialize_route_inputs(
                route,
                zeroControls,
                noDiffusion,
                10.0f,
                kWidth,
                kHeight,
                inputs,
                diagnostic);
            if (!initialized) {
                results.record(
                    std::string("gate4/zero-work/") +
                        Spektrafilm::scan_route_key(route),
                    false,
                    diagnostic);
                continue;
            }
            auto request = inputs.request(kWidth, kHeight);
            auto frame = JuicerProcess::root().prepare_cuda_frame(
                fixture.context_key(),
                fixture.snapshot(inputs),
                request,
                {},
                fixture.stream_opaque(),
                diagnostic);
            const auto view = frame.scatter_halation_resources();
            const auto workspace = frame.workspace_lease();
            bool passed =
                inputs.product.recipe.spatialOptics.scatterHalation.hash == 0 &&
                !inputs.scatterDescriptor && frame.active() &&
                !workspace.active() && !view.descriptor &&
                !view.filterTemp && !view.weightedAccumulation &&
                !view.currentCarrier.redSensitive &&
                !view.currentCarrier.greenSensitive &&
                !view.currentCarrier.blueSensitive;
            std::string finishDiagnostic;
            passed = frame.finish(
                         fixture.stream_opaque(),
                         finishDiagnostic) &&
                     passed;
            results.record(
                std::string("gate4/zero-work/") +
                    Spektrafilm::scan_route_key(route),
                passed,
                finishDiagnostic.empty() ? diagnostic : finishDiagnostic);
        }
    }

    void run_route_lifecycle_rows_impl(Results& results) {
        constexpr int kWidth = 16;
        constexpr int kHeight = 12;
        CudaPreparedFixture fixture;
        std::string diagnostic;
        if (!fixture.initialize(diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        PreparedInputs inputs;
        if (!initialize_prepared_inputs(
                false,
                false,
                true,
                kWidth,
                kHeight,
                inputs,
                diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        auto request = inputs.request(kWidth, kHeight);
        auto failedFrame = JuicerProcess::root().prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(inputs),
            request,
            {},
            fixture.stream_opaque(),
            diagnostic);
        auto failedView = failedFrame.scatter_halation_resources();
        failedView.filterTemp = nullptr;
        const auto failedLaunch = JuicerCuda::launch_scatter_halation(
            failedView,
            static_cast<cudaStream_t>(fixture.stream_opaque()));
        const bool failedWasActive = failedFrame.active();
        failedFrame.abort();
        results.record(
            "gate4/lifecycle/launcher-failure-abort",
            failedWasActive &&
                failedLaunch.status == cudaErrorInvalidValue &&
                !failedFrame.active() &&
                !failedFrame.scatter_halation_resources().descriptor,
            "ordinary prepared view binding failure retires through abort");

        auto transitionFrame = JuicerProcess::root().prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(inputs),
            request,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const auto transitionView = transitionFrame.scatter_halation_resources();
        const auto carrierR = transitionView.currentCarrier.redSensitive;
        bool transitioned =
            transitionFrame.active() &&
            upload_and_launch_dedicated(
                transitionView,
                static_cast<cudaStream_t>(fixture.stream_opaque()),
                diagnostic);
        const auto workspace = transitionFrame.workspace_lease();
        std::string transitionDiagnostic;
        transitioned = transitioned && workspace.active() &&
                       transitionFrame.stage_optical_workspace(
                           workspace,
                           fixture.stream_opaque(),
                           transitionDiagnostic) &&
                       transitionFrame.scatter_halation_resources()
                               .currentCarrier.redSensitive == carrierR;
        std::string finishDiagnostic;
        transitioned = transitionFrame.finish(
                           fixture.stream_opaque(),
                           finishDiagnostic) &&
                       transitioned;
        results.record(
            "gate4/lifecycle/downstream-scratch-transition",
            transitioned,
            transitionDiagnostic.empty() ? finishDiagnostic
                                         : transitionDiagnostic);
    }

    void print_results(const Results& results) {
        for (const auto& result : results.cases()) {
            std::cout << (result.passed ? "PASS " : "FAIL ") << result.name;
            if (!result.detail.empty()) {
                std::cout << " -- " << result.detail;
            }
            std::cout << '\n';
        }
        std::cout << "SUMMARY cases=" << results.cases().size()
                  << " failures=" << results.failure_count() << '\n';
    }

} // namespace

namespace ScatterHalationValidation {

    struct PreparedRouteInputs::Impl {
        FocusedRenderStateBuildProduct product;
        Scanner::ScannerSpectralLutDescriptor scannerDescriptor{};
        std::optional<ScatterHalationFrameDescriptor> scatterDescriptor;
        std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusionFrameSet;
    };

    PreparedRouteInputs::PreparedRouteInputs(std::unique_ptr<Impl> impl)
        : _impl(std::move(impl)) {}

    PreparedRouteInputs::~PreparedRouteInputs() = default;
    PreparedRouteInputs::PreparedRouteInputs(PreparedRouteInputs&&) noexcept =
        default;
    PreparedRouteInputs& PreparedRouteInputs::operator=(
        PreparedRouteInputs&&) noexcept = default;

    JuicerProcess::Root::CudaFramePreparationRequest
    PreparedRouteInputs::request(int width, int height) const {
        JuicerProcess::Root::CudaFramePreparationRequest value{};
        value.recipe = &_impl->product.recipe;
        value.exposureTables = &_impl->product.payload.exposureTables;
        value.spdSInv = _impl->product.payload.spdSInv.data();
        value.filmRawConfig = &_impl->product.payload.filmRawConfig;
        value.scannerTables = &_impl->product.payload.scannerTables;
        value.scannerColor = &_impl->product.payload.scannerColor;
        value.scannerLutDescriptor = &_impl->scannerDescriptor;
        value.diffusionFrameSetDescriptor =
            _impl->diffusionFrameSet ? &*_impl->diffusionFrameSet : nullptr;
        value.scatterHalationDescriptor =
            _impl->scatterDescriptor ? &*_impl->scatterDescriptor : nullptr;
        value.requestedWidth = width;
        value.requestedHeight = height;
        return value;
    }

    JuicerCuda::ResourceManager::SubmissionSnapshot
    PreparedRouteInputs::submission_snapshot(
        const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t identity) const {
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot{};
        snapshot.instanceToken.value = 0x4741544534524f55ull;
        snapshot.frameToken.value = identity;
        snapshot.snapshotId = identity;
        snapshot.deviceContextKey = contextKey;
        snapshot.contextEpoch = 1;
        snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
            _impl->product.payload.uploadCoreHash,
            _impl->product.recipe.dirCouplers.hash,
            _impl->product.payload.scannerHash,
            0);
        return snapshot;
    }

    const ScatterHalationFrameDescriptor*
    PreparedRouteInputs::scatter_descriptor() const noexcept {
        return _impl->scatterDescriptor ? &*_impl->scatterDescriptor : nullptr;
    }

    const Spektrafilm::DiffusionFrameSetDescriptor*
    PreparedRouteInputs::diffusion_frame_set() const noexcept {
        return _impl->diffusionFrameSet ? &*_impl->diffusionFrameSet : nullptr;
    }

    PreparedRouteInputs build_prepared_route_inputs(
        Spektrafilm::ScanRoute route,
        const ScatterHalationControls& controls,
        const Spektrafilm::DiffusionFilterAuthoredControls& cameraDiffusion,
        float pixelSizeUm,
        int width,
        int height) {
        ParamSnapshot snapshot;
        snapshot.scanRoute = route;
        snapshot.scatterHalationControls = controls;
        snapshot.cameraDiffusion = cameraDiffusion;
        snapshot.gateWeaveAmount = 0.0;
        auto impl = std::make_unique<PreparedRouteInputs::Impl>();
        std::string diagnostic;
        const bool printRoute = Spektrafilm::scan_route_is_print(route);
        const bool built =
            printRoute
                ? build_print_render_state_product(
                      snapshot, impl->product, diagnostic)
                : build_direct_render_state_product(
                      snapshot, impl->product, diagnostic);
        if (!built) {
            throw std::runtime_error(diagnostic);
        }
        if (!Spektrafilm::build_scatter_halation_frame_descriptor(
                impl->product.recipe.spatialOptics.scatterHalation,
                pixelSizeUm,
                impl->scatterDescriptor,
                diagnostic)) {
            throw std::runtime_error(diagnostic);
        }
        const Scanner::DirectScannerSpectralLutDescriptorInput directInput{
            &impl->product.recipe.profileRoute,
            &impl->product.recipe.densityBounds,
            &impl->product.recipe.scannerOutput};
        const Scanner::PrintScannerSpectralLutDescriptorInput printInput{
            &impl->product.recipe.profileRoute,
            &impl->product.recipe.densityBounds,
            &impl->product.recipe.scannerOutput};
        const bool scannerBuilt =
            printRoute
                ? Scanner::build_print_scanner_spectral_lut_descriptor(
                      printInput, impl->scannerDescriptor, diagnostic)
                : Scanner::build_direct_scanner_spectral_lut_descriptor(
                      directInput, impl->scannerDescriptor, diagnostic);
        if (!scannerBuilt) {
            throw std::runtime_error(diagnostic);
        }
        if (cameraDiffusion.active &&
            (!Spektrafilm::build_diffusion_frame_set_descriptor(
                 impl->product.recipe.spatialOptics,
                 route,
                 static_cast<double>(pixelSizeUm),
                 Spektrafilm::DiffusionFrameDomain{0, 0, width, height},
                 impl->diffusionFrameSet,
                 diagnostic) ||
             !impl->diffusionFrameSet || !impl->diffusionFrameSet->camera)) {
            throw std::runtime_error(
                diagnostic.empty() ? "missing camera diffusion frame set"
                                   : diagnostic);
        }
        return PreparedRouteInputs(std::move(impl));
    }

    void Results::record(std::string name, bool passed, std::string detail) {
        _cases.push_back(CaseResult{std::move(name), passed, std::move(detail)});
    }

    int Results::failure_count() const noexcept {
        int failures = 0;
        for (const CaseResult& result : _cases) {
            failures += result.passed ? 0 : 1;
        }
        return failures;
    }

    const std::vector<CaseResult>& Results::cases() const noexcept {
        return _cases;
    }

    void run_unpublished_route_rows(
        const Arguments& arguments,
        Results& results) {
        (void)arguments;
        run_unpublished_route_rows_impl(results);
        run_scanner_post_effect_cuda_rows(results);
    }

    void run_zero_work_rows(const Arguments& arguments, Results& results) {
        (void)arguments;
        run_zero_work_rows_impl(results);
    }

    void run_lifecycle_rows(const Arguments& arguments, Results& results) {
        (void)arguments;
        run_prepared_gate_2_rows(results);
        JuicerProcess::root().ensure_bootstrap();
        run_route_lifecycle_rows_impl(results);
    }

} // namespace ScatterHalationValidation

int main(int argc, char** argv) {
    try {
        const Arguments arguments = parse_arguments(argc, argv);
        std::filesystem::create_directories(arguments.scratchRoot);
        JuicerProcess::root().ensure_bootstrap();

        Results results;
        const auto run_host_gate_1 = [&] {
            run_control_rows(results);
            run_build_product_rows(results);
            run_profile_rows(arguments, results);
            run_recipe_rows(results);
            run_descriptor_rows(results);
            run_pending_admission_rows(results);
            run_camera_hash_rows(results);
        };
        if (arguments.caseGroup == "host-gate-1") {
            run_host_gate_1();
        } else if (arguments.caseGroup == "prepared-gate-2") {
            run_prepared_gate_2_rows(results);
        } else if (arguments.caseGroup == "focused-cuda-gate-3") {
            ScatterHalationValidation::run_focused_cuda_gate_3_rows(
                arguments,
                results);
        } else if (arguments.caseGroup == "unpublished-routes") {
            ScatterHalationValidation::run_unpublished_route_rows(
                arguments,
                results);
        } else if (arguments.caseGroup == "captured-carrier") {
            ScatterHalationValidation::run_captured_carrier_rows(
                arguments,
                results);
        } else if (arguments.caseGroup == "zero-work") {
            ScatterHalationValidation::run_zero_work_rows(arguments, results);
        } else if (arguments.caseGroup == "lifecycle") {
            ScatterHalationValidation::run_lifecycle_rows(arguments, results);
        } else if (arguments.caseGroup == "performance") {
            ScatterHalationValidation::run_performance_rows(arguments, results);
        } else {
            run_host_gate_1();
            run_prepared_gate_2_rows(results);
            JuicerProcess::root().ensure_bootstrap();
            ScatterHalationValidation::run_focused_cuda_gate_3_rows(
                arguments,
                results);
            ScatterHalationValidation::run_unpublished_route_rows(
                arguments,
                results);
            ScatterHalationValidation::run_captured_carrier_rows(
                arguments,
                results);
            ScatterHalationValidation::run_zero_work_rows(arguments, results);
            ScatterHalationValidation::run_lifecycle_rows(arguments, results);
        }
        print_results(results);
        if (arguments.caseGroup != "prepared-gate-2") {
            JuicerProcess::root().shutdown();
        }
        return results.failure_count() == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        JuicerProcess::root().shutdown();
        return 2;
    }
}
