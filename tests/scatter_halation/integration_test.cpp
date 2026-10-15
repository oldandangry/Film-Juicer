#include "integration_test.h"
#include "../ofx/render_assertions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
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

#include "Cuda/JuicerCudaFailure.h"

#include "ColorTransforms.h"
#include "Hash.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "juicer_cuda_owner.h"
#include "../ffi/juicer_test_api.h"
#include "ProfileAssets.h"
#include "ScatterHalation.h"
#include "Cuda/Film/JuicerCudaScatterHalation.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "nlohmann/json.hpp"

namespace JuicerProcess::TestSupport {

    class RootLifetimeObserver final {
    public:
        struct Snapshot {
            bool contextEntryPresent = false;
            bool frameOwnerPresent = false;
            long frameOwnerUseCount = 0;
            std::uint64_t contextEpoch = 0;
            std::uint64_t filmDensityCurvesHash = 0;
            std::size_t nativeAllocationCount = 0;
            std::size_t pendingFrameUseEventCount = 0;
            std::size_t pendingScanErrorReadbackCount = 0;
            bool pendingScanErrorReadbackOwned = false;
            std::string pendingScanErrorProfileKey;
            std::uint64_t pendingScanErrorContextEpoch = 0;
            cudaError_t pendingScanErrorEventQuery =
                cudaErrorInvalidResourceHandle;
            bool pendingScanErrorValueReady = false;
            int pendingScanErrorValue = 0;
            std::size_t retireQueueCount = 0;
            std::size_t retireBytes = 0;
            bool expectedRetireAddressPresent = false;
            JuicerCuda::DeviceLedgerSnapshot ledger{};
            std::uint64_t contextLedgerRecordCount = 0;
        };

        static Snapshot snapshot(
            Root& root,
            const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
            std::uintptr_t expectedRetireAddress = 0) {
            Snapshot result{};
            std::lock_guard<std::mutex> rootLock(root._cudaResourcesMutex);
            const auto ledgerIt = root._cudaDeviceLedgers.find(contextKey.deviceId);
            if (ledgerIt != root._cudaDeviceLedgers.end() && ledgerIt->second) {
                result.ledger = ledgerIt->second->snapshot();
            }
            for (const auto& [key, entry] : root._cudaContextResources) {
                if (key.deviceContextKey != contextKey) {
                    continue;
                }
                result.contextEntryPresent = true;
                result.contextEpoch = key.contextEpoch;
                result.frameOwnerPresent = static_cast<bool>(entry.frameOwner);
                result.frameOwnerUseCount = entry.frameOwner.use_count();
                if (entry.frameOwner) {
                    {
                        std::lock_guard<std::mutex> resourceLock(entry.frameOwner->m);
                        result.pendingFrameUseEventCount =
                            entry.frameOwner->pendingFrameUseEvents.size();
                        result.pendingScanErrorReadbackCount =
                            entry.frameOwner->pendingScanErrorReadbacks.size();
                        if (!entry.frameOwner->pendingScanErrorReadbacks.empty()) {
                            const auto& readback =
                                entry.frameOwner->pendingScanErrorReadbacks.front();
                            result.pendingScanErrorReadbackOwned =
                                readback.host && readback.eventOpaque;
                            result.pendingScanErrorProfileKey =
                                readback.identity.profileKey;
                            result.pendingScanErrorContextEpoch =
                                readback.identity.contextEpoch;
                            if (readback.eventOpaque) {
                                result.pendingScanErrorEventQuery =
                                    cudaEventQuery(reinterpret_cast<cudaEvent_t>(
                                        readback.eventOpaque));
                            }
                            if (readback.host &&
                                result.pendingScanErrorEventQuery ==
                                    cudaSuccess) {
                                result.pendingScanErrorValueReady = true;
                                result.pendingScanErrorValue = *readback.host;
                            }
                        }
                        result.filmDensityCurvesHash =
                            entry.frameOwner->filmDensityCurvesHash;
                        result.retireQueueCount =
                            entry.frameOwner->retireQueue.size();
                        result.retireBytes = entry.frameOwner->retireBytes;
                        result.expectedRetireAddressPresent =
                            expectedRetireAddress != 0 &&
                            std::any_of(
                                entry.frameOwner->retireQueue.begin(),
                                entry.frameOwner->retireQueue.end(),
                                [&](const JuicerCuda::Resources::RetireEntry&
                                        retired) {
                                    return reinterpret_cast<std::uintptr_t>(
                                               retired.ptr) ==
                                           expectedRetireAddress;
                                });
                    }
                    {
                        std::lock_guard<std::mutex> allocationLock(
                            entry.frameOwner->deviceAllocationRecordsMutex);
                        result.nativeAllocationCount =
                            entry.frameOwner->deviceAllocationRecords.size();
                    }
                }
                if (ledgerIt != root._cudaDeviceLedgers.end() && ledgerIt->second) {
                    result.contextLedgerRecordCount =
                        ledgerIt->second->record_count_for_context(
                            contextKey,
                            key.contextEpoch);
                }
                break;
            }
            return result;
        }
    };

} // namespace JuicerProcess::TestSupport

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
            arguments.caseGroup == "host-contracts" ||
            arguments.caseGroup == "prepared-frame" ||
            arguments.caseGroup == "terminal-retention" ||
            arguments.caseGroup == "focused-cuda-reference" ||
            arguments.caseGroup == "route-boundaries" ||
            arguments.caseGroup == "captured-carrier" ||
            arguments.caseGroup == "zero-work" ||
            arguments.caseGroup == "lifecycle" ||
            arguments.caseGroup == "performance" ||
            arguments.caseGroup == "all";
        const bool fixtureRequired =
            arguments.caseGroup == "focused-cuda-reference" ||
            arguments.caseGroup == "captured-carrier" ||
            arguments.caseGroup == "performance" ||
            arguments.caseGroup == "all";
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
                "expected --case-group host-contracts|prepared-frame|"
                "focused-cuda-reference|route-boundaries|captured-carrier|"
                "zero-work|lifecycle|performance|all "
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

        const auto check_gamma_success = [&results](
                                             std::string name,
                                             Spektrafilm::ScanRoute route,
                                             double filmGamma,
                                             double printGamma) {
            ParamSnapshot snapshot;
            snapshot.scanRoute = route;
            std::string diagnostic;
            const bool success = set_gamma_snapshot_values(
                filmGamma,
                printGamma,
                snapshot,
                diagnostic);
            const bool printRetained = std::isnan(printGamma)
                                           ? std::isnan(snapshot.printGammaFactor)
                                           : snapshot.printGammaFactor == printGamma;
            results.record(
                std::move(name),
                success &&
                    snapshot.filmGammaFactor == static_cast<float>(filmGamma) &&
                    printRetained,
                diagnostic);
        };
        const auto check_gamma_failure = [&results](
                                             std::string name,
                                             Spektrafilm::ScanRoute route,
                                             double filmGamma,
                                             double printGamma) {
            ParamSnapshot snapshot;
            snapshot.scanRoute = route;
            std::string diagnostic;
            const bool success = set_gamma_snapshot_values(
                filmGamma,
                printGamma,
                snapshot,
                diagnostic);
            results.record(
                std::move(name),
                !success && !diagnostic.empty(),
                diagnostic);
        };
        check_gamma_success(
            "controls/film-gamma-minimum",
            Spektrafilm::ScanRoute::NegativeDirectScan,
            Spektrafilm::kFilmGammaFactorMinimum,
            std::numeric_limits<double>::quiet_NaN());
        check_gamma_success(
            "controls/film-gamma-maximum",
            Spektrafilm::ScanRoute::PositiveDirectScan,
            Spektrafilm::kFilmGammaFactorMaximum,
            std::numeric_limits<double>::infinity());
        check_gamma_success(
            "controls/print-gamma-minimum",
            Spektrafilm::ScanRoute::NegativePrintScan,
            1.0,
            Spektrafilm::kPrintGammaFactorMinimum);
        check_gamma_success(
            "controls/print-gamma-maximum",
            Spektrafilm::ScanRoute::PositivePrintScan,
            1.0,
            Spektrafilm::kPrintGammaFactorMaximum);
        check_gamma_failure(
            "controls/film-gamma-immediately-below-minimum",
            Spektrafilm::ScanRoute::NegativeDirectScan,
            std::nextafter(Spektrafilm::kFilmGammaFactorMinimum, 0.0),
            1.0);
        check_gamma_failure(
            "controls/film-gamma-immediately-above-maximum",
            Spektrafilm::ScanRoute::NegativeDirectScan,
            std::nextafter(Spektrafilm::kFilmGammaFactorMaximum, 5.0),
            1.0);
        check_gamma_failure(
            "controls/film-gamma-nan",
            Spektrafilm::ScanRoute::NegativeDirectScan,
            std::numeric_limits<double>::quiet_NaN(),
            1.0);
        check_gamma_failure(
            "controls/film-gamma-infinity",
            Spektrafilm::ScanRoute::NegativeDirectScan,
            std::numeric_limits<double>::infinity(),
            1.0);
        check_gamma_failure(
            "controls/print-gamma-immediately-below-minimum",
            Spektrafilm::ScanRoute::NegativePrintScan,
            1.0,
            std::nextafter(Spektrafilm::kPrintGammaFactorMinimum, 0.0));
        check_gamma_failure(
            "controls/print-gamma-immediately-above-maximum",
            Spektrafilm::ScanRoute::NegativePrintScan,
            1.0,
            std::nextafter(Spektrafilm::kPrintGammaFactorMaximum, 3.0));
        check_gamma_failure(
            "controls/print-gamma-nan",
            Spektrafilm::ScanRoute::NegativePrintScan,
            1.0,
            std::numeric_limits<double>::quiet_NaN());
        check_gamma_failure(
            "controls/print-gamma-infinity",
            Spektrafilm::ScanRoute::NegativePrintScan,
            1.0,
            std::numeric_limits<double>::infinity());
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

    bool contains_text(const std::string& text, std::string_view expected) {
        return text.find(expected) != std::string::npos;
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

    class FilmProfileOwner {
    public:
        ~FilmProfileOwner() noexcept {
            const FjStatus status = fj_test_film_profile_release(std::exchange(_owner, nullptr), nullptr);
            if (status.category != FJ_STATUS_SUCCESS || status.api != FJ_API_NONE || status.native_code != 0) {
                std::fprintf(stderr, "profile fixture release failed: category=%u api=%u code=%d\n", status.category, status.api, status.native_code);
                std::abort();
            }
        }
        FilmProfileOwner() = default;
        FilmProfileOwner(const FilmProfileOwner&) = delete;
        FilmProfileOwner& operator=(const FilmProfileOwner&) = delete;

        FjStatus load_profile_view(const std::filesystem::path& root, const std::string& key, std::string& diagnostic) {
            const std::string path = root.generic_string();
            std::array<char, 512> bytes{};
            FjErrorBuffer error{bytes.data(), bytes.size(), 0};
            FjStatus status = fj_test_film_profile_acquire({path.data(), path.size()}, {key.data(), key.size()}, &_owner, &error);
            if (status.category == FJ_STATUS_SUCCESS) {
                status = fj_test_film_profile_view(_owner, &_view, &error);
            }
            // This object already owns any published handle if formatting throws.
            diagnostic.assign(bytes.data(), error.length);
            return status;
        }

        const FjFilmFixtureView& view() const noexcept {
            return _view;
        }
        bool has_owner() const noexcept {
            return _owner != nullptr;
        }

    private:
        FjFilmProfile* _owner = nullptr;
        FjFilmFixtureView _view{};
    };

    void run_profile_rows(const Arguments& arguments, Results& results) {
        const std::filesystem::path source =
            arguments.resourceRoot / "profiles" / "kodak_portra_400.json";
        const nlohmann::json completeProfile = load_json(source);
        const auto stage_profile = [&](const std::string& key, const nlohmann::json& profile) {
            const auto root = arguments.scratchRoot / "profile-catalogs" / key;
            std::filesystem::create_directories(root / "profiles");
            for (const char* defaultKey : {"kodak_portra_400", "kodak_portra_endura"}) {
                std::filesystem::copy_file(
                    arguments.resourceRoot / "profiles" / (std::string(defaultKey) + ".json"),
                    root / "profiles" / (std::string(defaultKey) + ".json"),
                    std::filesystem::copy_options::overwrite_existing);
            }
            write_json(root / "profiles" / (key + ".json"), profile);
            return root;
        };

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
                std::string("recipe_profile_") + mapping.use + "_" + mapping.antihalation;
            nlohmann::json profile = completeProfile;
            profile["info"]["stock"] = key;
            profile["info"]["name"] = key;
            profile["info"]["use"] = mapping.use;
            profile["info"]["antihalation"] = mapping.antihalation;
            const auto root = stage_profile(key, profile);
            std::string diagnostic;
            FilmProfileOwner loaded;
            const FjStatus status = loaded.load_profile_view(root, key, diagnostic);
            bool passed = status.category == FJ_STATUS_SUCCESS;
            if (passed) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    passed = passed &&
                             float_bits(loaded.view().halation_first_sigma_um[channel]) ==
                                 float_bits(mapping.sigma) &&
                             float_bits(loaded.view().halation_primary_amount[channel]) ==
                                 float_bits(mapping.strength[channel]);
                }
            }
            results.record(
                "profile/" + std::string(mapping.use) + "-" + mapping.antihalation,
                passed,
                diagnostic);
        }

        nlohmann::json defaulted = completeProfile;
        const std::string defaultedKey = "recipe_profile_missing_metadata_defaults";
        defaulted["info"]["stock"] = defaultedKey;
        defaulted["info"]["name"] = defaultedKey;
        defaulted["info"].erase("use");
        defaulted["info"].erase("antihalation");
        const auto defaultedRoot = stage_profile(defaultedKey, defaulted);
        std::string diagnostic;
        FilmProfileOwner defaultedProfile;
        const FjStatus defaultedStatus = defaultedProfile.load_profile_view(defaultedRoot, defaultedKey, diagnostic);
        const auto& defaultedView = defaultedProfile.view();
        bool defaultedPassed = defaultedStatus.category == FJ_STATUS_SUCCESS &&
                               defaultedView.use == FJ_PROFILE_USE_STILL &&
                               defaultedView.antihalation == FJ_PROFILE_ANTIHALATION_WEAK;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            constexpr std::array<float, 3> kAmounts{0.08f, 0.02f, 0.0f};
            defaultedPassed = defaultedPassed &&
                              float_bits(defaultedView.halation_first_sigma_um[channel]) == float_bits(65.0f) &&
                              float_bits(defaultedView.halation_primary_amount[channel]) == float_bits(kAmounts[channel]);
        }
        results.record("profile/missing-metadata-defaults-still-weak", defaultedPassed, diagnostic);

        const auto check_unsupported = [&](const char* field, const char* value) {
            const std::string key = std::string("recipe_profile_unsupported_") + field;
            nlohmann::json profile = completeProfile;
            profile["info"]["stock"] = key;
            profile["info"]["name"] = key;
            profile["info"][field] = value;
            const auto root = stage_profile(key, profile);
            std::string failure;
            FilmProfileOwner loaded;
            const FjStatus status = loaded.load_profile_view(root, key, failure);
            results.record(
                "profile/unsupported-" + std::string(field) + "-rejected",
                status.category == FJ_STATUS_PREPARATION_FAILURE && !loaded.has_owner() && !failure.empty(),
                failure);
        };
        check_unsupported("use", "unsupported");
        check_unsupported("antihalation", "unsupported");

        struct CoefficientCase {
            const char* name;
            const char* arrayName;
            double value;
            bool completes;
        };
        const std::array<CoefficientCase, 5> coefficientCases{{{"zero-sigma", "sigmas", 0.0, true},
                                                               {"negative-sigma", "sigmas", -0.01, true},
                                                               {"unrepresentable-center", "centers", std::numeric_limits<double>::max(), true},
                                                               {"unrepresentable-amplitude", "amplitudes", std::numeric_limits<double>::max(), false},
                                                               {"unrepresentable-sigma", "sigmas", std::numeric_limits<double>::max(), true}}};
        // C1/C4: narrowing precedes z=(exposure-center)/sigma and amplitude*CDF.
        // erfc(+inf)=0, erfc(-inf)=2, erfc(0)=1 establish exact special cases.
        const float center = completeProfile["data"]["density_curves_model"]["centers"][0][0].get<float>();
        const float amplitude = completeProfile["data"]["density_curves_model"]["amplitudes"][0][0].get<float>();
        for (const CoefficientCase& testCase : coefficientCases) {
            const std::string key = std::string("recipe_profile_") + testCase.name;
            nlohmann::json profile = completeProfile;
            profile["info"]["stock"] = key;
            profile["info"]["name"] = key;
            profile["data"]["density_curves_model"][testCase.arrayName][0][0] = testCase.value;
            const auto root = stage_profile(key, profile);
            std::string detail;
            FilmProfileOwner loaded;
            const FjStatus status = loaded.load_profile_view(root, key, detail);
            bool passed = status.category == (testCase.completes ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
            if (testCase.completes && status.category == FJ_STATUS_SUCCESS) {
                const auto& view = loaded.view();
                const auto layer = view.density_curves_layers[0][0];
                passed = passed && layer.count == view.log_exposure.count && layer.count > 0;
                for (std::size_t sample = 0; passed && sample < layer.count; ++sample) {
                    const float density = layer.data[sample];
                    if (testCase.value == 0.0) {
                        const float exposure = view.log_exposure.data[sample];
                        // The original Portra axis has no exact-center sample.
                        passed = exposure != center && float_bits(density) == float_bits(exposure < center ? 0.0f : amplitude);
                    } else if (testCase.value < 0.0) {
                        passed = std::isfinite(density) && density >= 0.0f && density <= amplitude &&
                                 (sample == 0 || layer.data[sample - 1] >= density);
                    } else {
                        const float expected = std::string_view(testCase.arrayName) == "centers" ? 0.0f : 0.5f * amplitude;
                        passed = float_bits(density) == float_bits(expected);
                    }
                }
            } else if (!testCase.completes) {
                passed = passed && !loaded.has_owner() && !detail.empty() &&
                         contains_text(detail, "Density { index: 0, source: NonfiniteLayer { channel: 0, layer: 0 }");
            }
            results.record("profile/" + std::string(testCase.name) + (testCase.completes ? "-completed" : "-computed-failure"), passed, detail);
        }

        const auto check_completion = [&](const char* name, nlohmann::json profile, bool completes, const char* reason) {
            const std::string key = std::string("recipe_profile_") + name;
            profile["info"]["stock"] = key;
            const auto root = stage_profile(key, profile);
            FilmProfileOwner loaded;
            std::string detail;
            const FjStatus status = loaded.load_profile_view(root, key, detail);
            bool passed = status.category == (completes ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE);
            if (completes && status.category == FJ_STATUS_SUCCESS) {
                passed = loaded.view().log_exposure.count == 1 &&
                         loaded.view().density_curves_layers[0][0].count == 1 &&
                         float_bits(loaded.view().density_curves_layers[0][0].data[0]) == float_bits(0.5f * amplitude);
            } else {
                passed = passed && !loaded.has_owner() && !detail.empty() && contains_text(detail, reason);
            }
            results.record("profile/" + std::string(name), passed, detail);
        };
        nlohmann::json atCenter = completeProfile;
        atCenter["data"]["log_exposure"] = {completeProfile["data"]["density_curves_model"]["centers"][0][0]};
        atCenter["data"]["density_curves_model"]["sigmas"][0][0] = -0.01;
        check_completion("negative-sigma-center-half", atCenter, true, "");
        atCenter["data"]["density_curves_model"]["sigmas"][0][0] = 0.0;
        check_completion("zero-sigma-center-computed-failure", atCenter, false, "NonfiniteLayer { channel: 0, layer: 0 }");
        nlohmann::json malformed = completeProfile;
        malformed["data"]["log_exposure"] = {1.0, 0.0};
        check_completion("descending-axis-completion-failure", malformed, false, "DescendingAxis { index: 1 }");
        malformed = completeProfile;
        malformed["data"]["wavelengths"].erase(80);
        check_completion("spectral-shape-decode-failure", malformed, false, "wavelengths");
        malformed = completeProfile;
        malformed["data"]["density_curves_model"]["sigmas"][0][0] = "unsupported";
        check_completion("coefficient-representation-decode-failure", malformed, false, "data.density_curves_model requires ModelCoefficients");

        nlohmann::json endpoints = completeProfile;
        const std::string endpointKey = "recipe_profile_infinite_axis_layer_order";
        endpoints["info"]["stock"] = endpointKey;
        endpoints["data"]["log_exposure"] = {-1e40, 0.0, 1e40};
        constexpr std::array<std::array<float, 3>, 3> kAmplitudes{{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}}};
        endpoints["data"]["density_curves_model"]["centers"] = std::array<std::array<float, 3>, 3>{};
        endpoints["data"]["density_curves_model"]["sigmas"] = std::array<std::array<float, 3>, 3>{{{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}}};
        endpoints["data"]["density_curves_model"]["amplitudes"] = kAmplitudes;
        FilmProfileOwner endpointProfile;
        std::string endpointDetail;
        const FjStatus endpointStatus = endpointProfile.load_profile_view(stage_profile(endpointKey, endpoints), endpointKey, endpointDetail);
        bool endpointPassed = endpointStatus.category == FJ_STATUS_SUCCESS;
        if (endpointPassed) {
            const auto& endpointView = endpointProfile.view();
            endpointPassed = endpointView.source_log_exposure.count == 3 && endpointView.log_exposure.count == 3 &&
                             endpointView.density_curves_cmy.count == 9 && endpointView.log_exposure.data[0] == -std::numeric_limits<float>::infinity() &&
                             endpointView.log_exposure.data[2] == std::numeric_limits<float>::infinity() &&
                             endpointView.source_log_exposure.data[0] == -1e40 && endpointView.source_log_exposure.data[2] == 1e40;
            for (std::size_t channel = 0; endpointPassed && channel < 3; ++channel) {
                for (std::size_t sample = 0; endpointPassed && sample < 3; ++sample) {
                    const float cdf = static_cast<float>(sample) * 0.5f;
                    float total = 0.0f;
                    for (std::size_t layer = 0; layer < 3; ++layer) {
                        const auto curve = endpointView.density_curves_layers[layer][channel];
                        const float expected = kAmplitudes[channel][layer] * cdf;
                        endpointPassed = endpointPassed && curve.count == 3 && float_bits(curve.data[sample]) == float_bits(expected);
                        total += expected;
                    }
                    endpointPassed = endpointPassed && float_bits(endpointView.density_curves_cmy.data[sample * 3 + channel]) == float_bits(total);
                }
            }
        }
        results.record("profile/ordered-infinity-axis-and-layer-cmy-order", endpointPassed, endpointDetail);

        const std::string variableAxisKey = "recipe_profile_variable_axis";
        nlohmann::json variableAxis = completeProfile;
        variableAxis["info"]["stock"] = variableAxisKey;
        variableAxis["info"]["name"] = variableAxisKey;
        std::vector<double> logExposure;
        logExposure.reserve(17u);
        for (int sample = 0; sample < 17; ++sample) {
            logExposure.push_back(-4.0 + static_cast<double>(sample) * 0.5);
        }
        logExposure[8] = logExposure[7];
        variableAxis["data"]["log_exposure"] = logExposure;
        const auto variableAxisRoot = stage_profile(variableAxisKey, variableAxis);
        std::string variableAxisDiagnostic;
        FilmProfileOwner variableAxisProfile;
        const FjStatus variableAxisStatus = variableAxisProfile.load_profile_view(variableAxisRoot, variableAxisKey, variableAxisDiagnostic);
        const auto& view = variableAxisProfile.view();
        bool variableAxisAccepted = variableAxisStatus.category == FJ_STATUS_SUCCESS &&
                                    view.source_log_exposure.count == 17u && view.log_exposure.count == 17u &&
                                    view.density_curves_cmy.count == 51u &&
                                    view.source_log_exposure.data[7] == view.source_log_exposure.data[8] &&
                                    view.log_exposure.data[7] == view.log_exposure.data[8] &&
                                    std::isnan(view.channel_density_cmy.data[0]) && std::isnan(view.base_density.data[0]);
        if (variableAxisStatus.category == FJ_STATUS_SUCCESS) {
            for (const auto& layer : view.density_curves_layers) {
                for (const auto& channel : layer) {
                    variableAxisAccepted = variableAxisAccepted && channel.count == 17u;
                }
            }
        }
        results.record(
            "profile/variable-axis-duplicate-and-nullable-spectral-accepted",
            variableAxisAccepted,
            variableAxisDiagnostic);
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
            directBuilt &&
                !Spektrafilm::scan_route_is_print(
                    directProduct.recipe.profileRoute.scanRoute) &&
                directProduct.recipe.hash != 0 &&
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
            printBuilt &&
                Spektrafilm::scan_route_is_print(
                    printProduct.recipe.profileRoute.scanRoute) &&
                printProduct.recipe.hash != 0 &&
                printProduct.payload.uploadCoreHash != 0 &&
                printProduct.payload.scannerHash != 0,
            printDiagnostic);

        struct GammaBuildCase {
            const char* name;
            Spektrafilm::ScanRoute route;
            float filmGamma;
            double printGamma;
        };
        constexpr std::array<GammaBuildCase, 4> gammaBuildCases{{{"build-product/gamma-negative-direct-film-minimum", Spektrafilm::ScanRoute::NegativeDirectScan, static_cast<float>(Spektrafilm::kFilmGammaFactorMinimum), 1.0},
                                                                 {"build-product/gamma-positive-direct-film-maximum", Spektrafilm::ScanRoute::PositiveDirectScan, static_cast<float>(Spektrafilm::kFilmGammaFactorMaximum), 1.0},
                                                                 {"build-product/gamma-negative-print-minimum", Spektrafilm::ScanRoute::NegativePrintScan, 1.0f, Spektrafilm::kPrintGammaFactorMinimum},
                                                                 {"build-product/gamma-positive-print-maximum", Spektrafilm::ScanRoute::PositivePrintScan, 1.0f, Spektrafilm::kPrintGammaFactorMaximum}}};
        for (const GammaBuildCase& testCase : gammaBuildCases) {
            ParamSnapshot gammaSnapshot;
            gammaSnapshot.scanRoute = testCase.route;
            gammaSnapshot.filmGammaFactor = testCase.filmGamma;
            gammaSnapshot.printGammaFactor = testCase.printGamma;
            if (Spektrafilm::scan_route_metadata(testCase.route).capturePolarity ==
                Spektrafilm::ProfilePolarity::Positive) {
                gammaSnapshot.filmProfileKey = "fujifilm_provia_100f";
            }
            FocusedRenderStateBuildProduct gammaProduct;
            std::string gammaDiagnostic;
            const bool gammaBuilt = Spektrafilm::scan_route_is_print(testCase.route)
                                        ? build_print_render_state_product(
                                              gammaSnapshot,
                                              gammaProduct,
                                              gammaDiagnostic)
                                        : build_direct_render_state_product(
                                              gammaSnapshot,
                                              gammaProduct,
                                              gammaDiagnostic);
            results.record(
                testCase.name,
                gammaBuilt,
                gammaDiagnostic);
        }

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
            requestedBuilt && requestedProduct.recipe.hash != 0 &&
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

        // Keep an independent equal-valued object so this remains an identity oracle.
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        ParamSnapshot equalDirect = direct;
        ParamSnapshot differentFilm = direct;
        differentFilm.filmProfileKey = "kodak_portra_160";
        ParamSnapshot inactivePrintSelection = direct;
        inactivePrintSelection.printProfileKey = "kodak_supra_endura";
        ParamSnapshot alternatePrint = print;
        alternatePrint.printProfileKey = "kodak_supra_endura";
        results.record(
            "profile-selection/meaningful-hash-identity",
            hash_params(direct) == hash_params(equalDirect) &&
                hash_params(direct) != hash_params(differentFilm) &&
                hash_params(direct) == hash_params(inactivePrintSelection) &&
                hash_params(print) != hash_params(alternatePrint),
            "film keys and active print keys must distinguish identity; inactive print selection must not");

        ParamSnapshot missingProfile = direct;
        missingProfile.filmProfileKey = "missing_s07_profile";
        FocusedRenderStateBuildProduct missingProduct;
        std::string missingDiagnostic;
        const bool missingBuilt = build_direct_render_state_product(
            missingProfile,
            missingProduct,
            missingDiagnostic);
        InstanceState missingState;
        seed_valid_pending(missingState, missingProfile);
        const PendingRenderAdmissionResult missingAdmission =
            admit_pending_render_state(missingState);
        results.record(
            "profile-selection/missing-selected-asset-fails-construction",
            !missingBuilt && !missingDiagnostic.empty() &&
                missingAdmission.status == PendingRenderAdmissionStatus::RebuildFailed &&
                !missingAdmission.diagnostic.empty() &&
                !missingAdmission.directState && !missingAdmission.printState,
            missingDiagnostic.empty() ? missingAdmission.diagnostic : missingDiagnostic);
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

        InstanceState failedAfterState;
        seed_valid_pending(failedAfterState, direct);
        const PendingRenderAdmissionResult validBeforeFailure =
            admit_pending_render_state(failedAfterState);
        ParamSnapshot missingAfterValid = direct;
        missingAfterValid.filmProfileKey = "missing_after_valid_profile";
        seed_valid_pending(failedAfterState, missingAfterValid);
        const PendingRenderAdmissionResult failedCurrent =
            admit_pending_render_state(failedAfterState);
        results.record(
            "pending/rebuild-failure-does-not-admit-previous-state",
            validBeforeFailure.status == PendingRenderAdmissionStatus::AdmittedDirect &&
                validBeforeFailure.directState &&
                validBeforeFailure.directState->recipe.hash != 0 &&
                failedCurrent.status == PendingRenderAdmissionStatus::RebuildFailed &&
                !failedCurrent.directState && !failedCurrent.printState,
            failedCurrent.diagnostic);

        InstanceState retryState;
        ParamSnapshot edited = direct;
        edited.cameraFilmFormatLongEdgeMm = 36.0f;
        seed_valid_pending(retryState, direct);
        PendingRenderAdmissionResult retryResult;
        struct CaptureGate {
            InstanceState* selected = nullptr;
            std::mutex mutex;
            std::condition_variable condition;
            bool captured = false;
            bool released = false;
        } gate;
        gate.selected = &retryState;
        set_pending_capture_test_hook(
            [](InstanceState& capturedState, void* context) {
                auto& captureGate = *static_cast<CaptureGate*>(context);
                if (&capturedState != captureGate.selected) {
                    return;
                }
                std::unique_lock<std::mutex> lock(captureGate.mutex);
                captureGate.captured = true;
                captureGate.condition.notify_one();
                captureGate.condition.wait(lock, [&] {
                    return captureGate.released;
                });
            },
            &gate);
        bool capturedBeforeEdit = false;
        {
            std::unique_lock<std::mutex> holdRebuild(retryState.rebuildMutex);
            std::thread admission([&] {
                retryResult = admit_pending_render_state(retryState);
            });
            {
                std::unique_lock<std::mutex> lock(gate.mutex);
                capturedBeforeEdit = gate.condition.wait_for(
                    lock,
                    std::chrono::seconds(10),
                    [&] {
                        return gate.captured;
                    });
            }
            if (capturedBeforeEdit) {
                seed_valid_pending(retryState, edited);
            }
            {
                std::lock_guard<std::mutex> lock(gate.mutex);
                gate.released = true;
            }
            gate.condition.notify_one();
            holdRebuild.unlock();
            admission.join();
        }
        set_pending_capture_test_hook(nullptr, nullptr);
        results.record(
            "pending/edit-during-rebuild-retries-current",
            capturedBeforeEdit &&
                retryResult.status == PendingRenderAdmissionStatus::AdmittedDirect &&
                retryResult.directState &&
                retryResult.snapshot.cameraFilmFormatLongEdgeMm == 36.0f &&
                retryState.lastHash.load(std::memory_order_acquire) ==
                    hash_params(edited) &&
                retryState.buildCounterNext.load(std::memory_order_relaxed) >= 2,
            retryResult.diagnostic);
    }

    struct FreshIdentity {
        std::uint64_t recipe;
        std::uint64_t filmRaw;
        std::uint64_t printSeedInput;
        std::uint64_t fixedGlareSeed;
    };

    void run_parameter_identity_pair(
        Results& results,
        std::string_view name,
        const ParamSnapshot& a,
        const ParamSnapshot& b,
        bool signedZero,
        std::optional<std::array<FreshIdentity, 2>> accepted) {
        const bool printRoute = Spektrafilm::scan_route_is_print(a.scanRoute);
        FocusedRenderStateBuildProduct freshA;
        FocusedRenderStateBuildProduct freshB;
        std::string diagnosticA;
        std::string diagnosticB;
        const bool builtA = printRoute
                                ? build_print_render_state_product(a, freshA, diagnosticA)
                                : build_direct_render_state_product(a, freshA, diagnosticA);
        const bool builtB = printRoute
                                ? build_print_render_state_product(b, freshB, diagnosticB)
                                : build_direct_render_state_product(b, freshB, diagnosticB);
        if (!builtA || !builtB) {
            results.record(std::string(name) + "/fresh-build", false, diagnosticA + " " + diagnosticB);
            return;
        }
        std::ostringstream identity;
        identity << std::hex << "A recipe=" << freshA.recipe.hash
                 << " filmRaw=" << freshA.recipe.filmRaw.hash
                 << " printSeedInput=" << freshA.recipe.print.hash
                 << " B recipe=" << freshB.recipe.hash
                 << " filmRaw=" << freshB.recipe.filmRaw.hash
                 << " printSeedInput=" << freshB.recipe.print.hash;
        const bool freshDiffer = freshA.recipe.hash != freshB.recipe.hash;
        results.record(std::string(name) + "/fresh-distinct",
                       freshDiffer,
                       identity.str());
        const auto agrees_with_accepted = [&](const RenderRecipe& recipe,
                                              const FreshIdentity& expected) {
            const std::uint64_t fixedSeed = printRoute
                                                ? Hash::hash_uint64_values({recipe.print.hash, 37, 5, 9})
                                                : 0;
            return RenderAssertions::identities_match(
                {recipe.hash, recipe.filmRaw.hash, recipe.print.hash, fixedSeed},
                {expected.recipe, expected.filmRaw, expected.printSeedInput, expected.fixedGlareSeed});
        };
        if (accepted) {
            results.record(std::string(name) + "/accepted-rust-identities",
                           agrees_with_accepted(freshA.recipe, (*accepted)[0]) &&
                               agrees_with_accepted(freshB.recipe, (*accepted)[1]),
                           identity.str());
        }
        for (bool reverse : {false, true}) {
            InstanceState state;
            seed_valid_pending(state, reverse ? b : a);
            const PendingRenderAdmissionResult first = admit_pending_render_state(state);
            seed_valid_pending(state, reverse ? a : b);
            const PendingRenderAdmissionResult destination =
                admit_pending_render_state(state);
            const RenderRecipe* actual = printRoute
                                             ? (destination.printState
                                                    ? &destination.printState->recipe
                                                    : nullptr)
                                             : (destination.directState
                                                    ? &destination.directState->recipe
                                                    : nullptr);
            const RenderRecipe& expected = reverse ? freshA.recipe : freshB.recipe;
            const std::uint64_t firstBuild = printRoute
                                                 ? (first.printState ? first.printState->buildCounter : 0)
                                                 : (first.directState ? first.directState->buildCounter : 0);
            const std::uint64_t destinationBuild = printRoute
                                                       ? (destination.printState
                                                              ? destination.printState->buildCounter
                                                              : 0)
                                                       : (destination.directState
                                                              ? destination.directState->buildCounter
                                                              : 0);
            results.record(
                std::string(name) + (reverse ? "/B-to-A" : "/A-to-B"),
                freshDiffer && firstBuild != 0 && destinationBuild > firstBuild &&
                    actual && actual->hash == expected.hash &&
                    actual->filmRaw.hash == expected.filmRaw.hash &&
                    actual->print.hash == expected.print.hash &&
                    (!signedZero ||
                     float_bits(actual->filmRaw.manualExposureCompensationEv) ==
                         float_bits(expected.filmRaw.manualExposureCompensationEv)),
                destination.diagnostic);
        }
    }

    void run_parameter_identity_rows(Results& results) {
        std::ifstream fixture(JUICER_PARAMETER_IDENTITY_PATH);
        if (!fixture) {
            throw std::runtime_error("Rust parameter identity fixture unavailable");
        }
        const auto identities = nlohmann::json::parse(fixture);
        const auto pair = [&](const char* name) {
            std::array<FreshIdentity, 2> expected{};
            for (std::size_t i = 0; i < expected.size(); ++i) {
                const auto& row = identities.at(std::string(name) + (i == 0 ? "/A" : "/B"));
                expected[i] = {row.at("recipe").get<std::uint64_t>(), row.at("filmRaw").get<std::uint64_t>(), row.at("printSeedInput").get<std::uint64_t>(), row.at("fixedGlareSeed").get<std::uint64_t>()};
            }
            return expected;
        };
        const auto kDirectSmall = pair("direct-exposure-small");
        const auto kDirectZero = pair("direct-exposure-zero");
        const auto kPrintSmall = pair("print-exposure-small");
        const auto kPrintZero = pair("print-exposure-zero");
        const auto kPrintMediumSmall = pair("print-medium-exposure-small");
        for (bool printRoute : {false, true}) {
            ParamSnapshot a = direct_snapshot();
            if (printRoute) {
                a.scanRoute = Spektrafilm::ScanRoute::NegativePrintScan;
            }
            a.cameraExposureCompensationEv = 1.0;
            ParamSnapshot b = a;
            b.cameraExposureCompensationEv = 1.00001;
            run_parameter_identity_pair(
                results, printRoute ? "identity/print-exposure-small" : "identity/direct-exposure-small", a, b, false, printRoute ? kPrintSmall : kDirectSmall);

            a.cameraExposureCompensationEv = 0.0;
            b = a;
            b.cameraExposureCompensationEv = -0.0;
            run_parameter_identity_pair(
                results, printRoute ? "identity/print-exposure-zero" : "identity/direct-exposure-zero", a, b, true, printRoute ? kPrintZero : kDirectZero);
        }

        ParamSnapshot a = direct_snapshot();
        a.scanRoute = Spektrafilm::ScanRoute::NegativePrintScan;
        a.printExposure = 1.0;
        ParamSnapshot b = a;
        b.printExposure = 1.00001;
        run_parameter_identity_pair(
            results, "identity/print-medium-exposure-small", a, b, false, kPrintMediumSmall);

        InstanceState sameEffectiveState;
        a = direct_snapshot();
        a.cameraExposureCompensationEv = 1.0;
        b = a;
        b.cameraExposureCompensationEv = std::nextafter(1.0, 2.0);
        seed_valid_pending(sameEffectiveState, a);
        const PendingRenderAdmissionResult first =
            admit_pending_render_state(sameEffectiveState);
        seed_valid_pending(sameEffectiveState, b);
        const PendingRenderAdmissionResult sameEffective =
            admit_pending_render_state(sameEffectiveState);
        results.record(
            "identity/same-retained-float32-reuses",
            a.cameraExposureCompensationEv != b.cameraExposureCompensationEv &&
                float_bits(a.cameraExposureCompensationEv) ==
                    float_bits(b.cameraExposureCompensationEv) &&
                hash_params(a) == hash_params(b) &&
                first.directState && sameEffective.directState &&
                first.directState->buildCounter ==
                    sameEffective.directState->buildCounter,
            sameEffective.diagnostic);

        ParamSnapshot inactive = a;
        inactive.cameraFilterUV[0] = 0.0;
        inactive.grainControls.particleAreaUm2 =
            std::nextafter(inactive.grainControls.particleAreaUm2, 100.0f);
        inactive.printProfileKey = "kodak_supra_endura";
        results.record(
            "identity/inactive-family-equivalence",
            !a.cameraFilterOverride && !a.grainControls.active &&
                hash_params(a) == hash_params(inactive),
            "inactive camera filter, grain, and print selection remain outside the key");
    }

    void run_parameter_sign_rows(Results& results) {
        enum class Setup : std::uint8_t {
            Direct,
            Print,
            Grain,
            Effects,
            Filter
        };
        struct SignCase {
            const char* name;
            Setup setup;
            void (*set)(ParamSnapshot&, double);
        };
        const std::array<SignCase, 17> cases{{{"print-preflash-exposure", Setup::Print, [](ParamSnapshot& p, double v) {
                                                   p.printPreflashExposure = v;
                                               }},
                                              {"print-ui-y", Setup::Print, [](ParamSnapshot& p, double v) {
                                                   p.printUiYmcCc[0] = v;
                                               }},
                                              {"preflash-m", Setup::Print, [](ParamSnapshot& p, double v) {
                                                   p.preflashMFilterCc = v;
                                               }},
                                              {"preflash-y", Setup::Print, [](ParamSnapshot& p, double v) {
                                                   p.preflashYFilterCc = v;
                                               }},
                                              {"glare-percent", Setup::Print, [](ParamSnapshot& p, double v) {
                                                   p.glarePercent = v;
                                               }},
                                              {"glare-roughness", Setup::Print, [](ParamSnapshot& p, double v) {
                                                   p.glareRoughness = v;
                                               }},
                                              {"glare-blur", Setup::Print, [](ParamSnapshot& p, double v) {
                                                   p.glareBlurSigmaPx = v;
                                               }},
                                              {"scanner-lens-blur", Setup::Direct, [](ParamSnapshot& p, double v) {
                                                   p.scannerLensBlurSigmaPx = v;
                                               }},
                                              {"scanner-unsharp-sigma", Setup::Direct, [](ParamSnapshot& p, double v) {
                                                   p.scannerUnsharpMask[0] = v;
                                               }},
                                              {"scanner-unsharp-amount", Setup::Direct, [](ParamSnapshot& p, double v) {
                                                   p.scannerUnsharpMask[1] = v;
                                               }},
                                              {"scanner-black-level", Setup::Direct, [](ParamSnapshot& p, double v) {
                                                   p.scannerBlackLevel = v;
                                               }},
                                              {"scanner-white-level", Setup::Direct, [](ParamSnapshot& p, double v) {
                                                   p.scannerWhiteLevel = v;
                                               }},
                                              {"gate-weave", Setup::Effects, [](ParamSnapshot& p, double v) {
                                                   p.gateWeaveAmount = v;
                                               }},
                                              {"film-dust", Setup::Effects, [](ParamSnapshot& p, double v) {
                                                   p.filmDustAmount = static_cast<float>(v);
                                               }},
                                              {"grain-amplitude", Setup::Grain, [](ParamSnapshot& p, double v) {
                                                   p.grainControls.amplitude = static_cast<float>(v);
                                               }},
                                              {"camera-filter-uv", Setup::Filter, [](ParamSnapshot& p, double v) {
                                                   p.cameraFilterUV[0] = v;
                                               }},
                                              {"camera-filter-ir", Setup::Filter, [](ParamSnapshot& p, double v) {
                                                   p.cameraFilterIR[0] = v;
                                               }}}};
        for (const SignCase& signCase : cases) {
            ParamSnapshot positive = direct_snapshot();
            if (signCase.setup == Setup::Print) {
                positive.scanRoute = Spektrafilm::ScanRoute::NegativePrintScan;
            } else if (signCase.setup == Setup::Grain) {
                positive.grainControls.active = true;
            } else if (signCase.setup == Setup::Effects) {
                positive.filmDustAmount = 1.0f;
                positive.gateWeaveAmount = 1.0;
            } else if (signCase.setup == Setup::Filter) {
                positive.cameraFilterOverride = true;
            }
            signCase.set(positive, 0.0);
            ParamSnapshot negative = positive;
            signCase.set(negative, -0.0);
            FocusedRenderStateBuildProduct freshPositive;
            FocusedRenderStateBuildProduct freshNegative;
            std::string positiveDiagnostic;
            std::string negativeDiagnostic;
            const bool printRoute = signCase.setup == Setup::Print;
            const bool positiveBuilt = printRoute
                                           ? build_print_render_state_product(positive, freshPositive, positiveDiagnostic)
                                           : build_direct_render_state_product(positive, freshPositive, positiveDiagnostic);
            const bool negativeBuilt = printRoute
                                           ? build_print_render_state_product(negative, freshNegative, negativeDiagnostic)
                                           : build_direct_render_state_product(negative, freshNegative, negativeDiagnostic);
            const std::string name = std::string("identity/sign/") + signCase.name;
            if (!positiveBuilt || !negativeBuilt) {
                positiveDiagnostic.append(" ").append(negativeDiagnostic);
                results.record(name + "/fresh-build", false, positiveDiagnostic);
                continue;
            }
            const bool recipeDiffers = freshPositive.recipe.hash != freshNegative.recipe.hash;
            const bool keyDiffers = hash_params(positive) != hash_params(negative);
            results.record(name + "/key-follows-recipe", recipeDiffers == keyDiffers, std::string("recipeDiffers=") + (recipeDiffers ? "1" : "0") + " keyDiffers=" + (keyDiffers ? "1" : "0"));
            if (recipeDiffers) {
                run_parameter_identity_pair(
                    results, name, positive, negative, true, std::nullopt);
            }
        }
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

    void run_capacity_recovery_classification_rows(Results& results) {
        using JuicerCuda::allocation_capacity_exhausted;
        const auto capacity = JuicerCuda::runtime_failure_status(cudaErrorMemoryAllocation);
        results.record(
            "recovery/capacity-producer-category",
            allocation_capacity_exhausted({capacity, "renamed diagnostic"}) &&
                allocation_capacity_exhausted({{FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "device_cap_exceeded"}),
            "typed capacity activates retired-allocation reap and bounded retry");
        results.record(
            "recovery/non-capacity-errors-do-not-retry",
            !allocation_capacity_exhausted({{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "out of memory memory allocation device_cap_exceeded"}),
            "diagnostic wording cannot select capacity retry");
    }

    struct PreparedInputs {
        FocusedRenderStateBuildProduct product;
        Scanner::ScannerSpectralLutDescriptor scannerDescriptor{};
        std::optional<ScatterHalationFrameDescriptor> scatterDescriptor;
        std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusionFrameSet;

        JuicerProcess::Root::CudaFramePreparationRequest request(
            ScatterHalationValidation::ImageExtent extent) const {
            JuicerProcess::Root::CudaFramePreparationRequest value{};
            value.recipe = &product.recipe;
            value.exposureTables = &product.payload.exposureTables;
            value.filmRawConfig = &product.payload.filmRawConfig;
            value.filmTcLut = product.payload.filmTcLut
                                  ? &*product.payload.filmTcLut
                                  : nullptr;
            value.printMainIlluminant = product.payload.printMainIlluminant
                                            ? &*product.payload.printMainIlluminant
                                            : nullptr;
            value.scannerTables = &product.payload.scannerTables;
            value.scannerColor = &product.payload.scannerColor;
            value.scannerLutDescriptor = &scannerDescriptor;
            value.outputBoundaryTable = product.payload.outputBoundaryTable.get();
            value.diffusionFrameSetDescriptor =
                diffusionFrameSet ? &*diffusionFrameSet : nullptr;
            value.scatterHalationDescriptor =
                scatterDescriptor ? &*scatterDescriptor : nullptr;
            value.requestedWidth = extent.width;
            value.requestedHeight = extent.height;
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

    bool wait_for_cuda_event(
        cudaEvent_t event,
        std::chrono::milliseconds timeout) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            const cudaError_t status = cudaEventQuery(event);
            if (status == cudaSuccess) {
                return true;
            }
            if (status != cudaErrorNotReady) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return false;
    }

    class FinitePostPrepareUpload final {
    public:
        ~FinitePostPrepareUpload() {
            release();
            const bool completed =
                _completionEvent &&
                wait_for_cuda_event(
                    _completionEvent,
                    std::chrono::seconds(12));
            if (completed && _staging) {
                (void)cudaFreeHost(_staging);
                _staging = nullptr;
            }
            if (_completionEvent) {
                (void)cudaEventDestroy(_completionEvent);
            }
            if (_gateEvent) {
                (void)wait_for_cuda_event(
                    _gateEvent,
                    std::chrono::seconds(12));
                (void)cudaEventDestroy(_gateEvent);
            }
            if (_gateStream) {
                (void)cudaStreamDestroy(_gateStream);
            }
        }

        bool enqueue(
            cudaStream_t targetStream,
            float* destination,
            const std::vector<float>& source,
            std::string& diagnostic) {
            diagnostic.clear();
            if (!targetStream || !destination || source.empty() ||
                _gateStream || _gateEvent || _completionEvent || _staging) {
                diagnostic = "invalid post-prepare upload request";
                return false;
            }
            const std::size_t bytes = source.size() * sizeof(float);
            cudaError_t status = cudaHostAlloc(
                &_staging,
                bytes,
                cudaHostAllocPortable);
            if (status == cudaSuccess) {
                std::memcpy(_staging, source.data(), bytes);
                status = cudaStreamCreateWithFlags(
                    &_gateStream,
                    cudaStreamNonBlocking);
            }
            if (status == cudaSuccess) {
                status = cudaEventCreateWithFlags(
                    &_gateEvent,
                    cudaEventDisableTiming);
            }
            if (status == cudaSuccess) {
                status = cudaEventCreateWithFlags(
                    &_completionEvent,
                    cudaEventDisableTiming);
            }
            if (status == cudaSuccess) {
                status = cudaLaunchHostFunc(
                    _gateStream,
                    &FinitePostPrepareUpload::host_gate,
                    this);
            }
            if (status == cudaSuccess) {
                status = cudaEventRecord(_gateEvent, _gateStream);
            }
            if (status == cudaSuccess) {
                status = cudaStreamWaitEvent(targetStream, _gateEvent, 0);
            }
            if (status == cudaSuccess) {
                status = cudaMemcpyAsync(
                    destination,
                    _staging,
                    bytes,
                    cudaMemcpyHostToDevice,
                    targetStream);
            }
            if (status == cudaSuccess) {
                status = cudaEventRecord(_completionEvent, targetStream);
            }
            if (status == cudaSuccess) {
                return true;
            }
            diagnostic = std::string("post-prepare upload setup failed: ") +
                         cudaGetErrorString(status);
            release();
            return false;
        }

        cudaError_t query() const noexcept {
            return _completionEvent
                       ? cudaEventQuery(_completionEvent)
                       : cudaErrorInvalidResourceHandle;
        }

        bool wait(std::chrono::milliseconds timeout) const {
            return _completionEvent &&
                   wait_for_cuda_event(_completionEvent, timeout);
        }

        void release() noexcept {
            try {
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    _released = true;
                }
                _condition.notify_all();
            } catch (...) {
                _releaseFailed.store(true, std::memory_order_release);
            }
        }

        bool self_released() const noexcept {
            return _selfReleased.load(std::memory_order_acquire);
        }

        bool release_failed() const noexcept {
            return _releaseFailed.load(std::memory_order_acquire);
        }

    private:
        static void CUDART_CB host_gate(void* opaque) {
            auto& upload = *static_cast<FinitePostPrepareUpload*>(opaque);
            std::unique_lock<std::mutex> lock(upload._mutex);
            if (!upload._condition.wait_for(
                    lock,
                    std::chrono::seconds(10),
                    [&] {
                        return upload._released;
                    })) {
                upload._selfReleased.store(true, std::memory_order_release);
                upload._released = true;
            }
        }

        mutable std::mutex _mutex;
        std::condition_variable _condition;
        cudaStream_t _gateStream = nullptr;
        cudaEvent_t _gateEvent = nullptr;
        cudaEvent_t _completionEvent = nullptr;
        void* _staging = nullptr;
        bool _released = false;
        std::atomic<bool> _selfReleased{false};
        std::atomic<bool> _releaseFailed{false};
    };


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

    void run_prepared_frame_rows(Results& results) {
        constexpr int kWidth = 32;
        constexpr int kHeight = 24;
        const std::uintptr_t planeBytes =
            static_cast<std::uintptr_t>(kWidth) * kHeight * sizeof(float);

        CudaPreparedFixture fixture;
        JuicerCuda::Failure diagnostic;
        if (!fixture.initialize(diagnostic.diagnostic)) {
            throw std::runtime_error(diagnostic.diagnostic);
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
                diagnostic.diagnostic)) {
            throw std::runtime_error(diagnostic.diagnostic);
        }
        auto zeroRequest = zeroInputs.request({kWidth, kHeight});
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
            diagnostic.diagnostic);
        JuicerCuda::Failure terminalDiagnostic;
        const bool zeroFinished =
            zeroFrame.finish(fixture.stream_opaque(), terminalDiagnostic) &&
            fixture.synchronize(terminalDiagnostic.diagnostic);
        results.record(
            "prepared/finish-cleanup-zero",
            zeroFinished &&
                zeroFrame.scatter_halation_resources().descriptor == nullptr,
            terminalDiagnostic.diagnostic);

        PreparedInputs dedicatedInputs;
        if (!initialize_prepared_inputs(
                false,
                false,
                true,
                kWidth,
                kHeight,
                dedicatedInputs,
                diagnostic.diagnostic)) {
            throw std::runtime_error(diagnostic.diagnostic);
        }
        auto dedicatedRequest = dedicatedInputs.request({kWidth, kHeight});
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
                dedicatedView.descriptor &&
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
        terminalDiagnostic = {};
        const bool dedicatedFinished =
            dedicatedFrame.finish(fixture.stream_opaque(), terminalDiagnostic) &&
            fixture.synchronize(terminalDiagnostic.diagnostic);
        results.record(
            "prepared/finish-cleanup",
            dedicatedFinished &&
                dedicatedFrame.scatter_halation_resources().descriptor == nullptr,
            terminalDiagnostic.diagnostic);

        PreparedInputs cameraInputs;
        if (!initialize_prepared_inputs(
                false,
                true,
                true,
                kWidth,
                kHeight,
                cameraInputs,
                diagnostic.diagnostic)) {
            throw std::runtime_error(diagnostic.diagnostic);
        }
        auto cameraRequest = cameraInputs.request({kWidth, kHeight});
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
        terminalDiagnostic = {};
        const bool cameraFinished =
            cameraFrame.finish(fixture.stream_opaque(), terminalDiagnostic) &&
            fixture.synchronize(terminalDiagnostic.diagnostic);
        results.record(
            "prepared/camera-finish-cleanup",
            cameraFinished &&
                cameraFrame.scatter_halation_resources().descriptor == nullptr &&
                !cameraFrame.diffusion_resources().active,
            terminalDiagnostic.diagnostic);

        auto cameraOverlapFirst = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(cameraInputs),
            cameraRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const std::string cameraOverlapFirstDiagnostic = diagnostic.diagnostic;
        auto cameraOverlapSecond = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(cameraInputs),
            cameraRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const std::string cameraOverlapSecondDiagnostic = diagnostic.diagnostic;
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
        terminalDiagnostic = {};
        const bool cameraOverlapFinished =
            cameraOverlapFirst.finish(
                fixture.stream_opaque(),
                terminalDiagnostic) &&
            fixture.synchronize(terminalDiagnostic.diagnostic);
        results.record(
            "prepared/camera-overlap-finish-cleanup",
            cameraOverlapFinished,
            terminalDiagnostic.diagnostic);

        auto absentRequest = dedicatedInputs.request({kWidth, kHeight});
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
                contains_text(diagnostic.diagnostic, "MissingRequiredResource") &&
                contains_text(diagnostic.diagnostic, "field=descriptor") &&
                absentFrame.scatter_halation_resources().descriptor == nullptr,
            diagnostic.diagnostic);

        auto presentForZeroRequest = zeroInputs.request({kWidth, kHeight});
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
                contains_text(diagnostic.diagnostic, "field=recipe_hash") &&
                contains_text(diagnostic.diagnostic, "expected=0") &&
                presentForZeroFrame.scatter_halation_resources().descriptor ==
                    nullptr,
            diagnostic.diagnostic);

        auto mismatched = *dedicatedInputs.scatterDescriptor;
        ++mismatched.recipeHash;
        auto mismatchRequest = dedicatedInputs.request({kWidth, kHeight});
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
                contains_text(diagnostic.diagnostic, "field=recipe_hash") &&
                mismatchFrame.scatter_halation_resources().descriptor == nullptr,
            diagnostic.diagnostic);

        auto zeroExtentRequest = dedicatedInputs.request({0, kHeight});
        auto zeroExtentFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            zeroExtentRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        results.record(
            "prepared/zero-extent",
            !zeroExtentFrame.active() && contains_text(diagnostic.diagnostic, "field=extent"),
            diagnostic.diagnostic);
        auto negativeExtentRequest = dedicatedInputs.request({-1, kHeight});
        auto negativeExtentFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            negativeExtentRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        results.record(
            "prepared/negative-extent",
            !negativeExtentFrame.active() && contains_text(diagnostic.diagnostic, "field=extent"),
            diagnostic.diagnostic);

        auto overflowRequest = dedicatedInputs.request(
            {std::numeric_limits<int>::max(),
             std::numeric_limits<int>::max()});
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
                contains_text(diagnostic.diagnostic, "ExactAdmissionFailure") &&
                contains_text(diagnostic.diagnostic, "failed_fact=filter_bytes") &&
                contains_text(diagnostic.diagnostic, "bytes_admitted_before_failure=0") &&
                contains_text(diagnostic.diagnostic, "physical_allocation_count=2"),
            diagnostic.diagnostic);

        auto incompatibleFrameSet = *cameraInputs.diffusionFrameSet;
        incompatibleFrameSet.camera->stage =
            Spektrafilm::DiffusionLinearStage::EnlargerPrintLinear;
        auto incompatibleRequest = cameraInputs.request({kWidth, kHeight});
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
                contains_text(diagnostic.diagnostic, "ExactAdmissionFailure") &&
                contains_text(diagnostic.diagnostic, "failed_fact=camera_carrier_stage") &&
                contains_text(diagnostic.diagnostic, "bytes_admitted_before_failure=0"),
            diagnostic.diagnostic);

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
        terminalDiagnostic = {};
        const bool overlapFinished =
            overlapFirst.finish(fixture.stream_opaque(), terminalDiagnostic) &&
            overlapSecond.finish(fixture.stream_opaque(), terminalDiagnostic) &&
            fixture.synchronize(terminalDiagnostic.diagnostic);
        results.record(
            "prepared/overlap-finish-cleanup",
            overlapFinished,
            terminalDiagnostic.diagnostic);

        auto nullStreamFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            dedicatedRequest,
            {},
            nullptr,
            diagnostic);
        terminalDiagnostic = {};
        const bool nullStreamFinished =
            nullStreamFrame.active() &&
            nullStreamFrame.finish(nullptr, terminalDiagnostic) &&
            cudaDeviceSynchronize() == cudaSuccess;
        results.record(
            "prepared/null-stream-finish",
            nullStreamFinished,
            terminalDiagnostic.diagnostic);

        cudaStream_t secondStream = nullptr;
        const cudaError_t secondStreamStatus =
            cudaStreamCreateWithFlags(&secondStream, cudaStreamNonBlocking);
        bool sequentialStreamsFinished = secondStreamStatus == cudaSuccess;
        if (sequentialStreamsFinished) {
            auto firstStreamFrame = root.prepare_cuda_frame(
                fixture.context_key(),
                fixture.snapshot(dedicatedInputs),
                dedicatedRequest,
                {},
                fixture.stream_opaque(),
                diagnostic);
            sequentialStreamsFinished =
                firstStreamFrame.active() &&
                firstStreamFrame.finish(fixture.stream_opaque(), terminalDiagnostic);
            if (sequentialStreamsFinished) {
                auto secondStreamFrame = root.prepare_cuda_frame(
                    fixture.context_key(),
                    fixture.snapshot(dedicatedInputs),
                    dedicatedRequest,
                    {},
                    secondStream,
                    diagnostic);
                sequentialStreamsFinished =
                    secondStreamFrame.active() &&
                    secondStreamFrame.finish(secondStream, terminalDiagnostic) &&
                    fixture.synchronize(terminalDiagnostic.diagnostic) &&
                    cudaStreamSynchronize(secondStream) == cudaSuccess;
            }
            (void)cudaStreamDestroy(secondStream);
        }
        results.record(
            "prepared/sequential-streams-same-context",
            sequentialStreamsFinished,
            sequentialStreamsFinished ? "both streams finished in one exact context"
                                      : terminalDiagnostic.diagnostic);

        auto abortFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            dedicatedRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const bool abortWasActive = abortFrame.active();
        abortFrame.abort();
        terminalDiagnostic = {};
        results.record(
            "prepared/abort-cleanup",
            abortWasActive && !abortFrame.active() &&
                abortFrame.scatter_halation_resources().descriptor == nullptr &&
                fixture.synchronize(terminalDiagnostic.diagnostic),
            terminalDiagnostic.diagnostic);

        auto dependentRequest = dedicatedInputs.request({kWidth, kHeight});
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

        {
            std::optional<PreparedInputs> pendingUploadInputs;
            pendingUploadInputs.emplace();
            if (!initialize_prepared_inputs(
                    false,
                    false,
                    true,
                    kWidth,
                    kHeight,
                    *pendingUploadInputs,
                    diagnostic.diagnostic)) {
                throw std::runtime_error(diagnostic.diagnostic);
            }
            auto pendingUploadRequest =
                pendingUploadInputs->request({kWidth, kHeight});
            const std::string expectedRetainedProfileKey =
                pendingUploadInputs->product.recipe.profileRoute.filmProfileKey;
            auto pendingUploadFrame = root.prepare_cuda_frame(
                fixture.context_key(),
                fixture.snapshot(*pendingUploadInputs),
                pendingUploadRequest,
                {},
                fixture.stream_opaque(),
                diagnostic);
            auto finalCpuBorrow =
                pendingUploadFrame.focused_resources();
            const std::uint64_t borrowedScannerColorHash =
                finalCpuBorrow.scannerColor
                    ? finalCpuBorrow.scannerColor->hash
                    : 0;
            auto mutablePreparedView =
                pendingUploadFrame.scatter_halation_resources();
            const std::uintptr_t pendingCarrierAddress =
                reinterpret_cast<std::uintptr_t>(
                    mutablePreparedView.currentCarrier.redSensitive);
            int* pendingScanErrorFlag = nullptr;
            const bool scanErrorPrepared =
                pendingUploadFrame.prepare_scan_error_stage(
                    pendingScanErrorFlag,
                    fixture.stream_opaque(),
                    diagnostic) == JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Ready;
            std::vector<float> retainedCarrierUpload(
                static_cast<std::size_t>(kWidth * kHeight),
                0.0f);
            FinitePostPrepareUpload outstandingUpload;
            const bool uploadEnqueued =
                scanErrorPrepared && finalCpuBorrow.active &&
                mutablePreparedView.currentCarrier.redSensitive &&
                outstandingUpload.enqueue(
                    static_cast<cudaStream_t>(fixture.stream_opaque()),
                    mutablePreparedView.currentCarrier.redSensitive,
                    retainedCarrierUpload,
                    diagnostic.diagnostic);
            const bool scanErrorFinalized =
                uploadEnqueued &&
                pendingUploadFrame.finalize_scan_error_stage(
                    pendingScanErrorFlag,
                    fixture.stream_opaque(),
                    diagnostic);
            const cudaError_t initialUploadQuery =
                scanErrorFinalized ? outstandingUpload.query()
                                   : cudaErrorInvalidResourceHandle;
            const auto beforePendingFinish =
                JuicerProcess::TestSupport::RootLifetimeObserver::snapshot(
                    root,
                    fixture.context_key());
            results.record(
                "prepared/pending-upload-queried",
                pendingUploadFrame.active() && scanErrorFinalized &&
                    initialUploadQuery == cudaErrorNotReady &&
                    !outstandingUpload.self_released() &&
                    !outstandingUpload.release_failed(),
                "a pinned upload into Root-owned prepared carrier storage is incomplete at the queried event");
            results.record(
                "prepared/last-legal-cpu-borrow",
                finalCpuBorrow.active && borrowedScannerColorHash != 0 &&
                    borrowedScannerColorHash ==
                        pendingUploadInputs->product.payload.scannerColor.hash,
                "focused CPU route objects are borrowed before prepared-frame finish");
            results.record(
                "prepared/native-owner-ledger-before-finish",
                beforePendingFinish.contextEntryPresent &&
                    beforePendingFinish.frameOwnerPresent &&
                    beforePendingFinish.frameOwnerUseCount >= 2 &&
                    beforePendingFinish.nativeAllocationCount != 0 &&
                    beforePendingFinish.ledger.chargedBytes != 0 &&
                    beforePendingFinish.contextLedgerRecordCount != 0,
                "prepared frame and Root share the native owner while upload is pending");

            terminalDiagnostic = {};
            const bool pendingFinished = pendingUploadFrame.finish(
                fixture.stream_opaque(),
                terminalDiagnostic);
            finalCpuBorrow = {};
            mutablePreparedView = {};
            pendingUploadRequest = {};
            pendingUploadInputs.reset();
            const cudaError_t postExpiryUploadQuery =
                outstandingUpload.query();
            const auto afterPendingFinish =
                JuicerProcess::TestSupport::RootLifetimeObserver::snapshot(
                    root,
                    fixture.context_key(),
                    pendingCarrierAddress);
            results.record(
                "prepared/finish-retains-native-owner-until-event",
                pendingFinished && !pendingUploadFrame.active() &&
                    afterPendingFinish.contextEntryPresent &&
                    afterPendingFinish.frameOwnerUseCount == 1 &&
                    afterPendingFinish.nativeAllocationCount != 0 &&
                    afterPendingFinish.pendingFrameUseEventCount != 0 &&
                    afterPendingFinish.pendingScanErrorReadbackCount != 0 &&
                    afterPendingFinish.pendingScanErrorReadbackOwned &&
                    afterPendingFinish.pendingScanErrorProfileKey ==
                        expectedRetainedProfileKey &&
                    afterPendingFinish.pendingScanErrorContextEpoch ==
                        afterPendingFinish.contextEpoch &&
                    afterPendingFinish.pendingScanErrorEventQuery ==
                        cudaErrorNotReady &&
                    !afterPendingFinish.pendingScanErrorValueReady &&
                    afterPendingFinish.retireQueueCount != 0 &&
                    afterPendingFinish.retireBytes != 0 &&
                    afterPendingFinish.expectedRetireAddressPresent &&
                    afterPendingFinish.ledger.chargedBytes != 0 &&
                    afterPendingFinish.contextLedgerRecordCount != 0 &&
                    postExpiryUploadQuery == cudaErrorNotReady &&
                    !outstandingUpload.self_released(),
                terminalDiagnostic.diagnostic);
            results.record(
                "prepared/caller-storage-expired-with-owned-readback",
                !pendingUploadInputs.has_value() &&
                    postExpiryUploadQuery == cudaErrorNotReady &&
                    afterPendingFinish.pendingScanErrorReadbackOwned &&
                    afterPendingFinish.pendingScanErrorProfileKey ==
                        expectedRetainedProfileKey &&
                    afterPendingFinish.pendingScanErrorContextEpoch != 0 &&
                    !outstandingUpload.self_released(),
                "caller preparation storage expired while Root-owned readback identity and completion remained pending");

            outstandingUpload.release();
            const bool uploadCompleted =
                outstandingUpload.wait(std::chrono::seconds(10));
            auto completedReadback =
                JuicerProcess::TestSupport::RootLifetimeObserver::snapshot(
                    root,
                    fixture.context_key());
            const auto readbackDeadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (completedReadback.pendingScanErrorEventQuery ==
                       cudaErrorNotReady &&
                   std::chrono::steady_clock::now() < readbackDeadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                completedReadback =
                    JuicerProcess::TestSupport::RootLifetimeObserver::snapshot(
                        root,
                        fixture.context_key());
            }
            results.record(
                "prepared/pending-upload-eventual-completion",
                uploadCompleted && !outstandingUpload.self_released() &&
                    !outstandingUpload.release_failed(),
                "queried upload completed within the finite timeout after release");
            results.record(
                "prepared/owned-readback-eventual-completion",
                uploadCompleted &&
                    completedReadback.pendingScanErrorReadbackOwned &&
                    completedReadback.pendingScanErrorEventQuery == cudaSuccess &&
                    completedReadback.pendingScanErrorValueReady &&
                    completedReadback.pendingScanErrorValue == 0 &&
                    completedReadback.pendingScanErrorProfileKey ==
                        expectedRetainedProfileKey &&
                    completedReadback.pendingScanErrorContextEpoch ==
                        completedReadback.contextEpoch &&
                    completedReadback.ledger.chargedBytes != 0 &&
                    completedReadback.contextLedgerRecordCount != 0 &&
                    !outstandingUpload.self_released(),
                "production-owned scan-error staging completed with its retained value and identity within the finite timeout");
        }

        terminalDiagnostic = {};
        const bool failureInjected =
            JuicerProcess::TestSupport::arm_context_drain_failure_once(
                fixture.context_key());
        const bool controlledDrainFailed =
            failureInjected &&
            !root.retire_idle_context(
                fixture.context_key().deviceId,
                fixture.context_key().contextOpaque,
                terminalDiagnostic.diagnostic);
        const auto afterControlledFailure =
            JuicerProcess::TestSupport::RootLifetimeObserver::snapshot(
                root,
                fixture.context_key());
        results.record(
            "prepared/controlled-drain-failure-retains-ownership",
            controlledDrainFailed &&
                contains_text(
                    terminalDiagnostic.diagnostic,
                    "test-injected CUDA context owner drain failure") &&
                afterControlledFailure.contextEntryPresent &&
                afterControlledFailure.frameOwnerPresent &&
                afterControlledFailure.nativeAllocationCount != 0 &&
                afterControlledFailure.pendingScanErrorReadbackCount != 0 &&
                afterControlledFailure.ledger.chargedBytes != 0 &&
                afterControlledFailure.contextLedgerRecordCount != 0,
            terminalDiagnostic.diagnostic);

        JuicerProcess::TestSupport::clear_context_drain_failure();
        terminalDiagnostic = {};
        const bool idleRetired =
            controlledDrainFailed &&
            root.retire_idle_context(
                fixture.context_key().deviceId,
                fixture.context_key().contextOpaque,
                terminalDiagnostic.diagnostic);
        const auto afterIdleRetire =
            JuicerProcess::TestSupport::RootLifetimeObserver::snapshot(
                root,
                fixture.context_key());
        results.record(
            "prepared/controlled-drain-failure-recovery",
            idleRetired && !afterIdleRetire.contextEntryPresent &&
                afterIdleRetire.ledger.chargedBytes == 0 &&
                afterIdleRetire.ledger.recordCount == 0 &&
                JuicerCuda::ResourceManager::global_state()
                        .pinnedStagingBytes.load(std::memory_order_acquire) == 0,
            terminalDiagnostic.diagnostic);
        results.record(
            "prepared/idle-context-retire",
            idleRetired,
            terminalDiagnostic.diagnostic);

        auto resetFrame = root.prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(dedicatedInputs),
            dedicatedRequest,
            {},
            fixture.stream_opaque(),
            diagnostic);
        terminalDiagnostic = {};
        const bool resetReady =
            resetFrame.active() &&
            resetFrame.finish(fixture.stream_opaque(), terminalDiagnostic) &&
            fixture.synchronize(terminalDiagnostic.diagnostic);
        const bool resetRetired =
            resetReady &&
            root.retire_reset_context(
                fixture.context_key().deviceId,
                fixture.context_key().contextOpaque,
                terminalDiagnostic.diagnostic);
        results.record(
            "prepared/reset-context-retire",
            resetRetired,
            terminalDiagnostic.diagnostic);
    }

    void run_terminal_retention_rows(Results& results, JuicerCuda::Owner& owner) {
        CudaPreparedFixture fixture;
        JuicerCuda::Failure diagnostic;
        PreparedInputs inputs;
        if (!fixture.initialize(diagnostic.diagnostic) ||
            !initialize_prepared_inputs(false, false, true, 32, 24, inputs, diagnostic.diagnostic)) {
            throw std::runtime_error(diagnostic.diagnostic);
        }
        auto& root = JuicerProcess::root();
        {
            auto frame = root.prepare_cuda_frame(fixture.context_key(), fixture.snapshot(inputs), inputs.request({32, 24}), {}, fixture.stream_opaque(), diagnostic);
            if (!frame.active() || !frame.record_use(fixture.stream_opaque(), diagnostic) ||
                !frame.finish(fixture.stream_opaque(), diagnostic) || !fixture.synchronize(diagnostic.diagnostic)) {
                throw std::runtime_error(diagnostic.diagnostic);
            }
        }
        const auto before = JuicerProcess::TestSupport::RootLifetimeObserver::snapshot(root, fixture.context_key());
        const bool injected = JuicerProcess::TestSupport::arm_context_drain_failure_once(fixture.context_key());
        FjCuda* borrow = JuicerCuda::borrowed_owner();
        std::array<char, 512> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        const auto blocked = fj_cuda_shutdown(borrow, &error);
        const auto after = JuicerProcess::TestSupport::RootLifetimeObserver::snapshot(root, fixture.context_key());
        results.record("terminal/gpu-failed-shutdown-retains-graph",
                       injected && blocked.category == FJ_STATUS_PREPARATION_FAILURE && blocked.api == FJ_API_NONE &&
                           blocked.native_code == 0 && JuicerCuda::borrowed_owner() == borrow &&
                           !root.begin_frame_preparation().active() && before.nativeAllocationCount != 0 &&
                           after.nativeAllocationCount == before.nativeAllocationCount &&
                           after.ledger.chargedBytes == before.ledger.chargedBytes &&
                           after.contextLedgerRecordCount == before.contextLedgerRecordCount,
                       message.data());
        JuicerProcess::TestSupport::clear_context_drain_failure();
        const auto consumed = owner.close(&error);
        const auto retained = JuicerProcess::TestSupport::RootLifetimeObserver::snapshot(root, fixture.context_key());
        results.record("terminal/gpu-blocked-destroy-does-not-retry",
                       consumed.category == blocked.category && consumed.api == blocked.api && consumed.native_code == blocked.native_code &&
                           JuicerCuda::borrowed_owner() == borrow && retained.nativeAllocationCount == before.nativeAllocationCount &&
                           retained.contextLedgerRecordCount == before.contextLedgerRecordCount &&
                           retained.ledger.chargedBytes == before.ledger.chargedBytes,
                       message.data());
        results.record("terminal/gpu-consumed-wrapper-empty", owner.close().category == FJ_STATUS_SUCCESS, "consumed wrapper and its destructor cannot retry the retained graph");
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

    void run_route_boundary_rows_impl(Results& results) {
        constexpr int kWidth = 16;
        constexpr int kHeight = 12;
        CudaPreparedFixture fixture;
        JuicerCuda::Failure diagnostic;
        if (!fixture.initialize(diagnostic.diagnostic)) {
            throw std::runtime_error(diagnostic.diagnostic);
        }
        const ScatterHalationControls controls = convert_or_throw(
            ScatterHalationRawControls{true, 1.0, 1.0, 1.0, 1.0});
        const Spektrafilm::DiffusionFilterAuthoredControls noDiffusion{};
        for (const Spektrafilm::ScanRouteMetadata& metadata :
             Spektrafilm::kScanRouteMatrix) {
            PreparedInputs inputs;
            diagnostic = {};
            const bool initialized = initialize_route_inputs(
                metadata.route,
                controls,
                noDiffusion,
                10.0f,
                kWidth,
                kHeight,
                inputs,
                diagnostic.diagnostic);
            bool passed = initialized && inputs.scatterDescriptor.has_value() &&
                          inputs.product.recipe.profileRoute.scanRoute ==
                              metadata.route;
            if (passed) {
                auto request = inputs.request({kWidth, kHeight});
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
                             diagnostic.diagnostic);
                JuicerCuda::Failure finishDiagnostic;
                passed = frame.finish(
                             fixture.stream_opaque(),
                             finishDiagnostic) &&
                         passed;
                if (!finishDiagnostic.diagnostic.empty()) {
                    diagnostic = finishDiagnostic;
                }
            }
            results.record(
                std::string("route-boundary/") + metadata.key,
                passed,
                diagnostic.diagnostic);
        }
    }

    void run_zero_work_rows_impl(Results& results) {
        constexpr int kWidth = 16;
        constexpr int kHeight = 12;
        CudaPreparedFixture fixture;
        JuicerCuda::Failure diagnostic;
        if (!fixture.initialize(diagnostic.diagnostic)) {
            throw std::runtime_error(diagnostic.diagnostic);
        }
        const ScatterHalationControls zeroControls = convert_or_throw(
            ScatterHalationRawControls{true, 0.0, 1.0, 0.0, 1.0});
        const Spektrafilm::DiffusionFilterAuthoredControls noDiffusion{};
        for (const Spektrafilm::ScanRoute route :
             {Spektrafilm::ScanRoute::NegativeDirectScan,
              Spektrafilm::ScanRoute::NegativePrintScan}) {
            PreparedInputs inputs;
            diagnostic = {};
            const bool initialized = initialize_route_inputs(
                route,
                zeroControls,
                noDiffusion,
                10.0f,
                kWidth,
                kHeight,
                inputs,
                diagnostic.diagnostic);
            if (!initialized) {
                results.record(
                    std::string("zero-work/") +
                        Spektrafilm::scan_route_key(route),
                    false,
                    diagnostic.diagnostic);
                continue;
            }
            auto request = inputs.request({kWidth, kHeight});
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
            JuicerCuda::Failure finishDiagnostic;
            passed = frame.finish(
                         fixture.stream_opaque(),
                         finishDiagnostic) &&
                     passed;
            results.record(
                std::string("zero-work/") +
                    Spektrafilm::scan_route_key(route),
                passed,
                finishDiagnostic.diagnostic.empty() ? diagnostic.diagnostic : finishDiagnostic.diagnostic);
        }
    }

    void run_route_lifecycle_rows_impl(Results& results) {
        constexpr int kWidth = 16;
        constexpr int kHeight = 12;
        CudaPreparedFixture fixture;
        JuicerCuda::Failure diagnostic;
        if (!fixture.initialize(diagnostic.diagnostic)) {
            throw std::runtime_error(diagnostic.diagnostic);
        }
        PreparedInputs inputs;
        if (!initialize_prepared_inputs(
                false,
                false,
                true,
                kWidth,
                kHeight,
                inputs,
                diagnostic.diagnostic)) {
            throw std::runtime_error(diagnostic.diagnostic);
        }
        auto request = inputs.request({kWidth, kHeight});
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
            "lifecycle/launcher-failure-abort",
            failedWasActive &&
                failedLaunch.status == cudaErrorInvalidValue &&
                !failedFrame.active() &&
                !failedFrame.scatter_halation_resources().descriptor,
            "ordinary prepared view binding failure retires through abort");

        auto enqueuedFrame = JuicerProcess::root().prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(inputs),
            request,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const auto enqueuedView = enqueuedFrame.scatter_halation_resources();
        const std::size_t carrierBytes =
            static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(kHeight) * sizeof(float);
        const cudaStream_t stream = static_cast<cudaStream_t>(fixture.stream_opaque());
        bool enqueueSucceeded = enqueuedFrame.active();
        for (float* plane : {enqueuedView.currentCarrier.redSensitive,
                             enqueuedView.currentCarrier.greenSensitive,
                             enqueuedView.currentCarrier.blueSensitive}) {
            enqueueSucceeded = enqueueSucceeded && plane &&
                               cudaMemsetAsync(plane, 0, carrierBytes, stream) == cudaSuccess;
        }
        if (enqueueSucceeded) {
            enqueueSucceeded = JuicerCuda::launch_scatter_halation(enqueuedView, stream).status == cudaSuccess;
        }
        enqueuedFrame.abort();
        const bool postEnqueueAbort =
            enqueueSucceeded && !enqueuedFrame.active() &&
            !enqueuedFrame.scatter_halation_resources().descriptor &&
            fixture.synchronize(diagnostic.diagnostic);
        results.record(
            "lifecycle/after-enqueue-abort",
            postEnqueueAbort,
            diagnostic.diagnostic);

        auto transitionFrame = JuicerProcess::root().prepare_cuda_frame(
            fixture.context_key(),
            fixture.snapshot(inputs),
            request,
            {},
            fixture.stream_opaque(),
            diagnostic);
        const auto transitionView = transitionFrame.scatter_halation_resources();
        auto* const carrierR = transitionView.currentCarrier.redSensitive;
        bool transitioned =
            transitionFrame.active() &&
            upload_and_launch_dedicated(
                transitionView,
                static_cast<cudaStream_t>(fixture.stream_opaque()),
                diagnostic.diagnostic);
        const auto workspace = transitionFrame.workspace_lease();
        JuicerCuda::Failure transitionDiagnostic;
        transitioned = transitioned && workspace.active() &&
                       transitionFrame.stage_optical_workspace(
                           workspace,
                           fixture.stream_opaque(),
                           transitionDiagnostic) &&
                       transitionFrame.scatter_halation_resources()
                               .currentCarrier.redSensitive == carrierR;
        JuicerCuda::Failure finishDiagnostic;
        transitioned = transitionFrame.finish(
                           fixture.stream_opaque(),
                           finishDiagnostic) &&
                       transitioned;
        results.record(
            "lifecycle/downstream-scratch-transition",
            transitioned,
            transitionDiagnostic.diagnostic.empty() ? finishDiagnostic.diagnostic
                                                    : transitionDiagnostic.diagnostic);
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

    void report_fatal_and_shutdown(const char* detail) noexcept {
        std::fprintf(stderr, "fatal: %s\n", detail);
        try {
            fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
        } catch (...) {
            std::fputs("fatal: shutdown failed during error handling\n", stderr);
        }
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
    PreparedRouteInputs::request(ImageExtent extent) const {
        JuicerProcess::Root::CudaFramePreparationRequest value{};
        value.recipe = &_impl->product.recipe;
        value.exposureTables = &_impl->product.payload.exposureTables;
        value.filmRawConfig = &_impl->product.payload.filmRawConfig;
        value.filmTcLut = _impl->product.payload.filmTcLut
                              ? &*_impl->product.payload.filmTcLut
                              : nullptr;
        value.printMainIlluminant = _impl->product.payload.printMainIlluminant
                                        ? &*_impl->product.payload.printMainIlluminant
                                        : nullptr;
        value.scannerTables = &_impl->product.payload.scannerTables;
        value.scannerColor = &_impl->product.payload.scannerColor;
        value.scannerLutDescriptor = &_impl->scannerDescriptor;
        value.outputBoundaryTable =
            _impl->product.payload.outputBoundaryTable.get();
        value.diffusionFrameSetDescriptor =
            _impl->diffusionFrameSet ? &*_impl->diffusionFrameSet : nullptr;
        value.scatterHalationDescriptor =
            _impl->scatterDescriptor ? &*_impl->scatterDescriptor : nullptr;
        value.requestedWidth = extent.width;
        value.requestedHeight = extent.height;
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

    const RenderRecipe& PreparedRouteInputs::recipe() const noexcept {
        return _impl->product.recipe;
    }

    PreparedRouteInputs build_prepared_route_inputs(
        Spektrafilm::ScanRoute route,
        const ScatterHalationControls& controls,
        const Spektrafilm::DiffusionFilterAuthoredControls& cameraDiffusion,
        float pixelSizeUm,
        ImageExtent extent) {
        return build_prepared_route_inputs(
            route,
            controls,
            cameraDiffusion,
            Spektrafilm::DirCouplersControls{},
            pixelSizeUm,
            extent);
    }

    PreparedRouteInputs build_prepared_route_inputs(
        Spektrafilm::ScanRoute route,
        const ScatterHalationControls& controls,
        const Spektrafilm::DiffusionFilterAuthoredControls& cameraDiffusion,
        const Spektrafilm::DirCouplersControls& dirCouplers,
        float pixelSizeUm,
        ImageExtent extent,
        float filmGammaFactor,
        const std::string& filmProfileKey) {
        ParamSnapshot snapshot;
        snapshot.scanRoute = route;
        if (!filmProfileKey.empty()) {
            snapshot.filmProfileKey = filmProfileKey;
        }
        snapshot.scatterHalationControls = controls;
        snapshot.cameraDiffusion = cameraDiffusion;
        snapshot.dirCouplers = dirCouplers;
        snapshot.filmGammaFactor = filmGammaFactor;
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
                 Spektrafilm::DiffusionFrameDomain{
                     0,
                     0,
                     extent.width,
                     extent.height},
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

    void run_route_boundary_rows(
        const Arguments& arguments,
        Results& results) {
        (void)arguments;
        run_route_boundary_rows_impl(results);
        run_scanner_post_effect_cuda_rows(results);
    }

    void run_zero_work_rows(const Arguments& arguments, Results& results) {
        (void)arguments;
        run_zero_work_rows_impl(results);
    }

    void run_lifecycle_rows(const Arguments& arguments, Results& results) {
        (void)arguments;
        run_prepared_frame_rows(results);
        JuicerProcess::root().ensure_bootstrap();
        run_route_lifecycle_rows_impl(results);
    }

} // namespace ScatterHalationValidation

int main(int argc, char** argv) noexcept {
    JuicerCuda::Owner cudaOwner;
    try {
        cudaOwner.create(JuicerProcess::data_directory());
        const Arguments arguments = parse_arguments(argc, argv);
        std::filesystem::create_directories(arguments.scratchRoot);
        JuicerProcess::root().ensure_bootstrap();

        Results results;
        const auto run_host_contracts = [&] {
            run_control_rows(results);
            run_build_product_rows(results);
            run_profile_rows(arguments, results);
            run_recipe_rows(results);
            run_descriptor_rows(results);
            run_pending_admission_rows(results);
            run_camera_hash_rows(results);
            run_parameter_identity_rows(results);
            run_parameter_sign_rows(results);
            run_capacity_recovery_classification_rows(results);
        };
        if (arguments.caseGroup == "host-contracts") {
            run_host_contracts();
        } else if (arguments.caseGroup == "prepared-frame") {
            run_prepared_frame_rows(results);
        } else if (arguments.caseGroup == "terminal-retention") {
            run_terminal_retention_rows(results, cudaOwner);
        } else if (arguments.caseGroup == "focused-cuda-reference") {
            ScatterHalationValidation::run_focused_cuda_reference_rows(
                arguments,
                results);
        } else if (arguments.caseGroup == "route-boundaries") {
            ScatterHalationValidation::run_route_boundary_rows(
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
            run_host_contracts();
            run_prepared_frame_rows(results);
            JuicerProcess::root().ensure_bootstrap();
            ScatterHalationValidation::run_focused_cuda_reference_rows(
                arguments,
                results);
            ScatterHalationValidation::run_route_boundary_rows(
                arguments,
                results);
            ScatterHalationValidation::run_captured_carrier_rows(
                arguments,
                results);
            ScatterHalationValidation::run_zero_work_rows(arguments, results);
            ScatterHalationValidation::run_lifecycle_rows(arguments, results);
        }
        if (results.cases().empty()) {
            results.record(
                "harness/non-empty-selection",
                false,
                "selected case group produced no test rows");
        }
        if (arguments.caseGroup != "terminal-retention") {
            const auto closed = fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
            std::vector<JuicerCuda::ResourceManager::DeviceContextKey> live;
            JuicerCuda::ResourceManager::registry_snapshot_context_keys(live);
            results.record("terminal/native-shutdown-last", closed.category == FJ_STATUS_SUCCESS && live.empty(), "typed shutdown closes the owner after all fixture work");
        }
        print_results(results);
        return results.failure_count() == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        report_fatal_and_shutdown(error.what());
        return 2;
    } catch (...) {
        report_fatal_and_shutdown("unknown exception");
        return 2;
    }
}
