#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <variant>

#include "Cuda/JuicerCudaResources.h"
#include "Scanner.h"

namespace JuicerCuda {

    // C++ producers retain typed rows. Foreign scalar tables retain their byte
    // representation, so no flat float span crosses a nested native array object.
    // Only fixed triplets are copied when consumed; all table storage stays borrowed.
    class ThreeChannelSamplesView {
    public:
        ThreeChannelSamplesView() = default;
        explicit ThreeChannelSamplesView(std::span<const std::array<float, 3>> rows);
        explicit ThreeChannelSamplesView(std::span<const float> scalars);
        explicit ThreeChannelSamplesView(std::span<const std::byte> scalarBytes);
        std::span<const std::byte> object_bytes() const noexcept;
        std::size_t sample_count() const noexcept;
        std::size_t scalar_count() const noexcept;
        std::array<float, 3> sample(std::size_t index) const;

    private:
        std::variant<std::span<const std::array<float, 3>>, std::span<const std::byte>> _samples;
    };

    // These views borrow scoped host storage only for the immediate preparation call.
    // Upload staging retains its existing copy/fallback behavior and completion ownership.
    struct FilmExposureResourceInput {
        Spektrafilm::RgbToRawMethod rgbToRawMethod = Spektrafilm::RgbToRawMethod::Hanatos2025;
        ThreeChannelSamplesView finalSensitivityRgb;
        std::uint64_t finalSensitivityHash = 0;
        std::uint64_t tcLutHash = 0;
    };

    struct FilmDevelopmentResourceInput {
        std::span<const float> logExposure;
        ThreeChannelSamplesView normalizedDensityCurvesRgb;
        std::array<std::array<std::span<const float>, 3>, 3> densityCurvesLayers;
        std::uint64_t normalizedDensityCurvesHash = 0;
        std::uint64_t densityCurvesLayersHash = 0;
        bool densityCurvesLayersRequired = false;
    };

    struct DirResourceInput {
        std::array<std::span<const float>, 3> compensatedDensityCurveAxesRgb;
        std::uint64_t compensatedDensityCurveAxesHash = 0;
        std::uint64_t hash = 0;
        bool active = false;
    };

    struct OutputGamutResourceInput {
        std::span<const float> cmax;
        std::uint64_t tableHash = 0;
        std::uint64_t recipeHash = 0;
        bool enabled = false;
    };

    struct FocusedRouteResourceInput {
        Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
        FilmExposureResourceInput filmRaw;
        FilmDevelopmentResourceInput filmDevelop;
        DirResourceInput dirCouplers;
        DensityBoundsRecipe densityBounds;
        Scanner::ScannerSpectralLutDescriptor scannerLutDescriptor;
        Scanner::SpectralTablesView scannerTables;
        std::span<const float> exposureIlluminant;
        std::span<const float> filmTcLut;
        std::span<const float> mallettBasis;
        int exposureSampleCount = 0;
        int mallettRows = 0;
        int mallettColumns = 0;
        bool mallettAvailable = false;
        bool wantDensityLayers = false;
        OutputGamutResourceInput outputGamut;
    };

    struct PrintResourceInput {
        PrintResourceDescriptors descriptors;
        ThreeChannelSamplesView filmChannelDensityCmy;
        std::span<const float> filmBaseDensity;
        std::span<const float> printLogExposure;
        ThreeChannelSamplesView printDensityCurvesCmy;
        ThreeChannelSamplesView printSensitivityCmy;
        std::span<const float> mainIlluminant;
        std::span<const float> preflashIlluminant;
        std::array<float, 3> preflashRawCmy{};
        float factorMidgray = 1.0f;
        float factorMidgrayComp = 1.0f;
        float normalizer = 1.0f;
    };

    struct StaticNoiseInput {
        std::span<const std::uint8_t> stbn;
        std::span<const std::uint8_t> wangTiles;
        std::span<const std::uint8_t> wangLut;
        int stbnWidth = 0;
        int stbnHeight = 0;
        int stbnFrames = 0;
        int wangWidth = 0;
        int wangHeight = 0;
        int wangCount = 0;
        int wangColors = 0;
    };

    bool build_focused_route_resource_input(
        const FocusedRouteResourcePreparation& request,
        FocusedRouteResourceInput& input,
        std::string& outError);

    bool build_static_noise_input(
        const JuicerAssets::StaticNoisePayloadSet& payloads,
        StaticNoiseInput& input,
        std::string& outError);

    // The returned illuminant span borrows the caller's storage through the immediate call.
    // Production preparation invokes this derivation only for missing resource descriptors.
    bool build_print_resource_input(
        const PrintResourcePreparation& request,
        PrintResourceInput& input,
        std::array<float, 81>& preflashIlluminant,
        std::string& outError);

} // namespace JuicerCuda
