#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ProfileAssets.h"
#include "ScanRoute.h"

// RenderRecipe owner map:
// - ProfileRoute owns selected profile identity, typed metadata, immutable selected payload
//   references, asset tokens, and resolved ScanRoute.
// - FilmRawRecipe owns direct-route input color, RGB-to-raw, sensitivity, and exposure ordering.
// - FilmDevelopRecipe owns direct-route authored/normalized film density development inputs.
// - DirRecipe owns development-inhibitor release behavior when Phase 5 introduces it.
// - DensityBoundsRecipe owns final route/media scanner and enlarger density bounds.
// - PrintRecipe owns print exposure, filters, development, and print-route policy when Phase 4 introduces it.
// - ScannerOutputRecipe owns scanner/output policy; scanner LUT resources own their descriptor.
// - OpticsRecipe owns lens, halation, scattering, and diffusion behavior when Phase 6 introduces it.
// - GrainContract/GrainRecipe own density_min and visual-grain behavior; density_min is not profile digest data.
// - FrameRequest owns frame-local extent, pixel size, temporal tokens, and metering request facts.
// - PreparedCudaFrame/context resource internals own durable GPU handles, scratch, staging, and views.
// This is source-adjacent orientation, not a runtime registry.
namespace Spektrafilm {

    inline constexpr const char kScannerPostEffectsNotImplementedForPhase3[] =
        "ScannerPostEffectsNotImplementedForPhase3";
    inline constexpr const char kDirNotImplementedForPhase3[] =
        "DirNotImplementedForPhase3";
    inline constexpr const char kPositiveCorrectionNotImplementedForPhase3[] =
        "PositiveCorrectionNotImplementedForPhase3";
    inline constexpr const char kQuantizedMedianNotAcceptedForPhase3[] =
        "QuantizedMedianNotAcceptedForPhase3";

    enum class RgbToRawMethod : std::uint8_t {
        Hanatos2025,
        Mallett2019
    };

    enum class AutoExposureMethod : std::uint8_t {
        CenterWeighted,
        Average,
        Median,
        Partial,
        Matrix,
        MultiZone,
        HighlightWeighted
    };

    enum class DensityMedium : std::uint8_t {
        Film,
        Print
    };

    enum class DensityBoundsSource : std::uint8_t {
        DirectFilmGrainContractAndAuthoredCurves,
        EnlargerFilmGrainContractAndAuthoredCurves,
        PrintMediaAuthoredCurves
    };

    enum class ScannerPostEffectDisposition : std::uint8_t {
        Identity,
        Implemented,
        BlockedNotImplementedForPhase3
    };

} // namespace Spektrafilm

struct ProfileRoute {
    std::string filmProfileKey;
    std::string printProfileKey;
    Spektrafilm::ScanRoute scanRoute = Spektrafilm::kDefaultScanRoute;
    Spektrafilm::ProfileSupport captureSupport = Spektrafilm::ProfileSupport::Unsupported;
    Spektrafilm::ProfileStage captureStage = Spektrafilm::ProfileStage::Unsupported;
    Spektrafilm::ProfilePolarity capturePolarity = Spektrafilm::ProfilePolarity::Unsupported;
    Profiles::ProfileUse captureUse = Profiles::ProfileUse::Unsupported;
    Profiles::ProfileAntihalation captureAntihalation = Profiles::ProfileAntihalation::Unsupported;
    Profiles::ProfileChannelModel captureChannelModel = Profiles::ProfileChannelModel::Unsupported;
    std::uint64_t filmProfileAssetVersionToken = 0;
    std::uint64_t printProfileAssetVersionToken = 0;
    std::shared_ptr<const Profiles::ValidatedFilmProfile> filmProfile;
    std::shared_ptr<const Profiles::ValidatedPrintProfile> printProfile;
    bool directRoutePrintProfileExcluded = false;
    bool directRouteNeutralCalibrationExcluded = false;
    std::uint64_t hash = 0;
};

struct CameraBandPassRecipe {
    bool active = false;
    std::array<float, 3> uv{{0.0f, 410.0f, 8.0f}};
    std::array<float, 3> ir{{0.0f, 675.0f, 15.0f}};
};

struct HanatosAdaptationRecipe {
    bool applyWindow = true;
    bool applySurface = false;
    float spectralGaussianBlur = 0.0f;
    std::array<float, 4> windowParams{};
    std::array<std::array<float, 15>, 3> surfaceParams{};
    std::string referenceIlluminant;
};

struct HighlightBoostRecipe {
    float boostEv = 0.0f;
    float boostRange = 0.3f;
    float protectEv = 4.0f;
};

struct FilmRawRecipe {
    int inputColorSpace = 0;
    bool inputCctfDecoding = false;
    Spektrafilm::RgbToRawMethod rgbToRawMethod = Spektrafilm::RgbToRawMethod::Hanatos2025;
    bool autoExposureEnabled = true;
    Spektrafilm::AutoExposureMethod autoExposureMethod = Spektrafilm::AutoExposureMethod::CenterWeighted;
    float manualExposureCompensationEv = 0.0f;
    float filmFormatLongEdgeMm = 36.0f;
    CameraBandPassRecipe cameraBandPass;
    HanatosAdaptationRecipe hanatos;
    HighlightBoostRecipe highlightBoost;
    // Loader-linear sensitivity is retained as provenance. finalSensitivity is the only
    // sensitivity payload/resource identity consumed by direct film-raw processing.
    std::array<std::array<float, 3>, 81> linearSensitivity{};
    std::uint64_t linearSensitivityHash = 0;
    std::array<std::array<float, 3>, 81> finalSensitivity{};
    std::uint64_t finalSensitivityHash = 0;
    std::uint64_t hanatosLutHash = 0;
    float mallettGreenMidgrayScale = 1.0f;
    std::uint64_t hash = 0;
};

struct FilmDevelopRecipe {
    Spektrafilm::ProfilePolarity polarity = Spektrafilm::ProfilePolarity::Unsupported;
    std::vector<float> logExposure;
    std::vector<std::array<float, 3>> authoredDensityCurves;
    std::vector<std::array<float, 3>> normalizedDensityCurves;
    std::array<float, 3> densityCurveGamma{{1.0f, 1.0f, 1.0f}};
    std::array<float, 3> authoredMinCmy{};
    std::array<float, 3> authoredMaxCmy{};
    std::uint64_t authoredDensityCurvesHash = 0;
    std::uint64_t normalizedDensityCurvesHash = 0;
    std::uint64_t hash = 0;
};

struct GrainContract {
    std::array<float, 3> densityMinCmy{{0.07f, 0.08f, 0.12f}};
};

struct DensityBoundsRecipe {
    Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
    Spektrafilm::DensityMedium medium = Spektrafilm::DensityMedium::Film;
    Spektrafilm::ProfilePolarity polarity = Spektrafilm::ProfilePolarity::Unsupported;
    Spektrafilm::DensityBoundsSource source =
        Spektrafilm::DensityBoundsSource::DirectFilmGrainContractAndAuthoredCurves;
    std::array<float, 3> dataMinCmy{};
    std::array<float, 3> dataMaxCmy{};
    std::array<float, 3> invSpanCmy{};
    std::array<float, 3> authoredMinCmy{};
    std::array<float, 3> authoredMaxCmy{};
    std::uint64_t hash = 0;
};

struct ScannerOutputRecipe {
    Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
    Spektrafilm::DensityMedium medium = Spektrafilm::DensityMedium::Film;
    Spektrafilm::ProfilePolarity polarity = Spektrafilm::ProfilePolarity::Unsupported;
    std::string viewingIlluminant;
    std::uint32_t lutResolution = 17;
    int outputColorSpace = 0;
    bool outputCctfEncoding = true;
    bool outputLinearPassThrough = false;
    bool directGlareDisabled = true;
    float lensBlurSigmaPx = 0.55f;
    float unsharpSigmaPx = 0.7f;
    float unsharpAmount = 1.0f;
    Spektrafilm::ScannerPostEffectDisposition postEffectsDisposition =
        Spektrafilm::ScannerPostEffectDisposition::BlockedNotImplementedForPhase3;
    std::string blockingDiagnostic = Spektrafilm::kScannerPostEffectsNotImplementedForPhase3;
    std::uint64_t hash = 0;
};

struct RenderRecipe {
    ProfileRoute profileRoute;
    FilmRawRecipe filmRaw;
    FilmDevelopRecipe filmDevelop;
    DensityBoundsRecipe densityBounds;
    ScannerOutputRecipe scannerOutput;
    bool directStructuralReady = false;
    bool directPixelAcceptance = false;
    std::uint64_t hash = 0;
};

namespace Spektrafilm {

    using ::DensityBoundsRecipe;
    using ::FilmDevelopRecipe;
    using ::FilmRawRecipe;
    using ::GrainContract;
    using ::ProfileRoute;
    using ::RenderRecipe;
    using ::ScannerOutputRecipe;

    struct DirectRecipeBuildInput {
        std::string filmProfileKey;
        std::string printProfileKey;
        ScanRoute scanRoute = kDefaultScanRoute;
        std::shared_ptr<const Profiles::ValidatedFilmProfile> filmProfile;
        GrainContract grainContract;
        bool directRoutePrintProfileExcluded = false;
        bool directRouteNeutralCalibrationExcluded = false;
        int spectralUpsamplingMode = 0;
        int inputColorSpace = 0;
        bool inputCctfDecoding = false;
        bool cameraAutoExposureEnabled = true;
        int cameraMeteringMethod = 0;
        float manualExposureCompensationEv = 0.0f;
        float filmFormatLongEdgeMm = 36.0f;
        bool cameraFilterOverride = false;
        std::array<double, 3> cameraFilterUV{{1.0, 410.0, 8.0}};
        std::array<double, 3> cameraFilterIR{{1.0, 675.0, 15.0}};
        std::array<float, 81> referenceIlluminant{};
        bool referenceIlluminantValid = false;
        std::uint32_t scannerLutResolution = 17;
        int outputColorSpace = 0;
        bool outputCctfEncoding = true;
        bool outputLinearPassThrough = false;
        float scannerLensBlurSigmaPx = 0.55f;
        float scannerUnsharpSigmaPx = 0.7f;
        float scannerUnsharpAmount = 1.0f;
    };

    struct DirectRecipeBuildResult {
        RenderRecipe recipe;
        std::string diagnostic;
        bool valid = false;
    };

    inline RenderRecipe make_render_recipe(ProfileRoute profileRoute) {
        RenderRecipe recipe{};
        recipe.profileRoute = std::move(profileRoute);
        return recipe;
    }

    inline RenderRecipe make_render_recipe(
        std::string filmProfileKey,
        std::string printProfileKey,
        ScanRoute scanRoute) {
        ProfileRoute profileRoute{};
        profileRoute.filmProfileKey = std::move(filmProfileKey);
        profileRoute.printProfileKey = std::move(printProfileKey);
        profileRoute.scanRoute = scanRoute;
        return make_render_recipe(std::move(profileRoute));
    }

    DirectRecipeBuildResult build_direct_render_recipe(const DirectRecipeBuildInput& input);

} // namespace Spektrafilm
