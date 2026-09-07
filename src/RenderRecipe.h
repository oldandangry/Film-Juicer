#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "DiffusionHostBehavior.h"
#include "ProfileAssets.h"
#include "ScanRoute.h"
#include "ScatterHalation.h"

// RenderRecipe owner map:
// - ProfileRoute owns selected profile identity, typed metadata, immutable selected payload
//   references, asset tokens, and resolved ScanRoute.
// - FilmRawRecipe owns direct-route input color, RGB-to-raw, sensitivity, and exposure ordering.
// - FilmDevelopRecipe owns direct-route authored/normalized film density development inputs.
// - DirCouplersRecipe owns development-inhibitor release behavior.
// - DensityBoundsRecipe owns final route/media scanner and enlarger density bounds.
// - PrintRecipe owns print filters, neutral calibration, exposure ordering, illuminant identity,
//   and print-medium handoff identity. Print preparation and launch remain downstream owners.
// - ScannerOutputRecipe owns scanner/output policy; scanner LUT resources own their descriptor.
// - OpticsRecipe owns lens, halation, scattering, and diffusion behavior when Phase 6 introduces it.
// - GrainContract owns Spektrafilm density_min; VisualGrainRecipe owns Film-Juicer grain behavior.
//   The visual particle density minimum is independent of scanner/enlarger density bounds.
// - DirectFrameRequest and PrintFrameRequest own frame-local extent, pixel size, and temporal tokens.
// - PreparedCudaFrame/context resource internals own durable GPU handles, scratch, staging, and views.
// This is source-adjacent orientation, not a runtime registry.
namespace Spektrafilm {

    inline constexpr const char kQuantizedMedianNotAcceptedForPhase3[] =
        "QuantizedMedianNotAcceptedForPhase3";
    inline constexpr const char kExactOpticsNotImplementedForPhase6[] =
        "ExactOpticsNotImplementedForPhase6";

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

    enum class NeutralCalibrationStatus : std::uint8_t {
        MissingFile,
        MissingEntry,
        Calibrated
    };

    enum class PrintNormalizationMode : std::uint8_t {
        None,
        CompensationOnly,
        NormalizeOnly,
        NormalizeAndCompensate
    };

    enum class SpatialOpticsDomain : std::uint8_t {
        FilmLinearExposure,
        PrintLinearExposure
    };

    enum class SpatialOpticsComponent : std::uint8_t {
        CameraDiffusion,
        CameraLensBlur,
        InEmulsionScatterHalation,
        EnlargerDiffusion
    };

} // namespace Spektrafilm

struct ProfileRoute {
    std::string filmProfileKey;
    std::string printProfileKey;
    Spektrafilm::ScanRoute scanRoute = Spektrafilm::kDefaultScanRoute;
    Spektrafilm::ProfilePolarity capturePolarity = Spektrafilm::ProfilePolarity::Unsupported;
    std::uint64_t filmProfileAssetVersionToken = 0;
    std::uint64_t printProfileAssetVersionToken = 0;
    std::shared_ptr<const Profiles::ValidatedFilmProfile> filmProfile;
    std::shared_ptr<const Profiles::ValidatedPrintProfile> printProfile;
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

struct DiffusionFilterOpticsRecipe {
    Spektrafilm::SpatialOpticsDomain domain =
        Spektrafilm::SpatialOpticsDomain::FilmLinearExposure;
    Spektrafilm::DiffusionFilterResolvedParameters resolved;
    std::uint64_t hash = 0;
};

struct CameraLensBlurOpticsRecipe {
    float sigmaUm = 0.0f;
    std::uint64_t hash = 0;
};

struct SpatialOptics {
    DiffusionFilterOpticsRecipe cameraDiffusion;
    CameraLensBlurOpticsRecipe cameraLensBlur;
    ScatterHalationOpticsRecipe scatterHalation;
    DiffusionFilterOpticsRecipe enlargerDiffusion;
    std::uint64_t hash = 0;
};

namespace Spektrafilm {

    inline constexpr std::uint32_t kDiffusionFrameDescriptorSchemaVersion = 2;

    enum class DiffusionLinearStage : std::uint8_t {
        CameraFilmLinear,
        EnlargerPrintLinear
    };

    struct DiffusionFrameDomain {
        int originX = 0;
        int originY = 0;
        int width = 0;
        int height = 0;

        friend bool operator==(
            const DiffusionFrameDomain&,
            const DiffusionFrameDomain&) = default;
    };

    struct DiffusionStageFrameDescriptor {
        DiffusionLinearStage stage = DiffusionLinearStage::CameraFilmLinear;
        double scatterFraction = 0.0;
        DiffusionPsfSampleDescriptor sample;
        std::uint64_t hash = 0;
    };

    struct DiffusionFrameSetDescriptor {
        ScanRoute route = kDefaultScanRoute;
        DiffusionFrameDomain fullFrame;
        std::optional<DiffusionStageFrameDescriptor> camera;
        std::optional<DiffusionStageFrameDescriptor> enlarger;
        std::uint64_t hash = 0;
    };

} // namespace Spektrafilm

struct FilmRawRecipe {
    int inputColorSpace = 0;
    bool inputCctfDecoding = false;
    Spektrafilm::RgbToRawMethod rgbToRawMethod = Spektrafilm::RgbToRawMethod::Hanatos2025;
    bool autoExposureEnabled = true;
    Spektrafilm::AutoExposureMethod autoExposureMethod = Spektrafilm::AutoExposureMethod::CenterWeighted;
    float manualExposureCompensationEv = 0.0f;
    float filmFormatLongEdgeMm = 35.0f;
    CameraBandPassRecipe cameraBandPass;
    HanatosAdaptationRecipe hanatos;
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
    std::array<std::array<std::vector<float>, 3>, 3> densityCurvesLayers{};
    bool densityCurvesLayersRequired = false;
    std::array<float, 3> densityCurveGamma{{1.0f, 1.0f, 1.0f}};
    std::array<float, 3> authoredMinCmy{};
    std::array<float, 3> authoredMaxCmy{};
    std::uint64_t normalizedDensityCurvesHash = 0;
    std::uint64_t densityCurvesLayersHash = 0;
    std::uint64_t hash = 0;
};

struct DirCouplersControls {
    bool active = true;
    float amount = 1.0f;
    float inhibitionSameLayer = 1.0f;
    float inhibitionInterlayer = 1.0f;
    float diffusionSizeUm = 20.0f;
    bool gammaUseStock = true;
    std::array<float, 3> gammaSameLayerRgb{{0.336f, 0.319f, 0.273f}};
    std::array<float, 2> gammaInterlayerRToGb{{0.353f, 0.302f}};
    std::array<float, 2> gammaInterlayerGToRb{{0.154f, 0.353f}};
    std::array<float, 2> gammaInterlayerBToRg{{0.168f, 0.226f}};
};

struct DirCouplersRecipe {
    Spektrafilm::ProfilePolarity polarity = Spektrafilm::ProfilePolarity::Unsupported;
    bool active = false;
    std::array<std::array<float, 3>, 3> matrixRgb{};
    float diffusionSizeUm = 0.0f;
    float diffusionTailUm = 0.0f;
    float diffusionTailWeight = 0.0f;
    std::array<float, 3> densityMaxRgb{};
    std::vector<std::array<float, 3>> precorrectedDensityCurves;
    std::uint64_t precorrectedDensityCurvesHash = 0;
    std::uint64_t hash = 0;
};

namespace Spektrafilm {

    enum class DirSourceContract : std::uint8_t {
        None,
        FilmLogRawToInitialDensityCmy
    };

    enum class DirBoundaryMode : std::uint8_t {
        None,
        SpektrafilmReferencePerOperator
    };

    enum class DirReferenceOperator : std::uint8_t {
        None,
        Identity,
        SpektrafilmSmallFirReflect,
        SpektrafilmLargeYvvReplicate
    };

    enum class DirFilterBackend : std::uint8_t {
        None,
        SmallFir,
        StrictYvvChannelsAliasedForward
    };

    enum class DirScratchTier : std::uint8_t {
        Tier0,
        Tier1F,
        Tier1IChannels,
        Tier2,
        Unsupported
    };

    enum class DirApproximationMarker : std::uint8_t {
        None,
        SpektrafilmStrict
    };

    enum class DirDescriptorSupport : std::uint8_t {
        Supported,
        Inactive,
        UnsupportedPartialRenderWindow,
        UnsupportedScratchTier
    };

    struct DirFrameExtent {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
    };

    struct DirGaussianComponentPlan {
        float sigmaPixels = 0.0f;
        float weight = 0.0f;
        DirReferenceOperator referenceOperator = DirReferenceOperator::None;
        DirFilterBackend backend = DirFilterBackend::None;
        DirFilterBackend targetBackend = DirFilterBackend::None;
        DirScratchTier targetScratchTier = DirScratchTier::Tier0;
    };

    struct DirFilterPlan {
        static constexpr int kMaxComponents = 4;

        int componentCount = 0;
        std::array<DirGaussianComponentPlan, kMaxComponents> components{};
    };

    struct DirScratchPlaneRoles {
        int rawCorrectionPlanes = 0;
        int filteredCorrectionPlanes = 0;
        int filterTempPlanes = 0;
        int cachedLogRawPlanes = 0;

        int total_float_planes() const noexcept {
            return rawCorrectionPlanes +
                   filteredCorrectionPlanes +
                   filterTempPlanes +
                   cachedLogRawPlanes;
        }
    };

    inline bool spatial_dir_roles_match_tier(
        DirScratchTier tier,
        const DirScratchPlaneRoles& roles) noexcept {
        if (tier == DirScratchTier::Tier0) {
            return roles.total_float_planes() == 0;
        }
        if (roles.filteredCorrectionPlanes != 3) {
            return false;
        }
        if (tier == DirScratchTier::Tier1F) {
            return roles.rawCorrectionPlanes == 3 &&
                   roles.filterTempPlanes == 1 &&
                   roles.cachedLogRawPlanes == 0;
        }
        if (tier == DirScratchTier::Tier1IChannels) {
            return roles.cachedLogRawPlanes == 0 &&
                   ((roles.rawCorrectionPlanes == 3 &&
                     roles.filterTempPlanes >= 1 &&
                     roles.filterTempPlanes <= 3) ||
                    (roles.rawCorrectionPlanes == 1 &&
                     roles.filterTempPlanes == 1));
        }
        if (tier == DirScratchTier::Tier2) {
            if (roles.cachedLogRawPlanes == 2) {
                return roles.rawCorrectionPlanes == 3 &&
                       roles.filterTempPlanes == 3;
            }
            return roles.cachedLogRawPlanes == 3 &&
                   roles.rawCorrectionPlanes == 3 &&
                   roles.filterTempPlanes >= 1 &&
                   roles.filterTempPlanes <= 3;
        }
        return false;
    }

    inline const char* to_cstr(DirSourceContract value) noexcept {
        switch (value) {
            case DirSourceContract::None:
                return "none";
            case DirSourceContract::FilmLogRawToInitialDensityCmy:
                return "film_log_raw_to_initial_density_cmy";
            default:
                return "unknown";
        }
    }

    inline const char* to_cstr(DirBoundaryMode value) noexcept {
        switch (value) {
            case DirBoundaryMode::None:
                return "none";
            case DirBoundaryMode::SpektrafilmReferencePerOperator:
                return "spektrafilm_reference_per_operator";
            default:
                return "unknown";
        }
    }

    inline const char* to_cstr(DirReferenceOperator value) noexcept {
        switch (value) {
            case DirReferenceOperator::None:
                return "none";
            case DirReferenceOperator::Identity:
                return "identity";
            case DirReferenceOperator::SpektrafilmSmallFirReflect:
                return "spektrafilm_small_fir_reflect";
            case DirReferenceOperator::SpektrafilmLargeYvvReplicate:
                return "spektrafilm_large_yvv_replicate";
            default:
                return "unknown";
        }
    }

    inline const char* to_cstr(DirFilterBackend value) noexcept {
        switch (value) {
            case DirFilterBackend::None:
                return "none";
            case DirFilterBackend::SmallFir:
                return "small_fir";
            case DirFilterBackend::StrictYvvChannelsAliasedForward:
                return "strict_yvv_channels_aliased_forward";
            default:
                return "unknown";
        }
    }

    inline const char* to_cstr(DirScratchTier value) noexcept {
        switch (value) {
            case DirScratchTier::Tier0:
                return "Tier0";
            case DirScratchTier::Tier1F:
                return "Tier1F";
            case DirScratchTier::Tier1IChannels:
                return "Tier1IChannels";
            case DirScratchTier::Tier2:
                return "Tier2";
            case DirScratchTier::Unsupported:
                return "unsupported";
            default:
                return "unknown";
        }
    }

    inline const char* to_cstr(DirApproximationMarker value) noexcept {
        switch (value) {
            case DirApproximationMarker::None:
                return "none";
            case DirApproximationMarker::SpektrafilmStrict:
                return "spektrafilm_strict";
            default:
                return "unknown";
        }
    }

    inline const char* to_cstr(DirDescriptorSupport value) noexcept {
        switch (value) {
            case DirDescriptorSupport::Supported:
                return "supported";
            case DirDescriptorSupport::Inactive:
                return "inactive";
            case DirDescriptorSupport::UnsupportedPartialRenderWindow:
                return "unsupported_partial_render_window";
            case DirDescriptorSupport::UnsupportedScratchTier:
                return "unsupported_scratch_tier";
            default:
                return "unknown";
        }
    }

} // namespace Spektrafilm

struct SpatialDirDescriptor {
    static constexpr std::array<float, 3> kExponentialAmplitudes{{0.1633f, 0.6496f, 0.1870f}};
    static constexpr std::array<float, 3> kExponentialSigmaRatios{{0.5360f, 1.5236f, 2.7684f}};
    std::uint64_t dirRecipeHash = 0;
    Spektrafilm::DirSourceContract sourceContract = Spektrafilm::DirSourceContract::None;
    Spektrafilm::DirBoundaryMode boundaryMode = Spektrafilm::DirBoundaryMode::None;
    Spektrafilm::DirScratchTier scratchTier = Spektrafilm::DirScratchTier::Tier0;
    Spektrafilm::DirApproximationMarker approximation = Spektrafilm::DirApproximationMarker::None;
    Spektrafilm::DirDescriptorSupport support = Spektrafilm::DirDescriptorSupport::Inactive;
    Spektrafilm::DirFrameExtent renderExtent{};
    Spektrafilm::DirFrameExtent fullFrameExtent{};
    Spektrafilm::DirFrameExtent filterDomainExtent{};
    Spektrafilm::DirFilterPlan filterPlan{};
    Spektrafilm::DirScratchPlaneRoles planeRoles{};
    Spektrafilm::DirScratchTier targetScratchTier = Spektrafilm::DirScratchTier::Tier0;
    Spektrafilm::DirScratchPlaneRoles targetPlaneRoles{};
    const char* traceRouteLabel = nullptr;
    float gaussianSigmaPixels = 0.0f;
    std::array<float, 3> exponentialSigmaPixels{};
    float gaussianWeight = 0.0f;
    std::array<float, 3> exponentialWeights{};
    std::uint64_t hash = 0;
};

struct VisualGrainControls {
    bool active = false;
    bool sublayersActive = true;
    float particleAreaUm2 = 0.2f;
    std::array<float, 3> particleScaleCmy{{0.8f, 1.0f, 2.0f}};
    std::array<float, 3> particleScaleLayers{{2.5f, 1.0f, 0.5f}};
    std::array<float, 3> visualParticleDensityMinCmy{{0.07f, 0.08f, 0.12f}};
    std::array<float, 3> uniformityCmy{{0.97f, 0.97f, 0.99f}};
    float correlationSigmaPx = 0.65f;
    float dyeCloudBlurUm = 1.0f;
    std::array<float, 2> microStructure{{0.2f, 30.0f}};
    int nSubLayers = 1;
    float amplitude = 1.0f;
    float chromaMix = 0.0f;
    float chromaSharedWeight = 1.0f;
    float chromaIndependentWeight = 0.0f;
    float coarseWeight = 0.0f;
    float midWeight = 0.0f;
    float sizeMixScale = 1.0f;
    float clumpTemporalMix = 0.0f;
    float clumpMorphPeriodSec = 8.0f;
    int debugView = 0;
};

struct VisualGrainRecipe {
    bool active = false;
    bool sublayersActive = false;
    std::array<bool, 3> grainLayerAxisFinite{};
    std::array<std::array<float, 16>, 3> grainLayerAxisBlockPrefixMax{};
    float particleAreaUm2 = 0.0f;
    std::array<float, 3> particleScaleCmy{};
    std::array<float, 3> particleScaleLayers{};
    std::array<float, 3> visualParticleDensityMinCmy{};
    std::array<float, 3> uniformityCmy{};
    float correlationSigmaPx = 0.0f;
    float dyeCloudBlurUm = 0.0f;
    std::array<float, 2> microStructure{};
    int nSubLayers = 0;
    float amplitude = 0.0f;
    float chromaMix = 0.0f;
    float chromaSharedWeight = 0.0f;
    float chromaIndependentWeight = 0.0f;
    float fineWeight = 0.0f;
    float midWeight = 0.0f;
    float coarseWeight = 0.0f;
    float sizeMixScale = 1.0f;
    float clumpTemporalMix = 0.0f;
    float clumpMorphPeriodSec = 0.0f;
    int debugView = 0;
    std::uint64_t densityCurvesLayersHash = 0;
    std::uint64_t hash = 0;
};

struct DefectDustRecipe final {
    float cellWidthMm = 0.0f;
    float cellHeightMm = 0.0f;
    float slotProbability = 0.0f;
    float softnessMinMm = 0.0f;
    float softnessMaxMm = 0.0f;
    float softnessSizeCapFraction = 0.0f;
    float supportXMm = 0.0f;
    float supportYMm = 0.0f;
    float fiberFraction = 0.0f;
    float fiberDriftFraction = 0.0f;
    float fiberFirstKnotMin = 0.0f;
    float fiberFirstKnotMax = 0.0f;
    float fiberSecondKnotMin = 0.0f;
    float fiberSecondKnotMax = 0.0f;
    float fiberInteriorWidthMinFraction = 0.0f;
    float fiberInteriorWidthMaxFraction = 0.0f;
    float diameterMinMm = 0.0f;
    float diameterBulkMaxMm = 0.0f;
    float diameterMaxMm = 0.0f;
    float diameterTailFraction = 0.0f;
    float fiberLengthMinMm = 0.0f;
    float fiberLengthMaxMm = 0.0f;
    float fiberWidthMinMm = 0.0f;
    float fiberWidthMaxMm = 0.0f;
    float opacityFaintCumulative = 0.0f;
    float opacityIntermediateCumulative = 0.0f;
    float compactOpacityMin = 0.0f;
    float compactOpacityFaintEnd = 0.0f;
    float compactOpacityIntermediateEnd = 0.0f;
    float compactOpacityMax = 0.0f;
    float fiberOpacityMin = 0.0f;
    float fiberOpacityFaintEnd = 0.0f;
    float fiberOpacityIntermediateEnd = 0.0f;
    float fiberOpacityMax = 0.0f;
    float compactDominantAspectMin = 0.0f;
    float compactDominantAspectMax = 0.0f;
    float compactSubsidiaryScaleMin = 0.0f;
    float compactSubsidiaryScaleMax = 0.0f;
    float compactSubsidiaryAspectMin = 0.0f;
    float compactSubsidiaryAspectMax = 0.0f;
    float compactSubsidiaryOffsetMax = 0.0f;
    float compactSubsidiaryAngleMaxRadians = 0.0f;
};

struct DefectScratchRecipe final {
    float cellWidthMm = 0.0f;
    float cellHeightMm = 0.0f;
    float slotProbability = 0.0f;
    float softnessMinMm = 0.0f;
    float softnessMaxMm = 0.0f;
    float softnessSizeCapFraction = 0.0f;
    float supportXMm = 0.0f;
    float supportYMm = 0.0f;
    float lengthMinMm = 0.0f;
    float lengthBulkMaxMm = 0.0f;
    float lengthMaxMm = 0.0f;
    float lengthTailFraction = 0.0f;
    float widthMinMm = 0.0f;
    float widthBulkMaxMm = 0.0f;
    float widthMaxMm = 0.0f;
    float widthTailFraction = 0.0f;
    float driftFraction = 0.0f;
    float firstKnotMin = 0.0f;
    float firstKnotMax = 0.0f;
    float secondKnotMin = 0.0f;
    float secondKnotMax = 0.0f;
    float interiorWidthMinFraction = 0.0f;
    float interiorWidthMaxFraction = 0.0f;
    float interiorDepthMinFraction = 0.0f;
    float interiorDepthMaxFraction = 0.0f;
    float endpointAbruptProbability = 0.0f;
    float interruptionProbability = 0.0f;
    float gapCenterMin = 0.0f;
    float gapCenterMax = 0.0f;
    float gapSpanMin = 0.0f;
    float gapSpanMax = 0.0f;
    float scuffProbability = 0.0f;
    float scuffLengthMaxMm = 0.0f;
    float scuffAngleMaxRadians = 0.0f;
    float strengthMin = 0.0f;
    float strengthMax = 0.0f;
};

struct FilmJuicerEffectsRecipe final {
    DefectDustRecipe filmDust{};
    DefectScratchRecipe filmScratch{};
    DefectDustRecipe gateDust{};
    DefectScratchRecipe gateScratch{};
    double gateWeaveAmount = 0.0;
    bool active = false;
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
    bool blackCorrection = false;
    bool whiteCorrection = false;
    float blackLevel = 0.01f;
    float whiteLevel = 0.98f;
    bool glareActive = false;
    float glarePercent = 0.0f;
    float glareRoughness = 0.0f;
    float glareBlurSigmaPx = 0.0f;
    float lensBlurSigmaPx = 0.0f;
    float unsharpSigmaPx = 0.7f;
    float unsharpAmount = 0.7f;
    std::uint64_t hash = 0;
};

struct CmyCcTriplet {
    float c = 0.0f;
    float m = 0.0f;
    float y = 0.0f;
};

struct DichroicFilterRecipe {
    std::array<float, 4> customEdgesNm{{516.0f, 500.0f, 610.0f, 607.0f}};
    std::array<float, 4> customTransitionsNm{{12.0f, 8.0f, 8.0f, 8.0f}};
    std::uint64_t hash = 0;
};

struct PrintFilterRecipe {
    CmyCcTriplet mainCmyCc;
    CmyCcTriplet preflashCmyCc;
    DichroicFilterRecipe dichroic;
    std::uint64_t hash = 0;
};

struct PrintExposureRecipe {
    float printExposure = 1.0f;
    float preflashExposure = 0.0f;
    float cameraExposureCompensationEv = 0.0f;
    bool normalizePrintExposure = true;
    bool printExposureCompensation = true;
    Spektrafilm::PrintNormalizationMode normalizationMode =
        Spektrafilm::PrintNormalizationMode::NormalizeAndCompensate;
    std::uint64_t hash = 0;
};

struct PrintIlluminantRecipe {
    std::string key = "TH-KG3";
    std::uint64_t hash = 0;
};

struct PrintRecipe {
    PrintFilterRecipe filters;
    PrintExposureRecipe exposure;
    PrintIlluminantRecipe illuminant;
    std::uint64_t hash = 0;
};

struct RenderRecipe {
    ProfileRoute profileRoute;
    FilmRawRecipe filmRaw;
    SpatialOptics spatialOptics;
    FilmDevelopRecipe filmDevelop;
    DirCouplersRecipe dirCouplers;
    VisualGrainRecipe visualGrain;
    FilmJuicerEffectsRecipe filmJuicerEffects;
    GrainContract grainContract;
    DensityBoundsRecipe enlargerFilmBounds;
    DensityBoundsRecipe densityBounds;
    ScannerOutputRecipe scannerOutput;
    PrintRecipe print;
    bool directStructuralReady = false;
    bool printStructuralReady = false;
    std::uint64_t hash = 0;
};

namespace Spektrafilm {

    using ::DensityBoundsRecipe;
    using ::DirCouplersControls;
    using ::DirCouplersRecipe;
    using ::FilmDevelopRecipe;
    using ::FilmJuicerEffectsRecipe;
    using ::FilmRawRecipe;
    using ::GrainContract;
    using ::PrintRecipe;
    using ::ProfileRoute;
    using ::RenderRecipe;
    using ::ScannerOutputRecipe;
    using ::SpatialDirDescriptor;
    using ::SpatialOptics;
    using ::VisualGrainControls;
    using ::VisualGrainRecipe;

    struct SpatialOpticsControls {
        DiffusionFilterAuthoredControls cameraDiffusion;
        float cameraLensBlurUm = 0.0f;
        ScatterHalationControls scatterHalation;
        DiffusionFilterAuthoredControls enlargerDiffusion;
    };

    struct FilmFoundationBuildInput {
        std::string filmProfileKey;
        ScanRoute scanRoute = kDefaultScanRoute;
        std::shared_ptr<const Profiles::ValidatedFilmProfile> filmProfile;
        VisualGrainControls visualGrain;
        float filmDustAmount = 0.0f;
        float filmScratchAmount = 0.0f;
        float gateDustAmount = 0.0f;
        float gateScratchAmount = 0.0f;
        double gateWeaveAmount = 0.0;
        GrainContract grainContract;
        DirCouplersControls dirCouplers;
        SpatialOpticsControls spatialOptics;
        int spectralUpsamplingMode = 0;
        int inputColorSpace = 0;
        bool inputCctfDecoding = false;
        bool applyHanatos2025AdaptationWindow = true;
        bool applyHanatos2025AdaptationSurface = false;
        bool cameraAutoExposureEnabled = true;
        int cameraMeteringMethod = 0;
        float manualExposureCompensationEv = 0.0f;
        float filmFormatLongEdgeMm = 35.0f;
        bool cameraFilterOverride = false;
        std::array<double, 3> cameraFilterUV{{1.0, 410.0, 8.0}};
        std::array<double, 3> cameraFilterIR{{1.0, 675.0, 15.0}};
        std::array<float, 81> referenceIlluminant{};
        bool referenceIlluminantValid = false;
    };

    struct DirectRecipeBuildInput {
        FilmFoundationBuildInput film;
        std::uint32_t scannerLutResolution = 17;
        int outputColorSpace = 0;
        bool outputCctfEncoding = true;
        bool scannerBlackCorrection = false;
        bool scannerWhiteCorrection = false;
        float scannerBlackLevel = 0.01f;
        float scannerWhiteLevel = 0.98f;
        float scannerLensBlurSigmaPx = 0.0f;
        float scannerUnsharpSigmaPx = 0.7f;
        float scannerUnsharpAmount = 0.7f;
    };

    struct DirectRecipeBuildResult {
        RenderRecipe recipe;
        std::string diagnostic;
        bool valid = false;
    };

    struct PrintRecipeBuildInput {
        FilmFoundationBuildInput film;
        std::string printProfileKey;
        std::shared_ptr<const Profiles::ValidatedPrintProfile> printProfile;
        NeutralCalibrationStatus neutralCalibrationStatus = NeutralCalibrationStatus::MissingEntry;
        CmyCcTriplet currentNeutralCmyCc{0.0f, 65.0f, 55.0f};
        CmyCcTriplet calibratedNeutralCmyCc{0.0f, 65.0f, 55.0f};
        // UI remains Y/M/C. These values are converted once to internal C/M/Y here.
        std::array<float, 3> uiYmcCc{};
        float preflashMFilterCc = 0.0f;
        float preflashYFilterCc = 0.0f;
        float printExposure = 1.0f;
        float preflashExposure = 0.0f;
        bool normalizePrintExposure = true;
        bool printExposureCompensation = true;
        std::string printIlluminantKey = "TH-KG3";
        std::uint32_t scannerLutResolution = 17;
        int outputColorSpace = 0;
        bool outputCctfEncoding = true;
        bool scannerBlackCorrection = false;
        bool scannerWhiteCorrection = false;
        float scannerBlackLevel = 0.01f;
        float scannerWhiteLevel = 0.98f;
        bool glareActive = true;
        float glarePercent = 0.03f;
        float glareRoughness = 0.7f;
        float glareBlurSigmaPx = 0.5f;
        float scannerLensBlurSigmaPx = 0.0f;
        float scannerUnsharpSigmaPx = 0.7f;
        float scannerUnsharpAmount = 0.7f;
    };

    struct PrintRecipeBuildResult {
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
    PrintRecipeBuildResult build_print_render_recipe(const PrintRecipeBuildInput& input);
    float print_exposure_normalizer(
        PrintNormalizationMode mode,
        float factorMidgray,
        float factorMidgrayComp);
    bool build_diffusion_frame_set_descriptor(
        const SpatialOptics& optics,
        ScanRoute route,
        double pixelSizeUm,
        DiffusionFrameDomain fullFrame,
        std::optional<DiffusionFrameSetDescriptor>& out,
        std::string& diagnostic);
    bool build_spatial_dir_descriptor(
        const DirCouplersRecipe& recipe,
        float pixelSizeUm,
        DirFrameExtent renderExtent,
        DirFrameExtent fullFrameExtent,
        const char* traceRouteLabel,
        SpatialDirDescriptor& out);
    bool preflight_camera_lens_blur(
        const CameraLensBlurOpticsRecipe& recipe,
        std::string& outDiagnostic);

} // namespace Spektrafilm
