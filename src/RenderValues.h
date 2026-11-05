#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "DiffusionHostBehavior.h"
#include "GamutCompression.h"
#include "ScanRoute.h"
#include "ScatterHalation.h"

namespace JuicerDir {
    class MathOutcome;
}
struct FjDirFrameMath;

// Production photographic policy and profile ownership live in Rust PreparedState.
// Native descriptors and coefficients bind CUDA execution. Frame resources, scratch,
// and asynchronous retirement remain Root/context owned.
namespace Spektrafilm {

    inline constexpr const char kQuantizedMedianNotAcceptedForPhase3[] =
        "QuantizedMedianNotAcceptedForPhase3";
    inline constexpr const char kExactOpticsNotImplementedForPhase6[] =
        "ExactOpticsNotImplementedForPhase6";

    enum class RgbToRawMethod : std::uint8_t {
        Hanatos2025,
        Mallett2019,
        Arctic2026beta04
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

    enum class SpatialOpticsDomain : std::uint8_t {
        FilmLinearExposure,
        PrintLinearExposure
    };

} // namespace Spektrafilm

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

enum class DirNonlinearMode : std::uint8_t {
    Inactive,
    NegativeDonorLangmuir,
    PositiveReceiverLangmuir
};

namespace Spektrafilm {

    enum class DirReferenceOperator : std::uint8_t {
        None,
        Identity,
        SpektrafilmSmallFirReflect,
        SpektrafilmLargeYvvReflect
    };

    enum class DirScratchTier : std::uint8_t {
        Tier0,
        Tier1F,
        Tier1IChannels,
        Tier2,
        Unsupported
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
        struct IirCoefficients {
            double feedforward = 0.0;
            std::array<double, 3> feedback{};
            double normalizationDenominator = 0.0;
            double feedforwardNumerator = 0.0;
            std::array<double, 3> feedbackNumerators{};
        };

        float sigmaPixels = 0.0f;
        float weight = 0.0f;
        int radius = 0;
        DirReferenceOperator referenceOperator = DirReferenceOperator::None;
        DirScratchTier targetScratchTier = DirScratchTier::Tier0;
        IirCoefficients iir{};
        double boundaryTruncationAccuracy = 1.0e-11;
        double boundaryCertificationTolerance = 5.0e-7;
        std::uint32_t boundaryDerivationVersion = 1u;
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

} // namespace Spektrafilm

struct SpatialDirDescriptor {
    std::uint64_t dirRecipeHash = 0;
    Spektrafilm::DirScratchTier scratchTier = Spektrafilm::DirScratchTier::Tier0;
    Spektrafilm::DirDescriptorSupport support = Spektrafilm::DirDescriptorSupport::Inactive;
    Spektrafilm::DirFrameExtent renderExtent{};
    Spektrafilm::DirFrameExtent fullFrameExtent{};
    Spektrafilm::DirFrameExtent filterDomainExtent{};
    Spektrafilm::DirFilterPlan filterPlan{};
    Spektrafilm::DirScratchPlaneRoles planeRoles{};
    Spektrafilm::DirScratchTier targetScratchTier = Spektrafilm::DirScratchTier::Tier0;
    Spektrafilm::DirScratchPlaneRoles targetPlaneRoles{};
    const char* traceRouteLabel = nullptr;
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

struct OutputGamutRecipe {
    bool enabled = false;
    float lightnessKneeThreshold = 0.95f;
    float lightnessKneeLimit = 1.0f;
    float lightnessKneePower = 1.6f;
    float chromaKneeThreshold = 0.95f;
    float chromaKneeLimit = 1.0f;
    float chromaKneePower = 1.6f;
    int outputColorSpace = 0;
    Gamut::OutputGamutTransform transform;
    std::uint64_t transformTableVersionHash = 0;
    std::uint64_t hash = 0;
};

struct CmyCcTriplet {
    float c = 0.0f;
    float m = 0.0f;
    float y = 0.0f;
};

namespace Spektrafilm {

    using ::DirNonlinearMode;
    using ::FilmJuicerEffectsRecipe;
    using ::SpatialDirDescriptor;
    using ::SpatialOptics;

    // Native workspace policy for an already resolved nonempty DIR operator plan.
    void resolve_spatial_dir_scratch(SpatialDirDescriptor& descriptor) noexcept;

    bool compose_spatial_dir_descriptor(
        const JuicerDir::MathOutcome&, const char* traceRouteLabel, SpatialDirDescriptor& out);

    // S4.E consumes an already-computed Rust frame result; native code retains
    // final scratch/role/alias and execution hash binding.
    bool compose_spatial_dir_descriptor(
        const FjDirFrameMath&, float pixelSizeUm, const char* traceRouteLabel, SpatialDirDescriptor& out);

    bool preflight_camera_lens_blur(
        const CameraLensBlurOpticsRecipe& recipe,
        std::string& outDiagnostic);

} // namespace Spektrafilm
