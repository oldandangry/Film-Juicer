#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ProfileCatalog.h"

namespace Profiles {

    struct ProfileInfoSummary {
        std::string stock;
        std::string name;
        std::string type;
    };

    struct ProfileGlare {
        bool active = false;
        float percent = 0.0f;
        float roughness = 0.0f;
        float blur = 0.0f;
        float compensationRemovalFactor = 0.0f;
        float compensationRemovalDensity = 0.0f;
        float compensationRemovalTransition = 0.0f;
    };

    struct GrainMetadata {
        bool active = false;
        bool sublayersActive = false;
        float agxParticleAreaUm2 = 0.0f;
        std::array<float, 3> agxParticleScale{{0.0f, 0.0f, 0.0f}};
        std::array<float, 3> agxParticleScaleLayers{{0.0f, 0.0f, 0.0f}};
        std::array<float, 3> densityMin{{0.0f, 0.0f, 0.0f}};
        std::array<float, 3> uniformity{{0.0f, 0.0f, 0.0f}};
        float amplitude = 1.0f;          // Grain amplitude scalar (OD delta multiplier).
        float chroma = 1.0f;             // Grain chroma correlation mix (0=shared, 1=independent).
        float chromaSharedWeight = 0.0f; // sqrt(1 - chroma)
        float chromaIndWeight = 1.0f;    // sqrt(chroma)
        float blur = 0.0f;               // Grain blur sigma in pixels.
        float blurDyeCloudsUm = 0.0f;    // Dye-cloud blur sigma scale in pixels (legacy _um name).
        float sizeMixWeight = 0.30f;
        float sizeMixWeightMid = 0.0f;
        float sizeMixScale = 3.0f;
        float clumpTemporalMix = 0.30f;
        float clumpMorphPeriodSec = 8.0f;
        bool breathingDebug = false;
        int debugView = 0;
        std::array<float, 2> microStructure{{0.0f, 0.0f}}; // [cell_um, clump_sigma_x1e-3]
        int nSubLayers = 1;
        float filmDustAmount = 0.0f;
        float gateDustAmount = 0.0f;
        float filmScratchAmount = 0.0f;
        float gateScratchAmount = 0.0f;
    };

    struct HalationMetadata {
        bool active = false;
        std::array<float, 3> strength{{0.0f, 0.0f, 0.0f}};
        std::array<float, 3> sizeUm{{0.0f, 0.0f, 0.0f}};
        std::array<float, 3> scatteringStrength{{0.0f, 0.0f, 0.0f}};
        std::array<float, 3> scatteringSizeUm{{0.0f, 0.0f, 0.0f}};
    };

    struct DirCouplersProfile {
        bool hasData = false;
        bool active = false;
        float amount = 1.0f;
        std::array<float, 3> ratioRGB{{1.0f, 1.0f, 1.0f}};
        float diffusionInterlayer = 0.0f;
        float diffusionSizeUm = 0.0f;
        float highExposureShift = 0.0f;
    };

    struct MaskingCouplersProfile {
        bool hasData = false;
        std::vector<float> crossOverPoints;
        std::vector<float> transitionWidths;
        std::array<std::vector<std::array<float, 3>>, 3> gaussianModel{};
    };

    struct AgxFilmProfile {
        std::vector<std::pair<float, float>> dyeC;
        std::vector<std::pair<float, float>> dyeM;
        std::vector<std::pair<float, float>> dyeY;
        std::vector<std::pair<float, float>> baseMin;
        std::vector<std::pair<float, float>> baseMid;
        float dyeDensityMinFactor = 1.0f;
        std::array<float, 3> gammaFactor{{1.0f, 1.0f, 1.0f}};
        bool hasGammaFactor = false;

        float glareCompensationFactor = 0.0f;
        float glareCompensationDensity = 1.2f;
        float glareCompensationTransition = 0.3f;
        bool hasGlareCompensation = false;

        std::string densitometer;
        std::vector<float> densityMidNeutral;
        std::vector<float> logExposureMidNeutral;
        std::string referenceIlluminant;
        std::string viewingIlluminant;

        std::vector<std::pair<float, float>> logSensR;
        std::vector<std::pair<float, float>> logSensG;
        std::vector<std::pair<float, float>> logSensB;

        std::vector<std::pair<float, float>> densityCurveR;
        std::vector<std::pair<float, float>> densityCurveG;
        std::vector<std::pair<float, float>> densityCurveB;
        std::array<std::array<std::vector<std::pair<float, float>>, 3>, 3> densityCurvesLayers{}; // [layer][channel]
        bool hasDensityCurvesLayers = false;

        DirCouplersProfile dirCouplers;
        MaskingCouplersProfile maskingCouplers;

        std::array<float, 3> cameraFilterUV{{1.0f, 410.0f, 8.0f}};
        std::array<float, 3> cameraFilterIR{{1.0f, 675.0f, 15.0f}};
        bool hasCameraFilterUV = false;
        bool hasCameraFilterIR = false;

        std::string type;
        ProfileGlare glare;
        GrainMetadata grain;
        HalationMetadata halation;
        bool hasGlare = false;
        bool hasGrain = false;
        bool hasHalation = false;
    };

    enum class ProfileRole : unsigned char {
        Film,
        Print
    };

    enum class ProfileUse : unsigned char {
        Still,
        Cine,
        Unsupported
    };

    enum class ProfileAntihalation : unsigned char {
        Strong,
        Weak,
        No,
        Unsupported
    };

    enum class ProfileChannelModel : unsigned char {
        Color,
        Bw,
        Unsupported
    };

    enum class NeutralCalibrationStatus : unsigned char {
        NotConsumedInPhase2B,
        ExcludedForDirectRoute,
        OptionalPresent,
        OptionalMissing
    };

    struct DensitometerLabel {
        std::string value = "status_M";
    };

    struct IlluminantKey {
        std::string value;
    };

    struct SpektrafilmProfileInfo {
        std::string stock;
        std::string name;
        Spektrafilm::ProfileSupport support = Spektrafilm::ProfileSupport::Film;
        Spektrafilm::ProfileStage stage = Spektrafilm::ProfileStage::Filming;
        Spektrafilm::ProfilePolarity type = Spektrafilm::ProfilePolarity::Negative;
        ProfileUse use = ProfileUse::Still;
        ProfileAntihalation antihalation = ProfileAntihalation::Weak;
        ProfileChannelModel channelModel = ProfileChannelModel::Color;
        DensitometerLabel densitometer{};
        IlluminantKey referenceIlluminant{"D55"};
        IlluminantKey viewingIlluminant{"D50"};
        float logSensitivityDensityOverMin = 0.2f;
        bool supportDefaulted = false;
        bool stageDefaulted = false;
        bool typeDefaulted = false;
        bool useDefaulted = false;
        bool antihalationDefaulted = false;
        bool channelModelDefaulted = false;
        bool densitometerDefaulted = false;
        bool referenceIlluminantDefaulted = false;
        bool viewingIlluminantDefaulted = false;
        bool logSensitivityDensityOverMinDefaulted = false;
    };

    struct SpektrafilmProfileSamples {
        std::array<float, 81> wavelengths{};
        std::array<std::array<float, 3>, 81> logSensitivity{};
        std::array<std::array<float, 3>, 81> linearSensitivity{};
        std::array<std::array<float, 3>, 81> channelDensity{};
        std::array<float, 81> baseDensity{};
        std::vector<float> logExposure;
        std::vector<std::array<float, 3>> densityCurves;
        std::vector<std::array<std::array<float, 3>, 3>> densityCurvesLayers;
        std::array<float, 4> hanatos2025AdaptationWindowParams{};
        std::array<std::array<float, 15>, 3> hanatos2025AdaptationSurfaceParams{};
        bool hasDensityCurvesLayers = false;
        bool hasHanatos2025AdaptationWindowParams = false;
        bool hasHanatos2025AdaptationSurfaceParams = false;
    };

    struct SpektrafilmFilmData : SpektrafilmProfileSamples {
    };

    struct SpektrafilmPrintData : SpektrafilmProfileSamples {
    };

    struct GrainContract {
        std::array<float, 3> density_min{{0.07f, 0.08f, 0.12f}};
        bool densityCurvesLayersAuthored = false;
    };

    struct ProfileDigest {
        ProfileRole profileRole = ProfileRole::Film;
        NeutralCalibrationStatus optionalNeutralCalibrationStatus = NeutralCalibrationStatus::NotConsumedInPhase2B;
        std::array<float, 3> gammaSamelayerRgb{{0.336f, 0.319f, 0.273f}};
        std::array<float, 2> gammaInterlayerRToGb{{0.353f, 0.302f}};
        std::array<float, 2> gammaInterlayerGToRb{{0.154f, 0.353f}};
        std::array<float, 2> gammaInterlayerBToRg{{0.168f, 0.226f}};
        std::string dirGammaSource = "negative-default";
        std::array<float, 3> halationFirstSigmaUm{{65.0f, 65.0f, 65.0f}};
        std::array<float, 3> halationStrength{{0.08f, 0.02f, 0.0f}};
        bool halationPresetApplied = true;
        bool hanatosWindowAuthored = false;
        bool hanatosSurfaceAuthored = false;
        bool hanatosRuntimeApplyWindowDefault = true;
        bool hanatosRuntimeApplySurfaceDefault = false;
        float hanatosSpectralGaussianBlurDefault = 0.0f;
        GrainContract grainContract{};
    };

    struct SelectedProfileDiagnostic {
        std::string message;
        std::string profileKey;
        std::string sourcePath;
        std::string field;
        std::string route;
        bool failed = false;
    };

    struct ValidatedFilmProfile {
        SpektrafilmProfileInfo info;
        SpektrafilmFilmData data;
        ProfileDigest digest;
        SelectedProfileDiagnostic diagnostic;
        std::uint64_t assetVersionToken = 0;
        std::string sourcePath;
    };

    struct ValidatedPrintProfile {
        SpektrafilmProfileInfo info;
        SpektrafilmPrintData data;
        ProfileDigest digest;
        SelectedProfileDiagnostic diagnostic;
        std::uint64_t assetVersionToken = 0;
        std::string sourcePath;
    };

    bool load_agx_film_profile_json(const std::string& jsonPath, AgxFilmProfile& outProfile);

    bool load_profile_info(const std::string& jsonPath, ProfileInfoSummary& outInfo);

    bool load_validated_film_profile_json(
        const std::string& jsonPath,
        ValidatedFilmProfile& outProfile,
        std::string* outDiagnostic = nullptr);

    bool load_validated_print_profile_json(
        const std::string& jsonPath,
        ValidatedPrintProfile& outProfile,
        std::string* outDiagnostic = nullptr);

    ProfileDigest build_profile_digest(const SpektrafilmProfileInfo& info, ProfileRole role);

} // namespace Profiles
