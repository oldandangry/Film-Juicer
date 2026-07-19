#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "ProfileAssets.h"

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
        float printShadowCompensationFactor = 0.0f;
        float printShadowCompensationDensity = 0.0f;
        float printShadowCompensationTransition = 0.0f;
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
        float blurDyeCloudsUm = 0.0f;    // Dye-cloud blur sigma scale in pixels.
        float sizeMixWeight = 0.30f;
        float sizeMixWeightMid = 0.0f;
        float sizeMixScale = 3.0f;
        float clumpTemporalMix = 0.30f;
        float clumpMorphPeriodSec = 8.0f;
        int debugView = 0;
        std::array<float, 2> microStructure{{0.0f, 0.0f}}; // [cell_um, clump_sigma_x1e-3]
        int nSubLayers = 1;
        float filmDustAmount = 0.0f;
        float gateDustAmount = 0.0f;
        float filmScratchAmount = 0.0f;
        float gateScratchAmount = 0.0f;
    };

    // UI halation controls; profile-derived defaults come from ProfileDigest antihalation presets.
    struct HalationMetadata {
        bool active = false;
        std::array<float, 3> primaryAmount{{0.0f, 0.0f, 0.0f}};
        std::array<float, 3> sizeUm{{0.0f, 0.0f, 0.0f}};
        std::array<float, 3> secondaryAmount{{0.0f, 0.0f, 0.0f}};
        std::array<float, 3> secondarySizeUm{{0.0f, 0.0f, 0.0f}};
    };

    // Profile-authored DIR coefficients consumed by the recipe-owned DIR runtime.
    struct DirProfile {
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

    struct SpektrafilmProfileJson {
        std::vector<std::pair<float, float>> dyeC;
        std::vector<std::pair<float, float>> dyeM;
        std::vector<std::pair<float, float>> dyeY;
        std::vector<std::pair<float, float>> baseDensityMin;
        std::vector<std::pair<float, float>> baseDensityMid;
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

        DirProfile dirCouplers;
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

    bool load_spektrafilm_profile_json(const std::string& jsonPath, SpektrafilmProfileJson& outProfile);

    bool load_profile_info(const std::string& jsonPath, ProfileInfoSummary& outInfo);

    bool load_validated_film_profile_json(
        const std::string& jsonPath,
        ValidatedFilmProfile& outProfile,
        std::string* outDiagnostic = nullptr);

    bool load_validated_print_profile_json(
        const std::string& jsonPath,
        ValidatedPrintProfile& outProfile,
        std::string* outDiagnostic = nullptr);

} // namespace Profiles
