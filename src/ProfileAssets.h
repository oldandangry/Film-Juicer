#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "ProfileCatalog.h"
#include "ScanRoute.h"

namespace JuicerAssets {
    class PrintProfileSource;
}

namespace Profiles {
    struct NamedIlluminant {};
    struct BlackbodyIlluminant {
        double temperatureKelvin;
    };
    struct ProfileIlluminant {
        std::string value;
        std::variant<NamedIlluminant, BlackbodyIlluminant> kind;
    };

    struct FilmProfileInfo {
        std::string stock;
        Spektrafilm::ProfileSupport support = Spektrafilm::ProfileSupport::Film;
        Spektrafilm::ProfileStage stage = Spektrafilm::ProfileStage::Filming;
        Spektrafilm::ProfilePolarity type = Spektrafilm::ProfilePolarity::Negative;
        ProfileIlluminant referenceIlluminant{"D55", NamedIlluminant{}};
        ProfileIlluminant viewingIlluminant{"D50", NamedIlluminant{}};
    };
    struct PrintProfileInfo {
        std::string stock;
        Spektrafilm::ProfileStage stage = Spektrafilm::ProfileStage::Printing;
        ProfileIlluminant viewingIlluminant{"D50", NamedIlluminant{}};
    };
    struct FilmProfileSamples {
        std::array<float, 81> wavelengths{};
        std::array<std::array<float, 3>, 81> linearSensitivity{};
        std::array<std::array<float, 3>, 81> channelDensity{};
        std::array<float, 81> baseDensity{};
        std::vector<float> logExposure;
        std::vector<std::array<float, 3>> densityCurves;
        std::array<std::array<std::vector<float>, 3>, 3> densityCurvesLayers{}; // [layer][channel][sample]
        std::array<float, 4> hanatos2025AdaptationWindowParams{};
        std::array<std::array<float, 15>, 3> hanatos2025AdaptationSurfaceParams{};
        bool hasHanatos2025AdaptationWindowParams = false;
        bool hasHanatos2025AdaptationSurfaceParams = false;
    };
    struct PrintProfileSamples {
        std::array<std::array<float, 3>, 81> linearSensitivity{};
        std::array<std::array<float, 3>, 81> channelDensity{};
        std::array<float, 81> baseDensity{};
        std::vector<float> logExposure;
        std::vector<std::array<float, 3>> densityCurves;
    };

    struct ProfileDigest {
        std::array<float, 3> gammaSamelayerRgb{{0.341f, 0.324f, 0.273f}};
        std::array<float, 2> gammaInterlayerRToGb{{0.355f, 0.305f}};
        std::array<float, 2> gammaInterlayerGToRb{{0.154f, 0.358f}};
        std::array<float, 2> gammaInterlayerBToRg{{0.171f, 0.225f}};
        std::array<float, 3> halationFirstSigmaUm{{65.0f, 65.0f, 65.0f}};
        std::array<float, 3> halationPrimaryAmount{{0.08f, 0.02f, 0.0f}};
        float hanatosSpectralGaussianBlurDefault = 0.0f;
    };

    struct FilmProfile {
        FilmProfileInfo info;
        FilmProfileSamples data;
        ProfileDigest digest;
        std::uint64_t assetVersionToken = 0;
    };
    struct PrintProfile {
        PrintProfileInfo info;
        PrintProfileSamples data;
        std::uint64_t assetVersionToken = 0;
    };
    struct PrintDensityCurves {
        std::vector<std::array<float, 3>> totals;
        std::uint64_t hash = 0;
    };

    struct SelectedProfileRequest {
        std::string filmProfileKey;
        std::string printProfileKey;
        Spektrafilm::ScanRoute scanRoute = Spektrafilm::kDefaultScanRoute;
    };

    struct SelectedProfileResult {
        std::shared_ptr<const FilmProfile> filmProfile;
        std::shared_ptr<const JuicerAssets::PrintProfileSource> printSource;
        bool valid = false;
        std::string diagnostic;
    };

} // namespace Profiles
