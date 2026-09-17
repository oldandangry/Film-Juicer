#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ProfileCatalog.h"
#include "ScanRoute.h"

namespace Profiles {

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
        IlluminantKey referenceIlluminant{"D55"};
        IlluminantKey viewingIlluminant{"D50"};
    };

    struct SpektrafilmProfileSamples {
        std::array<float, 81> wavelengths{};
        std::array<std::array<float, 3>, 81> logSensitivity{};
        std::array<std::array<float, 3>, 81> linearSensitivity{};
        std::array<std::array<float, 3>, 81> channelDensity{};
        std::array<float, 81> baseDensity{};
        std::vector<float> logExposure;
        std::vector<std::array<float, 3>> densityCurves;
        std::array<std::array<std::vector<float>, 3>, 3> densityCurvesLayers{}; // [sublayer][channel]
        std::array<float, 4> hanatos2025AdaptationWindowParams{};
        std::array<std::array<float, 15>, 3> hanatos2025AdaptationSurfaceParams{};
        bool hasHanatos2025AdaptationWindowParams = false;
        bool hasHanatos2025AdaptationSurfaceParams = false;
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

    inline constexpr std::uint64_t kDensityCurveEvaluatorVersion = 1u;

    struct DensityCurveModel {
        std::array<std::array<double, 3>, 3> centers{};
        std::array<std::array<double, 3>, 3> amplitudes{};
        std::array<std::array<double, 3>, 3> sigmas{};
    };

    struct DensityCurveSample {
        std::array<float, 3> total{};
        std::array<std::array<float, 3>, 3> layers{}; // [layer][channel]
        bool valid = false;
    };

    DensityCurveSample evaluate_density_curve_sample(
        const DensityCurveModel& model,
        Spektrafilm::ProfilePolarity polarity,
        double sourceLogExposure);

    struct ValidatedFilmProfile {
        SpektrafilmProfileInfo info;
        SpektrafilmProfileSamples data;
        std::vector<double> sourceLogExposure;
        DensityCurveModel densityModel;
        ProfileDigest digest;
        std::uint64_t assetVersionToken = 0;
    };

    struct ValidatedPrintProfile {
        SpektrafilmProfileInfo info;
        SpektrafilmProfileSamples data;
        std::vector<double> sourceLogExposure;
        DensityCurveModel densityModel;
        std::uint64_t assetVersionToken = 0;
    };

    struct SelectedProfileRequest {
        std::string filmProfileKey;
        std::string printProfileKey;
        Spektrafilm::ScanRoute scanRoute = Spektrafilm::kDefaultScanRoute;
    };

    struct SelectedProfileResult {
        std::shared_ptr<const ValidatedFilmProfile> filmProfile;
        std::shared_ptr<const ValidatedPrintProfile> printProfile;
        bool valid = false;
        std::string diagnostic;
    };

    class ProfileAssetStore {
    public:
        ProfileAssetStore();
        ~ProfileAssetStore();

        std::shared_ptr<const ValidatedFilmProfile> load_film_profile_by_key(
            const Spektrafilm::ProfileCatalog& catalog,
            const std::string& key,
            std::string* outDiagnostic = nullptr);
        std::shared_ptr<const ValidatedPrintProfile> load_print_profile_by_key(
            const Spektrafilm::ProfileCatalog& catalog,
            const std::string& key,
            std::string* outDiagnostic = nullptr);
        SelectedProfileResult selected_profiles_for_route(
            const Spektrafilm::ProfileCatalog& catalog,
            const SelectedProfileRequest& request);
        void release_cached_payloads() noexcept;

    private:
        struct CacheState;
        std::unique_ptr<CacheState> _cache;
    };

} // namespace Profiles
