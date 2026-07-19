#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ProfileCatalog.h"
#include "ScanRoute.h"

namespace Profiles {

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
        bool supportDefaulted = false;
        bool stageDefaulted = false;
        bool typeDefaulted = false;
        bool useDefaulted = false;
        bool antihalationDefaulted = false;
        bool channelModelDefaulted = false;
        bool referenceIlluminantDefaulted = false;
        bool viewingIlluminantDefaulted = false;
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
        bool hasDensityCurvesLayers = false;
        bool densityCurvesLayersMalformed = false;
        std::string densityCurvesLayersDiagnostic;
        std::array<float, 4> hanatos2025AdaptationWindowParams{};
        std::array<std::array<float, 15>, 3> hanatos2025AdaptationSurfaceParams{};
        bool hasHanatos2025AdaptationWindowParams = false;
        bool hasHanatos2025AdaptationSurfaceParams = false;
    };

    struct SpektrafilmFilmData : SpektrafilmProfileSamples {
    };

    struct SpektrafilmPrintData : SpektrafilmProfileSamples {
    };

    struct ProfileDigest {
        ProfileRole profileRole = ProfileRole::Film;
        std::array<float, 3> gammaSamelayerRgb{{0.336f, 0.319f, 0.273f}};
        std::array<float, 2> gammaInterlayerRToGb{{0.353f, 0.302f}};
        std::array<float, 2> gammaInterlayerGToRb{{0.154f, 0.353f}};
        std::array<float, 2> gammaInterlayerBToRg{{0.168f, 0.226f}};
        std::string dirGammaSource = "negative-default";
        std::array<float, 3> halationFirstSigmaUm{{65.0f, 65.0f, 65.0f}};
        std::array<float, 3> halationPrimaryAmount{{0.08f, 0.02f, 0.0f}};
        bool halationPresetApplied = true;
        float hanatosSpectralGaussianBlurDefault = 0.0f;
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

    struct SelectedProfileRequest {
        std::string filmProfileKey;
        std::string printProfileKey;
        Spektrafilm::ScanRoute scanRoute = Spektrafilm::kDefaultScanRoute;
    };

    struct SelectedProfileResult {
        std::shared_ptr<const ValidatedFilmProfile> filmProfile;
        std::shared_ptr<const ValidatedPrintProfile> printProfile;
        bool directRoutePrintProfileExcluded = false;
        bool directRouteNeutralCalibrationExcluded = false;
        bool valid = false;
        std::string diagnostic;
    };

    ProfileDigest build_profile_digest(const SpektrafilmProfileInfo& info, ProfileRole role);
    std::uint64_t build_profile_asset_version_token(
        const SpektrafilmProfileInfo& info,
        const SpektrafilmProfileSamples& data);

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
