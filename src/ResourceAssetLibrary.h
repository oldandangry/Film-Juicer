#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ProfileAssets.h"
#include "ProfileCatalog.h"
#include "ScanRoute.h"
#include "SpectralData.h"

namespace JuicerAssets {

    struct StbnNoisePayload {
        std::vector<std::uint8_t> data;
        int width = 512;
        int height = 512;
        int frames = 256;
        bool valid = false;
        std::string error;
        std::uint64_t version = 0;
    };

    struct WangNoisePayload {
        std::vector<std::uint8_t> tiles;
        std::vector<std::uint8_t> lut;
        int width = 0;
        int height = 0;
        int count = 0;
        int colors = 0;
        bool valid = false;
        std::string error;
        std::uint64_t version = 0;
    };

    struct StaticNoisePayloadSet {
        StbnNoisePayload stbn;
        WangNoisePayload wang;
        std::uint64_t version = 0;
    };

    struct IlluminantFilterCurveSet {
        Spectral::Curve d65;
        Spectral::Curve d55;
        Spectral::Curve d50;
        Spectral::Curve tungsten;
        Spectral::Curve kinoton75P;
        Spectral::Curve tungstenKg3;
        Spectral::Curve tungstenKg3Lens;
        std::uint64_t version = 0;
    };

    enum class NeutralPrintCalibrationStatus : unsigned char {
        MissingFile,
        MissingEntry,
        Found,
        Malformed
    };

    struct NeutralPrintCalibrationResult {
        NeutralPrintCalibrationStatus status =
            NeutralPrintCalibrationStatus::MissingEntry;
        std::array<float, 3> cmyCc{};
        std::string diagnostic;
    };

    using SelectedProfileRequest = Profiles::SelectedProfileRequest;
    using SelectedProfileResult = Profiles::SelectedProfileResult;

    class Library {
    public:
        struct StaticNoiseAssetSet;
        struct IlluminantFilterAssetSet;

        static constexpr std::uint64_t kProcessAssetVersion = 1ull;

        explicit Library(std::string dataDir);
        ~Library();

        const Spektrafilm::ProfileCatalog& spektrafilm_profile_catalog();
        std::shared_ptr<const Profiles::ValidatedFilmProfile>
        selected_film_profile_for_key(const std::string& key);
        SelectedProfileResult selected_profiles_for_route(
            const SelectedProfileRequest& request);
        std::shared_ptr<const StaticNoisePayloadSet> static_noise_payloads();
        const IlluminantFilterCurveSet& illuminant_filter_curves();
        NeutralPrintCalibrationResult neutral_print_calibration(
            const std::string& printProfileKey,
            const std::string& printIlluminantKey,
            const std::string& filmProfileKey);
        void release_cached_payloads() noexcept;

    private:
        void ensure_catalogs();
        void ensure_static_noise_assets();
        void ensure_illuminant_filter_assets();
        void load_catalogs();
        void load_static_noise_assets();
        void load_illuminant_filter_assets();

        struct StaticNoisePayloadCacheState;
        struct IlluminantFilterCurveCacheState;
        struct NeutralPrintCalibrationCacheState;

        std::once_flag _catalogOnce;
        std::once_flag _staticNoiseOnce;
        std::once_flag _illuminantFilterOnce;
        std::string _dataDir;
        Spektrafilm::ProfileCatalog _spektrafilmProfileCatalog;
        std::unique_ptr<StaticNoiseAssetSet> _staticNoiseAssets;
        std::unique_ptr<IlluminantFilterAssetSet> _illuminantFilterAssets;
        std::unique_ptr<StaticNoisePayloadCacheState> _staticNoisePayloadCache;
        std::unique_ptr<IlluminantFilterCurveCacheState>
            _illuminantFilterCurveCache;
        std::unique_ptr<NeutralPrintCalibrationCacheState>
            _neutralPrintCalibrationCache;
        std::unique_ptr<Profiles::ProfileAssetStore> _selectedProfileAssets;
    };

} // namespace JuicerAssets
