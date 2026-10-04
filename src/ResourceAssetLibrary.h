#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "GamutCompression.h"
#include "ProfileAssets.h"
#include "ProfileCatalog.h"
#include "ScanRoute.h"
#include "RustAssetBridge.h"
#include "SpectralData.h"

namespace JuicerAssets {

    struct IlluminantFilterCurveSet {
        Spectral::Curve d65;
        Spectral::Curve d55;
        Spectral::Curve d50;
        Spectral::Curve tungsten;
        Spectral::Curve kinoton75P;
        Spectral::Curve tungstenKg3;
        Spectral::Curve tungstenKg3Lens;
    };

    using SelectedProfileRequest = Profiles::SelectedProfileRequest;
    using SelectedProfileResult = Profiles::SelectedProfileResult;

    class Library {
    public:
        explicit Library(const std::filesystem::path& resourceRoot);
        ~Library();

        Spectral::ReconstructionLut copy_hanatos_lut();
        Spectral::ReconstructionLut copy_arctic_lut();
        Spectral::MallettBasis copy_mallett_basis();
        Spectral::CMFTriplets copy_cmf_triplets();
        const Spektrafilm::ProfileCatalog& spektrafilm_profile_catalog();
        std::shared_ptr<const Profiles::FilmProfile>
        selected_film_profile_for_key(const std::string& key);
        SelectedProfileResult selected_profiles_for_route(
            const SelectedProfileRequest& request);
        NoiseSource noise();
        std::shared_ptr<const IlluminantFilterCurveSet> illuminant_filter_curves();
        std::shared_ptr<const Gamut::InputCompressionHull>
        input_compression_hull();
        std::shared_ptr<const Gamut::OutputBoundaryTable>
        output_boundary_table(
            const Gamut::OutputGamutTransform& transform,
            std::string& diagnostic);
        NeutralPrintCalibrationResult neutral_print_calibration(
            const std::string& printStock,
            const std::string& printIlluminantKey,
            const std::string& filmStock);
        FjStatus release_cached_payloads(FjErrorBuffer* error = nullptr) noexcept;
        FjStatus close(FjErrorBuffer* error = nullptr) noexcept;

    private:
        void ensure_catalogs();
        void load_catalogs();

        struct IlluminantFilterCurveCacheState;
        struct InputCompressionHullCacheState;
        struct OutputBoundaryTableCacheState;

        std::once_flag _catalogOnce;
        AssetBridge _bridge;
        Spektrafilm::ProfileCatalog _spektrafilmProfileCatalog;
        std::unique_ptr<IlluminantFilterCurveCacheState>
            _illuminantFilterCurveCache;
        std::unique_ptr<InputCompressionHullCacheState>
            _inputCompressionHullCache;
        std::unique_ptr<OutputBoundaryTableCacheState>
            _outputBoundaryTableCache;
    };

} // namespace JuicerAssets
