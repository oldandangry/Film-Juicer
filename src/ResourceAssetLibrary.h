#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <span>
#include <vector>

#include "GamutCompression.h"
#include "ProfileAssets.h"
#include "ProfileCatalog.h"
#include "ScanRoute.h"
#include "RustAssetBridge.h"
#include "SpectralData.h"

namespace JuicerAssets {

    struct IlluminantCurveView {
        std::span<const float> lambda_nm;
        std::span<const float> linear;
        // Existing native scalar/illuminant transports still require an owning
        // curve. This explicit copy retires with those S4.E integration callers.
        Spectral::Curve copy_curve() const;
    };

    class IlluminantFilterCurveSet final {
    public:
        ~IlluminantFilterCurveSet();
        IlluminantFilterCurveSet(const IlluminantFilterCurveSet&) = delete;
        IlluminantFilterCurveSet& operator=(const IlluminantFilterCurveSet&) = delete;
        const IlluminantCurveView d65, d55, d50, tungsten, kinoton75P, tungstenKg3, tungstenKg3Lens;
        const std::uintptr_t entryIdentity;

    private:
        friend class AssetBridge;
        IlluminantFilterCurveSet(FjIlluminantsOwner*, const FjIlluminantsView&) noexcept;
        FjIlluminantsOwner* _owner;
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
        std::shared_ptr<const Gamut::InputHullHold>
        input_compression_hull();
        std::shared_ptr<const Gamut::OutputBoundaryHold>
        output_boundary_table(
            const Gamut::OutputGamutTransform& transform,
            std::string& diagnostic);
        NeutralPrintCalibrationResult neutral_print_calibration(
            const std::string& printStock,
            const std::string& printIlluminantKey,
            const std::string& filmStock);
        FjStatus release_cached_payloads(FjErrorBuffer* error = nullptr) noexcept;
        FjStatus close(FjErrorBuffer* error = nullptr) noexcept;
        FjAssets* borrow_rust_assets() const noexcept;

    private:
        void ensure_catalogs();
        void load_catalogs();

        std::once_flag _catalogOnce;
        AssetBridge _bridge;
        Spektrafilm::ProfileCatalog _spektrafilmProfileCatalog;
    };

} // namespace JuicerAssets
