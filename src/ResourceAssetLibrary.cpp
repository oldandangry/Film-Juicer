#include "ResourceAssetLibrary.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <limits>
#include <utility>

#include "Logging.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/JuicerCudaFailure.h"

namespace JuicerAssets {
#if defined(JUICER_FILM_RECIPE_STATE_TEST_HOOK)
    namespace RecipeStateTest {
        void selected_source(std::shared_ptr<const FilmSource>& source);
    }
#endif
    Library::Library(const std::filesystem::path& resourceRoot) : _bridge(resourceRoot) {}

    Library::~Library() = default;

    FjAssets* Library::borrow_rust_assets() const noexcept {
        return _bridge.borrow_assets();
    }

    FjStatus Library::close(FjErrorBuffer* error) noexcept {
        return _bridge.close(error);
    }

    void Library::ensure_catalogs() {
        std::call_once(_catalogOnce, [this]() {
            load_catalogs();
        });
    }


    void Library::load_catalogs() {
        _spektrafilmProfileCatalog = _bridge.load_catalog();
        // Optional trace formatting must not reopen a completed once-publication
        // or replace its retained catalog owner after a diagnostic allocation.
        try {
            if (!JTRACE_ENABLED(1)) {
                return;
            }
            if (!_spektrafilmProfileCatalog.valid) {
                JTRACE("CATALOG", "spektrafilm profile catalog unavailable: " + _spektrafilmProfileCatalog.failure);
                return;
            }
            std::ostringstream oss;
            oss << "spektrafilm profile catalog film=" << _spektrafilmProfileCatalog.filmProfiles.size()
                << " print=" << _spektrafilmProfileCatalog.printProfiles.size()
                << " defaultFilm=" << (_spektrafilmProfileCatalog.defaultFilmPresent ? 1 : 0)
                << " defaultPrint=" << (_spektrafilmProfileCatalog.defaultPrintPresent ? 1 : 0);
            JTRACE("CATALOG", oss.str());
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }


    const Spektrafilm::ProfileCatalog& Library::spektrafilm_profile_catalog() {
        ensure_catalogs();
        return _spektrafilmProfileCatalog;
    }

    FjStatus Library::film_defaults(
        const std::string& key,
        FjFilmDigest& out,
        FjErrorBuffer& error) {
        return _bridge.film_defaults(key, out, error);
    }

    std::shared_ptr<const Profiles::FilmProfile>
    Library::selected_film_profile_for_key(const std::string& key) {
        ensure_catalogs();
        try {
            return _bridge.film(key);
        } catch (const JuicerCuda::ExecutionFailure& failure) {
            if (failure.failure.status.category != FJ_STATUS_PREPARATION_FAILURE) {
                throw;
            }
            JTRACE("PROFILE", failure.failure.diagnostic);
            return {};
        }
    }

    SelectedProfileResult Library::selected_profiles_for_route(
        const SelectedProfileRequest& request) {
        ensure_catalogs();
        SelectedProfileResult result;
        try {
            result.filmSource = _bridge.film_source(request.filmProfileKey);
#if defined(JUICER_FILM_RECIPE_STATE_TEST_HOOK)
            RecipeStateTest::selected_source(result.filmSource);
#endif
            if (Spektrafilm::scan_route_is_print(request.scanRoute)) {
                result.printSource = _bridge.print(request.printProfileKey);
            }
            result.valid = true;
        } catch (const JuicerCuda::ExecutionFailure& failure) {
            if (failure.failure.status.category != FJ_STATUS_PREPARATION_FAILURE) {
                throw;
            }
            result.diagnostic = failure.failure.diagnostic;
        }
        return result;
    }

    NoiseSource Library::noise() {
        return _bridge.noise();
    }

    std::shared_ptr<const IlluminantFilterCurveSet> Library::illuminant_filter_curves() {
        return _bridge.illuminants();
    }

    std::shared_ptr<const Gamut::InputHullHold> Library::input_compression_hull() {
        return _bridge.input_hull();
    }

    std::shared_ptr<const Gamut::OutputBoundaryHold> Library::output_boundary_table(
        const Gamut::OutputGamutTransform& transform, std::string& diagnostic) {
        return _bridge.output_boundary(transform, diagnostic);
    }

    NeutralPrintCalibrationResult Library::neutral_print_calibration(const std::string& printStock,
                                                                     const std::string& printIlluminantKey,
                                                                     const std::string& filmStock) {
        return _bridge.neutral_print_calibration(printStock, printIlluminantKey, filmStock);
    }

    FjStatus Library::release_cached_payloads(FjErrorBuffer* error) noexcept {
        return _bridge.release_cached_payloads(error);
    }

} // namespace JuicerAssets

namespace JuicerAssets {
    Spectral::ReconstructionLut Library::copy_hanatos_lut() {
        return _bridge.copy_hanatos_lut();
    }
    Spectral::ReconstructionLut Library::copy_arctic_lut() {
        return _bridge.copy_arctic_lut();
    }
    Spectral::MallettBasis Library::copy_mallett_basis() {
        return _bridge.copy_mallett_basis();
    }
    Spectral::CMFTriplets Library::copy_cmf_triplets() {
        return _bridge.copy_cmf_triplets();
    }
} // namespace JuicerAssets
