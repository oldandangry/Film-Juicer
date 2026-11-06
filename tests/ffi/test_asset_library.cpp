#include "test_asset_library.h"

#include <filesystem>
#include <memory>
#include <sstream>

#include "Logging.h"

namespace JuicerAssets {
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
