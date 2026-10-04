#include "ResourceAssetLibrary.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <utility>

#include "Logging.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/JuicerCudaFailure.h"
#include "Illuminants.h"

namespace JuicerAssets {

    namespace {
        std::optional<std::vector<std::pair<float, float>>> copy_available_csv(AssetBridge& bridge, CsvSource source) {
            try {
                return bridge.copy_csv_pairs(source);
            } catch (const JuicerCuda::ExecutionFailure& failure) {
                if (failure.failure.status.category != FJ_STATUS_PREPARATION_FAILURE) {
                    throw;
                }
                JTRACE("ILLUM", failure.failure.diagnostic);
                return std::nullopt;
            }
        }

        IlluminantFilterCurveSet load_illuminant_filter_curves(AssetBridge& bridge) {
            IlluminantFilterCurveSet curves;
            const auto build = [&](CsvSource source, std::string_view label) {
                const auto pairs = copy_available_csv(bridge, source);
                return pairs ? Spectral::build_illuminant_curve(*pairs, label) : Spectral::Curve{};
            };
            curves.d65 = build(CsvSource::D65, "D65");
            curves.d55 = build(CsvSource::D55, "D55");
            curves.d50 = build(CsvSource::D50, "D50");
            curves.tungsten = build(CsvSource::T, "T");
            curves.kinoton75P = build(CsvSource::K75p, "K75P");
            auto kg3 = copy_available_csv(bridge, CsvSource::Kg3);
            if (kg3) {
                curves.tungstenKg3 = Spectral::build_tungsten_kg3_curve(*kg3, "KG3");
            } else {
                // The second consumer may make its ordinary acquisition after failure.
                kg3 = copy_available_csv(bridge, CsvSource::Kg3);
            }
            if (kg3) {
                auto input = Spectral::prepare_tungsten_kg3_lens_input(*kg3, "KG3");
                if (input) {
                    const auto lens = copy_available_csv(bridge, CsvSource::Canon24F28Is);
                    if (lens) {
                        curves.tungstenKg3Lens = Spectral::build_tungsten_kg3_lens_curve(std::move(*input), *lens, "Canon 24 F2.8 IS");
                    }
                }
            }
            return curves;
        }

        bool curve_is_on_reference_axis(const Spectral::Curve& curve) {
            const size_t expected = static_cast<size_t>(Spectral::gShape.K);
            return curve.lambda_nm.size() == expected && curve.linear.size() == expected;
        }

        bool illuminant_filter_curves_complete(const IlluminantFilterCurveSet& curves) {
            return curve_is_on_reference_axis(curves.d65) &&
                   curve_is_on_reference_axis(curves.d55) &&
                   curve_is_on_reference_axis(curves.d50) &&
                   curve_is_on_reference_axis(curves.tungsten) &&
                   curve_is_on_reference_axis(curves.kinoton75P) &&
                   curve_is_on_reference_axis(curves.tungstenKg3) &&
                   curve_is_on_reference_axis(curves.tungstenKg3Lens);
        }

    } // namespace

    struct Library::IlluminantFilterCurveCacheState {
        std::mutex mutex;
        std::shared_ptr<const IlluminantFilterCurveSet> curves;
    };

    struct Library::InputCompressionHullCacheState {
        std::mutex mutex;
        std::shared_ptr<const Gamut::InputCompressionHull> hull;
    };

    struct Library::OutputBoundaryTableCacheState {
        std::mutex mutex;
        std::array<std::shared_ptr<const Gamut::OutputBoundaryTable>,
                   OutputEncoding::kColorSpaceCount>
            tables;
    };

    Library::Library(const std::filesystem::path& resourceRoot)
        : _bridge(resourceRoot),
          _illuminantFilterCurveCache(
              std::make_unique<IlluminantFilterCurveCacheState>()),
          _inputCompressionHullCache(
              std::make_unique<InputCompressionHullCacheState>()),
          _outputBoundaryTableCache(
              std::make_unique<OutputBoundaryTableCacheState>()) {
    }

    Library::~Library() = default;

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
            result.filmProfile = _bridge.film(request.filmProfileKey);
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
        {
            std::lock_guard<std::mutex> lock(_illuminantFilterCurveCache->mutex);
            if (_illuminantFilterCurveCache->curves) {
                return _illuminantFilterCurveCache->curves;
            }
        }
        const auto candidate = std::make_shared<const IlluminantFilterCurveSet>(load_illuminant_filter_curves(_bridge));
        const bool complete = illuminant_filter_curves_complete(*candidate);
#if defined(JUICER_ILLUMINANT_TEST_HOOK)
        IlluminantTest::before_curve_publication(candidate);
#endif
        std::shared_ptr<const IlluminantFilterCurveSet> selected;
        {
            std::lock_guard<std::mutex> lock(_illuminantFilterCurveCache->mutex);
            if (!_illuminantFilterCurveCache->curves && complete) {
                _illuminantFilterCurveCache->curves = candidate;
            }
            selected = _illuminantFilterCurveCache->curves;
        }
        return selected ? selected : candidate;
    }

    std::shared_ptr<const Gamut::InputCompressionHull>
    Library::input_compression_hull() {
        {
            std::lock_guard<std::mutex> lock(_inputCompressionHullCache->mutex);
            if (_inputCompressionHullCache->hull) {
                return _inputCompressionHullCache->hull;
            }
        }
        const auto curves = illuminant_filter_curves();
        auto candidate = std::make_shared<Gamut::InputCompressionHull>();
        if (!Gamut::build_input_compression_hull(
                Spectral::gXBar,
                Spectral::gYBar,
                Spectral::gZBar,
                curves->d65,
                *candidate)) {
            return {};
        }
        std::lock_guard<std::mutex> lock(_inputCompressionHullCache->mutex);
        if (!_inputCompressionHullCache->hull) {
            _inputCompressionHullCache->hull = std::move(candidate);
        }
        return _inputCompressionHullCache->hull;
    }

    std::shared_ptr<const Gamut::OutputBoundaryTable>
    Library::output_boundary_table(
        const Gamut::OutputGamutTransform& transform,
        std::string& diagnostic) {
        diagnostic.clear();
        const int outputIndex =
            OutputEncoding::toIndex(transform.outputColorSpace);
        if (!transform.valid || transform.hash == 0 || outputIndex < 0 ||
            outputIndex >= static_cast<int>(OutputEncoding::kColorSpaceCount)) {
            diagnostic =
                "ResourceDescriptorMismatch component=output_boundary_cache field=transform";
            return {};
        }
        const std::size_t index = static_cast<std::size_t>(outputIndex);
        std::lock_guard<std::mutex> lock(_outputBoundaryTableCache->mutex);
        auto& cached = _outputBoundaryTableCache->tables[index];
        if (!cached) {
            auto candidate = std::make_shared<Gamut::OutputBoundaryTable>();
            if (!Gamut::build_output_boundary_table(transform, *candidate)) {
                diagnostic = candidate->diagnostic;
                return {};
            }
            cached = std::move(candidate);
        }
        if (cached->transformHash != transform.hash) {
            diagnostic =
                "ResourceDescriptorMismatch component=output_boundary_cache field=published_identity";
            return {};
        }
        if (cached->contractHash !=
            Gamut::output_boundary_contract_hash(transform)) {
            diagnostic =
                "ResourceDescriptorMismatch component=output_boundary_cache field=contract_identity";
            return {};
        }
        return cached;
    }

    NeutralPrintCalibrationResult Library::neutral_print_calibration(const std::string& printStock,
                                                                     const std::string& printIlluminantKey,
                                                                     const std::string& filmStock) {
        return _bridge.neutral_print_calibration(printStock, printIlluminantKey, filmStock);
    }

    FjStatus Library::release_cached_payloads(FjErrorBuffer* error) noexcept {
        FjStatus first{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        const auto cleanup = [&](auto operation) {
            try {
                operation();
            } catch (const std::bad_alloc&) {
                if (first.category == FJ_STATUS_SUCCESS) {
                    first = JuicerCuda::write_status({FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "host cache cleanup allocation failed", error);
                }
            } catch (const std::exception& detail) {
                if (first.category == FJ_STATUS_SUCCESS) {
                    first = JuicerCuda::write_status({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, detail.what(), error);
                }
            } catch (...) {
                if (first.category == FJ_STATUS_SUCCESS) {
                    first = JuicerCuda::write_status({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "host cache cleanup failed", error);
                }
            }
        };
        cleanup([&] {
            if (_illuminantFilterCurveCache) {
                std::shared_ptr<const IlluminantFilterCurveSet> detached;
                {
                    std::lock_guard<std::mutex> lock(_illuminantFilterCurveCache->mutex);
                    detached.swap(_illuminantFilterCurveCache->curves);
                }
            }
        });
        cleanup([&] {
            if (_outputBoundaryTableCache) {
                decltype(_outputBoundaryTableCache->tables) detached;
                {
                    std::lock_guard<std::mutex> lock(_outputBoundaryTableCache->mutex);
                    detached.swap(_outputBoundaryTableCache->tables);
                }
            }
        });
        const auto result = _bridge.release_cached_payloads(first.category == FJ_STATUS_SUCCESS ? error : nullptr);
        return first.category == FJ_STATUS_SUCCESS ? result : first;
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
