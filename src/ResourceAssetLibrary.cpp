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
#include "Illuminants.h"
#include "RustGamutBridge.h"

namespace JuicerAssets {
#if defined(JUICER_OUTPUT_GAMUT_TEST_HOOK)
    namespace OutputGamutTest {
        void enter_lock() noexcept;
        void leave_lock() noexcept;
        void observe_generation(std::uint64_t& generation) noexcept;
        void after_miss(std::uint64_t generation);
        void after_selection(bool warm, std::uint64_t generation, const std::shared_ptr<const Gamut::OutputBoundaryTable>& hold);
        void before_illuminant_clear();
        void after_clear(bool exhausted, std::uint64_t generation) noexcept;
        struct LockObservation {
            LockObservation() noexcept {
                enter_lock();
            }
            ~LockObservation() {
                leave_lock();
            }
        };
    } // namespace OutputGamutTest
#endif


    namespace {
        std::optional<CsvRows> available_csv_rows(AssetBridge& bridge, CsvSource source) {
            try {
                return bridge.csv_rows(source);
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
                const auto pairs = available_csv_rows(bridge, source);
                return pairs ? JuicerIlluminant::from_samples(*pairs, label) : Spectral::Curve{};
            };
            curves.d65 = build(CsvSource::D65, "D65");
            curves.d55 = build(CsvSource::D55, "D55");
            curves.d50 = build(CsvSource::D50, "D50");
            curves.tungsten = build(CsvSource::T, "T");
            curves.kinoton75P = build(CsvSource::K75p, "K75P");
            auto kg3 = available_csv_rows(bridge, CsvSource::Kg3);
            if (kg3) {
                curves.tungstenKg3 = JuicerIlluminant::tungsten_kg3(*kg3, "KG3");
            } else {
                // The second consumer may make its ordinary acquisition after failure.
                auto retry = available_csv_rows(bridge, CsvSource::Kg3);
                if (retry) {
                    kg3.emplace(std::move(*retry));
                }
            }
            if (kg3) {
                auto input = JuicerIlluminant::prepare_lens(*kg3, "KG3");
                if (input) {
                    const auto lens = available_csv_rows(bridge, CsvSource::Canon24F28Is);
                    if (lens) {
                        curves.tungstenKg3Lens = JuicerIlluminant::finish_lens(std::move(*input), *lens, "Canon 24 F2.8 IS");
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

    // FJ_TEMP_BRIDGE: sole native derived illuminant cache; remove S4.E.
    struct Library::IlluminantFilterCurveCacheState {
        std::mutex mutex;
        std::shared_ptr<const IlluminantFilterCurveSet> curves;
    };

    struct Library::InputCompressionHullCacheState {
        std::mutex mutex;
        std::shared_ptr<const Gamut::InputCompressionHull> hull;
    };

    // FJ_TEMP_BRIDGE: sole native output boundary cache; remove S4.E.
    struct Library::OutputBoundaryTableCacheState {
        std::mutex mutex;
        std::uint64_t generation = 0;
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
            result.filmSource = _bridge.film_source(request.filmProfileKey);
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
        if (!JuicerGamut::input_hull(
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
        if (!transform.valid || transform.hash == 0 || transform.contractHash == 0 || outputIndex < 0 ||
            outputIndex >= static_cast<int>(OutputEncoding::kColorSpaceCount)) {
            diagnostic =
                "ResourceDescriptorMismatch component=output_boundary_cache field=transform";
            return {};
        }
        const std::size_t index = static_cast<std::size_t>(outputIndex);
        const auto expectedTransform = transform.hash;
        const auto expectedContract = transform.contractHash;
        const auto identities_match = [&](const std::shared_ptr<const Gamut::OutputBoundaryTable>& hold) {
            if (hold->transformHash != expectedTransform) {
                diagnostic = "ResourceDescriptorMismatch component=output_boundary_cache field=published_identity";
                return false;
            }
            if (hold->contractHash != expectedContract) {
                diagnostic = "ResourceDescriptorMismatch component=output_boundary_cache field=contract_identity";
                return false;
            }
            return true;
        };
        std::shared_ptr<const Gamut::OutputBoundaryTable> selected;
        std::uint64_t generation = 0;
        {
            std::lock_guard<std::mutex> lock(_outputBoundaryTableCache->mutex);
#if defined(JUICER_OUTPUT_GAMUT_TEST_HOOK)
            OutputGamutTest::LockObservation observed;
            OutputGamutTest::observe_generation(_outputBoundaryTableCache->generation);
#endif
            selected = _outputBoundaryTableCache->tables[index];
            generation = _outputBoundaryTableCache->generation;
        }
        if (selected) {
#if defined(JUICER_OUTPUT_GAMUT_TEST_HOOK)
            OutputGamutTest::after_selection(true, generation, selected);
#endif
            if (!identities_match(selected))
                return {};
            return selected;
        }
#if defined(JUICER_OUTPUT_GAMUT_TEST_HOOK)
        OutputGamutTest::after_miss(generation);
#endif
        auto candidate = std::make_shared<Gamut::OutputBoundaryTable>();
        if (!JuicerGamut::output_boundary(transform, *candidate)) {
            diagnostic = candidate->diagnostic;
            return {};
        }
        bool stale = false;
        {
            std::lock_guard<std::mutex> lock(_outputBoundaryTableCache->mutex);
#if defined(JUICER_OUTPUT_GAMUT_TEST_HOOK)
            OutputGamutTest::LockObservation observed;
#endif
            stale = generation != _outputBoundaryTableCache->generation;
            if (!stale) {
                auto& cached = _outputBoundaryTableCache->tables[index];
                if (!cached)
                    cached = candidate;
                selected = cached;
            }
        }
        // Clear changes publication eligibility, never a completed private result.
        // Once selected, a hold remains authoritative even across a later clear.
        if (stale)
            return candidate;
#if defined(JUICER_OUTPUT_GAMUT_TEST_HOOK)
        OutputGamutTest::after_selection(false, generation, selected);
#endif
        if (!identities_match(selected))
            return {};
        return selected;
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
#if defined(JUICER_OUTPUT_GAMUT_TEST_HOOK)
                OutputGamutTest::before_illuminant_clear();
#endif
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
                bool exhausted = false;
#if defined(JUICER_OUTPUT_GAMUT_TEST_HOOK)
                std::uint64_t generation = 0;
#endif
                {
                    std::lock_guard<std::mutex> lock(_outputBoundaryTableCache->mutex);
#if defined(JUICER_OUTPUT_GAMUT_TEST_HOOK)
                    OutputGamutTest::LockObservation observed;
#endif
                    exhausted = _outputBoundaryTableCache->generation == std::numeric_limits<std::uint64_t>::max();
                    if (!exhausted) {
                        ++_outputBoundaryTableCache->generation;
                        detached.swap(_outputBoundaryTableCache->tables);
                    }
#if defined(JUICER_OUTPUT_GAMUT_TEST_HOOK)
                    generation = _outputBoundaryTableCache->generation;
#endif
                }
#if defined(JUICER_OUTPUT_GAMUT_TEST_HOOK)
                OutputGamutTest::after_clear(exhausted, generation);
#endif
                if (exhausted && first.category == FJ_STATUS_SUCCESS) {
                    first = JuicerCuda::write_status({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "output boundary cache generation exhausted", error);
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
