#include "ProcessRuntime.h"

#include <array>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <string_view>
#include <utility>

#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/JuicerCudaFailure.h"
#include "Logging.h"
#include "NativePath.h"

namespace JuicerProcess {
    struct ProcessRuntime::CatalogPresentation {
        std::once_flag once;
        Spektrafilm::ProfileCatalog catalog;
    };

    namespace {
        FjStatus internal_failure() noexcept {
            return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        }

        void report_cleanup(FjStatus result) noexcept {
            if (result.category != FJ_STATUS_SUCCESS) {
                try {
                    JTRACE("RUNTIME", "process runtime cleanup failed category=" + std::to_string(result.category));
                } catch (...) {
                    JuicerLogging::discard_current_exception();
                }
            }
        }

        struct CatalogDeleter {
            void operator()(FjCatalog* owner) const noexcept {
                report_cleanup(fj_legacy_catalog_release(owner, nullptr));
            }
        };

        struct FilmProfileDeleter {
            void operator()(FjFilmProfile* owner) const noexcept {
                report_cleanup(fj_legacy_film_profile_release(owner, nullptr));
            }
        };

        [[noreturn]] void fail(FjStatus status, std::string_view message) {
            if (status.category == FJ_STATUS_ALLOCATION_FAILURE) {
                throw std::bad_alloc();
            }
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, message);
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }

        std::string copy_text(FjStringView view) {
            if (view.count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) ||
                (view.count == 0 ? view.data != nullptr : !view.data)) {
                fail(internal_failure(), "invalid Rust catalog text view");
            }
            if (view.count > std::string().max_size()) {
                fail({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "profile text exceeds native representation");
            }
            return view.count == 0 ? std::string() : std::string(view.data, view.count);
        }
    } // namespace

    ProcessRuntime::ProcessRuntime() noexcept = default;

    ProcessRuntime::~ProcessRuntime() noexcept {
        if (_runtime) {
            // Destruction is only legal after host callback exclusion. A rejected
            // precheck leaves the opaque owner retained; this destructor never retries.
            report_cleanup(close());
        }
    }

    ProcessRuntime::ProcessRuntime(ProcessRuntime&& other) noexcept
        : _runtime(std::exchange(other._runtime, nullptr)), _catalog(std::move(other._catalog)) {}

    ProcessRuntime& ProcessRuntime::operator=(ProcessRuntime&& other) noexcept {
        if (this != &other) {
            if (_runtime) {
                const FjStatus result = close();
                if (result.category != FJ_STATUS_SUCCESS) {
                    report_cleanup(result);
                    return *this;
                }
            }
            _runtime = std::exchange(other._runtime, nullptr);
            _catalog = std::move(other._catalog);
        }
        return *this;
    }

    void ProcessRuntime::create(const std::filesystem::path& resourceRoot) {
        if (_runtime) {
            fail({FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, "process runtime is already initialized");
        }
        JuicerAssets::NativePathArgument path(resourceRoot);
        auto catalog = std::make_unique<CatalogPresentation>();
        std::array<char, 1024> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        FjRuntime* runtime = nullptr;
        const FjStatus status = fj_legacy_runtime_create(path.view(), &runtime, &error);
        if (status.category != FJ_STATUS_SUCCESS) {
            fail(status, {message.data(), error.length});
        }
        if (!runtime) {
            fail(internal_failure(), "Rust runtime creation returned no owner");
        }
        _runtime = runtime;
        _catalog = std::move(catalog);
    }

    FjRuntimeView ProcessRuntime::borrow() const {
        std::array<char, 1024> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        FjRuntimeView view{};
        const FjStatus status = borrow(view, &error);
        if (status.category != FJ_STATUS_SUCCESS) {
            fail(status, {message.data(), error.length});
        }
        return view;
    }

    FjStatus ProcessRuntime::borrow(FjRuntimeView& out, FjErrorBuffer* error) const noexcept {
        out = {};
        if (!_runtime) {
            return JuicerCuda::write_status({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0},
                                            "process runtime is unavailable",
                                            error);
        }
        return fj_legacy_runtime_borrow(_runtime, &out, error);
    }

    FjStatus ProcessRuntime::check_close(FjErrorBuffer* error) const noexcept {
        if (!_runtime) {
            return JuicerCuda::write_status({FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}, error);
        }
        return fj_legacy_runtime_check_close(_runtime, error);
    }

    FjStatus ProcessRuntime::shutdown(FjErrorBuffer* error) noexcept {
        if (!_runtime) {
            return JuicerCuda::write_status({FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}, error);
        }
        return fj_legacy_runtime_shutdown(_runtime, error);
    }

    FjStatus ProcessRuntime::close(FjErrorBuffer* error) noexcept {
        if (!_runtime) {
            return JuicerCuda::write_status({FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}, error);
        }
        std::array<char, 1024> message{};
        FjErrorBuffer precheck{message.data(), message.size(), 0};
        const FjStatus checked = fj_legacy_runtime_check_close(_runtime, &precheck);
        if (checked.category != FJ_STATUS_SUCCESS) {
            return JuicerCuda::write_status(checked, {message.data(), precheck.length}, error);
        }
        FjRuntime* runtime = std::exchange(_runtime, nullptr);
        const FjStatus result = fj_legacy_runtime_destroy(runtime, error);
        _catalog.reset();
        return result;
    }

    const Spektrafilm::ProfileCatalog& ProcessRuntime::spektrafilm_profile_catalog() {
        if (!_runtime || !_catalog) {
            fail({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "process runtime is unavailable for profile catalog access");
        }
        std::call_once(_catalog->once, [this]() {
            load_catalog();
        });
        return _catalog->catalog;
    }

    void ProcessRuntime::load_catalog() {
        FjRuntimeView runtimeView{};
        std::array<char, 4096> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        const FjStatus borrowed = borrow(runtimeView, &error);
        if (borrowed.category != FJ_STATUS_SUCCESS || !runtimeView.assets) {
            fail(borrowed.category == FJ_STATUS_SUCCESS ? internal_failure() : borrowed,
                 {message.data(), error.length});
        }
        FjCatalog* acquired = nullptr;
        FjCatalogCounts counts{};
        const FjStatus result = fj_legacy_catalog_acquire(runtimeView.assets, &acquired, &counts, &error);
        std::unique_ptr<FjCatalog, CatalogDeleter> owner(acquired);
        Spektrafilm::ProfileCatalog candidate;
        if (result.category == FJ_STATUS_PREPARATION_FAILURE) {
#if defined(JUICER_CATALOG_TEST_HOOK)
            JuicerAssets::CatalogTest::before_source_diagnostic();
#endif
            candidate.failure.assign(message.data(), error.length);
            _catalog->catalog = std::move(candidate);
            trace_catalog(_catalog->catalog);
            return;
        }
        if (result.category != FJ_STATUS_SUCCESS || !owner) {
            fail(result.category == FJ_STATUS_SUCCESS ? internal_failure() : result,
                 {message.data(), error.length});
        }
#if defined(JUICER_CATALOG_TEST_HOOK)
        JuicerAssets::CatalogTest::before_conversion(counts);
#endif
        if (counts.film_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            counts.print_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            fail({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "catalog count exceeds OFX option index width");
        }
        const auto copy_entries = [&](std::uint32_t role, std::vector<Spektrafilm::ProfileCatalogEntry>& entries, std::size_t count) {
            entries.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                FjCatalogEntryView view{};
                const FjStatus entryResult = fj_legacy_catalog_entry(owner.get(), role, i, &view, &error);
                if (entryResult.category != FJ_STATUS_SUCCESS) {
                    fail(entryResult, {message.data(), error.length});
                }
                Spektrafilm::ProfilePolarity polarity;
                switch (view.polarity) {
                    case FJ_POLARITY_NEGATIVE:
                        polarity = Spektrafilm::ProfilePolarity::Negative;
                        break;
                    case FJ_POLARITY_POSITIVE:
                        polarity = Spektrafilm::ProfilePolarity::Positive;
                        break;
                    default:
                        fail(internal_failure(), "invalid Rust catalog polarity");
                }
                entries.push_back({copy_text(view.key), copy_text(view.label), polarity});
            }
        };
        copy_entries(FJ_PROFILE_ROLE_FILM, candidate.filmProfiles, counts.film_count);
        copy_entries(FJ_PROFILE_ROLE_PRINT, candidate.printProfiles, counts.print_count);
        candidate.valid = true;
        candidate.defaultFilmPresent = true;
        candidate.defaultPrintPresent = true;
#if defined(JUICER_PROFILE_CONVERSION_TEST_HOOK)
        JuicerAssets::ProfileTest::before_catalog_publication();
#endif
        const FjStatus released = fj_legacy_catalog_release(owner.release(), &error);
        if (released.category != FJ_STATUS_SUCCESS) {
            fail(released, {message.data(), error.length});
        }
        _catalog->catalog = std::move(candidate);
        trace_catalog(_catalog->catalog);
    }

    void ProcessRuntime::trace_catalog(const Spektrafilm::ProfileCatalog& catalog) noexcept {
        try {
            if (!JTRACE_ENABLED(1)) {
                return;
            }
            std::ostringstream trace;
            trace << "spektrafilm profile catalog film=" << catalog.filmProfiles.size()
                  << " print=" << catalog.printProfiles.size()
                  << " defaultFilm=" << (catalog.defaultFilmPresent ? 1 : 0)
                  << " defaultPrint=" << (catalog.defaultPrintPresent ? 1 : 0);
            JTRACE("CATALOG", trace.str());
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    FjStatus ProcessRuntime::film_defaults(const std::string& key, FjFilmDigest& out, FjErrorBuffer& error) const noexcept {
        out = {};
        FjRuntimeView view{};
        const FjStatus borrowed = borrow(view, &error);
        if (borrowed.category != FJ_STATUS_SUCCESS) {
            return borrowed;
        }
        FjFilmProfile* rawOwner = nullptr;
        FjStatus status = fj_legacy_film_profile_acquire(
            view.assets,
            {key.empty() ? nullptr : key.data(), key.size()},
            &rawOwner,
            &error);
        std::unique_ptr<FjFilmProfile, FilmProfileDeleter> owner(rawOwner);
        if (status.category != FJ_STATUS_SUCCESS) {
            return status;
        }
        if (!owner) {
            return internal_failure();
        }
        FjFilmProfileView profile{};
        status = fj_legacy_film_profile_view(owner.get(), &profile, &error);
        if (status.category != FJ_STATUS_SUCCESS) {
            return status;
        }
        const FjFilmDigest defaults = profile.digest;
        FjFilmProfile* released = owner.release();
        status = fj_legacy_film_profile_release(released, &error);
        if (status.category == FJ_STATUS_SUCCESS) {
            out = defaults;
        }
        return status;
    }
} // namespace JuicerProcess
