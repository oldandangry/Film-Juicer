#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "ProfileCatalog.h"
#include "juicer_legacy_api.h"

#if defined(JUICER_CATALOG_TEST_HOOK)
namespace JuicerAssets::CatalogTest {
    void before_conversion(FjCatalogCounts& counts);
    void before_source_diagnostic();
} // namespace JuicerAssets::CatalogTest
#endif
#if defined(JUICER_PROFILE_CONVERSION_TEST_HOOK)
namespace JuicerAssets::ProfileTest {
    void before_catalog_publication();
}
#endif

namespace JuicerProcess {

    class ProcessRuntime final {
    public:
        ProcessRuntime() noexcept;
        ~ProcessRuntime() noexcept;
        ProcessRuntime(const ProcessRuntime&) = delete;
        ProcessRuntime& operator=(const ProcessRuntime&) = delete;
        ProcessRuntime(ProcessRuntime&& other) noexcept;
        ProcessRuntime& operator=(ProcessRuntime&& other) noexcept;

        void create(const std::filesystem::path& resourceRoot);

        FjRuntimeView borrow() const;
        FjStatus borrow(FjRuntimeView& out, FjErrorBuffer* error = nullptr) const noexcept;
        FjStatus check_close(FjErrorBuffer* error = nullptr) const noexcept;
        FjStatus shutdown(FjErrorBuffer* error = nullptr) noexcept;
        FjStatus close(FjErrorBuffer* error = nullptr) noexcept;

        const Spektrafilm::ProfileCatalog& spektrafilm_profile_catalog();
        FjStatus film_defaults(const std::string& key, FjFilmDigest& out, FjErrorBuffer& error) const noexcept;

    private:
        struct CatalogPresentation;
        void load_catalog();
        void trace_catalog(const Spektrafilm::ProfileCatalog& catalog) noexcept;

        FjRuntime* _runtime = nullptr;
        std::unique_ptr<CatalogPresentation> _catalog;
    };

} // namespace JuicerProcess
