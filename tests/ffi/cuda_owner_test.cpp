#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "ProcessRoot.h"
#include "RustAssetBridge.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "juicer_cuda_api.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "case_results.h"

namespace JuicerProcess::TestSupport {

    class RootLifetimeObserver final {
    public:
        static bool has_only_host_metadata(const Root& root) {
            return root._assets && root._cudaContextResources.empty() &&
                   root._cudaDeviceLedgers.empty() && root._activeFramePreparations == 0;
        }
    };

} // namespace JuicerProcess::TestSupport

namespace {

    void require(bool condition, const char* message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    void require_no_owner() {
        try {
            (void)JuicerProcess::root();
        } catch (const std::logic_error&) {
            return;
        }
        throw std::runtime_error("root accessor constructed or retained an unexpected owner");
    }

    void check_failed_close() {
        // Only registry metadata is used. An active submission rejects retirement
        // before CUDA discovery; the retained graph lasts until process exit.
        static int contextToken = 0;
        const JuicerCuda::ResourceManager::DeviceContextKey key{0, &contextToken};
        JuicerProcess::Root* retained = nullptr;
        {
            JuicerCuda::Owner owner;
            owner.create("missing-blocked-runtime-resources");
            retained = &JuicerProcess::root();
            JuicerCuda::ResourceManager::RegistryContextSnapshot snapshot{};
            require(JuicerCuda::ResourceManager::registry_begin_submission(key, snapshot),
                    "could not establish registry submission");
            const auto failed = fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
            require(failed.category == FJ_STATUS_PREPARATION_FAILURE && failed.api == FJ_API_NONE && failed.native_code == 0,
                    "borrowed shutdown lost code-less registry failure");
            require(&JuicerProcess::root() == retained,
                    "failed close deleted or unregistered the owner");
            require(!retained->begin_frame_preparation().active(),
                    "failed close reopened frame admission");
            require(JuicerCuda::ResourceManager::registry_note_submission_end(key),
                    "could not end registry submission");
            const auto consumed = owner.close();
            require(consumed.category == failed.category && consumed.api == failed.api && consumed.native_code == failed.native_code,
                    "blocked destroy retried shutdown or lost its failure");
            require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0, "failed terminal close retained Rust ownership");
            require(owner.close().category == FJ_STATUS_SUCCESS, "repeated consumed close failed");
        }
        require(&JuicerProcess::root() == retained &&
                    !retained->begin_frame_preparation().active(),
                "destruction retried the consumed owner or reopened admission");
        JuicerCuda::ResourceManager::RegistryContextSnapshot snapshot{};
        require(JuicerCuda::ResourceManager::registry_begin_owner_retire(key, snapshot),
                "consumed close or destruction retried context retirement");
        FjCuda* replacement = nullptr;
        const FjStatus result = fj_cuda_create(JuicerAssets::NativePathArgument(std::filesystem::path("other")).view(), &replacement, nullptr);
        require(result.category == FJ_STATUS_PREPARATION_FAILURE && !replacement,
                "failed close allowed a replacement owner");
    }

} // namespace

static_assert(!std::is_constructible_v<JuicerProcess::Root, std::filesystem::path>);
static_assert(!std::is_destructible_v<JuicerProcess::Root>);
static_assert(!std::is_copy_constructible_v<JuicerCuda::Owner>);

int main(int argc, char** argv) {
    try {
        JuicerTests::CaseLog case_log;
        if (argc > 1 && std::string_view(argv[argc - 1]) == "--list-cases") {
            for (const char* id : {"noncreating_access", "invalid_creation", "metadata_only_create", "scoped_destroy", "failed_terminal_close"})
                JuicerTests::list_case("cuda_owner", id);
            return 0;
        }
        JuicerTests::CaseEvent access_event("cuda_owner", "noncreating_access");
        require_no_owner();
        fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
        {
            JuicerCuda::Owner empty;
            require(empty.close().category == FJ_STATUS_SUCCESS, "empty close failed");
        }
        require_no_owner();
        access_event.passed();

        JuicerTests::CaseEvent invalid_event("cuda_owner", "invalid_creation");
        std::array<char, 8> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 99};
        FjCuda* rejected = nullptr;
        JuicerTests::Cohort invalid_creation("cuda_owner.invalid_create_variant", {"null-path-data", "null-output", "embedded-path-nul"});
        auto invalid_path_event = invalid_creation.case_event("null-path-data");
        FjStatus result = fj_cuda_create(FjPathView{nullptr, 1, FJ_PATH_UNIX_BYTES}, &rejected, &error);
        require(result.category == FJ_STATUS_UNSUPPORTED_INPUT && !rejected,
                "invalid creation published an owner");
        require(error.length == diagnostic.size() - 1 && diagnostic.back() == '\0',
                "creation diagnostic did not truncate/terminate");
        require_no_owner();
        invalid_path_event.passed();
        auto null_output_event = invalid_creation.case_event("null-output");
        result = fj_cuda_create(JuicerAssets::NativePathArgument(std::filesystem::path("path")).view(), nullptr, nullptr);
        require(result.category == FJ_STATUS_UNSUPPORTED_INPUT, "null output accepted");
        null_output_event.passed();
        auto embedded_nul_event = invalid_creation.case_event("embedded-path-nul");
        const std::array<char, 3> embeddedNul{{'a', '\0', 'b'}};
        result = fj_cuda_create(JuicerAssets::NativePathArgument(std::filesystem::path(std::string(embeddedNul.data(), embeddedNul.size()))).view(), &rejected, nullptr);
        require(result.category == FJ_STATUS_UNSUPPORTED_INPUT && !rejected,
                "embedded NUL accepted in data directory");
        require_no_owner();
        embedded_nul_event.passed();
        invalid_event.passed();

        JuicerTests::CaseEvent metadata_event("cuda_owner", "metadata_only_create");
        const std::array<std::string, 2> directories{"missing-first-runtime-resources", "missing-reloaded-runtime-resources"};
        JuicerTests::Cohort create_variants("cuda_owner.metadata_create_variant", {directories[0], directories[1]});
        for (const auto& directory : directories) {
            auto variant_event = create_variants.case_event(directory);
            JuicerCuda::Owner owner;
            owner.create(directory);
            auto& root = JuicerProcess::root();
            require(&root == &JuicerProcess::Root::instance(), "root accessors disagree");
            require(JuicerProcess::TestSupport::RootLifetimeObserver::has_only_host_metadata(root),
                    "creation discovered CUDA resources or failed to copy directory metadata");
            result = fj_cuda_create(JuicerAssets::NativePathArgument(std::filesystem::path("other")).view(), &rejected, &error);
            require(result.category == FJ_STATUS_PREPARATION_FAILURE && !rejected,
                    "duplicate runtime was accepted");
            require(&root == &JuicerProcess::root(), "duplicate creation replaced the owner");
            require(owner.close().category == FJ_STATUS_SUCCESS, "metadata-only owner failed to close");
            require_no_owner();
            require(owner.close().category == FJ_STATUS_SUCCESS, "consumed owner close was not harmless");
            variant_event.passed();
        }
        metadata_event.passed();
        JuicerTests::CaseEvent scoped_event("cuda_owner", "scoped_destroy");
        {
            JuicerCuda::Owner owner;
            owner.create("missing-scoped-runtime-resources");
        }
        require_no_owner();
        scoped_event.passed();
        JuicerTests::CaseEvent close_event("cuda_owner", "failed_terminal_close");
        check_failed_close();
        close_event.passed();
        std::puts("PASS CUDA owner: noncreating access/teardown, failed/duplicate create, metadata-only creation, shared borrowing, close/reload, scoped destruction and consume-once failed close");
        case_log.finish();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL CUDA owner: %s\n", error.what());
        return 1;
    }
}
