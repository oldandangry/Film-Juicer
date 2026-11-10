#include <array>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "ProcessRoot.h"
#include "owner_fixture.h"
#include "native_path.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "juicer_cuda_api.h"
#include "juicer_cuda_owner.h"
#include "juicer_test_api.h"
#include "case_results.h"
#include "runtime_c.h"


namespace JuicerProcess::TestSupport {
    class RootLifetimeObserver final {
    public:
        static bool has_only_native_metadata(const Root& root) {
            return root._cudaContextResources.empty() && root._cudaDeviceLedgers.empty() && root._activeFramePreparations == 0;
        }
    };
} // namespace JuicerProcess::TestSupport

namespace {

    void require(bool condition, const char* message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    void require_empty(const FjRuntimeView& view) {
        require(!view.assets && !view.cuda, "empty Runtime borrow published a partial view");
    }

    void check_failed_close() {
        // Registry metadata rejects retirement before CUDA discovery. The Rust
        // Runtime still consumes its host ownership when native destroy fails.
        static int contextToken = 0;
        const JuicerCuda::ResourceManager::DeviceContextKey key{0, &contextToken};
        JuicerTestSupport::RuntimeFixture runtime;
        runtime.create("missing-blocked-runtime-resources");
        JuicerCuda::ResourceManager::RegistryContextSnapshot snapshot{};
        require(JuicerCuda::ResourceManager::registry_begin_submission(key, snapshot),
                "could not establish registry submission");
        const FjStatus shutdown = runtime.shutdown();
        require(shutdown.category == FJ_STATUS_PREPARATION_FAILURE && shutdown.api == FJ_API_NONE && shutdown.native_code == 0,
                "borrowed shutdown lost code-less registry failure");
        {
            const FjRuntimeView view = runtime.borrow();
            JuicerProcess::Root& root = JuicerCuda::borrow_root(view.cuda);
            FjRuntimeView liveView{};
            require(runtime.borrow(liveView).category == FJ_STATUS_SUCCESS &&
                        liveView.assets == view.assets && liveView.cuda == view.cuda,
                    "failed borrowed shutdown invalidated the Runtime view");
            require(&JuicerCuda::borrow_root(liveView.cuda) == &root,
                    "failed close deleted or unregistered the native graph");
            require(!root.begin_frame_preparation().active(),
                    "failed close reopened frame admission");
        }
        require(JuicerCuda::ResourceManager::registry_note_submission_end(key),
                "could not end registry submission");
        const FjStatus consumed = runtime.close();
        require(consumed.category == shutdown.category && consumed.api == shutdown.api && consumed.native_code == shutdown.native_code,
                "blocked destroy retried shutdown or lost its failure");
        require(fj_test_assets_live_owners() == 0 && fj_test_catalog_live_owners() == 0,
                "failed terminal close retained Rust ownership");
        require(runtime.close().category == FJ_STATUS_SUCCESS, "repeated consumed close failed");
        require(JuicerCuda::ResourceManager::registry_begin_owner_retire(key, snapshot),
                "consumed close retried context retirement");
        FjCuda* replacement = nullptr;
        const FjStatus result = fj_cuda_create(&replacement, nullptr);
        require(result.category == FJ_STATUS_PREPARATION_FAILURE && !replacement,
                "failed close allowed a replacement native graph");
    }

    FjStatus create_runtime(FjPathView path, FjRuntime** out, FjErrorBuffer* error) {
        return fj_test_runtime_create(path, out, error);
    }

} // namespace

static_assert(!std::is_copy_constructible_v<JuicerTestSupport::RuntimeFixture>);
static_assert(std::is_move_constructible_v<JuicerTestSupport::RuntimeFixture>);
static_assert(!std::is_constructible_v<JuicerProcess::Root>);
static_assert(!std::is_destructible_v<JuicerProcess::Root>);

int main(int argc, char** argv) {
    try {
        JuicerTests::CaseLog case_log;
        if (argc > 1 && std::string_view(argv[argc - 1]) == "--list-cases") {
            for (const char* id : {"noncreating_access", "invalid_creation", "metadata_only_create", "scoped_destroy", "failed_terminal_close"})
                JuicerTests::list_case("cuda_owner", id);
            return 0;
        }
        JuicerTests::CaseEvent access_event("cuda_owner", "noncreating_access");
        JuicerTestSupport::RuntimeFixture empty;
        FjRuntimeView emptyView{};
        std::array<char, 64> emptyDiagnostic{'s', 't', 'a', 'l', 'e'};
        FjErrorBuffer emptyError{emptyDiagnostic.data(), emptyDiagnostic.size(), 99};
        require(empty.borrow(emptyView, &emptyError).category == FJ_STATUS_PREPARATION_FAILURE,
                "empty ProcessRuntime admitted a borrow");
        require_empty(emptyView);
        require(std::string_view(emptyDiagnostic.data(), emptyError.length) == "process runtime is unavailable",
                "empty ProcessRuntime left a stale diagnostic");
        require(empty.close().category == FJ_STATUS_SUCCESS, "empty ProcessRuntime close failed");
        require(empty.close().category == FJ_STATUS_SUCCESS, "repeated empty close failed");
        access_event.passed();

        JuicerTests::CaseEvent invalid_event("cuda_owner", "invalid_creation");
        std::array<char, 8> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 99};
        FjRuntime* rejected = nullptr;
        JuicerTests::Cohort invalid_creation("cuda_owner.invalid_create_variant", {"null-path-data", "null-output", "embedded-path-nul"});
        auto invalid_path_event = invalid_creation.case_event("null-path-data");
        FjStatus result = create_runtime(FjPathView{nullptr, 1, FJ_PATH_UNIX_BYTES}, &rejected, &error);
        require(result.category == FJ_STATUS_UNSUPPORTED_INPUT && !rejected,
                "invalid Runtime path published an owner");
        require(error.length == diagnostic.size() - 1 && diagnostic.back() == '\0',
                "Runtime creation diagnostic did not truncate/terminate");
        invalid_path_event.passed();
        auto null_output_event = invalid_creation.case_event("null-output");
        result = create_runtime(JuicerAssets::NativePathArgument(std::filesystem::path("path")).view(), nullptr, nullptr);
        require(result.category == FJ_STATUS_UNSUPPORTED_INPUT, "null Runtime output accepted");
        null_output_event.passed();
        auto embedded_nul_event = invalid_creation.case_event("embedded-path-nul");
        const std::array<char, 3> embeddedNul{{'a', '\0', 'b'}};
        result = create_runtime(JuicerAssets::NativePathArgument(std::filesystem::path(std::string(embeddedNul.data(), embeddedNul.size()))).view(), &rejected, nullptr);
        require(result.category == FJ_STATUS_UNSUPPORTED_INPUT && !rejected,
                "embedded NUL accepted in Runtime resource path");
        embedded_nul_event.passed();
        invalid_event.passed();

        JuicerTests::CaseEvent metadata_event("cuda_owner", "metadata_only_create");
        result = fj_test_runtime_c_consumer(
            JuicerAssets::NativePathArgument(std::filesystem::path("missing-c-runtime-consumer-resources")).view());
        require(result.category == FJ_STATUS_SUCCESS && result.api == FJ_API_NONE && result.native_code == 0,
                "C11 Runtime create/borrow/check/shutdown/destroy consumer failed");
        FjCuda* native = nullptr;
        result = fj_cuda_create(&native, &error);
        require(result.category == FJ_STATUS_SUCCESS && native,
                "metadata-only native create failed");
        require(fj_cuda_destroy(native, nullptr).category == FJ_STATUS_SUCCESS,
                "metadata-only native destroy failed");
        native = nullptr;
        require(fj_cuda_create(&native, nullptr).category == FJ_STATUS_SUCCESS && native,
                "metadata-only native recreation failed");
        FjErrorBuffer malformedDestroy{nullptr, 1, 99};
        require(fj_cuda_destroy(native, &malformedDestroy).category == FJ_STATUS_UNSUPPORTED_INPUT && malformedDestroy.length == 0,
                "malformed native destroy did not consume ownership with the preserved diagnostic status");
        const std::array<std::string, 2> directories{"missing-first-runtime-resources", "missing-reloaded-runtime-resources"};
        JuicerTests::Cohort create_variants("cuda_owner.metadata_create_variant", {directories[0], directories[1]});
        for (const auto& directory : directories) {
            auto variant_event = create_variants.case_event(directory);
            JuicerTestSupport::RuntimeFixture runtime;
            runtime.create(directory);
            {
                const FjRuntimeView view = runtime.borrow();
                require(view.assets && view.cuda, "Runtime borrow did not return both owners");
                JuicerProcess::Root& root = JuicerCuda::borrow_root(view.cuda);
                require(JuicerProcess::TestSupport::RootLifetimeObserver::has_only_native_metadata(root),
                        "Runtime creation acquired native context/resources before rendering");
                require(root.begin_frame_preparation().active(),
                        "metadata-only Runtime creation did not admit frame preparation");
                FjCuda* duplicate = nullptr;
                result = fj_cuda_create(&duplicate, &error);
                require(result.category == FJ_STATUS_PREPARATION_FAILURE && !duplicate,
                        "duplicate native registration was accepted");
                FjRuntime* duplicateRuntime = nullptr;
                const std::filesystem::path duplicateDirectory("missing-duplicate-runtime-resources");
                const JuicerAssets::NativePathArgument duplicatePath(duplicateDirectory);
                result = fj_test_runtime_create(duplicatePath.view(), &duplicateRuntime, &error);
                require(result.category == FJ_STATUS_PREPARATION_FAILURE && !duplicateRuntime && fj_test_assets_live_owners() == 1,
                        "duplicate Runtime creation published an owner or retained partial Assets");
                FjRuntimeView repeatedView{};
                require(runtime.borrow(repeatedView).category == FJ_STATUS_SUCCESS &&
                            repeatedView.assets == view.assets && repeatedView.cuda == view.cuda,
                        "duplicate create replaced or leaked the Runtime");
                require(&JuicerCuda::borrow_root(repeatedView.cuda) == &root,
                        "duplicate create replaced the bound Root");
            }
            require(runtime.close().category == FJ_STATUS_SUCCESS, "metadata-only Runtime failed to close");
            require(runtime.close().category == FJ_STATUS_SUCCESS, "consumed Runtime close was not harmless");
            variant_event.passed();
        }
        metadata_event.passed();
        JuicerTests::CaseEvent scoped_event("cuda_owner", "scoped_destroy");
        {
            JuicerTestSupport::RuntimeFixture runtime;
            runtime.create("missing-scoped-runtime-resources");
            require(runtime.borrow().assets && runtime.borrow().cuda, "scoped Runtime borrow failed");
        }
        require(fj_test_assets_live_owners() == 0, "scoped Runtime destruction retained Rust Assets");
        scoped_event.passed();
        JuicerTests::CaseEvent close_event("cuda_owner", "failed_terminal_close");
        check_failed_close();
        close_event.passed();
        std::puts("PASS CUDA owner: noncreating Runtime access/teardown, invalid/runtime and metadata-only creation, duplicate registration, shared view, reload and failed close");
        case_log.finish();
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL CUDA owner: %s\n", error.what());
        return 1;
    }
}
