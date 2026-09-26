#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "ProcessRoot.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "juicer_cuda_api.h"
#include "juicer_cuda_owner.h"

namespace JuicerProcess::TestSupport {

    class RootLifetimeObserver final {
    public:
        static bool has_only_host_metadata(const Root& root, const std::string& directory) {
            return root._dataDir == directory && root._cudaContextResources.empty() &&
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
            require(!owner.close(), "close reported success with an active submission");
            require(&JuicerProcess::root() == retained,
                    "failed close deleted or unregistered the owner");
            require(!retained->begin_frame_preparation().active(),
                    "failed close reopened frame admission");
            require(JuicerCuda::ResourceManager::registry_note_submission_end(key),
                    "could not end registry submission");
            require(owner.close(), "repeated consumed close failed");
        }
        require(&JuicerProcess::root() == retained &&
                    !retained->begin_frame_preparation().active(),
                "destruction retried the consumed owner or reopened admission");
        JuicerCuda::ResourceManager::RegistryContextSnapshot snapshot{};
        require(JuicerCuda::ResourceManager::registry_begin_owner_retire(key, snapshot),
                "consumed close or destruction retried context retirement");
        FjCuda* replacement = nullptr;
        const FjStatus result = fj_cuda_create(FjStringView{"other", 5}, &replacement, nullptr);
        require(result.category == FJ_STATUS_PREPARATION_FAILURE && !replacement,
                "failed close allowed a replacement owner");
    }

} // namespace

static_assert(!std::is_constructible_v<JuicerProcess::Root, std::string>);
static_assert(!std::is_destructible_v<JuicerProcess::Root>);
static_assert(!std::is_copy_constructible_v<JuicerCuda::Owner>);

int main() {
    try {
        require_no_owner();
        JuicerProcess::shutdown_if_initialized();
        {
            JuicerCuda::Owner empty;
            require(empty.close(), "empty close failed");
        }
        require_no_owner();

        std::array<char, 8> diagnostic{};
        FjErrorBuffer error{diagnostic.data(), diagnostic.size(), 99};
        FjCuda* rejected = nullptr;
        FjStatus result = fj_cuda_create(FjStringView{nullptr, 1}, &rejected, &error);
        require(result.category == FJ_STATUS_UNSUPPORTED_INPUT && !rejected,
                "invalid creation published an owner");
        require(error.length == diagnostic.size() - 1 && diagnostic.back() == '\0',
                "creation diagnostic did not truncate/terminate");
        require_no_owner();
        result = fj_cuda_create(FjStringView{"path", 4}, nullptr, nullptr);
        require(result.category == FJ_STATUS_UNSUPPORTED_INPUT, "null output accepted");
        const std::array<char, 3> embeddedNul{{'a', '\0', 'b'}};
        result = fj_cuda_create(FjStringView{embeddedNul.data(), embeddedNul.size()}, &rejected, nullptr);
        require(result.category == FJ_STATUS_UNSUPPORTED_INPUT && !rejected,
                "embedded NUL accepted in data directory");
        require_no_owner();

        for (const std::string directory : {"missing-first-runtime-resources", "missing-reloaded-runtime-resources"}) {
            JuicerCuda::Owner owner;
            owner.create(directory);
            auto& root = JuicerProcess::root();
            require(&root == &JuicerProcess::Root::instance(), "root accessors disagree");
            require(JuicerProcess::TestSupport::RootLifetimeObserver::has_only_host_metadata(root, directory),
                    "creation discovered CUDA resources or failed to copy directory metadata");
            result = fj_cuda_create(FjStringView{"other", 5}, &rejected, &error);
            require(result.category == FJ_STATUS_PREPARATION_FAILURE && !rejected,
                    "duplicate runtime was accepted");
            require(&root == &JuicerProcess::root(), "duplicate creation replaced the owner");
            require(owner.close(), "metadata-only owner failed to close");
            require_no_owner();
            require(owner.close(), "consumed owner close was not harmless");
        }
        {
            JuicerCuda::Owner owner;
            owner.create("missing-scoped-runtime-resources");
        }
        require_no_owner();
        check_failed_close();
        std::puts("PASS CUDA owner: noncreating access/teardown, failed/duplicate create, metadata-only creation, shared borrowing, close/reload, scoped destruction and consume-once failed close");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL CUDA owner: %s\n", error.what());
        return 1;
    }
}
