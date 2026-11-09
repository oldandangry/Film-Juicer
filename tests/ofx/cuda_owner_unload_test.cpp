#include <array>
#include <cstdio>
#include <exception>
#include <stdexcept>

#include "instance_host.h"
#include "module_test.h"

#include "ProcessRoot.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "juicer_cuda_api.h"
#include "juicer_cuda_owner.h"
#include "../ffi/juicer_test_api.h"
#include "../ffi/case_results.h"

namespace {

    void require(bool condition, const char* message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    FjRuntimeView loaded_runtime() {
        FjRuntimeView view{};
        view.cuda = fj_test_ofx_native();
        view.assets = static_cast<const FjAssets*>(fj_test_ofx_assets());
        require(view.cuda && view.assets, "loaded OFX Runtime borrow failed");
        return view;
    }

    void require_no_owner() {
        FjCuda* probe = nullptr;
        require(fj_cuda_create(&probe, nullptr).category == FJ_STATUS_SUCCESS && probe,
                "module unload retained an unexpected native registration");
        require(fj_cuda_destroy(probe, nullptr).category == FJ_STATUS_SUCCESS,
                "empty module unload probe could not close");
    }

} // namespace

int main(int argc, char** argv) try {
    constexpr std::array<std::string_view, 6> operations{
        "entry-factory", "noncreating-unload", "load-close-reload", "failed-unload-retention", "consumed-unload", "replacement-blocked"};
    if (JuicerTests::list_requested(argc, argv)) {
        for (const auto operation : operations)
            JuicerTests::list_case("ofx.owner_unload", operation);
        return 0;
    }
    JuicerTests::CaseLog case_log;
    try {
        JuicerTests::CaseEvent entry("ofx.owner_unload", operations[0]);
        require(OfxGetNumberOfPlugins() == 1, "expected one plugin");
        OfxPlugin* plugin = OfxGetPlugin(0);
        require(plugin && plugin->mainEntry, "plugin entry point is missing");
        ofx_test::InstanceHost host;
        entry.passed();

        // Exercise the actual standard module lifecycle and its terminal precheck.
        JuicerTests::CaseEvent empty_unload("ofx.owner_unload", operations[1]);
        require(plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr) == kOfxStatOK, "unbound unload failed");
        require_no_owner();
        empty_unload.passed();
        JuicerTests::CaseEvent reload("ofx.owner_unload", operations[2]);
        plugin->setHost(host.binding());
        require(plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr) == kOfxStatOK, "module load failed");
        require(plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr) == kOfxStatOK, "module unload failed");
        require_no_owner();
        require(plugin->mainEntry(kOfxActionLoad, nullptr, nullptr, nullptr) == kOfxStatOK, "module reload failed");
        reload.passed();

        // Exercise the real module unload while a native callback is active.
        // Its terminal precheck must retain the Runtime and its Assets.
        JuicerTests::CaseEvent failed_unload("ofx.owner_unload", operations[3]);
        {
            const FjRuntimeView activeView = loaded_runtime();
            JuicerCuda::NativeCall active(activeView.cuda);
            JuicerProcess::Root& activeRoot = JuicerCuda::borrow_root(activeView.cuda);
            require(plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr) == kOfxStatErrFatal,
                    "reentrant module unload did not reject terminal close");
            const FjRuntimeView stillLive = loaded_runtime();
            require(stillLive.assets == activeView.assets && stillLive.cuda == activeView.cuda,
                    "reentrant module unload invalidated the borrowed Runtime view");
            require(&JuicerCuda::borrow_root(stillLive.cuda) == &activeRoot,
                    "reentrant module unload replaced the bound native Root");
        }
        FjCuda* retainedCuda = nullptr;
        {
            const FjRuntimeView afterCallback = loaded_runtime();
            require(afterCallback.assets && afterCallback.cuda,
                    "rejected module unload did not retain its Runtime after callback return");
            retainedCuda = afterCallback.cuda;
        }

        // Metadata-only admission forces shutdown rejection before CUDA access.
        static int contextToken = 0;
        const JuicerCuda::ResourceManager::DeviceContextKey key{0, &contextToken};
        JuicerCuda::ResourceManager::RegistryContextSnapshot snapshot{};
        require(JuicerCuda::ResourceManager::registry_begin_submission(key, snapshot),
                "could not establish registry submission");
        require(plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr) == kOfxStatErrFatal,
                "failed native shutdown did not return fatal OFX unload status");
        require(!JuicerCuda::borrow_root(retainedCuda).begin_frame_preparation().active(),
                "failed module unload reopened native admission");
        require(!fj_test_ofx_assets() && !fj_test_ofx_native(),
                "failed legal module close did not consume its Runtime");
        failed_unload.passed();

        JuicerTests::CaseEvent consumed_unload("ofx.owner_unload", operations[4]);
        require(JuicerCuda::ResourceManager::registry_note_submission_end(key),
                "could not end registry submission");
        require(plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr) == kOfxStatOK, "consumed module unload failed");
        require(JuicerCuda::ResourceManager::registry_begin_owner_retire(key, snapshot),
                "repeated module unload retried context retirement");
        consumed_unload.passed();
        JuicerTests::CaseEvent replacement_blocked("ofx.owner_unload", operations[5]);
        FjCuda* replacement = nullptr;
        const FjStatus result = fj_cuda_create(&replacement, nullptr);
        require(result.category == FJ_STATUS_PREPARATION_FAILURE && !replacement,
                "failed module unload allowed a replacement owner");
        replacement_blocked.passed();
        case_log.finish();
        std::puts("PASS module unload: noncreating teardown, close/reload, fatal OFX status and consume-once retention");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL module unload: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fputs("FAIL module unload: unknown exception\n", stderr);
        return 1;
    }
} catch (const std::exception& error) {
    std::fprintf(stderr, "OFX owner test setup/output: %s\n", error.what());
    return 1;
} catch (...) {
    std::fputs("OFX owner test setup/output failed\n", stderr);
    return 1;
}
