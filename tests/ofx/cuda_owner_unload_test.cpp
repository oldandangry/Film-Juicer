#include <array>
#include <cstdio>
#include <exception>
#include <stdexcept>

#include "ofxsImageEffect.h"

#include "ProcessRoot.h"
#include "RustAssetBridge.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "juicer_cuda_api.h"
#include "../ffi/case_results.h"

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
        throw std::runtime_error("factory unload retained an unexpected owner");
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
        OFX::PluginFactoryArray factories;
        OFX::Plugin::getPluginIDs(factories);
        require(factories.size() == 1, "expected one factory");
        OFX::PluginFactory& factory = *factories.front();
        entry.passed();

        // Load directly to exercise the real owner without installing host suites.
        // The failed unload uses the real entry point and its exception mapping.
        JuicerTests::CaseEvent empty_unload("ofx.owner_unload", operations[1]);
        factory.unload();
        require_no_owner();
        empty_unload.passed();
        JuicerTests::CaseEvent reload("ofx.owner_unload", operations[2]);
        factory.load();
        factory.unload();
        require_no_owner();
        factory.load();
        auto& retained = JuicerProcess::root();
        reload.passed();

        // Metadata-only admission forces shutdown rejection before CUDA access.
        JuicerTests::CaseEvent failed_unload("ofx.owner_unload", operations[3]);
        static int contextToken = 0;
        const JuicerCuda::ResourceManager::DeviceContextKey key{0, &contextToken};
        JuicerCuda::ResourceManager::RegistryContextSnapshot snapshot{};
        require(JuicerCuda::ResourceManager::registry_begin_submission(key, snapshot),
                "could not establish registry submission");
        require(plugin->mainEntry(kOfxActionUnload, nullptr, nullptr, nullptr) == kOfxStatErrFatal,
                "failed native shutdown did not return fatal OFX unload status");
        require(&JuicerProcess::root() == &retained &&
                    !retained.begin_frame_preparation().active(),
                "failed factory unload lost the graph or reopened admission");
        failed_unload.passed();

        JuicerTests::CaseEvent consumed_unload("ofx.owner_unload", operations[4]);
        require(JuicerCuda::ResourceManager::registry_note_submission_end(key),
                "could not end registry submission");
        factory.unload();
        require(&JuicerProcess::root() == &retained,
                "factory retried its consumed native owner");
        require(JuicerCuda::ResourceManager::registry_begin_owner_retire(key, snapshot),
                "repeated factory unload retried context retirement");
        consumed_unload.passed();
        JuicerTests::CaseEvent replacement_blocked("ofx.owner_unload", operations[5]);
        FjCuda* replacement = nullptr;
        const FjStatus result = fj_cuda_create(JuicerAssets::NativePathArgument(std::filesystem::path("other")).view(), &replacement, nullptr);
        require(result.category == FJ_STATUS_PREPARATION_FAILURE && !replacement,
                "failed factory unload allowed a replacement owner");
        replacement_blocked.passed();
        case_log.finish();
        std::puts("PASS factory unload: noncreating teardown, close/reload, fatal OFX status and consume-once retention");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL factory unload: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fputs("FAIL factory unload: unknown exception\n", stderr);
        return 1;
    }
} catch (const std::exception& error) {
    std::fprintf(stderr, "OFX owner test setup/output: %s\n", error.what());
    return 1;
} catch (...) {
    std::fputs("OFX owner test setup/output failed\n", stderr);
    return 1;
}
