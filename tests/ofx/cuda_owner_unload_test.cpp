#include <cstdio>
#include <exception>
#include <stdexcept>

#include "ofxsImageEffect.h"

#include "ProcessRoot.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "juicer_cuda_api.h"

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

int main() {
    try {
        require(OfxGetNumberOfPlugins() == 1, "expected one plugin");
        OfxPlugin* plugin = OfxGetPlugin(0);
        require(plugin && plugin->mainEntry, "plugin entry point is missing");
        OFX::PluginFactoryArray factories;
        OFX::Plugin::getPluginIDs(factories);
        require(factories.size() == 1, "expected one factory");
        OFX::PluginFactory& factory = *factories.front();

        // Load directly to exercise the real owner without installing host suites.
        // The failed unload uses the real entry point and its exception mapping.
        factory.unload();
        require_no_owner();
        factory.load();
        factory.unload();
        require_no_owner();
        factory.load();
        auto& retained = JuicerProcess::root();

        // Metadata-only admission forces shutdown rejection before CUDA access.
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

        require(JuicerCuda::ResourceManager::registry_note_submission_end(key),
                "could not end registry submission");
        factory.unload();
        require(&JuicerProcess::root() == &retained,
                "factory retried its consumed native owner");
        require(JuicerCuda::ResourceManager::registry_begin_owner_retire(key, snapshot),
                "repeated factory unload retried context retirement");
        FjCuda* replacement = nullptr;
        const FjStatus result = fj_cuda_create(FjStringView{"other", 5}, &replacement, nullptr);
        require(result.category == FJ_STATUS_PREPARATION_FAILURE && !replacement,
                "failed factory unload allowed a replacement owner");
        std::puts("PASS factory unload: noncreating teardown, close/reload, fatal OFX status and consume-once retention");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL factory unload: %s\n", error.what());
        return 1;
    } catch (...) {
        std::fputs("FAIL factory unload: unknown exception\n", stderr);
        return 1;
    }
}
