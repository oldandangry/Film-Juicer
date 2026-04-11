#include "ProcessRoot.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "JuicerState.h"
#include "Logging.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#endif

namespace JuicerProcess {

    namespace {

        inline int bool_to_i32(bool value) noexcept {
            return value ? 1 : 0;
        }

    } // namespace

    Root& Root::instance() noexcept {
        static Root root;
        return root;
    }

    Root& root() noexcept {
        return Root::instance();
    }

    void Root::ensure_bootstrap(BootstrapFn callback) {
        if (!callback) {
            return;
        }
        std::call_once(_bootstrapOnce, callback);
    }

    JuicerAssets::Library& Root::assets() noexcept {
        return _assets;
    }

    void Root::retire_idle_contexts(InstanceState& state) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        const bool traceInfo = JTRACE_ENABLED(1);
        std::vector<JuicerCuda::ResourceManager::DeviceContextKey> keys;
        try {
            std::lock_guard<std::mutex> lock(state.cudaMutex);
            keys.reserve(state.cudaByDevice.size());
            for (const auto& entry : state.cudaByDevice) {
                keys.emplace_back(entry.first);
            }
        } catch (...) {
            if (traceInfo) {
                JTRACE("MSLCY", "teardown_retire_idle_failed error=context_key_snapshot_failed");
            }
            return;
        }

        const JuicerCuda::ResourceManager::DeviceContextKey* keyData = keys.data();
        const size_t keyCount = keys.size();
        for (size_t i = 0; i < keyCount; ++i, ++keyData) {
            const auto& key = *keyData;
            std::string retireError;
            const bool retireOk = JuicerCuda::ResourceManager::command_retire_context_idle(key, retireError);
            if (!retireOk || !retireError.empty()) {
                if (traceInfo) {
                    const std::uintptr_t contextBits = reinterpret_cast<std::uintptr_t>(key.contextOpaque);
                    std::string msg;
                    msg.reserve(192);
                    msg = "teardown_retire_idle_failed device_id=";
                    msg += std::to_string(key.deviceId);
                    msg += " context=";
                    msg += std::to_string(contextBits);
                    msg += " accepted=";
                    msg += std::to_string(bool_to_i32(retireOk));
                    if (!retireError.empty()) {
                        msg += " error=";
                        msg += retireError;
                    }
                    JTRACE("MSLCY", msg);
                }
            }
        }
#else
        (void)state;
#endif
    }

} // namespace JuicerProcess
