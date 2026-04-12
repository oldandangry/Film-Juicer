#include "ProcessRoot.h"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Illuminants.h"
#include "JuicerState.h"
#include "Logging.h"
#include "SpectralData.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#endif

namespace JuicerProcess {

    namespace {

        inline int bool_to_i32(bool value) noexcept {
            return value ? 1 : 0;
        }

        void load_spectral_globals() {
            Spectral::SpectralMutationScope mutationScope(
                Spectral::SpectralMutationStage::Bootstrap,
                "process_bootstrap");
            (void)mutationScope;

            try {
                Spectral::lock_shape_to_reference_axis();
                const auto cmf = Spectral::load_csv_triplets(data_dir_string("cie1931_2deg.csv"));
                if (!Spectral::cmf_triplets_match_reference_axis(cmf)) {
                    JTRACE("INIT", "FATAL: CMF wavelengths do not match 380-780@5nm grid");
                    throw std::runtime_error("CMF grid mismatch");
                }
                Spectral::set_cie_1931_2deg_cmf(cmf.xbar, cmf.ybar, cmf.zbar);
                Spectral::ensure_precomputed_up_to_date();
                Spectral::disable_hanatos_if_reference_mismatch();
            } catch (...) {
            }

            try {
                const std::string lutPath = data_dir_string(
                    "luts",
                    "spectral_upsampling",
                    "irradiance_xy_tc.npy");
                Spectral::load_hanatos_spectra_lut(lutPath);
            } catch (...) {
                Spectral::set_hanatos_available(false);
            }

            try {
                const std::string basisPath = data_dir_string(
                    "luts",
                    "spectral_upsampling",
                    "mallett2019_basis.npy");
                Spectral::load_mallett2019_basis_npy(basisPath);
            } catch (...) {
                Spectral::set_mallett_available(false);
            }

            std::vector<std::pair<float, float>> kg3Pairs;
            try {
                kg3Pairs = Spectral::load_csv_pairs(data_dir_string(
                    "filters",
                    "heat_absorbing",
                    "schott",
                    "KG3.csv"));
            } catch (...) {
                kg3Pairs.clear();
            }
            if (kg3Pairs.empty()) {
                kg3Pairs = {
                    {Spectral::gShape.lambdaMin, 1.0f},
                    {Spectral::gShape.lambdaMax, 1.0f}};
            }
            Spectral::set_filter_KG3_from_pairs(kg3Pairs);
        }

    } // namespace

    Root::FramePreparationToken::FramePreparationToken(Root* root) noexcept
        : _root(root) {
    }

    Root::FramePreparationToken::~FramePreparationToken() {
        reset();
    }

    Root::FramePreparationToken::FramePreparationToken(FramePreparationToken&& other) noexcept
        : _root(std::exchange(other._root, nullptr)) {
    }

    Root::FramePreparationToken& Root::FramePreparationToken::operator=(FramePreparationToken&& other) noexcept {
        if (this != &other) {
            reset();
            _root = std::exchange(other._root, nullptr);
        }
        return *this;
    }

    bool Root::FramePreparationToken::active() const noexcept {
        return _root != nullptr;
    }

    void Root::FramePreparationToken::reset() noexcept {
        Root* root = _root;
        _root = nullptr;
        if (root) {
            root->finish_frame_preparation();
        }
    }

    Root& Root::instance() noexcept {
        static Root root;
        return root;
    }

    Root& root() noexcept {
        return Root::instance();
    }

    void Root::ensure_bootstrap() {
        resume_frame_preparation();
        std::call_once(_bootstrapOnce, load_spectral_globals);
    }

    void Root::shutdown() noexcept {
        stop_frame_preparation();
        wait_for_frame_preparation();
        retire_known_contexts();
        release_working_state_cores();
        _assets.release_cached_payloads();
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            _shutdownActive = false;
            _framePreparationCv.notify_all();
        } catch (...) {
        }
    }

    JuicerAssets::Library& Root::assets() noexcept {
        return _assets;
    }

    Root::FramePreparationToken Root::begin_frame_preparation() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            if (!_acceptFramePreparation) {
                return FramePreparationToken{};
            }
            ++_activeFramePreparations;
            return FramePreparationToken(this);
        } catch (...) {
            return FramePreparationToken{};
        }
    }

    bool Root::retire_idle_context(int deviceId, void* contextOpaque, std::string& outError) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        JuicerCuda::ResourceManager::DeviceContextKey key{};
        key.deviceId = deviceId;
        key.contextOpaque = contextOpaque;
        return JuicerCuda::ResourceManager::command_retire_context_idle(key, outError);
#else
        (void)deviceId;
        (void)contextOpaque;
        outError.clear();
        return true;
#endif
    }

    bool Root::retire_reset_context(int deviceId, void* contextOpaque, std::string& outError) noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        JuicerCuda::ResourceManager::DeviceContextKey key{};
        key.deviceId = deviceId;
        key.contextOpaque = contextOpaque;
        return JuicerCuda::ResourceManager::command_retire_context_reset(key, outError);
#else
        (void)deviceId;
        (void)contextOpaque;
        outError.clear();
        return true;
#endif
    }

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    void Root::destroy_cuda_resources(JuicerCuda::Resources* resources) noexcept {
        JuicerCuda::destroy(resources);
    }
#endif

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
            const bool retireOk = retire_idle_context(key.deviceId, key.contextOpaque, retireError);
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

    void Root::retire_known_contexts() noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        std::string retireError;
        (void)JuicerCuda::ResourceManager::command_retire_all_contexts_idle(retireError);
#endif
    }

    void Root::finish_frame_preparation() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            if (_activeFramePreparations > 0) {
                --_activeFramePreparations;
            }
            if (_activeFramePreparations == 0) {
                _framePreparationCv.notify_all();
            }
        } catch (...) {
        }
    }

    void Root::resume_frame_preparation() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            if (!_shutdownActive) {
                _acceptFramePreparation = true;
            }
        } catch (...) {
        }
    }

    void Root::stop_frame_preparation() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            _acceptFramePreparation = false;
            _shutdownActive = true;
            if (_activeFramePreparations == 0) {
                _framePreparationCv.notify_all();
            }
        } catch (...) {
        }
    }

    void Root::wait_for_frame_preparation() noexcept {
        try {
            std::unique_lock<std::mutex> lock(_framePreparationMutex);
            _framePreparationCv.wait(lock, [this]() {
                return _activeFramePreparations == 0;
            });
        } catch (...) {
        }
    }

} // namespace JuicerProcess
