#include "ProcessRoot.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "Illuminants.h"
#include "JuicerState.h"
#include "Logging.h"
#include "SpectralData.h"

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#endif

namespace JuicerProcess {

    namespace {

        inline int bool_to_i32(bool value) noexcept {
            return value ? 1 : 0;
        }

        std::string compute_process_data_dir() {
            namespace fs = std::filesystem;

#if defined(_WIN32)
            HMODULE module = nullptr;
            if (!GetModuleHandleExW(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCWSTR>(&compute_process_data_dir),
                    &module)) {
                return std::string();
            }

            std::wstring buffer(MAX_PATH, L'\0');
            DWORD length = 0;
            for (;;) {
                SetLastError(ERROR_SUCCESS);
                length = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
                if (length == 0) {
                    return std::string();
                }
                if (length < buffer.size()) {
                    buffer.resize(length);
                    break;
                }
                if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
                    buffer.resize(length);
                    break;
                }
                buffer.resize(buffer.size() * 2);
            }

            fs::path modulePath(buffer);
            fs::path moduleDir = modulePath.parent_path();
            if (moduleDir.empty()) {
                return std::string();
            }
            fs::path contentsDir = moduleDir.parent_path();
            if (contentsDir.empty()) {
                return std::string();
            }

            fs::path resourcesDir = (contentsDir / "Resources").lexically_normal();
            resourcesDir.make_preferred();
            std::wstring native = resourcesDir.native();
            if (!native.empty() && native.back() != L'\\') {
                native.push_back(L'\\');
            }

            if (native.empty()) {
                return std::string();
            }

            int required = WideCharToMultiByte(
                CP_UTF8,
                0,
                native.c_str(),
                static_cast<int>(native.size()),
                nullptr,
                0,
                nullptr,
                nullptr);
            if (required <= 0) {
                return std::string();
            }

            std::string path(static_cast<size_t>(required), '\0');
            WideCharToMultiByte(CP_UTF8, 0, native.c_str(), static_cast<int>(native.size()), path.data(), required, nullptr, nullptr);
            return path;
#else
            Dl_info info{};
            if (dladdr(reinterpret_cast<const void*>(&compute_process_data_dir), &info) == 0 || info.dli_fname == nullptr) {
                return std::string();
            }

            fs::path modulePath(info.dli_fname);
            fs::path moduleDir = modulePath.parent_path();
            if (moduleDir.empty()) {
                return std::string();
            }
            fs::path contentsDir = moduleDir.parent_path();
            if (contentsDir.empty()) {
                return std::string();
            }

            fs::path resourcesDir = (contentsDir / "Resources").lexically_normal();
            resourcesDir.make_preferred();
            std::string path = resourcesDir.u8string();
            if (!path.empty() && path.back() != '/') {
                path.push_back('/');
            }
            return path;
#endif
        }

        template <typename... Parts>
        std::string data_file_string(const std::string& dataDir, Parts&&... parts) {
            std::filesystem::path path(dataDir);
            ((path /= std::filesystem::path(std::forward<Parts>(parts))), ...);
            path.make_preferred();
            return path.string();
        }

        void load_spectral_globals(const std::string& dataDir) {
            Spectral::SpectralMutationScope mutationScope(
                Spectral::SpectralMutationStage::Bootstrap,
                "process_bootstrap");
            (void)mutationScope;

            try {
                Spectral::lock_shape_to_reference_axis();
                const auto cmf = Spectral::load_csv_triplets(data_file_string(dataDir, "cie1931_2deg.csv"));
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
                const std::string lutPath = data_file_string(
                    dataDir,
                    "luts",
                    "spectral_upsampling",
                    "irradiance_xy_tc.npy");
                Spectral::load_hanatos_spectra_lut(lutPath);
            } catch (...) {
                Spectral::set_hanatos_available(false);
            }

            try {
                const std::string basisPath = data_file_string(
                    dataDir,
                    "luts",
                    "spectral_upsampling",
                    "mallett2019_basis.npy");
                Spectral::load_mallett2019_basis_npy(basisPath);
            } catch (...) {
                Spectral::set_mallett_available(false);
            }

            std::vector<std::pair<float, float>> kg3Pairs;
            try {
                kg3Pairs = Spectral::load_csv_pairs(data_file_string(
                    dataDir,
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

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        bool resolve_cuda_frame_resources(
            InstanceState& instanceState,
            const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
            JuicerCuda::Resources*& outResources,
            std::string& outError) {
            std::lock_guard<std::mutex> lock(instanceState.cudaMutex);
            auto& slot = instanceState.cudaByDevice[deviceContextKey];
            if (!slot) {
                slot.reset(JuicerCuda::create());
                if (!slot) {
                    JTRACE("CUDA", "FATAL: failed to allocate CUDA resources");
                    outError = "failed to allocate CUDA resources";
                    return false;
                }
            }
            if (slot->deviceId < 0) {
                slot->deviceId = deviceContextKey.deviceId;
            }
            if (!slot->ownerContextOpaque) {
                slot->ownerContextOpaque = deviceContextKey.contextOpaque;
            }
            outResources = slot.get();
            if (!outResources) {
                outError = "CUDA resources missing after allocation";
                return false;
            }
            return true;
        }
#endif

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

#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
    struct Root::PreparedCudaFrame::State {
        Root* root = nullptr;
        JuicerCuda::Resources* resources = nullptr;
        JuicerCuda::ResourceManager::SubmissionTransaction transaction{};
        const char* failureStageTag = "prepare_frame";
        const char* failurePrefix = "CUDA prepared frame failed";

        void set_failure(const char* stageTag, const char* prefix) noexcept {
            failureStageTag = stageTag;
            failurePrefix = prefix;
        }
    };

    Root::PreparedCudaFrame::PreparedCudaFrame(std::unique_ptr<State> state) noexcept
        : _state(std::move(state)) {
    }

    Root::PreparedCudaFrame::~PreparedCudaFrame() {
        abort("prepared_frame_scope_exit");
    }

    Root::PreparedCudaFrame::PreparedCudaFrame(PreparedCudaFrame&& other) noexcept
        : _state(std::move(other._state)) {
    }

    Root::PreparedCudaFrame& Root::PreparedCudaFrame::operator=(PreparedCudaFrame&& other) noexcept {
        if (this != &other) {
            abort("prepared_frame_move_assignment");
            _state = std::move(other._state);
        }
        return *this;
    }

    bool Root::PreparedCudaFrame::active() const noexcept {
        return _state &&
               _state->resources &&
               _state->transaction.active &&
               !_state->transaction.committed;
    }

    bool Root::PreparedCudaFrame::finish(void* cudaStreamOpaque, std::string& outError) {
        outError.clear();
        if (!_state || !_state->root || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }
        return _state->root->commit_submission(_state->transaction, cudaStreamOpaque, outError);
    }

    void Root::PreparedCudaFrame::abort(const char* reason) noexcept {
        if (_state && _state->root && _state->transaction.active && !_state->transaction.committed) {
            _state->root->rollback_submission(_state->transaction, reason);
        }
    }

    bool Root::PreparedCudaFrame::prepare_current_medium(
        const WorkingState& workingState,
        bool negativeMedium,
        const JuicerCuda::ResourceManager::ScratchRequestDescriptor& scratchRequest,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        const char* stageTag = negativeMedium ? "command_ensure_current_medium_uploaded_negative"
                                              : "command_ensure_current_medium_uploaded_print";
        if (!JuicerCuda::ResourceManager::command_ensure_current_medium_uploaded(
                _state->transaction,
                *_state->resources,
                workingState,
                negativeMedium,
                scratchRequest,
                cudaStreamOpaque,
                outError)) {
            _state->set_failure(stageTag, "CUDA current-medium upload failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_scan_lut(
        const WorkingState& workingState,
        bool negativeMedium,
        const JuicerCuda::ResourceManager::ScratchRequestDescriptor& scratchRequest,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        const char* stageTag = negativeMedium ? "command_ensure_scan_lut_negative"
                                              : "command_ensure_scan_lut_print";
        const char* failurePrefix = negativeMedium ? "CUDA scan LUT upload failed"
                                                   : "CUDA print scan LUT upload failed";
        if (!JuicerCuda::ResourceManager::command_ensure_scan_lut(
                _state->transaction,
                *_state->resources,
                workingState,
                negativeMedium,
                scratchRequest,
                cudaStreamOpaque,
                outError)) {
            _state->set_failure(stageTag, failurePrefix);
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_optics_scratch(
        const JuicerCuda::ResourceManager::ScratchRequestDescriptor& scratchRequest,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        if (!JuicerCuda::ResourceManager::command_ensure_optics_scratch(
                _state->transaction,
                *_state->resources,
                scratchRequest,
                cudaStreamOpaque,
                outError)) {
            _state->set_failure(
                "command_ensure_optics_scratch",
                "CUDA optics scratch allocation failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_print_illuminant_filtered(
        const WorkingState& workingState,
        const Print::Runtime& printRuntime,
        const Print::Params& printParams,
        const JuicerCuda::ResourceManager::ScratchRequestDescriptor& scratchRequest,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        if (!JuicerCuda::ResourceManager::command_ensure_print_illuminant_filtered(
                _state->transaction,
                *_state->resources,
                workingState,
                printRuntime,
                printParams,
                scratchRequest,
                cudaStreamOpaque,
                outError)) {
            _state->set_failure(
                "command_ensure_print_illuminant_filtered",
                "CUDA print illuminant upload failed");
            return false;
        }
        return true;
    }

    JuicerCuda::Resources* Root::PreparedCudaFrame::resources() const noexcept {
        return _state ? _state->resources : nullptr;
    }

    JuicerCuda::ResourceManager::SubmissionTransaction& Root::PreparedCudaFrame::submission() noexcept {
        return _state->transaction;
    }

    const char* Root::PreparedCudaFrame::failure_stage_tag() const noexcept {
        return _state ? _state->failureStageTag : "prepare_frame";
    }

    const char* Root::PreparedCudaFrame::failure_prefix() const noexcept {
        return _state ? _state->failurePrefix : "CUDA prepared frame failed";
    }
#endif

    Root::ShutdownToken::ShutdownToken(Root* root) noexcept
        : _root(root) {
    }

    Root::ShutdownToken::~ShutdownToken() {
        reset();
    }

    Root::ShutdownToken::ShutdownToken(ShutdownToken&& other) noexcept
        : _root(std::exchange(other._root, nullptr)) {
    }

    Root::ShutdownToken& Root::ShutdownToken::operator=(ShutdownToken&& other) noexcept {
        if (this != &other) {
            reset();
            _root = std::exchange(other._root, nullptr);
        }
        return *this;
    }

    void Root::ShutdownToken::reset() noexcept {
        Root* root = _root;
        _root = nullptr;
        if (root) {
            root->finish_shutdown();
        }
    }

    Root& Root::instance() noexcept {
        static Root root;
        return root;
    }

    Root& root() noexcept {
        return Root::instance();
    }

    Root::Root()
        : _dataDir(compute_process_data_dir()), _assets(_dataDir) {
    }

    const std::string& Root::data_dir() const noexcept {
        return _dataDir;
    }

    void Root::ensure_bootstrap() {
        resume_frame_preparation();
        std::call_once(_bootstrapOnce, [this]() {
            load_spectral_globals(_dataDir);
        });
    }

    void Root::shutdown() noexcept {
        ShutdownToken shutdown = begin_shutdown();
        (void)shutdown;
        wait_for_frame_preparation();
        std::string retireError;
        if (!retire_known_contexts(retireError)) {
            set_shutdown_retire_blocked(true);
            if (JTRACE_ENABLED(1)) {
                std::string msg;
                msg.reserve(160);
                msg = "process_shutdown_retire_failed release_host_services=0";
                if (!retireError.empty()) {
                    msg += " error=";
                    msg += retireError;
                }
                JTRACE("MSLCY", msg);
            }
            return;
        }
        set_shutdown_retire_blocked(false);
        release_process_host_services();
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

    Root::PreparedCudaFrame Root::prepare_cuda_frame(
        InstanceState& instanceState,
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const WorkingState& workingState,
        const AutoExposureBufferRequest& autoExposureBufferRequest,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        std::unique_ptr<PreparedCudaFrame::State> state;
        try {
            state = std::make_unique<PreparedCudaFrame::State>();
        } catch (...) {
            outError = "failed to allocate CUDA prepared frame";
            return PreparedCudaFrame{};
        }

        PreparedCudaFrame frame(std::move(state));
        frame._state->root = this;
        frame._state->set_failure("prepare_frame", "CUDA prepared frame failed");
        if (!resolve_cuda_frame_resources(instanceState, deviceContextKey, frame._state->resources, outError)) {
            return frame;
        }
        if (!begin_submission(frame._state->transaction, snapshot, outError)) {
            frame._state->set_failure("begin_submission", "begin_submission failed");
            return frame;
        }
        if (!acquire_submission_plan(frame._state->transaction, outError)) {
            frame._state->set_failure("acquire_plan", "acquire_plan failed");
            frame.abort("prepared_frame_acquire_failed");
            return frame;
        }
        if (!JuicerCuda::ResourceManager::command_ensure_uploaded(
                frame._state->transaction,
                *frame._state->resources,
                workingState,
                cudaStreamOpaque,
                outError)) {
            frame._state->set_failure("command_ensure_uploaded", "CUDA WorkingState upload failed");
            frame.abort("prepared_frame_upload_failed");
            return frame;
        }
        if (!JuicerCuda::ResourceManager::command_ensure_scan_error_flag(
                frame._state->transaction,
                *frame._state->resources,
                cudaStreamOpaque,
                outError)) {
            frame._state->set_failure("command_ensure_scan_error_flag", "CUDA scan error flag allocation failed");
            frame.abort("prepared_frame_scan_error_flag_failed");
            return frame;
        }
        if (autoExposureBufferRequest.enabled &&
            !JuicerCuda::ResourceManager::command_ensure_auto_exposure_buffers(
                frame._state->transaction,
                *frame._state->resources,
                autoExposureBufferRequest.meterWidth,
                autoExposureBufferRequest.meterHeight,
                autoExposureBufferRequest.reusableKeyHash,
                cudaStreamOpaque,
                outError)) {
            frame._state->set_failure(
                "command_ensure_auto_exposure_buffers",
                "CUDA auto-exposure buffer allocation failed");
            frame.abort("prepared_frame_auto_exposure_failed");
            return frame;
        }
        return frame;
    }

    bool Root::begin_submission(
        JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        std::string& outError) {
        return JuicerCuda::ResourceManager::begin_submission(transaction, snapshot, outError);
    }

    bool Root::acquire_submission_plan(
        JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
        std::string& outError) {
        return JuicerCuda::ResourceManager::acquire_plan(transaction, outError);
    }

    bool Root::commit_submission(
        JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
        void* cudaStreamOpaque,
        std::string& outError) {
        return JuicerCuda::ResourceManager::commit_submission(transaction, cudaStreamOpaque, outError);
    }

    void Root::rollback_submission(
        JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
        const char* reason) noexcept {
        JuicerCuda::ResourceManager::rollback_submission(transaction, reason);
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

    bool Root::retire_known_contexts(std::string& outError) noexcept {
        outError.clear();
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        try {
            return JuicerCuda::ResourceManager::command_retire_all_contexts_idle(outError);
        } catch (...) {
            outError = "registry-wide context retire threw";
            return false;
        }
#else
        return true;
#endif
    }

    void Root::release_cuda_host_asset_caches() noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        JuicerCuda::purge_host_asset_caches_if_registry_idle("process_shutdown");
#endif
    }

    void Root::release_process_host_services() noexcept {
        release_working_state_cores();
        release_cuda_host_asset_caches();
        _assets.release_cached_payloads();
    }

    Root::ShutdownToken Root::begin_shutdown() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            _acceptFramePreparation = false;
            ++_activeShutdowns;
            if (_activeFramePreparations == 0) {
                _framePreparationCv.notify_all();
            }
            return ShutdownToken(this);
        } catch (...) {
            return ShutdownToken{};
        }
    }

    void Root::finish_shutdown() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            if (_activeShutdowns > 0) {
                --_activeShutdowns;
            }
            _framePreparationCv.notify_all();
        } catch (...) {
        }
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
            if (_activeShutdowns == 0 && !_shutdownRetireBlocked) {
                _acceptFramePreparation = true;
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

    void Root::set_shutdown_retire_blocked(bool blocked) noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            _shutdownRetireBlocked = blocked;
            if (blocked) {
                _acceptFramePreparation = false;
            }
        } catch (...) {
        }
    }

} // namespace JuicerProcess
