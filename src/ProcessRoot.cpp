#include "ProcessRoot.h"

#include <atomic>
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
#include <cuda_runtime.h>
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#endif

namespace JuicerProcess {

    namespace {

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
        Root::CudaResourceOwner resourceOwner;
        Root::CudaResourceOwner grainStaticOwner;
        JuicerCuda::Resources* resources = nullptr;
        JuicerCuda::Resources* grainStaticResources = nullptr;
        JuicerCuda::ResourceManager::SubmissionTransaction transaction{};
        const char* failureStageTag = "prepare_frame";
        const char* failurePrefix = "CUDA prepared frame failed";
        bool failureMarksContextLoss = true;

        void set_failure(const char* stageTag, const char* prefix, bool marksContextLoss = true) noexcept {
            failureStageTag = stageTag;
            failurePrefix = prefix;
            failureMarksContextLoss = marksContextLoss;
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

    Root::PreparedCudaFrame::WorkspaceLeaseMarker::WorkspaceLeaseMarker(
        const WorkspaceRequest& request,
        std::uint64_t leaseGeneration) noexcept
        : _request(request)
        , _leaseGeneration(leaseGeneration)
        , _active(leaseGeneration != 0) {
    }

    bool Root::PreparedCudaFrame::WorkspaceLeaseMarker::active() const noexcept {
        return _active;
    }

    bool Root::PreparedCudaFrame::WorkspaceLeaseMarker::has_any_family() const noexcept {
        return _active && (_request.needOptics || _request.needSpatialDir);
    }

    std::uint64_t Root::PreparedCudaFrame::WorkspaceLeaseMarker::lease_generation() const noexcept {
        return _active ? _leaseGeneration : 0;
    }

    bool Root::PreparedCudaFrame::active() const noexcept {
        return _state &&
               _state->resources &&
               _state->transaction.active &&
               !_state->transaction.committed;
    }

    Root::PreparedCudaFrame::WorkspaceLeaseMarker Root::PreparedCudaFrame::bind_workspace_request(
        const WorkspaceRequest& request) const noexcept {
        if (!active()) {
            return WorkspaceLeaseMarker{};
        }
        return WorkspaceLeaseMarker(request, _state->transaction.leaseGeneration);
    }

    JuicerCuda::ResourceManager::ScratchRequestDescriptor Root::PreparedCudaFrame::make_scratch_request_descriptor(
        const WorkspaceLeaseMarker& workspace) noexcept {
        return JuicerCuda::ResourceManager::make_scratch_request_descriptor(
            workspace._request.needOptics,
            workspace._request.needSpatialDir,
            workspace._request.requestedWidth,
            workspace._request.requestedHeight,
            workspace._request.needBlurred,
            workspace._request.needAux,
            workspace._request.needGrainTriplet,
            workspace._request.needGrainShared,
            workspace._request.needGateMask);
    }

    bool Root::PreparedCudaFrame::workspace_marker_matches_current_frame(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        return _state &&
               _state->resources &&
               _state->transaction.active &&
               !_state->transaction.committed &&
               workspace.active() &&
               workspace.lease_generation() == _state->transaction.leaseGeneration;
    }

    bool Root::PreparedCudaFrame::validate_workspace_lease_marker(
        const WorkspaceLeaseMarker& workspace,
        std::string& outError) const {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }
        if (!workspace.active()) {
            outError = "prepared frame workspace marker is not active";
            return false;
        }
        if (workspace.lease_generation() != _state->transaction.leaseGeneration) {
            outError = "prepared frame workspace marker does not match current lease";
            return false;
        }
        return true;
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
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }

        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_scratch_request_descriptor(workspace);
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
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }

        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_scratch_request_descriptor(workspace);
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
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }

        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_scratch_request_descriptor(workspace);
        if (!JuicerCuda::ResourceManager::command_ensure_optics_scratch(
                _state->transaction,
                *_state->resources,
                scratchRequest,
                cudaStreamOpaque,
                outError)) {
            const bool marksContextLoss = !JuicerCuda::ResourceManager::error_is_scratch_exhausted(outError);
            _state->set_failure(
                "command_ensure_optics_scratch",
                marksContextLoss
                    ? "CUDA optics scratch allocation failed"
                    : "CUDA optics scratch deferred by contention policy",
                marksContextLoss);
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_spatial_dir_scratch(
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }

        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_scratch_request_descriptor(workspace);
        if (!JuicerCuda::ResourceManager::command_ensure_spatial_dir_scratch(
                _state->transaction,
                *_state->resources,
                scratchRequest,
                cudaStreamOpaque,
                outError)) {
            const bool marksContextLoss = !JuicerCuda::ResourceManager::error_is_scratch_exhausted(outError);
            _state->set_failure(
                "command_ensure_spatial_dir_scratch",
                marksContextLoss
                    ? "CUDA spatial DIR scratch allocation failed"
                    : "CUDA spatial DIR scratch deferred by contention policy",
                marksContextLoss);
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_print_illuminant_filtered(
        const WorkingState& workingState,
        const Print::Runtime& printRuntime,
        const Print::Params& printParams,
        const WorkspaceLeaseMarker& workspace,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }

        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_scratch_request_descriptor(workspace);
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

    bool Root::PreparedCudaFrame::prepare_scan_error_stage(
        int*& outScanErrorFlag,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        outScanErrorFlag = nullptr;
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        JuicerCuda::Resources& resources = *_state->resources;
        outScanErrorFlag = resources.scanErrorFlag;
        if (!outScanErrorFlag) {
            _state->set_failure(
                "scan_error_flag_missing",
                "CUDA scan error validation failed",
                false);
            outError = "scan error flag missing after allocation";
            return false;
        }

        cudaEvent_t scanEvent = resources.scanErrorEventOpaque
            ? reinterpret_cast<cudaEvent_t>(resources.scanErrorEventOpaque)
            : nullptr;
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        if (resources.scanErrorPending && scanEvent && resources.scanErrorHost) {
            cudaError_t pollErr = cudaEventQuery(scanEvent);
            if (pollErr == cudaSuccess) {
                resources.scanErrorPending = 0;
                if (*resources.scanErrorHost != 0) {
                    JTRACE("CUDA", "FATAL: previous scan produced non-finite RGB");
                    _state->set_failure(
                        "scan_error_previous_readback",
                        "CUDA scan error validation failed",
                        false);
                    outError = "previous scan produced non-finite RGB";
                    return false;
                }
            }
            else if (pollErr == cudaErrorNotReady) {
                const cudaError_t waitErr = cudaStreamWaitEvent(stream, scanEvent, 0);
                if (waitErr != cudaSuccess) {
                    _state->set_failure(
                        "scan_error_stream_wait",
                        "CUDA scan error validation failed");
                    outError = "CUDA scan error stream wait failed";
                    return false;
                }
                resources.scanErrorPending = 0;
            }
            else {
                _state->set_failure(
                    "scan_error_event_query",
                    "CUDA scan error validation failed");
                outError = "CUDA scan error event query failed";
                return false;
            }
        }

        const cudaError_t flagErr = cudaMemsetAsync(outScanErrorFlag, 0, sizeof(int), stream);
        if (flagErr != cudaSuccess) {
            _state->set_failure(
                "scan_error_flag_memset",
                "CUDA scan error validation failed");
            outError = "CUDA scan error flag memset failed";
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::finalize_scan_error_stage(
        int* scanErrorFlag,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }
        if (!scanErrorFlag) {
            _state->set_failure(
                "scan_error_flag_missing",
                "CUDA scan error validation failed",
                false);
            outError = "scan error flag missing after allocation";
            return false;
        }

        JuicerCuda::Resources& resources = *_state->resources;
        cudaEvent_t scanEvent = resources.scanErrorEventOpaque
            ? reinterpret_cast<cudaEvent_t>(resources.scanErrorEventOpaque)
            : nullptr;
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        if (resources.scanErrorHost && scanEvent) {
            cudaError_t flagErr = cudaMemcpyAsync(
                resources.scanErrorHost,
                scanErrorFlag,
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream);
            if (flagErr != cudaSuccess) {
                _state->set_failure(
                    "scan_error_flag_readback",
                    "CUDA scan error validation failed");
                outError = "CUDA scan error flag readback failed";
                return false;
            }
            const cudaError_t evErr = cudaEventRecord(scanEvent, stream);
            if (evErr != cudaSuccess) {
                _state->set_failure(
                    "scan_error_event_record",
                    "CUDA scan error validation failed");
                outError = "CUDA scan error event record failed";
                return false;
            }
            resources.scanErrorPending = 1;
        }
        else {
            static std::atomic<bool> sScanErrorReadbackUnavailableWarned{ false };
            if (!sScanErrorReadbackUnavailableWarned.exchange(true)) {
                JTRACE("CUDA", "scan error host/event staging unavailable; skipping asynchronous scan-error readback validation");
            }
        }
        return true;
    }

    bool Root::PreparedCudaFrame::checkpoint_scratch_phase(
        const WorkspaceLeaseMarker& workspace,
        const char* stageTag,
        std::string& outError) {
        if (!validate_workspace_lease_marker(workspace, outError)) {
            return false;
        }

        const JuicerCuda::ResourceManager::ScratchRequestDescriptor scratchRequest =
            make_scratch_request_descriptor(workspace);
        const char* failureStageTag = stageTag ? stageTag : "command_checkpoint_scratch_phase";
        if (!JuicerCuda::ResourceManager::command_checkpoint_scratch_phase(
                _state->transaction,
                *_state->resources,
                scratchRequest,
                failureStageTag,
                outError)) {
            _state->set_failure(
                failureStageTag,
                "CUDA scratch phase checkpoint failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_spatial_dir_kernel(
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        if (!JuicerCuda::ResourceManager::command_ensure_spatial_dir_kernel(
                _state->transaction,
                *_state->resources,
                _state->resources->spatialDirKernel,
                sigma,
                cudaStreamOpaque,
                outError)) {
            _state->set_failure(
                "command_ensure_spatial_dir_kernel",
                "CUDA spatial DIR kernel upload failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_gaussian_kernel(
        JuicerCuda::Resources::DeviceGaussianKernel& kernel,
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        if (!JuicerCuda::ResourceManager::command_ensure_gaussian_kernel(
                _state->transaction,
                *_state->resources,
                kernel,
                sigma,
                cudaStreamOpaque,
                outError)) {
            _state->set_failure(
                "command_ensure_gaussian_kernel",
                "CUDA gaussian kernel upload failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::prepare_halation_kernel(
        JuicerCuda::Resources::DeviceGaussianKernel& kernel,
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        if (!JuicerCuda::ResourceManager::command_ensure_halation_kernel(
                _state->transaction,
                *_state->resources,
                kernel,
                sigma,
                cudaStreamOpaque,
                outError)) {
            _state->set_failure(
                "command_ensure_halation_kernel",
                "CUDA halation kernel upload failed");
            return false;
        }
        return true;
    }

    bool Root::PreparedCudaFrame::launch_base_pipeline_graph(
        JuicerCuda::PipelineRunParams& run,
        int renderModeKey,
        void* cudaStreamOpaque,
        int& outCudaErrorCode,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }

        if (!JuicerCuda::ResourceManager::command_launch_base_pipeline_graph(
                _state->transaction,
                run,
                renderModeKey,
                cudaStreamOpaque,
                outCudaErrorCode,
                outError)) {
            _state->set_failure(
                "command_launch_base_pipeline_graph",
                "CUDA base graph launch command failed");
            return false;
        }
        return true;
    }

    Root::PreparedCudaFrame::GrainStaticAssets Root::PreparedCudaFrame::grain_static_assets() const noexcept {
        GrainStaticAssets assets{};
        if (!_state || !_state->grainStaticResources || !_state->transaction.active || _state->transaction.committed) {
            return assets;
        }

        const JuicerCuda::Resources& resources = *_state->grainStaticResources;
        if (resources.stbnData && resources.stbnWidth > 0 && resources.stbnHeight > 0 && resources.stbnFrames > 0) {
            assets.stbn = resources.stbnData;
            assets.stbnWidth = resources.stbnWidth;
            assets.stbnHeight = resources.stbnHeight;
            assets.stbnFrames = resources.stbnFrames;
        }
        if (resources.wangTilesData && resources.wangLutData &&
            resources.wangWidth > 0 && resources.wangHeight > 0 &&
            resources.wangCount > 0 && resources.wangColors > 0) {
            assets.wangTiles = resources.wangTilesData;
            assets.wangLut = resources.wangLutData;
            assets.wangWidth = resources.wangWidth;
            assets.wangHeight = resources.wangHeight;
            assets.wangCount = resources.wangCount;
            assets.wangColors = resources.wangColors;
        }
        return assets;
    }

    Root::PreparedCudaFrame::DurableBundleView Root::PreparedCudaFrame::durable_bundle() const noexcept {
        DurableBundleView bundle{};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return bundle;
        }

        const JuicerCuda::Resources& resources = *_state->resources;
        bundle.film.densB = &resources.densB;
        bundle.film.densG = &resources.densG;
        bundle.film.densR = &resources.densR;
        bundle.film.dirDensB = &resources.dirDensB;
        bundle.film.dirDensG = &resources.dirDensG;
        bundle.film.dirDensR = &resources.dirDensR;
        bundle.film.sensB = &resources.sensB;
        bundle.film.sensG = &resources.sensG;
        bundle.film.sensR = &resources.sensR;
        bundle.film.hasDensityCurvesLayers = resources.hasDensityCurvesLayers;
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                bundle.film.densityCurvesLayers[layer][ch] = resources.densityCurvesLayers[layer][ch];
            }
        }
        bundle.film.tablesAx = resources.tablesAx;
        bundle.film.tablesAy = resources.tablesAy;
        bundle.film.tablesAz = resources.tablesAz;
        bundle.film.tablesIllum = resources.tablesIllum;
        bundle.film.tablesK = resources.tablesK;
        for (int i = 0; i < 9; ++i) {
            bundle.film.spdSInv[i] = resources.spdSInv[i];
        }
        bundle.film.hanatosLut = resources.hanatosLut;
        bundle.film.hanatosN = resources.hanatosN;
        bundle.film.hanatosLutIntegrated = resources.hanatosLutIntegrated;
        bundle.film.hanatosNIntegrated = resources.hanatosNIntegrated;
        bundle.film.mallettBasis = resources.mallettBasis;
        bundle.film.mallettBasisK = resources.mallettBasisK;

        bundle.scan.negativeMedium = &resources.scanNegative;
        bundle.scan.printMedium = &resources.scanPrint;
        bundle.scan.negativeLut = &resources.scanNegativeLut;
        bundle.scan.printLut = &resources.scanPrintLut;

        bundle.print.printIllumFiltered = resources.printIllumFiltered;
        bundle.print.printIllumK = resources.printIllumK;
        bundle.print.printSensC = &resources.printSensC;
        bundle.print.printSensM = &resources.printSensM;
        bundle.print.printSensY = &resources.printSensY;
        bundle.print.printDcC = &resources.printDcC;
        bundle.print.printDcM = &resources.printDcM;
        bundle.print.printDcY = &resources.printDcY;
        bundle.print.printGammaC = resources.printGammaC;
        bundle.print.printGammaM = resources.printGammaM;
        bundle.print.printGammaY = resources.printGammaY;
        for (int i = 0; i < 3; ++i) {
            bundle.print.printPreflashRaw[i] = resources.printPreflashRaw[i];
        }
        return bundle;
    }

    Root::PreparedCudaFrame::OpticsKernelView Root::PreparedCudaFrame::optics_kernels() const noexcept {
        OpticsKernelView kernels{};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return kernels;
        }

        const JuicerCuda::Resources& resources = *_state->resources;
        kernels.spatialDir = { resources.spatialDirKernel.weights, resources.spatialDirKernel.radius };
        kernels.scannerLensBlur = { resources.scannerLensBlurKernel.weights, resources.scannerLensBlurKernel.radius };
        kernels.scannerUnsharp = { resources.scannerUnsharpKernel.weights, resources.scannerUnsharpKernel.radius };
        kernels.scannerGlare = { resources.scannerGlareKernel.weights, resources.scannerGlareKernel.radius };
        kernels.grainBlur = { resources.grainBlurKernel.weights, resources.grainBlurKernel.radius };
        kernels.grainBlurMid = { resources.grainBlurKernelMid.weights, resources.grainBlurKernelMid.radius };
        kernels.grainBlurCoarse = { resources.grainBlurKernelCoarse.weights, resources.grainBlurKernelCoarse.radius };
        for (int layer = 0; layer < 3; ++layer) {
            for (int ch = 0; ch < 3; ++ch) {
                kernels.grainDye[layer][ch] = {
                    resources.grainDyeKernel[layer][ch].weights,
                    resources.grainDyeKernel[layer][ch].radius
                };
            }
        }
        for (int i = 0; i < 3; ++i) {
            kernels.halation[i] = { resources.halationKernel[i].weights, resources.halationKernel[i].radius };
            kernels.halationScatter[i] = {
                resources.halationScatterKernel[i].weights,
                resources.halationScatterKernel[i].radius
            };
        }
        return kernels;
    }

    Root::PreparedCudaFrame::AutoExposureBufferView Root::PreparedCudaFrame::auto_exposure_buffers() const noexcept {
        AutoExposureBufferView view{};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return view;
        }

        const JuicerCuda::Resources& resources = *_state->resources;
        view.scratch.partialsA = resources.autoExposureScratch.partialsA;
        view.scratch.partialsB = resources.autoExposureScratch.partialsB;
        view.scratch.partialCapacity = resources.autoExposureScratch.partialCapacity;
        view.scratch.maxYBits = resources.autoExposureScratch.maxYBits;
        view.scratch.histogram = resources.autoExposureScratch.histogram;
        view.scratch.weightsX = resources.autoExposureScratch.weightsX;
        view.scratch.weightsY = resources.autoExposureScratch.weightsY;
        view.deviceState.exposureScale = resources.autoExposureExposureScale;
        view.deviceState.autoEV = resources.autoExposureAutoEV;
        view.deviceState.valid = resources.autoExposureValid;
        view.weightsWidth = resources.autoExposureScratch.weightsWidth;
        view.weightsHeight = resources.autoExposureScratch.weightsHeight;
        view.keyHash = resources.autoExposureKeyHash;
        view.sliderEV = resources.autoExposureSliderEV;
        view.active =
            view.scratch.partialsA &&
            view.scratch.partialsB &&
            view.scratch.partialCapacity > 0 &&
            view.scratch.maxYBits &&
            view.scratch.histogram &&
            view.scratch.weightsX &&
            view.scratch.weightsY &&
            view.deviceState.exposureScale &&
            view.deviceState.autoEV &&
            view.deviceState.valid;
        return view;
    }

    void Root::PreparedCudaFrame::mark_auto_exposure_weights_built(
        int weightsWidth,
        int weightsHeight) noexcept {
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return;
        }
        JuicerCuda::Resources& resources = *_state->resources;
        resources.autoExposureScratch.weightsWidth = weightsWidth;
        resources.autoExposureScratch.weightsHeight = weightsHeight;
    }

    void Root::PreparedCudaFrame::mark_auto_exposure_metered(
        std::uint64_t keyHash,
        double sliderEV) noexcept {
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return;
        }
        JuicerCuda::Resources& resources = *_state->resources;
        resources.autoExposureKeyHash = keyHash;
        resources.autoExposureSliderEV = sliderEV;
    }

    void Root::PreparedCudaFrame::mark_auto_exposure_slider_updated(double sliderEV) noexcept {
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return;
        }
        _state->resources->autoExposureSliderEV = sliderEV;
    }

    Root::PreparedCudaFrame::SpatialDirScratchView Root::PreparedCudaFrame::spatial_dir_scratch(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        SpatialDirScratchView view{};
        if (!workspace_marker_matches_current_frame(workspace) || !workspace._request.needSpatialDir) {
            return view;
        }

        const JuicerCuda::Resources::DeviceSpatialDirScratch& scratch = _state->resources->spatialDirScratch;
        view.corrY = scratch.corrY;
        view.corrM = scratch.corrM;
        view.corrC = scratch.corrC;
        view.tmp = scratch.tmp;
        view.active = view.corrY && view.corrM && view.corrC && view.tmp;
        return view;
    }

    Root::PreparedCudaFrame::ScannerOpticsScratchView Root::PreparedCudaFrame::scanner_optics_scratch(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        ScannerOpticsScratchView view{};
        if (!workspace_marker_matches_current_frame(workspace) || !workspace._request.needOptics) {
            return view;
        }

        const JuicerCuda::Resources::DeviceOpticsScratch& scratch = _state->resources->scannerScratch;
        view.rgbR = scratch.rgbR;
        view.rgbG = scratch.rgbG;
        view.rgbB = scratch.rgbB;
        view.tmp = scratch.tmp;
        view.blurred = scratch.blurred;
        view.aux = scratch.aux;
        view.grainTmp = scratch.grainTmp;
        view.grainTmpShared = scratch.grainTmpShared;
        view.grainTmpMid = scratch.grainTmpMid;
        view.grainTmpCoarse = scratch.grainTmpCoarse;
        view.gateMask = scratch.gateMask;
        view.gateMaskWidth = scratch.gateWidth;
        view.gateMaskHeight = scratch.gateHeight;
        view.gateMaskHash = scratch.gateMaskHash;
        view.active = view.rgbR && view.rgbG && view.rgbB && view.tmp;
        view.hasGateMask = view.gateMask && view.gateMaskWidth > 0 && view.gateMaskHeight > 0;
        return view;
    }

    void Root::PreparedCudaFrame::mark_gate_mask_built(std::uint64_t gateMaskHash) noexcept {
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return;
        }
        _state->resources->scannerScratch.gateMaskHash = gateMaskHash;
    }

    JuicerCuda::Resources* Root::PreparedCudaFrame::runtime_resources() const noexcept {
        return _state ? _state->resources : nullptr;
    }

    const char* Root::PreparedCudaFrame::failure_stage_tag() const noexcept {
        return _state ? _state->failureStageTag : "prepare_frame";
    }

    const char* Root::PreparedCudaFrame::failure_prefix() const noexcept {
        return _state ? _state->failurePrefix : "CUDA prepared frame failed";
    }

    bool Root::PreparedCudaFrame::failure_marks_context_loss() const noexcept {
        return _state ? _state->failureMarksContextLoss : true;
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
    void Root::CudaResourcesDeleter::operator()(JuicerCuda::Resources* resources) const noexcept {
        JuicerCuda::destroy(resources);
    }

    bool Root::resolve_context_cuda_resources(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::uint64_t contextEpoch,
        ContextCudaResourceMap& contextMap,
        CudaResourceOwner& outResourceOwner,
        JuicerCuda::Resources*& outResources,
        std::string& outError) {
        outResourceOwner.reset();
        outResources = nullptr;
        outError.clear();
        if (deviceContextKey.deviceId < 0 || !deviceContextKey.contextOpaque) {
            outError = "invalid CUDA context key";
            return false;
        }
        if (contextEpoch == 0) {
            outError = "invalid CUDA context epoch";
            return false;
        }

        const ContextCudaResourceKey resourceKey{ deviceContextKey, contextEpoch };
        std::vector<CudaResourceOwner> retiredOwners;
        {
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            CudaResourceOwner& resourceOwner = contextMap[resourceKey];
            if (!resourceOwner) {
                CudaResourceOwner resources(JuicerCuda::create(), CudaResourcesDeleter{});
                if (!resources) {
                    JTRACE("CUDA", "FATAL: failed to allocate CUDA resources");
                    outError = "failed to allocate CUDA resources";
                    return false;
                }
                if (resources->deviceId < 0) {
                    resources->deviceId = deviceContextKey.deviceId;
                }
                if (!resources->ownerContextOpaque) {
                    resources->ownerContextOpaque = deviceContextKey.contextOpaque;
                }
                if (resources->deviceId != deviceContextKey.deviceId ||
                    resources->ownerContextOpaque != deviceContextKey.contextOpaque) {
                    outError = "CUDA resources resolved for a different context";
                    return false;
                }
                resourceOwner = std::move(resources);
            }

            outResourceOwner = resourceOwner;
            for (auto it = contextMap.begin(); it != contextMap.end();) {
                if (!(it->first.deviceContextKey == deviceContextKey) ||
                    it->first.contextEpoch >= contextEpoch) {
                    ++it;
                    continue;
                }
                if (it->second) {
                    retiredOwners.emplace_back(std::move(it->second));
                }
                it = contextMap.erase(it);
            }
        }

        outResources = outResourceOwner.get();
        if (!outResources) {
            outError = "CUDA resources missing after allocation";
            return false;
        }
        return true;
    }

    bool Root::resolve_cuda_frame_resources(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::uint64_t contextEpoch,
        CudaResourceOwner& outResourceOwner,
        JuicerCuda::Resources*& outResources,
        std::string& outError) {
        return resolve_context_cuda_resources(
            deviceContextKey,
            contextEpoch,
            _cudaResourcesByContext,
            outResourceOwner,
            outResources,
            outError);
    }

    bool Root::resolve_cuda_grain_static_resources(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::uint64_t contextEpoch,
        CudaResourceOwner& outResourceOwner,
        JuicerCuda::Resources*& outResources,
        std::string& outError) {
        return resolve_context_cuda_resources(
            deviceContextKey,
            contextEpoch,
            _cudaGrainStaticByContext,
            outResourceOwner,
            outResources,
            outError);
    }

    Root::PreparedCudaFrame Root::prepare_cuda_frame(
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
        if (!begin_submission(frame._state->transaction, snapshot, outError)) {
            frame._state->set_failure("begin_submission", "begin_submission failed");
            return frame;
        }
        if (!resolve_cuda_frame_resources(
                deviceContextKey,
                frame._state->transaction.snapshot.contextEpoch,
                frame._state->resourceOwner,
                frame._state->resources,
                outError)) {
            frame._state->set_failure("resolve_cuda_resources", "CUDA resource acquisition failed");
            frame.abort("prepared_frame_resource_acquire_failed");
            return frame;
        }
        if (!resolve_cuda_grain_static_resources(
                deviceContextKey,
                frame._state->transaction.snapshot.contextEpoch,
                frame._state->grainStaticOwner,
                frame._state->grainStaticResources,
                outError)) {
            frame._state->set_failure("resolve_grain_static_resources", "CUDA grain-static resource acquisition failed");
            frame.abort("prepared_frame_grain_static_acquire_failed");
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
        if (!JuicerCuda::ensure_grain_static_assets_uploaded(
                *frame._state->grainStaticResources,
                cudaStreamOpaque,
                outError)) {
            frame._state->set_failure(
                "ensure_grain_static_assets_uploaded",
                "CUDA grain-static asset upload failed");
            frame.abort("prepared_frame_grain_static_upload_failed");
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
