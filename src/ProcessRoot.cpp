#include "ProcessRoot.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
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
            } catch (const std::exception& ex) {
                Spectral::set_hanatos_available(false);
                Spectral::set_mallett_available(false);
#if JUICER_DIAGNOSTICS_COMPILED
                if (JTRACE_ENABLED(1)) {
                    std::string msg = "FATAL: spectral bootstrap failed: ";
                    msg += ex.what();
                    JTRACE("INIT", msg);
                }
#endif
            } catch (...) {
                Spectral::set_hanatos_available(false);
                Spectral::set_mallett_available(false);
                JTRACE("INIT", "FATAL: spectral bootstrap failed with unknown error");
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
        struct AutoExposureFrameWorkspace {
            JuicerCudaAutoExposureScratch scratch{};
            JuicerCudaAutoExposureDeviceState deviceState{};
            int weightsWidth = 0;
            int weightsHeight = 0;
            int weightsXCapacity = 0;
            int weightsYCapacity = 0;
            Spektrafilm::AutoExposureMethod method = Spektrafilm::AutoExposureMethod::CenterWeighted;
            std::uint64_t keyHash = 0;
            bool active = false;
        };

        struct ScanErrorFrameStage {
            int* deviceFlag = nullptr;
            int* hostFlag = nullptr;
            void* eventOpaque = nullptr;
            bool active = false;
            bool readbackPending = false;
        };

        struct FrameScratchWorkspace {
            WorkspaceRequest request{};
            JuicerCuda::Resources::DeviceOpticsScratch optics{};
            JuicerCuda::Resources::DeviceSpatialDirScratch spatialDir{};
            float* sharedTmpPlane = nullptr;
            int sharedTmpWidth = 0;
            int sharedTmpHeight = 0;
            std::size_t sharedTmpCapacityElements = 0;
            bool retainedLeaseActive = false;
            bool overflowActive = false;
        };

        Root* root = nullptr;
        Root::CudaResourceOwner resourceOwner;
        Root::CudaResourceOwner grainStaticOwner;
        JuicerCuda::Resources* resources = nullptr;
        JuicerCuda::Resources* grainStaticResources = nullptr;
        const Spectral::FilmRawConfig* directFilmRawConfig = nullptr;
        const Scanner::ColorRuntime* directScannerColor = nullptr;
        JuicerCuda::ResourceManager::SubmissionTransaction transaction{};
        ScanErrorFrameStage scanErrorStage{};
        AutoExposureFrameWorkspace autoExposureWorkspace{};
        FrameScratchWorkspace scratchWorkspace{};
        void* lastCudaStreamOpaque = nullptr;
        bool frameUseEventSubmitted = false;
        const char* failureStageTag = "prepare_frame";
        const char* failurePrefix = "CUDA prepared frame failed";
        bool failureMarksContextLoss = true;

        void set_failure(const char* stageTag, const char* prefix, bool marksContextLoss = true) noexcept {
            failureStageTag = stageTag;
            failurePrefix = prefix;
            failureMarksContextLoss = marksContextLoss;
        }

        void remember_stream(void* cudaStreamOpaque) noexcept {
            if (cudaStreamOpaque) {
                lastCudaStreamOpaque = cudaStreamOpaque;
            }
        }

        bool allocate_auto_exposure_workspace(
            const JuicerCuda::AutoExposurePreviewDescriptor& descriptor,
            std::string& outError);
        bool allocate_scan_error_stage(std::string& outError);
        bool release_scan_error_stage_after_use(
            void* cudaStreamOpaque,
            std::string& outError);
        bool release_auto_exposure_workspace_after_use(
            void* cudaStreamOpaque,
            std::string& outError);
        bool ensure_scratch_workspace(
            const WorkspaceRequest& request,
            void* cudaStreamOpaque,
            std::string& outError);
        bool release_scratch_workspace_after_use(
            void* cudaStreamOpaque,
            std::string& outError);
        bool submit_frame_use_event(
            void* cudaStreamOpaque,
            std::string& outError);
        void free_scan_error_stage_now() noexcept;
        void free_auto_exposure_workspace_now() noexcept;
        void free_scratch_workspace_now() noexcept;
    };

    bool Root::PreparedCudaFrame::State::allocate_scan_error_stage(std::string& outError) {
        outError.clear();
        free_scan_error_stage_now();

        ScanErrorFrameStage next{};
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&next.deviceFlag), sizeof(int));
        if (err != cudaSuccess) {
            outError = std::string("cudaMalloc(frame scan error flag) failed: ") +
                (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            next.deviceFlag = nullptr;
            return false;
        }

        err = cudaMallocHost(reinterpret_cast<void**>(&next.hostFlag), sizeof(int));
        if (err != cudaSuccess) {
            next.hostFlag = nullptr;
        }

        if (next.hostFlag) {
            cudaEvent_t ev = nullptr;
            err = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
            if (err == cudaSuccess && ev) {
                next.eventOpaque = reinterpret_cast<void*>(ev);
            }
        }

        next.active = true;
        scanErrorStage = next;
        return true;
    }

    bool Root::PreparedCudaFrame::State::release_scan_error_stage_after_use(
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        ScanErrorFrameStage& stage = scanErrorStage;
        if (!stage.active) {
            return true;
        }
        if (!resources) {
            outError = "CUDA resources unavailable for frame scan-error stage release";
            return false;
        }

        remember_stream(cudaStreamOpaque);
        void* retireStreamOpaque = cudaStreamOpaque ? cudaStreamOpaque : lastCudaStreamOpaque;
        if (stage.readbackPending && stage.hostFlag && stage.eventOpaque) {
            if (!JuicerCuda::retain_scan_error_readback(
                    *resources,
                    stage.hostFlag,
                    stage.eventOpaque,
                    outError)) {
                if (outError.empty()) {
                    outError = "frame scan-error readback retention failed";
                }
                return false;
            }
            stage.readbackPending = false;
        }

        if (stage.deviceFlag) {
            int* flag = stage.deviceFlag;
            if (!JuicerCuda::retire_frame_scratch_allocation(
                    *resources,
                    flag,
                    sizeof(int),
                    retireStreamOpaque,
                    "frame scan error flag",
                    outError)) {
                if (outError.empty()) {
                    outError = "frame scan-error flag retire failed";
                }
                return false;
            }
            stage.deviceFlag = nullptr;
        }

        if (stage.hostFlag) {
            cudaFreeHost(stage.hostFlag);
            stage.hostFlag = nullptr;
        }
        if (stage.eventOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(stage.eventOpaque);
            cudaEventDestroy(ev);
            stage.eventOpaque = nullptr;
        }
        stage = ScanErrorFrameStage{};
        return true;
    }

    void Root::PreparedCudaFrame::State::free_scan_error_stage_now() noexcept {
        ScanErrorFrameStage& stage = scanErrorStage;
        if (stage.deviceFlag) {
            cudaFree(stage.deviceFlag);
        }
        if (stage.hostFlag) {
            cudaFreeHost(stage.hostFlag);
        }
        if (stage.eventOpaque) {
            cudaEvent_t ev = reinterpret_cast<cudaEvent_t>(stage.eventOpaque);
            cudaEventDestroy(ev);
        }
        stage = ScanErrorFrameStage{};
    }

    bool Root::PreparedCudaFrame::State::allocate_auto_exposure_workspace(
        const JuicerCuda::AutoExposurePreviewDescriptor& descriptor,
        std::string& outError) {
        outError.clear();
        free_auto_exposure_workspace_now();
        const int meterWidth = descriptor.previewWidth;
        const int meterHeight = descriptor.previewHeight;
        if (meterWidth <= 0 || meterHeight <= 0) {
            outError = "auto-exposure meter dimensions invalid";
            return false;
        }

        const int blockX = 16;
        const int blockY = 16;
        const int gridX = (meterWidth + blockX - 1) / blockX;
        const int gridY = (meterHeight + blockY - 1) / blockY;
        const bool needsHistogram = descriptor.method == Spektrafilm::AutoExposureMethod::Median;
        const bool needsWeights = descriptor.method == Spektrafilm::AutoExposureMethod::CenterWeighted;
        const bool needsPartials = !needsHistogram;
        const int neededPartials = needsPartials ? gridX * gridY : 0;
        if (needsPartials && neededPartials <= 0) {
            outError = "auto-exposure partial count invalid";
            return false;
        }

        auto alloc_device = [&](auto*& ptr, std::size_t bytes, const char* label) -> bool {
            const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
            if (err != cudaSuccess) {
                outError = std::string("cudaMalloc(") + label + ") failed: " +
                    (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
                ptr = nullptr;
                return false;
            }
            return true;
        };

        AutoExposureFrameWorkspace next{};
        if (!alloc_device(next.deviceState.exposureScale, sizeof(float), "frame auto-exposure scale") ||
            !alloc_device(next.deviceState.autoEV, sizeof(double), "frame auto-exposure autoEV") ||
            !alloc_device(next.deviceState.valid, sizeof(int), "frame auto-exposure valid")) {
            autoExposureWorkspace = next;
            free_auto_exposure_workspace_now();
            return false;
        }
        if (needsHistogram &&
            (!alloc_device(next.scratch.maxYBits, sizeof(unsigned int), "frame auto-exposure maxYBits") ||
             !alloc_device(next.scratch.histogram, sizeof(unsigned int) * 2048u, "frame auto-exposure histogram"))) {
            autoExposureWorkspace = next;
            free_auto_exposure_workspace_now();
            return false;
        }
        if (needsWeights &&
            (!alloc_device(next.scratch.weightsX, static_cast<std::size_t>(meterWidth) * sizeof(float), "frame auto-exposure weightsX") ||
             !alloc_device(next.scratch.weightsY, static_cast<std::size_t>(meterHeight) * sizeof(float), "frame auto-exposure weightsY"))) {
            autoExposureWorkspace = next;
            free_auto_exposure_workspace_now();
            return false;
        }
        if (needsPartials &&
            (!alloc_device(
                 next.scratch.partialsA,
                 static_cast<std::size_t>(neededPartials) * sizeof(JuicerCudaAutoExposurePartial),
                 "frame auto-exposure partialsA") ||
             !alloc_device(
                 next.scratch.partialsB,
                 static_cast<std::size_t>(neededPartials) * sizeof(JuicerCudaAutoExposurePartial),
                 "frame auto-exposure partialsB"))) {
            autoExposureWorkspace = next;
            free_auto_exposure_workspace_now();
            return false;
        }

        next.scratch.partialCapacity = neededPartials;
        next.weightsXCapacity = needsWeights ? meterWidth : 0;
        next.weightsYCapacity = needsWeights ? meterHeight : 0;
        next.weightsWidth = 0;
        next.weightsHeight = 0;
        next.method = descriptor.method;
        next.keyHash = 0;
        next.active = true;
        autoExposureWorkspace = next;
        return true;
    }

    bool Root::PreparedCudaFrame::State::release_auto_exposure_workspace_after_use(
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        AutoExposureFrameWorkspace& workspace = autoExposureWorkspace;
        if (!workspace.active) {
            return true;
        }

        try {
            if (!resources) {
                outError = "CUDA resources unavailable for frame auto-exposure retire";
                return false;
            }

            remember_stream(cudaStreamOpaque);
            void* retireStreamOpaque = cudaStreamOpaque ? cudaStreamOpaque : lastCudaStreamOpaque;
            bool retiredAll = true;
            auto retire_ptr = [&](auto*& ptr, std::size_t bytes, const char* label) {
                if (!ptr || !retiredAll) {
                    return;
                }
                void* raw = ptr;
                std::string localError;
                if (JuicerCuda::retire_frame_scratch_allocation(
                        *resources,
                        raw,
                        bytes,
                        retireStreamOpaque,
                        label,
                        localError)) {
                    ptr = nullptr;
                    return;
                }
                retiredAll = false;
                if (localError.empty()) {
                    outError = std::string(label ? label : "frame auto-exposure buffer") +
                        " retire failed";
                } else {
                    outError = localError;
                }
            };

            retire_ptr(workspace.deviceState.exposureScale, sizeof(float), "frame auto-exposure scale");
            retire_ptr(workspace.deviceState.autoEV, sizeof(double), "frame auto-exposure autoEV");
            retire_ptr(workspace.deviceState.valid, sizeof(int), "frame auto-exposure valid");
            retire_ptr(workspace.scratch.maxYBits, sizeof(unsigned int), "frame auto-exposure maxYBits");
            retire_ptr(workspace.scratch.histogram, sizeof(unsigned int) * 2048u, "frame auto-exposure histogram");
            retire_ptr(
                workspace.scratch.weightsX,
                static_cast<std::size_t>(std::max(0, workspace.weightsXCapacity)) * sizeof(float),
                "frame auto-exposure weightsX");
            retire_ptr(
                workspace.scratch.weightsY,
                static_cast<std::size_t>(std::max(0, workspace.weightsYCapacity)) * sizeof(float),
                "frame auto-exposure weightsY");
            retire_ptr(
                workspace.scratch.partialsA,
                static_cast<std::size_t>(std::max(0, workspace.scratch.partialCapacity)) *
                    sizeof(JuicerCudaAutoExposurePartial),
                "frame auto-exposure partialsA");
            retire_ptr(
                workspace.scratch.partialsB,
                static_cast<std::size_t>(std::max(0, workspace.scratch.partialCapacity)) *
                    sizeof(JuicerCudaAutoExposurePartial),
                "frame auto-exposure partialsB");

            if (retiredAll) {
                workspace = AutoExposureFrameWorkspace{};
                return true;
            }
        } catch (...) {
            outError = "frame auto-exposure retire threw";
        }
        if (outError.empty()) {
            outError = "frame auto-exposure retire failed";
        }
        return false;
    }

    void Root::PreparedCudaFrame::State::free_auto_exposure_workspace_now() noexcept {
        AutoExposureFrameWorkspace& workspace = autoExposureWorkspace;
        if (workspace.scratch.partialsA) {
            cudaFree(workspace.scratch.partialsA);
        }
        if (workspace.scratch.partialsB) {
            cudaFree(workspace.scratch.partialsB);
        }
        if (workspace.scratch.maxYBits) {
            cudaFree(workspace.scratch.maxYBits);
        }
        if (workspace.scratch.histogram) {
            cudaFree(workspace.scratch.histogram);
        }
        if (workspace.scratch.weightsX) {
            cudaFree(workspace.scratch.weightsX);
        }
        if (workspace.scratch.weightsY) {
            cudaFree(workspace.scratch.weightsY);
        }
        if (workspace.deviceState.exposureScale) {
            cudaFree(workspace.deviceState.exposureScale);
        }
        if (workspace.deviceState.autoEV) {
            cudaFree(workspace.deviceState.autoEV);
        }
        if (workspace.deviceState.valid) {
            cudaFree(workspace.deviceState.valid);
        }
        workspace = AutoExposureFrameWorkspace{};
    }

    bool Root::PreparedCudaFrame::State::submit_frame_use_event(
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!resources || !transaction.active || transaction.committed) {
            outError = "prepared frame is not active for use-event submission";
            return false;
        }
        if (frameUseEventSubmitted) {
            return true;
        }
        remember_stream(cudaStreamOpaque);

        cudaEvent_t ev = nullptr;
        const cudaError_t createErr = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        if (createErr != cudaSuccess || !ev) {
            outError = std::string("cudaEventCreateWithFlags(frame use) failed: ") +
                (cudaGetErrorString(createErr) ? cudaGetErrorString(createErr) : "(unknown)");
            return false;
        }

        const cudaStream_t stream = cudaStreamOpaque
            ? reinterpret_cast<cudaStream_t>(cudaStreamOpaque)
            : nullptr;
        const cudaError_t recordErr = cudaEventRecord(ev, stream);
        if (recordErr != cudaSuccess) {
            cudaEventDestroy(ev);
            outError = std::string("cudaEventRecord(frame use) failed: ") +
                (cudaGetErrorString(recordErr) ? cudaGetErrorString(recordErr) : "(unknown)");
            return false;
        }

        void* eventOpaque = reinterpret_cast<void*>(ev);
        if (!JuicerCuda::retain_frame_use_event(*resources, eventOpaque, outError)) {
            if (eventOpaque) {
                cudaEventDestroy(reinterpret_cast<cudaEvent_t>(eventOpaque));
            }
            if (outError.empty()) {
                outError = "frame use event retention failed";
            }
            return false;
        }
        frameUseEventSubmitted = true;
        return true;
    }

    bool Root::PreparedCudaFrame::State::ensure_scratch_workspace(
        const WorkspaceRequest& request,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!request.needOptics && !request.needSpatialDir) {
            return true;
        }
        if (!resources) {
            outError = "CUDA resources unavailable for frame scratch lease";
            return false;
        }
        if (request.requestedWidth <= 0 || request.requestedHeight <= 0) {
            outError = "frame scratch dimensions invalid";
            return false;
        }

        auto requests_match = [](const WorkspaceRequest& a, const WorkspaceRequest& b) noexcept {
            return a.needOptics == b.needOptics &&
                a.needSpatialDir == b.needSpatialDir &&
                a.requestedWidth == b.requestedWidth &&
                a.requestedHeight == b.requestedHeight &&
                a.needBlurred == b.needBlurred &&
                a.needAux == b.needAux &&
                a.needGrainTriplet == b.needGrainTriplet &&
                a.needGrainShared == b.needGrainShared &&
                a.needGateMask == b.needGateMask;
        };

        if (scratchWorkspace.retainedLeaseActive || scratchWorkspace.overflowActive) {
            if (!requests_match(scratchWorkspace.request, request)) {
                outError = "prepared frame scratch workspace request mismatch";
                return false;
            }
            return true;
        }

        remember_stream(cudaStreamOpaque);
        bool retainedAcquired = false;
        if (!JuicerCuda::try_acquire_retained_frame_scratch_lease(
                *resources,
                transaction.leaseGeneration,
                cudaStreamOpaque,
                retainedAcquired,
                outError)) {
            return false;
        }
        if (retainedAcquired) {
            scratchWorkspace.request = request;
            scratchWorkspace.retainedLeaseActive = true;
            return true;
        }

        const std::size_t width = static_cast<std::size_t>(request.requestedWidth);
        const std::size_t height = static_cast<std::size_t>(request.requestedHeight);
        if (width > (std::numeric_limits<std::size_t>::max() / height)) {
            outError = "frame scratch element count overflow";
            return false;
        }
        const std::size_t requiredElements = width * height;
        if (requiredElements > (std::numeric_limits<std::size_t>::max() / sizeof(float))) {
            outError = "frame scratch byte count overflow";
            return false;
        }
        const std::size_t planeBytes = requiredElements * sizeof(float);

        FrameScratchWorkspace next{};
        next.request = request;
        next.overflowActive = true;

        auto alloc_float = [&](float*& ptr, std::size_t bytes, const char* label) {
            cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
            if (err == cudaSuccess && ptr) {
                return true;
            }
            outError = std::string("cudaMalloc(") + (label ? label : "frame scratch") + ") failed: " +
                (cudaGetErrorString(err) ? cudaGetErrorString(err) : "(unknown)");
            ptr = nullptr;
            return false;
        };
        auto fail_after_partial_alloc = [&]() {
            scratchWorkspace = next;
            free_scratch_workspace_now();
            return false;
        };

        if (!alloc_float(next.sharedTmpPlane, planeBytes, "frame shared tmp plane")) {
            return fail_after_partial_alloc();
        }
        next.sharedTmpWidth = request.requestedWidth;
        next.sharedTmpHeight = request.requestedHeight;
        next.sharedTmpCapacityElements = requiredElements;

        if (request.needOptics) {
            JuicerCuda::Resources::DeviceOpticsScratch& optics = next.optics;
            if (!alloc_float(optics.rgbR, planeBytes, "frame scannerScratch.rgbR") ||
                !alloc_float(optics.rgbG, planeBytes, "frame scannerScratch.rgbG") ||
                !alloc_float(optics.rgbB, planeBytes, "frame scannerScratch.rgbB")) {
                return fail_after_partial_alloc();
            }
            optics.tmp = next.sharedTmpPlane;
            optics.width = request.requestedWidth;
            optics.height = request.requestedHeight;
            optics.capacityElements = requiredElements;
            if (request.needBlurred &&
                !alloc_float(optics.blurred, planeBytes, "frame scannerScratch.blurred")) {
                return fail_after_partial_alloc();
            }
            if (request.needAux &&
                !alloc_float(optics.aux, planeBytes, "frame scannerScratch.aux")) {
                return fail_after_partial_alloc();
            }
            if (request.needGrainTriplet) {
                if (!alloc_float(optics.grainTmp, planeBytes, "frame scannerScratch.grainTmp") ||
                    !alloc_float(optics.grainTmpMid, planeBytes, "frame scannerScratch.grainTmpMid") ||
                    !alloc_float(optics.grainTmpCoarse, planeBytes, "frame scannerScratch.grainTmpCoarse")) {
                    return fail_after_partial_alloc();
                }
            }
            if (request.needGrainShared &&
                !alloc_float(optics.grainTmpShared, planeBytes, "frame scannerScratch.grainTmpShared")) {
                return fail_after_partial_alloc();
            }
            if (request.needGateMask) {
                const int gateWidth = (request.requestedWidth + 1) / 2;
                const int gateHeight = (request.requestedHeight + 1) / 2;
                const std::size_t gateElements =
                    static_cast<std::size_t>(gateWidth) * static_cast<std::size_t>(gateHeight);
                if (gateElements > (std::numeric_limits<std::size_t>::max() / sizeof(float))) {
                    outError = "frame gate-mask byte count overflow";
                    return fail_after_partial_alloc();
                }
                if (!alloc_float(
                        optics.gateMask,
                        gateElements * sizeof(float),
                        "frame scannerScratch.gateMask")) {
                    return fail_after_partial_alloc();
                }
                optics.gateWidth = gateWidth;
                optics.gateHeight = gateHeight;
                optics.gateMaskCapacityElements = gateElements;
                optics.gateMaskHash = 0;
            }
        }

        if (request.needSpatialDir) {
            JuicerCuda::Resources::DeviceSpatialDirScratch& spatialDir = next.spatialDir;
            if (!alloc_float(spatialDir.corrY, planeBytes, "frame spatial DIR corrY") ||
                !alloc_float(spatialDir.corrM, planeBytes, "frame spatial DIR corrM") ||
                !alloc_float(spatialDir.corrC, planeBytes, "frame spatial DIR corrC")) {
                return fail_after_partial_alloc();
            }
            spatialDir.tmp = next.sharedTmpPlane;
            spatialDir.width = request.requestedWidth;
            spatialDir.height = request.requestedHeight;
            spatialDir.capacityElements = requiredElements;
        }

        scratchWorkspace = next;
        return true;
    }

    bool Root::PreparedCudaFrame::State::release_scratch_workspace_after_use(
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        FrameScratchWorkspace& workspace = scratchWorkspace;
        if (!workspace.retainedLeaseActive && !workspace.overflowActive) {
            return true;
        }
        if (!resources) {
            outError = "CUDA resources unavailable for frame scratch release";
            return false;
        }

        remember_stream(cudaStreamOpaque);
        void* retireStreamOpaque = cudaStreamOpaque ? cudaStreamOpaque : lastCudaStreamOpaque;

        if (workspace.retainedLeaseActive) {
            if (!submit_frame_use_event(retireStreamOpaque, outError)) {
                if (outError.empty()) {
                    outError = "frame scratch use-event submission failed";
                }
                return false;
            }
            if (!JuicerCuda::release_retained_frame_scratch_lease(
                    *resources,
                    transaction.leaseGeneration,
                    outError)) {
                if (outError.empty()) {
                    outError = "retained frame scratch lease release failed";
                }
                return false;
            }
            workspace = FrameScratchWorkspace{};
            return true;
        }

        bool retiredAll = true;
        auto retire_ptr = [&](float*& ptr, std::size_t bytes, const char* label) {
            if (!ptr || !retiredAll) {
                return;
            }
            void* raw = ptr;
            std::string localError;
            if (JuicerCuda::retire_frame_scratch_allocation(
                    *resources,
                    raw,
                    bytes,
                    retireStreamOpaque,
                    label,
                    localError)) {
                ptr = nullptr;
                return;
            }
            retiredAll = false;
            outError = localError.empty()
                ? std::string(label ? label : "frame scratch") + " retire failed"
                : localError;
        };

        const std::size_t planeBytes = workspace.optics.capacityElements > 0
            ? workspace.optics.capacityElements * sizeof(float)
            : workspace.spatialDir.capacityElements * sizeof(float);
        retire_ptr(workspace.optics.rgbR, planeBytes, "frame scannerScratch.rgbR");
        retire_ptr(workspace.optics.rgbG, planeBytes, "frame scannerScratch.rgbG");
        retire_ptr(workspace.optics.rgbB, planeBytes, "frame scannerScratch.rgbB");
        retire_ptr(workspace.optics.blurred, planeBytes, "frame scannerScratch.blurred");
        retire_ptr(workspace.optics.aux, planeBytes, "frame scannerScratch.aux");
        retire_ptr(workspace.optics.grainTmp, planeBytes, "frame scannerScratch.grainTmp");
        retire_ptr(workspace.optics.grainTmpShared, planeBytes, "frame scannerScratch.grainTmpShared");
        retire_ptr(workspace.optics.grainTmpMid, planeBytes, "frame scannerScratch.grainTmpMid");
        retire_ptr(workspace.optics.grainTmpCoarse, planeBytes, "frame scannerScratch.grainTmpCoarse");
        retire_ptr(
            workspace.optics.gateMask,
            workspace.optics.gateMaskCapacityElements * sizeof(float),
            "frame scannerScratch.gateMask");
        retire_ptr(workspace.spatialDir.corrY, planeBytes, "frame spatial DIR corrY");
        retire_ptr(workspace.spatialDir.corrM, planeBytes, "frame spatial DIR corrM");
        retire_ptr(workspace.spatialDir.corrC, planeBytes, "frame spatial DIR corrC");
        retire_ptr(
            workspace.sharedTmpPlane,
            workspace.sharedTmpCapacityElements * sizeof(float),
            "frame shared tmp plane");

        if (retiredAll) {
            workspace = FrameScratchWorkspace{};
            return true;
        }
        if (outError.empty()) {
            outError = "frame scratch retire failed";
        }
        return false;
    }

    void Root::PreparedCudaFrame::State::free_scratch_workspace_now() noexcept {
        FrameScratchWorkspace& workspace = scratchWorkspace;
        if (workspace.optics.rgbR) {
            cudaFree(workspace.optics.rgbR);
        }
        if (workspace.optics.rgbG) {
            cudaFree(workspace.optics.rgbG);
        }
        if (workspace.optics.rgbB) {
            cudaFree(workspace.optics.rgbB);
        }
        if (workspace.optics.blurred) {
            cudaFree(workspace.optics.blurred);
        }
        if (workspace.optics.aux) {
            cudaFree(workspace.optics.aux);
        }
        if (workspace.optics.grainTmp) {
            cudaFree(workspace.optics.grainTmp);
        }
        if (workspace.optics.grainTmpShared) {
            cudaFree(workspace.optics.grainTmpShared);
        }
        if (workspace.optics.grainTmpMid) {
            cudaFree(workspace.optics.grainTmpMid);
        }
        if (workspace.optics.grainTmpCoarse) {
            cudaFree(workspace.optics.grainTmpCoarse);
        }
        if (workspace.optics.gateMask) {
            cudaFree(workspace.optics.gateMask);
        }
        if (workspace.spatialDir.corrY) {
            cudaFree(workspace.spatialDir.corrY);
        }
        if (workspace.spatialDir.corrM) {
            cudaFree(workspace.spatialDir.corrM);
        }
        if (workspace.spatialDir.corrC) {
            cudaFree(workspace.spatialDir.corrC);
        }
        if (workspace.sharedTmpPlane) {
            cudaFree(workspace.sharedTmpPlane);
        }
        workspace = FrameScratchWorkspace{};
    }

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
        JuicerCuda::ResourceManager::ScratchRequestBuildRequest request{};
        request.families.needOptics = workspace._request.needOptics;
        request.families.needSpatialDir = workspace._request.needSpatialDir;
        request.extent.requestedWidth = workspace._request.requestedWidth;
        request.extent.requestedHeight = workspace._request.requestedHeight;
        request.attachments.needBlurred = workspace._request.needBlurred;
        request.attachments.needAux = workspace._request.needAux;
        request.attachments.needGrainTriplet = workspace._request.needGrainTriplet;
        request.attachments.needGrainShared = workspace._request.needGrainShared;
        request.attachments.needGateMask = workspace._request.needGateMask;
        return JuicerCuda::ResourceManager::make_scratch_request_descriptor(
            request);
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
        _state->remember_stream(cudaStreamOpaque);
        std::string releaseError;
        if (!_state->release_scan_error_stage_after_use(cudaStreamOpaque, releaseError)) {
            outError = releaseError.empty() ? "frame scan-error stage release failed" : releaseError;
            return false;
        }
        if (!_state->release_scratch_workspace_after_use(cudaStreamOpaque, releaseError)) {
            outError = releaseError.empty() ? "frame scratch workspace release failed" : releaseError;
            return false;
        }
        if (!_state->release_auto_exposure_workspace_after_use(cudaStreamOpaque, releaseError)) {
            outError = releaseError.empty() ? "frame auto-exposure retire failed" : releaseError;
            if (JTRACE_ENABLED(1)) {
                std::string msg = "frame_auto_exposure_retire_failed finish=1 error=";
                msg += outError;
                JTRACE("CUDA", msg);
            }
            return false;
        }

        if (!_state->root->commit_submission(_state->transaction, cudaStreamOpaque, outError)) {
            return false;
        }
        return true;
    }

    void Root::PreparedCudaFrame::abort(const char* reason) noexcept {
        if (_state) {
            try {
                std::string releaseError;
                (void)_state->release_scan_error_stage_after_use(
                    _state->lastCudaStreamOpaque,
                    releaseError);
                if (!_state->release_scratch_workspace_after_use(
                        _state->lastCudaStreamOpaque,
                        releaseError) &&
                    JTRACE_ENABLED(1)) {
                    std::string msg = "frame_scratch_workspace_release_failed abort=1";
                    if (reason && reason[0]) {
                        msg += " reason=";
                        msg += reason;
                    }
                    if (!releaseError.empty()) {
                        msg += " error=";
                        msg += releaseError;
                    }
                    JTRACE("CUDA", msg);
                }
                if (!_state->release_auto_exposure_workspace_after_use(
                        _state->lastCudaStreamOpaque,
                        releaseError) &&
                    JTRACE_ENABLED(1)) {
                    std::string msg = "frame_auto_exposure_retire_failed abort=1";
                    if (reason && reason[0]) {
                        msg += " reason=";
                        msg += reason;
                    }
                    if (!releaseError.empty()) {
                        msg += " error=";
                        msg += releaseError;
                    }
                    JTRACE("CUDA", msg);
                }
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
        }
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
        if (!_state->ensure_scratch_workspace(workspace._request, cudaStreamOpaque, outError)) {
            _state->set_failure(
                "acquire_frame_scratch_workspace",
                "CUDA frame scratch workspace acquisition failed");
            return false;
        }
        if (_state->scratchWorkspace.overflowActive) {
            return true;
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
        if (!_state->ensure_scratch_workspace(workspace._request, cudaStreamOpaque, outError)) {
            _state->set_failure(
                "acquire_frame_scratch_workspace",
                "CUDA frame scratch workspace acquisition failed");
            return false;
        }
        if (_state->scratchWorkspace.overflowActive) {
            return true;
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

        bool previousScanErrorDetected = false;
        if (!JuicerCuda::poll_scan_error_readbacks(
                *_state->resources,
                cudaStreamOpaque,
                previousScanErrorDetected,
                outError)) {
            _state->set_failure(
                "scan_error_pending_readback",
                "CUDA scan error validation failed");
            return false;
        }
        if (previousScanErrorDetected) {
            JTRACE("CUDA", "FATAL: previous scan produced non-finite RGB");
            _state->set_failure(
                "scan_error_previous_readback",
                "CUDA scan error validation failed",
                false);
            outError = "previous scan produced non-finite RGB";
            return false;
        }

        State::ScanErrorFrameStage& stage = _state->scanErrorStage;
        outScanErrorFlag = stage.deviceFlag;
        if (!outScanErrorFlag) {
            _state->set_failure(
                "scan_error_flag_missing",
                "CUDA scan error validation failed",
                false);
            outError = "scan error flag missing after allocation";
            return false;
        }

        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
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

        State::ScanErrorFrameStage& stage = _state->scanErrorStage;
        if (scanErrorFlag != stage.deviceFlag) {
            _state->set_failure(
                "scan_error_flag_mismatch",
                "CUDA scan error validation failed",
                false);
            outError = "scan error flag does not match prepared frame stage";
            return false;
        }

        cudaEvent_t scanEvent = stage.eventOpaque
            ? reinterpret_cast<cudaEvent_t>(stage.eventOpaque)
            : nullptr;
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        if (stage.hostFlag && scanEvent) {
            cudaError_t flagErr = cudaMemcpyAsync(
                stage.hostFlag,
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
            stage.readbackPending = true;
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

    bool Root::PreparedCudaFrame::prepare_gaussian_kernel_slot(
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

    bool Root::PreparedCudaFrame::prepare_halation_kernel_slot(
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

    bool Root::PreparedCudaFrame::prepare_scanner_lens_blur_kernel(
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        return prepare_gaussian_kernel_slot(
            _state->resources->scannerLensBlurKernel,
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::prepare_scanner_unsharp_kernel(
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        return prepare_gaussian_kernel_slot(
            _state->resources->scannerUnsharpKernel,
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::prepare_scanner_glare_kernel(
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        return prepare_gaussian_kernel_slot(
            _state->resources->scannerGlareKernel,
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::prepare_grain_blur_kernel(
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        return prepare_gaussian_kernel_slot(
            _state->resources->grainBlurKernel,
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::prepare_grain_blur_mid_kernel(
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        return prepare_gaussian_kernel_slot(
            _state->resources->grainBlurKernelMid,
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::prepare_grain_blur_coarse_kernel(
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        return prepare_gaussian_kernel_slot(
            _state->resources->grainBlurKernelCoarse,
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::prepare_grain_dye_kernel(
        int layer,
        int channel,
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        if (layer < 0 || layer >= 3 || channel < 0 || channel >= 3) {
            outError = "invalid grain dye kernel slot";
            _state->set_failure(
                "prepare_grain_dye_kernel",
                "CUDA grain dye-cloud kernel upload failed");
            return false;
        }
        return prepare_gaussian_kernel_slot(
            _state->resources->grainDyeKernel[layer][channel],
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::prepare_halation_kernel(
        int channel,
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        if (channel < 0 || channel >= 3) {
            outError = "invalid halation kernel slot";
            _state->set_failure(
                "prepare_halation_kernel",
                "CUDA halation kernel upload failed");
            return false;
        }
        return prepare_halation_kernel_slot(
            _state->resources->halationKernel[channel],
            sigma,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::prepare_halation_scatter_kernel(
        int channel,
        float sigma,
        void* cudaStreamOpaque,
        std::string& outError) {
        if (!_state || !_state->resources) {
            outError = "prepared frame is not active";
            return false;
        }
        if (channel < 0 || channel >= 3) {
            outError = "invalid halation scatter kernel slot";
            _state->set_failure(
                "prepare_halation_scatter_kernel",
                "CUDA halation scatter kernel upload failed");
            return false;
        }
        return prepare_halation_kernel_slot(
            _state->resources->halationScatterKernel[channel],
            sigma,
            cudaStreamOpaque,
            outError);
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

    bool Root::PreparedCudaFrame::validate_density_primitives(
        const WorkingState& workingState,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }
        return JuicerCuda::validate_density_primitives(
            *_state->resources,
            workingState,
            cudaStreamOpaque,
            outError);
    }

    bool Root::PreparedCudaFrame::validate_print_primitives(
        const WorkingState& workingState,
        const Print::Runtime& printRuntime,
        const Print::Params& printParams,
        float midgrayFactor,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            outError = "prepared frame is not active";
            return false;
        }
        return JuicerCuda::validate_print_primitives(
            *_state->resources,
            workingState,
            printRuntime,
            printParams,
            midgrayFactor,
            cudaStreamOpaque,
            outError);
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

        const State::AutoExposureFrameWorkspace& workspace = _state->autoExposureWorkspace;
        view.scratch = workspace.scratch;
        view.deviceState = workspace.deviceState;
        view.weightsWidth = workspace.weightsWidth;
        view.weightsHeight = workspace.weightsHeight;
        view.keyHash = workspace.keyHash;
        const bool histogramReady =
            workspace.method != Spektrafilm::AutoExposureMethod::Median ||
            (view.scratch.maxYBits && view.scratch.histogram);
        const bool partialsReady =
            workspace.method == Spektrafilm::AutoExposureMethod::Median ||
            (view.scratch.partialsA && view.scratch.partialsB && view.scratch.partialCapacity > 0);
        const bool weightsReady =
            workspace.method != Spektrafilm::AutoExposureMethod::CenterWeighted ||
            (view.scratch.weightsX && view.scratch.weightsY);
        view.active =
            workspace.active &&
            histogramReady &&
            partialsReady &&
            weightsReady &&
            view.deviceState.exposureScale &&
            view.deviceState.autoEV &&
            view.deviceState.valid;
        return view;
    }

    Root::PreparedCudaFrame::UploadTraceView Root::PreparedCudaFrame::upload_trace_view() const noexcept {
        UploadTraceView view{};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return view;
        }

        std::lock_guard<std::mutex> lock(_state->resources->m);
        view.uploadedBuildCounter = _state->resources->uploadedBuildCounter;
        view.printIllumBuildCounter = _state->resources->printIllumBuildCounter;
        view.printIllumCoreHash = _state->resources->printIllumCoreHash;
        view.printIllumNeutralFilterHash = _state->resources->printIllumNeutralFilterHash;
        view.printIllumYShiftSteps = _state->resources->printIllumYShiftSteps;
        view.printIllumMShiftSteps = _state->resources->printIllumMShiftSteps;
        view.printIllumCShiftSteps = _state->resources->printIllumCShiftSteps;
        view.printPreflashValid = _state->resources->printPreflashValid;
        view.printPreflashKeyHash = _state->resources->printPreflashKeyHash;
        view.directUploadCounter = _state->resources->directUploadCounter;
        view.finalSensitivityHash = _state->resources->directFinalSensitivityHash;
        view.densityCurvesHash = _state->resources->directDensityCurvesHash;
        view.densityBoundsHash = _state->resources->directDensityBoundsHash;
        view.scannerDescriptorHash = _state->resources->directScannerDescriptorHash;
        view.active = true;
        return view;
    }

    Root::PreparedCudaFrame::DirectPreparedView Root::PreparedCudaFrame::direct_resources() const noexcept {
        DirectPreparedView view{};
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed ||
            !_state->directFilmRawConfig || !_state->directScannerColor) {
            return view;
        }

        const JuicerCuda::Resources& resources = *_state->resources;
        view.film.finalSensB = {resources.sensB.x, resources.sensB.y, resources.sensB.n, resources.sensB.domainBegin, resources.sensB.domainEnd};
        view.film.finalSensG = {resources.sensG.x, resources.sensG.y, resources.sensG.n, resources.sensG.domainBegin, resources.sensG.domainEnd};
        view.film.finalSensR = {resources.sensR.x, resources.sensR.y, resources.sensR.n, resources.sensR.domainBegin, resources.sensR.domainEnd};
        view.film.normalizedDensB = {resources.densB.x, resources.densB.y, resources.densB.n, resources.densB.domainBegin, resources.densB.domainEnd};
        view.film.normalizedDensG = {resources.densG.x, resources.densG.y, resources.densG.n, resources.densG.domainBegin, resources.densG.domainEnd};
        view.film.normalizedDensR = {resources.densR.x, resources.densR.y, resources.densR.n, resources.densR.domainBegin, resources.densR.domainEnd};
        view.film.tablesAx = resources.tablesAx;
        view.film.tablesAy = resources.tablesAy;
        view.film.tablesAz = resources.tablesAz;
        view.film.tablesIllum = resources.tablesIllum;
        view.film.tablesK = resources.tablesK;
        std::copy_n(resources.spdSInv, 9, view.film.spdSInv);
        view.film.hanatosLut = resources.hanatosLut;
        view.film.hanatosN = resources.hanatosN;
        view.film.hanatosLutIntegrated = resources.hanatosLutIntegrated;
        view.film.hanatosNIntegrated = resources.hanatosNIntegrated;
        view.film.mallettBasis = resources.mallettBasis;
        view.film.mallettBasisK = resources.mallettBasisK;
        std::copy_n(_state->directFilmRawConfig->inputRGBToXYZ.m, 9, view.film.inputRGBToXYZ);
        std::copy_n(_state->directFilmRawConfig->inputXYZAdapt.m, 9, view.film.inputXYZAdapt);
        view.film.applyInputChromaticAdapt = _state->directFilmRawConfig->applyInputChromaticAdapt ? 1 : 0;
        std::copy_n(resources.refIllumWhiteXYZ, 3, view.film.refIllumWhiteXYZ);
        view.film.finalSensitivityHash = resources.directFinalSensitivityHash;
        view.film.normalizedDensityCurvesHash = resources.directDensityCurvesHash;
        view.scanMedium = &resources.scanNegative;
        view.scanLut = &resources.scanNegativeLut;
        view.scannerColor = _state->directScannerColor;
        view.densityBoundsHash = resources.directDensityBoundsHash;
        view.scannerDescriptorHash = resources.directScannerDescriptorHash;
        view.selectedMethod = resources.directSelectedMethod;
        view.active = view.scanMedium && view.scanLut->canonical_ready() &&
                      view.densityBoundsHash != 0 && view.scannerDescriptorHash != 0;
        return view;
    }

    void Root::PreparedCudaFrame::mark_auto_exposure_weights_built(
        const AutoExposureWeightsExtent& weights) noexcept {
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return;
        }
        State::AutoExposureFrameWorkspace& workspace = _state->autoExposureWorkspace;
        workspace.weightsWidth = weights.width;
        workspace.weightsHeight = weights.height;
    }

    void Root::PreparedCudaFrame::mark_auto_exposure_metered(
        const AutoExposureMeteredResult& result) noexcept {
        if (!_state || !_state->resources || !_state->transaction.active || _state->transaction.committed) {
            return;
        }
        State::AutoExposureFrameWorkspace& workspace = _state->autoExposureWorkspace;
        workspace.keyHash = result.keyHash;
    }

    Root::PreparedCudaFrame::SpatialDirScratchView Root::PreparedCudaFrame::spatial_dir_scratch(
        const WorkspaceLeaseMarker& workspace) const noexcept {
        SpatialDirScratchView view{};
        if (!workspace_marker_matches_current_frame(workspace) || !workspace._request.needSpatialDir) {
            return view;
        }
        if (!_state->scratchWorkspace.retainedLeaseActive && !_state->scratchWorkspace.overflowActive) {
            return view;
        }

        const JuicerCuda::Resources::DeviceSpatialDirScratch& scratch =
            _state->scratchWorkspace.overflowActive
                ? _state->scratchWorkspace.spatialDir
                : _state->resources->spatialDirScratch;
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
        if (!_state->scratchWorkspace.retainedLeaseActive && !_state->scratchWorkspace.overflowActive) {
            return view;
        }

        const JuicerCuda::Resources::DeviceOpticsScratch& scratch =
            _state->scratchWorkspace.overflowActive
                ? _state->scratchWorkspace.optics
                : _state->resources->scannerScratch;
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
        if (_state->scratchWorkspace.overflowActive) {
            _state->scratchWorkspace.optics.gateMaskHash = gateMaskHash;
            return;
        }
        if (!_state->scratchWorkspace.retainedLeaseActive) {
            return;
        }
        _state->resources->scannerScratch.gateMaskHash = gateMaskHash;
    }

    void Root::PreparedCudaFrame::record_use(void* cudaStreamOpaque) noexcept {
        try {
            if (!_state ||
                !_state->resources ||
                !_state->transaction.active ||
                _state->transaction.committed ||
                _state->frameUseEventSubmitted) {
                return;
            }
            std::string ignoredError;
            (void)_state->submit_frame_use_event(cudaStreamOpaque, ignoredError);
        }
        catch (...) {
            JuicerLogging::discard_current_exception();
        }
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
        try {
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
            release_cuda_context_resource_owners();
            release_process_host_services();
        }
        catch (...) {
            JuicerLogging::discard_current_exception();
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
        frame._state->remember_stream(cudaStreamOpaque);
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
            // SF_TEMP_BRIDGE_StaticNoiseUpload owner=Phase3A direct-preparation audit:
            // allowed=existing blocked legacy prepare_cuda_frame only; hash_owner=none;
            // output_impact=no direct Phase3A pixels; removal gate=Phase3C before direct pixels.
            frame._state->set_failure(
                "ensure_grain_static_assets_uploaded",
                "CUDA grain-static asset upload failed");
            frame.abort("prepared_frame_grain_static_upload_failed");
            return frame;
        }
        if (!frame._state->allocate_scan_error_stage(outError)) {
            frame._state->set_failure(
                "allocate_scan_error_stage",
                "CUDA scan error staging allocation failed");
            frame.abort("prepared_frame_scan_error_flag_failed");
            return frame;
        }
        if (autoExposureBufferRequest.enabled &&
            !frame._state->allocate_auto_exposure_workspace(
                autoExposureBufferRequest.descriptor,
                outError)) {
            frame._state->set_failure(
                "allocate_auto_exposure_workspace",
                "CUDA auto-exposure workspace allocation failed");
            frame.abort("prepared_frame_auto_exposure_failed");
            return frame;
        }
        return frame;
    }

    Root::PreparedCudaFrame Root::prepare_cuda_frame(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const DirectCudaPreparationRequest& request,
        const AutoExposureBufferRequest& autoExposureBufferRequest,
        void* cudaStreamOpaque,
        std::string& outError) {
        outError.clear();
        std::unique_ptr<PreparedCudaFrame::State> state;
        try {
            state = std::make_unique<PreparedCudaFrame::State>();
        } catch (...) {
            outError = "failed to allocate direct CUDA prepared frame";
            return PreparedCudaFrame{};
        }

        PreparedCudaFrame frame(std::move(state));
        frame._state->root = this;
        frame._state->remember_stream(cudaStreamOpaque);
        frame._state->set_failure("prepare_cuda_frame_direct", "CUDA direct prepared frame failed");
        if (!begin_submission(frame._state->transaction, snapshot, outError)) {
            frame._state->set_failure("begin_submission_direct", "direct begin_submission failed");
            return frame;
        }
        if (!resolve_cuda_frame_resources(
                deviceContextKey,
                frame._state->transaction.snapshot.contextEpoch,
                frame._state->resourceOwner,
                frame._state->resources,
                outError)) {
            frame._state->set_failure("resolve_cuda_direct_resources", "CUDA direct resource acquisition failed");
            frame.abort("direct_prepared_frame_resource_acquire_failed");
            return frame;
        }
        if (!acquire_submission_plan(frame._state->transaction, outError)) {
            frame._state->set_failure("acquire_direct_plan", "direct acquire_plan failed");
            frame.abort("direct_prepared_frame_acquire_failed");
            return frame;
        }

        JuicerCuda::DirectResourcePreparation directRequest{};
        directRequest.recipe = request.recipe;
        directRequest.exposureTables = request.exposureTables;
        directRequest.spdSInv = request.spdSInv;
        directRequest.filmRawConfig = request.filmRawConfig;
        directRequest.scannerTables = request.scannerTables;
        directRequest.scannerColor = request.scannerColor;
        directRequest.scannerLutDescriptor = request.scannerLutDescriptor;
        if (!JuicerCuda::prepare_direct_resources(
                *frame._state->resources,
                directRequest,
                cudaStreamOpaque,
                outError)) {
            frame._state->set_failure("prepare_direct_resources", "CUDA direct resource preparation failed");
            frame.abort("direct_prepared_frame_upload_failed");
            return frame;
        }
        frame._state->directFilmRawConfig = request.filmRawConfig;
        frame._state->directScannerColor = request.scannerColor;
        if (!frame._state->allocate_scan_error_stage(outError)) {
            frame._state->set_failure("allocate_direct_scan_error_stage", "CUDA direct scan error staging allocation failed");
            frame.abort("direct_prepared_frame_scan_error_flag_failed");
            return frame;
        }
        if (autoExposureBufferRequest.enabled &&
            !frame._state->allocate_auto_exposure_workspace(autoExposureBufferRequest.descriptor, outError)) {
            frame._state->set_failure("allocate_direct_auto_exposure_workspace", "CUDA direct auto-exposure workspace allocation failed");
            frame.abort("direct_prepared_frame_auto_exposure_failed");
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
            JuicerLogging::discard_current_exception();
            try {
                outError = "registry-wide context retire threw";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
#else
        return true;
#endif
    }

    void Root::release_cuda_context_resource_owners() noexcept {
#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)
        try {
            ContextCudaResourceMap frameResources;
            ContextCudaResourceMap grainStaticResources;
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            frameResources.swap(_cudaResourcesByContext);
            grainStaticResources.swap(_cudaGrainStaticByContext);
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
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
            JuicerLogging::discard_current_exception();
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
            JuicerLogging::discard_current_exception();
        }
    }

    void Root::resume_frame_preparation() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            if (_activeShutdowns == 0 && !_shutdownRetireBlocked) {
                _acceptFramePreparation = true;
            }
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    void Root::wait_for_frame_preparation() noexcept {
        try {
            std::unique_lock<std::mutex> lock(_framePreparationMutex);
            _framePreparationCv.wait(lock, [this]() {
                return _activeFramePreparations == 0;
            });
        } catch (...) {
            JuicerLogging::discard_current_exception();
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
            JuicerLogging::discard_current_exception();
        }
    }

} // namespace JuicerProcess
