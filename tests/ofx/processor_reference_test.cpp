#include "../ffi/spatial_optics_fixture.h"
#include "../ffi/print_illuminant_fixture.h"
#include "../ffi/exposure_fixture.h"
#include "../ffi/prepared_boundary.h"
#include "../ffi/prepared_product_fixture.h"
#include "../ffi/prepared_frame_fixture.h"
#include "juicer_cuda_call_scope.h"
#include "juicer_cuda_api.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "render_assertions.h"
#include "nlohmann/json.hpp"

#include "Hash.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "JuicerState.h"
#include "ProcessRoot.h"
#include "juicer_cuda_owner.h"
#include "ColorTransforms.h"
#include "mainProcessing.h"
#include "ofxsSupportPrivate.h"

#if defined(JUICER_PREPARED_BOUNDARY_TEST)
#include "Cuda/JuicerCudaExecutor.h"
#endif

#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
#include <cstdarg>
#include <cstring>
#include "Cuda/JuicerCudaExecutor.h"

#include "RustAssetBridge.h"
#include "RustEffectsFrameBridge.h"
#include "RustGrainFrameBridge.h"
#include "../ffi/juicer_test_api.h"

namespace JuicerAssets::NoiseTest {
    enum class EarlierFailure : unsigned char {
        None,
        Missing,
        Preflight,
        Focused,
        Print
    };
    struct Observation {
        std::array<unsigned, 3> operations{};
        EarlierFailure earlier = EarlierFailure::None;
        std::uint32_t sourceFailure = FJ_STATUS_SUCCESS;
        unsigned exceptionDrops = 0;
        bool exceptionReleasedAfterRecovery = false;
        bool closeBeforeAdmission = false;
        bool corrupt = false;
        bool gateViolation = false;
        bool noiseStep = false;
        bool noiseStepGated = false;
        bool recoveryFinished = false;
        bool releaseAfterRecovery = false;
        FjStatus result{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
    };
    thread_local Observation* observation = nullptr;
    class SourceError final : public std::exception {
    public:
        ~SourceError() override {
            if (observation && _lifetime.use_count() == 1) {
                ++observation->exceptionDrops;
                observation->gateViolation = observation->gateViolation || JuicerCuda::calling_thread_native_gate_active();
                observation->exceptionReleasedAfterRecovery = observation->recoveryFinished;
            }
        }
        const char* what() const noexcept override {
            return "noise source fixture standard exception";
        }

    private:
        // Count final exception storage, including platforms that copy a throw temporary.
        std::shared_ptr<const unsigned char> _lifetime = std::make_shared<const unsigned char>(0);
    };
    void observe(Operation operation) noexcept {
        if (observation) {
            ++observation->operations[static_cast<unsigned>(operation)];
            observation->gateViolation = observation->gateViolation || JuicerCuda::calling_thread_native_gate_active();
            if (operation == Operation::Release) {
                observation->releaseAfterRecovery = observation->recoveryFinished;
            }
        }
    }
    void view(FjStaticNoise& view) {
        if (observation && observation->closeBeforeAdmission) {
            if (fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr).category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("fixture close before admission failed");
            }
        }
        if (observation && observation->sourceFailure != FJ_STATUS_SUCCESS) {
            if (observation->sourceFailure == FJ_STATUS_ALLOCATION_FAILURE) {
                throw std::bad_alloc{};
            }
            if (observation->sourceFailure == FJ_STATUS_INTERNAL_FAILURE) {
                throw SourceError{};
            }
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, {observation->sourceFailure, FJ_API_NONE, 0}, "noise source fixture failure");
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        if (observation && observation->corrupt) {
            view.stbn.count = 1;
        }
    }
    void projection_step() noexcept {
        if (observation) {
            observation->noiseStep = true;
            observation->noiseStepGated = JuicerCuda::calling_thread_native_gate_active();
        }
    }
    void outcome(FjStatus status) noexcept {
        if (observation) {
            observation->result = status;
        }
    }
} // namespace JuicerAssets::NoiseTest

namespace JuicerCuda::ExecutorTest {
    thread_local std::optional<WorkspaceHandoff> workspaceHandoff;
    void observe_workspace_handoff(WorkspaceHandoff handoff) noexcept {
        workspaceHandoff = handoff;
    }
    namespace {
        thread_local WorkspaceTransitionObservation s_workspace;
        struct WorkspaceRejectionState {
            unsigned count = 0;
            WorkspaceRejection kind = WorkspaceRejection::Capacity;
            std::optional<unsigned> candidate;
            bool armed = false;
        };
        thread_local WorkspaceRejectionState s_workspaceRejection;
    } // namespace
    void reset_workspace_transitions() noexcept {
        s_workspace = {};
        s_workspaceRejection = {};
        workspaceHandoff.reset();
    }
    WorkspaceTransitionObservation workspace_transitions() noexcept {
        return s_workspace;
    }
    bool arm_workspace_rejection(unsigned count, WorkspaceRejection kind) noexcept {
        if (count > 6 || (kind != WorkspaceRejection::Capacity && kind != WorkspaceRejection::InvalidValue && kind != WorkspaceRejection::ContextLoss) ||
            s_workspaceRejection.armed) {
            return false;
        }
        s_workspaceRejection = {count, kind, std::nullopt, true};
        return true;
    }
    void begin_workspace_candidate(unsigned index, const ResourceManager::WorkspaceLayout& layout) noexcept {
        if (index < s_workspace.candidates.size()) {
            s_workspace.candidateCount = index + 1;
            s_workspace.candidates[index].layout = layout;
            s_workspaceRejection.candidate = index;
        }
    }
    void end_workspace_candidate(bool success, const Failure& failure) noexcept {
        if (s_workspaceRejection.candidate) {
            auto& candidate = s_workspace.candidates[*s_workspaceRejection.candidate];
            candidate.succeeded = success;
            candidate.result = success ? FjStatus{FJ_STATUS_SUCCESS, FJ_API_NONE, 0} : failure.status;
            s_workspaceRejection.candidate.reset();
        }
    }
    bool inject_workspace_candidate_failure(Failure& failure) {
        const auto index = s_workspaceRejection.candidate;
        if (!index) {
            return false;
        }
        ++s_workspace.candidates[*index].backendCalls;
        if (!s_workspaceRejection.armed || *index >= s_workspaceRejection.count) {
            return false;
        }
        switch (s_workspaceRejection.kind) {
            case WorkspaceRejection::Capacity:
                failure = {{FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "workspace candidate capacity rejection"};
                break;
            case WorkspaceRejection::InvalidValue:
                failure = {runtime_failure_status(cudaErrorInvalidValue), "workspace candidate invalid-value rejection"};
                break;
            case WorkspaceRejection::ContextLoss:
                failure = {runtime_failure_status(cudaErrorContextIsDestroyed), "workspace candidate context-loss rejection"};
                break;
        }
        return true;
    }
    void observe_workspace_requested(const std::optional<ResourceManager::WorkspaceLayout>& layout) noexcept {
        s_workspace.requested = layout;
    }
    void observe_workspace_resource_acquired() noexcept {
        ++s_workspace.resourcesAcquired;
    }
    void observe_workspace_admitted(const ResourceManager::WorkspaceLayout& layout) noexcept {
        s_workspace.admitted = layout;
    }
    void observe_workspace_final(const ResourceManager::WorkspaceLayout& layout) noexcept {
        s_workspace.finalDevelopment = layout;
    }
    void observe_workspace_post_frame(const std::optional<ResourceManager::WorkspaceLayout>& layout) noexcept {
        s_workspace.postFrame = layout;
    }
    void observe_workspace_finish() noexcept {
        ++s_workspace.finishes;
    }
    void observe_workspace_dir_build(WorkspaceCacheChannels channels) noexcept {
        s_workspace.sourceCacheChannels = channels.source;
        s_workspace.finalCacheChannels = channels.final;
    }
    void observe_workspace_capture() noexcept {
        s_workspace.captureEnqueued = true;
    }
    void observe_workspace_scanner() noexcept {
        s_workspace.scannerEnqueued = true;
    }
    void observe_workspace_bindings(JuicerProcess::Root::PreparedCudaFrame& frame,
                                    const JuicerProcess::Root::PreparedCudaFrame::WorkspaceLeaseMarker& marker) noexcept {
        if (!s_workspace.admitted) {
            return;
        }
        const auto scanner = frame.scanner_workspace(marker);
        const auto dir = frame.spatial_dir_scratch(marker);
        const auto grain = frame.visual_grain_workspace(marker);
        const auto& layout = *s_workspace.admitted;
        s_workspace.bindingsObserved = true;
        s_workspace.scannerMapping = layout.scanner_storage() == ResourceManager::ScannerStorage::Absent
                                         ? !scanner.active
                                     : layout.scanner_storage() == ResourceManager::ScannerStorage::DirFiltered
                                         ? scanner.active && scanner.rgbR == dir.filteredCorrectionC && scanner.rgbG == dir.filteredCorrectionM && scanner.rgbB == dir.filteredCorrectionY
                                         : scanner.active && !scanner.rgbAliasedFromSpatialDirFiltered;
        const bool raw = layout.grain_storage() == ResourceManager::GrainStorage::DirRaw;
        s_workspace.grainMapping = layout.grain_storage() == ResourceManager::GrainStorage::Absent
                                       ? !grain.active
                                       : grain.active && grain.deltaAccum == (raw ? dir.rawCorrectionY : scanner.aux) &&
                                             grain.layerWork == (layout.grain_uses_layers() ? raw ? dir.rawCorrectionM : scanner.grainTmp : nullptr) &&
                                             grain.sharedDelta == (layout.grain_uses_shared() ? raw ? dir.rawCorrectionC : scanner.grainTmpShared : nullptr) &&
                                             (!raw || (!scanner.aux && !scanner.grainTmp && !scanner.grainTmpShared));
    }
    enum class Event : std::uint8_t {
        Injected,
        Message,
        Classified,
        FrameAborted,
        RecoveryStarted,
        RecoveryEnded,
        FatalMapped
    };
    enum class Delivery : std::uint8_t {
        Success,
        Failure,
        Throw
    };
    struct FailureObservation {
        std::array<Event, 8> events{};
        std::size_t count = 0;
        const char* diagnostic = nullptr;
        FjStatus status{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        const char* stage = nullptr;
        Delivery delivery = Delivery::Success;
        JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult scanResult = JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Failed;
        bool overflow = false;
        bool failureStageMatches = false;
        bool failureStatusMatches = false;
        bool nativeRecoveryPending = false;
        bool adapterRecoveryPending = false;
        bool messageMatches = false;
        bool armed = true;
        FjStatus classifiedStatus{};
        bool failureDiagnosticMatches = false;

        void record(Event event) noexcept {
            if (count < events.size()) {
                events[count++] = event;
            } else {
                overflow = true;
            }
        }
    };
    thread_local FailureObservation* observation = nullptr;

    namespace {
        thread_local DirFrameObservation s_dirFrameObservation;
        thread_local GrainFrameObservation s_grainFrameObservation;
        thread_local EffectsFrameObservation s_effectsFrameObservation;
    } // namespace
    void reset_effects_frame_observation() noexcept {
        s_effectsFrameObservation = {};
    }
    EffectsFrameObservation effects_frame_observation() noexcept {
        return s_effectsFrameObservation;
    }
    void observe_effects_frame(EffectsFrameObservation observation) noexcept {
        observation.count = std::min(s_effectsFrameObservation.count + 1, 2u);
        s_effectsFrameObservation = observation;
    }
    void reset_grain_frame_observation() noexcept {
        s_grainFrameObservation = {};
    }
    GrainFrameObservation grain_frame_observation() noexcept {
        return s_grainFrameObservation;
    }
    void observe_grain_frame(GrainFrameObservation observation) noexcept {
        observation.count = std::min(s_grainFrameObservation.count + 1, 2u);
        s_grainFrameObservation = observation;
    }
    void reset_dir_frame_observation() noexcept {
        s_dirFrameObservation = {};
    }
    DirFrameObservation dir_frame_observation() noexcept {
        return s_dirFrameObservation;
    }
    void observe_dir_frame(Spektrafilm::ScanRoute route, float pixelSizeUm, Spektrafilm::DirFrameExtent renderExtent, Spektrafilm::DirFrameExtent fullFrameExtent, bool success, const Spektrafilm::SpatialDirDescriptor& descriptor) noexcept {
        auto& out = s_dirFrameObservation;
        out.count = std::min(out.count + 1, 2u);
        out.route = route;
        out.pixelSizeBits = std::bit_cast<std::uint32_t>(pixelSizeUm);
        out.renderExtent = renderExtent;
        out.fullFrameExtent = fullFrameExtent;
        out.success = success;
        out.descriptor = descriptor;
    }

    struct ResourceFailures {
        std::optional<Failure> stbn;
        std::optional<Failure> wang;
        cudaError_t eventCreate = cudaSuccess;
        cudaError_t eventRecord = cudaSuccess;
        unsigned lutPosition = 0;
        unsigned lutAttempts = 0;
        int stbnAttempts = 0;
        int wangAttempts = 0;
        int injections = 0;

        void record_injection() noexcept {
            if (injections++ == 0 && observation) {
                observation->record(Event::Injected);
            }
        }
    };
    thread_local ResourceFailures* resourceFailures = nullptr;

    bool inject_grain_upload_failure(const char* label, Failure& failure) {
        if (!resourceFailures) {
            return false;
        }
        if (resourceFailures->lutPosition && std::string_view(label).starts_with("focused scan PCHIP")) {
            if (++resourceFailures->lutAttempts == resourceFailures->lutPosition) {
                failure = {{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "scanner LUT upload fixture failure"};
                resourceFailures->record_injection();
                return true;
            }
        }
        const std::optional<Failure>* injected = nullptr;
        if (std::strcmp(label, "STBN") == 0) {
            ++resourceFailures->stbnAttempts;
            injected = &resourceFailures->stbn;
        } else if (std::strcmp(label, "Wang.tiles") == 0) {
            ++resourceFailures->wangAttempts;
            injected = &resourceFailures->wang;
        }
        if (!injected || !injected->has_value()) {
            return false;
        }
        failure = **injected;
        resourceFailures->record_injection();
        return true;
    }

    bool inject_defect_event_create(cudaError_t& error) noexcept {
        if (!resourceFailures || resourceFailures->eventCreate == cudaSuccess) {
            return false;
        }
        error = resourceFailures->eventCreate;
        resourceFailures->eventCreate = cudaSuccess;
        resourceFailures->record_injection();
        return true;
    }

    bool inject_defect_event_record(cudaError_t& error) noexcept {
        if (!resourceFailures || resourceFailures->eventRecord == cudaSuccess) {
            return false;
        }
        error = resourceFailures->eventRecord;
        resourceFailures->eventRecord = cudaSuccess;
        resourceFailures->record_injection();
        return true;
    }

    bool inject_scan_error(Failure& failure, JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult& result) {
        if (!observation || !observation->armed) {
            return false;
        }
        observation->armed = false;
        failure.diagnostic = observation->diagnostic;
        failure.status = observation->status;
        result = observation->scanResult;
        observation->record(Event::Injected);
        return true;
    }
    void observe_classification(const char* stage, const Failure& failure, bool recoveryPending) noexcept {
        s_workspace.classified = failure.status;
        s_workspace.recoveryPending = recoveryPending;
        if (observation) {
            observation->failureStageMatches = std::strcmp(stage, observation->stage) == 0;
            observation->failureStatusMatches = failure.status.category == observation->status.category && failure.status.api == observation->status.api && failure.status.native_code == observation->status.native_code;
            observation->nativeRecoveryPending = recoveryPending;
            observation->classifiedStatus = failure.status;
            observation->failureDiagnosticMatches = failure.diagnostic == observation->diagnostic;
            observation->record(Event::Classified);
        }
    }
    void observe_frame_abort() noexcept {
        ++s_workspace.aborts;
        if (observation) {
            observation->record(Event::FrameAborted);
        }
    }
    void observe_recovery_start(bool pending) noexcept {
        if (observation) {
            observation->adapterRecoveryPending = pending;
            observation->record(Event::RecoveryStarted);
        }
    }
    void observe_recovery_end() noexcept {
        if (JuicerAssets::NoiseTest::observation) {
            JuicerAssets::NoiseTest::observation->recoveryFinished = true;
        }
        if (observation) {
            observation->record(Event::RecoveryEnded);
        }
    }
} // namespace JuicerCuda::ExecutorTest
#endif

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray&) {}

namespace {
    struct Properties {
        std::map<std::string, std::vector<std::string>> strings;
        std::map<std::string, std::vector<int>> ints;
        std::map<std::string, std::vector<double>> doubles;
        std::map<std::string, std::vector<void*>> pointers;
    };

    template <typename T>
    OfxStatus get_value(
        const std::map<std::string, std::vector<T>>& values,
        const char* name,
        int index,
        T* output) {
        if (!name || !output || index < 0) {
            return kOfxStatErrBadHandle;
        }
        const auto found = values.find(name);
        if (found == values.end() || static_cast<std::size_t>(index) >= found->second.size()) {
            std::cerr << "missing OFX property: " << name << "[" << index << "]\n";
            return kOfxStatErrUnknown;
        }
        *output = found->second[static_cast<std::size_t>(index)];
        return kOfxStatOK;
    }

    OfxStatus set_pointer(OfxPropertySetHandle handle, const char* name, int index, void* value) {
        if (!handle || !name || index < 0) {
            return kOfxStatErrBadHandle;
        }
        auto& items = reinterpret_cast<Properties*>(handle)->pointers[name];
        if (items.size() <= static_cast<std::size_t>(index)) {
            items.resize(static_cast<std::size_t>(index) + 1);
        }
        items[static_cast<std::size_t>(index)] = value;
        return kOfxStatOK;
    }

    OfxStatus get_pointer(OfxPropertySetHandle handle, const char* name, int index, void** output) {
        return handle
                   ? get_value(reinterpret_cast<Properties*>(handle)->pointers, name, index, output)
                   : kOfxStatErrBadHandle;
    }

    OfxStatus get_string(OfxPropertySetHandle handle, const char* name, int index, char** output) {
        if (!handle || !output) {
            return kOfxStatErrBadHandle;
        }
        std::string value;
        const OfxStatus status = get_value(reinterpret_cast<Properties*>(handle)->strings, name, index, &value);
        if (status != kOfxStatOK) {
            return status;
        }
        *output = const_cast<char*>(reinterpret_cast<Properties*>(handle)->strings[name][static_cast<std::size_t>(index)].c_str());
        return kOfxStatOK;
    }

    OfxStatus get_double(OfxPropertySetHandle handle, const char* name, int index, double* output) {
        return handle
                   ? get_value(reinterpret_cast<Properties*>(handle)->doubles, name, index, output)
                   : kOfxStatErrBadHandle;
    }

    OfxStatus get_int(OfxPropertySetHandle handle, const char* name, int index, int* output) {
        return handle
                   ? get_value(reinterpret_cast<Properties*>(handle)->ints, name, index, output)
                   : kOfxStatErrBadHandle;
    }

    OfxStatus effect_properties(OfxImageEffectHandle handle, OfxPropertySetHandle* output) {
        *output = reinterpret_cast<OfxPropertySetHandle>(handle);
        return kOfxStatOK;
    }

    OfxStatus effect_parameters(OfxImageEffectHandle handle, OfxParamSetHandle* output) {
        *output = reinterpret_cast<OfxParamSetHandle>(handle);
        return kOfxStatOK;
    }

    OfxStatus parameter_properties(OfxParamSetHandle handle, OfxPropertySetHandle* output) {
        *output = reinterpret_cast<OfxPropertySetHandle>(handle);
        return kOfxStatOK;
    }

    class ImageLeaseAudit final {
    public:
        void acquire(OfxPropertySetHandle handle, std::string label) {
            auto [position, inserted] =
                _leases.emplace(handle, Lease{std::move(label), 1, 0});
            if (!inserted) {
                position->second.acquisitions += 1;
                set_error(Error::DuplicateAcquisition);
            }
        }

        void release(OfxPropertySetHandle handle) noexcept {
            const auto found = _leases.find(handle);
            if (found == _leases.end()) {
                set_error(Error::UnknownRelease);
                return;
            }
            Lease& lease = found->second;
            lease.releases += 1;
            if (lease.releases > lease.acquisitions) {
                set_error(Error::DuplicateRelease);
            }
        }

        void require_complete(const char* caseName) const {
            if (_error != Error::None) {
                throw std::runtime_error(
                    std::string(caseName) + ": " + error_message());
            }
            if (_leases.size() != 2) {
                throw std::runtime_error(
                    std::string(caseName) +
                    ": expected two acquired image handles");
            }
            for (const auto& [handle, lease] : _leases) {
                (void)handle;
                if (lease.acquisitions != 1 || lease.releases != 1) {
                    throw std::runtime_error(
                        std::string(caseName) +
                        ": image lease was not released exactly once for " +
                        lease.label);
                }
            }
        }

    private:
        enum class Error : std::uint8_t {
            None,
            DuplicateAcquisition,
            UnknownRelease,
            DuplicateRelease
        };

        struct Lease {
            std::string label;
            int acquisitions = 0;
            int releases = 0;
        };

        void set_error(Error error) noexcept {
            if (_error == Error::None) {
                _error = error;
            }
        }

        const char* error_message() const noexcept {
            switch (_error) {
                case Error::DuplicateAcquisition:
                    return "duplicate image acquisition";
                case Error::UnknownRelease:
                    return "unknown image release";
                case Error::DuplicateRelease:
                    return "duplicate image release";
                case Error::None:
                    break;
            }
            return "image lease audit failed";
        }

        std::map<OfxPropertySetHandle, Lease> _leases;
        Error _error = Error::None;
    };

    thread_local ImageLeaseAudit* s_activeImageLeaseAudit = nullptr;

    class ScopedImageLeaseAudit final {
    public:
        explicit ScopedImageLeaseAudit(ImageLeaseAudit& audit) noexcept {
            s_activeImageLeaseAudit = &audit;
        }

        ~ScopedImageLeaseAudit() {
            s_activeImageLeaseAudit = nullptr;
        }

        ScopedImageLeaseAudit(const ScopedImageLeaseAudit&) = delete;
        ScopedImageLeaseAudit& operator=(const ScopedImageLeaseAudit&) = delete;
    };

    OfxStatus release_image(OfxPropertySetHandle handle) {
        if (!s_activeImageLeaseAudit) {
            return kOfxStatErrBadHandle;
        }
        s_activeImageLeaseAudit->release(handle);
        return kOfxStatOK;
    }

    struct AbortObservation {
        std::thread::id thread = std::this_thread::get_id();
        unsigned calls = 0;
        unsigned cancelAt = 0;
        unsigned throwAt = 0;
        bool sameThread = true;
    };
    thread_local AbortObservation* s_abortObservation = nullptr;

    int effect_abort(OfxImageEffectHandle) {
        if (!s_abortObservation) {
            return 0;
        }
        auto& observation = *s_abortObservation;
        ++observation.calls;
        observation.sameThread = observation.sameThread && observation.thread == std::this_thread::get_id();
        if (observation.calls == observation.throwAt) {
            throw std::runtime_error("abort suite exception");
        }
        return observation.calls == observation.cancelAt ? 1 : 0;
    }

#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters) Required OFX message-suite callback signature.
    OfxStatus failure_message(void*, const char* type, const char* id, const char* format, ...) {
        auto& observation = *JuicerCuda::ExecutorTest::observation;
        // Acquiring the real non-reentrant gate proves the adapter released it.
        JuicerCuda::NativeCall call(JuicerCuda::borrowed_owner());
        observation.record(JuicerCuda::ExecutorTest::Event::Message);
        std::vector<char> text(std::strlen(observation.diagnostic) + 1);
        va_list args;
        va_start(args, format);
        const int length = std::vsnprintf(text.data(), text.size(), format, args);
        va_end(args);
        observation.messageMatches = length >= 0 &&
                                     static_cast<std::size_t>(length) < text.size() &&
                                     std::strcmp(type, kOfxMessageError) == 0 &&
                                     std::strcmp(id, "FilmJuicerDeferredCudaFailure") == 0 &&
                                     std::strcmp(text.data(), observation.diagnostic) == 0;
        if (observation.delivery == JuicerCuda::ExecutorTest::Delivery::Throw) {
            throw std::runtime_error("injected host message exception");
        }
        return observation.delivery == JuicerCuda::ExecutorTest::Delivery::Failure
                   ? kOfxStatFailed
                   : kOfxStatOK;
    }
#endif

    class NarrowHost final {
    public:
        NarrowHost() {
            _properties.propSetPointer = set_pointer;
            _properties.propGetPointer = get_pointer;
            _properties.propGetString = get_string;
            _properties.propGetDouble = get_double;
            _properties.propGetInt = get_int;
            _effect.getPropertySet = effect_properties;
            _effect.getParamSet = effect_parameters;
            _effect.clipReleaseImage = release_image;
            _effect.abort = effect_abort;
            _parameters.paramSetGetPropertySet = parameter_properties;
            OFX::Private::gPropSuite = &_properties;
            OFX::Private::gEffectSuite = &_effect;
            OFX::Private::gParamSuite = &_parameters;
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
            _messages.message = failure_message;
            OFX::Private::gMessageSuite = &_messages;
#endif
        }

        ~NarrowHost() {
            OFX::Private::gPropSuite = nullptr;
            OFX::Private::gEffectSuite = nullptr;
            OFX::Private::gParamSuite = nullptr;
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
            OFX::Private::gMessageSuite = nullptr;
#endif
        }

    private:
        OfxPropertySuiteV1 _properties{};
        OfxImageEffectSuiteV1 _effect{};
        OfxParameterSuiteV1 _parameters{};
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
        OfxMessageSuiteV1 _messages{};
#endif
    };

    class NarrowEffect final : public OFX::ImageEffect {
    public:
        explicit NarrowEffect(OfxImageEffectHandle handle)
            : OFX::ImageEffect(handle) {}

        void render(const OFX::RenderArguments&) override {}
    };

    struct ImagePropertiesInput {
        OfxRectI bounds{};
        int components = 0;
        int rowBytes = 0;
    };

    void fill_image_properties(Properties& values, void* deviceData, const ImagePropertiesInput& input) {
        const std::array<int, 4> bounds{{input.bounds.x1, input.bounds.y1, input.bounds.x2, input.bounds.y2}};
        values.ints[kOfxImagePropRowBytes] = {input.rowBytes};
        values.ints[kOfxImagePropBounds] = std::vector<int>(bounds.begin(), bounds.end());
        values.ints[kOfxImagePropRegionOfDefinition] = std::vector<int>(bounds.begin(), bounds.end());
        values.doubles[kOfxImagePropPixelAspectRatio] = {1.0};
        values.doubles[kOfxImageEffectPropRenderScale] = {1.0, 1.0};
        values.strings[kOfxImageEffectPropComponents] = {
            input.components == 4 ? kOfxImageComponentRGBA : kOfxImageComponentRGB};
        values.strings[kOfxImageEffectPropPixelDepth] = {kOfxBitDepthFloat};
        values.strings[kOfxImageEffectPropPreMultiplication] = {kOfxImageOpaque};
        values.strings[kOfxImagePropField] = {kOfxImageFieldNone};
        values.strings[kOfxImagePropUniqueIdentifier] = {"pre-rust-processor-input"};
        values.pointers[kOfxImagePropData] = {deviceData};
    }

    void require_cuda(cudaError_t status, const char* action) {
        if (status != cudaSuccess) {
            throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
        }
    }

    constexpr int kWidth = 7;
    constexpr int kHeight = 5;
    constexpr float kCanary = -8.0f;

    struct Case {
        const char* name = nullptr;
        Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
        int reconstruction = 0;
        int components = 0;
        int originX = 0;
        int originY = 0;
        bool combined = false;
        bool signedZeroGlare = false;
        bool negativeZero = false;
        bool uniform = false;
        int width = kWidth;
        int height = kHeight;
        double time = 37.0;
        double scaleX = 1.0;
        double scaleY = 1.0;
        const std::vector<float>* sourcePixels = nullptr;
        int fullWidth = 0;
        int fullHeight = 0;
    };

    struct DeviceFrame {
        float* source = nullptr;
        float* destination = nullptr;
        cudaStream_t stream = nullptr;
        bool destinationInSource = false;

        ~DeviceFrame() {
            if (stream) {
                (void)cudaStreamSynchronize(stream);
            }
            if (destination && !destinationInSource) {
                (void)cudaFree(destination);
            }
            if (source) {
                (void)cudaFree(source);
            }
            if (stream) {
                (void)cudaStreamDestroy(stream);
            }
        }
    };

    ParamSnapshot parameters_for(const Case& test) {
        ParamSnapshot parameters;
        parameters.scanRoute = test.route;
        if (Spektrafilm::scan_route_metadata(test.route).capturePolarity ==
            Spektrafilm::ProfilePolarity::Positive) {
            parameters.filmProfileKey = "fujifilm_provia_100f";
        }
        parameters.spectralUpsamplingMode = test.reconstruction;
        parameters.cameraAutoExposureEnabled = 0;
        if (!test.combined) {
            parameters.scannerUnsharpMask = {0.0, 0.0};
        }
        parameters.gateWeaveAmount = 0.0;
        if (test.combined) {
            parameters.cameraDiffusion.active = true;
            parameters.grainControls.active = true;
            std::string diagnostic;
            if (!Spektrafilm::build_scatter_halation_controls(
                    ScatterHalationRawControls{true, 1.0, 1.0, 1.0, 1.0},
                    parameters.scatterHalationControls,
                    diagnostic)) {
                throw std::runtime_error(diagnostic);
            }
        }
        if (test.signedZeroGlare) {
            parameters.glareActive = true;
            parameters.glarePercent = 0.3;
            parameters.cameraExposureCompensationEv = test.negativeZero ? -0.0 : 0.0;
        }
        return parameters;
    }

    enum class ExecutionPath : std::uint8_t {
        Processor,
        Direct,
        Boundary,
        State,
        StateContract,
        StateLifetime
    };

    enum class DestinationLayout : std::uint8_t {
        Matching,
        Offset,
        Uncovered,
        DisjointRows
    };

    std::vector<float> render_case(const Case& test, const ParamSnapshot& parameters, InstanceState& state, bool emptyWindow = false, ExecutionPath path = ExecutionPath::Processor, DestinationLayout destinationLayout = DestinationLayout::Matching, [[maybe_unused]] std::uint64_t snapshotId = 1, [[maybe_unused]] JuicerCudaTest::StateLifetimeCase lifetimeCase = JuicerCudaTest::StateLifetimeCase::SuppliedStream) {
        const int width = test.width;
        const int height = test.height;
        const int pitch = width * test.components + 5;
        const std::size_t bytes = static_cast<std::size_t>(pitch * height) * sizeof(float);
        std::vector<float> input(static_cast<std::size_t>(pitch * height), -7.0f);
        const int border = destinationLayout == DestinationLayout::Offset ? 1 : 0;
        const int destinationPitch = pitch + 2 * border * test.components;
        const int destinationHeight = height + 2 * border;
        const std::size_t destinationBytes = static_cast<std::size_t>(destinationPitch * destinationHeight) * sizeof(float);
        std::vector<float> output(static_cast<std::size_t>(destinationPitch * destinationHeight), kCanary);
        std::vector<float> initialDestination(output);
        for (int y = 0; y < height; ++y) {
            const std::size_t rowOffset = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch);
            for (int x = 0; x < width; ++x) {
                const std::size_t pixelOffset = rowOffset +
                                                static_cast<std::size_t>(x) * static_cast<std::size_t>(test.components);
                for (int c = 0; c < test.components; ++c) {
                    input[pixelOffset + static_cast<std::size_t>(c)] =
                        c == 3 ? 0.125f + static_cast<float>((x + y) % 4) * 0.25f
                               : static_cast<float>(((test.uniform ? 0 : x + 3 * y) + 7 * c) % 19) / 19.0f;
                }
            }
        }
        if (test.sourcePixels) {
            const std::size_t rowSamples = static_cast<std::size_t>(width) * static_cast<std::size_t>(test.components);
            if (test.sourcePixels->size() != rowSamples * static_cast<std::size_t>(height)) {
                throw std::runtime_error("accepted capture input shape mismatch");
            }
            for (int y = 0; y < height; ++y) {
                const std::size_t sourceOffset = static_cast<std::size_t>(y) * rowSamples;
                const std::size_t destinationOffset = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch);
                std::copy_n(test.sourcePixels->data() + sourceOffset, rowSamples, input.data() + destinationOffset);
            }
        }
        DeviceFrame device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "create stream");
        device.destinationInSource = destinationLayout == DestinationLayout::DisjointRows;
        require_cuda(cudaMalloc(&device.source, bytes + (device.destinationInSource ? destinationBytes : 0)), "allocate source");
        if (device.destinationInSource) {
            device.destination = device.source + bytes / sizeof(float);
        } else {
            require_cuda(cudaMalloc(&device.destination, destinationBytes), "allocate destination");
        }
        require_cuda(cudaMemcpy(device.source, input.data(), bytes, cudaMemcpyHostToDevice), "copy source");
        require_cuda(cudaMemcpy(device.destination, initialDestination.data(), destinationBytes, cudaMemcpyHostToDevice), "copy destination");

#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        if (path == ExecutionPath::State || path == ExecutionPath::StateContract || path == ExecutionPath::StateLifetime) {
            const FjRect bounds{test.originX, test.originY, test.originX + width, test.originY + height};
            FjFrame frame{};
            frame.source = {reinterpret_cast<std::uintptr_t>(device.source), bounds, pitch * static_cast<std::ptrdiff_t>(sizeof(float)), static_cast<std::uint32_t>(test.components), FJ_DEPTH_FLOAT32};
            frame.destination = frame.source;
            frame.destination.address = reinterpret_cast<std::uintptr_t>(device.destination);
            frame.render_window = bounds;
            frame.full_frame_extent = bounds;
            frame.stream = reinterpret_cast<std::uintptr_t>(device.stream);
            frame.flags = FJ_FRAME_STREAM_PRESENT;
            frame.pixel_size_um = 35'000.0f / static_cast<float>(width);
            frame.time_frames = test.time;
            frame.frame_rate = 24.0;
            frame.session_seed = 0x20260923;
            frame.clip_token = 0x5312;
            std::string diagnostic;
            if (!JuicerCudaTest::execute_prepared_state(JuicerProcess::root().assets().borrow_rust_assets(), parameters, frame, 0x641207 + static_cast<int>(test.route), path == ExecutionPath::StateContract, path == ExecutionPath::StateLifetime, diagnostic, lifetimeCase)) {
                throw std::runtime_error(std::string(test.name) + ": held state: " + diagnostic);
            }
        } else
#endif
        {
            Properties effectProperties;
            effectProperties.strings[kOfxImageEffectPropContext] = {kOfxImageEffectContextFilter};
            Properties sourceProperties;
            Properties destinationProperties;
            const OfxRectI bounds{test.originX, test.originY, test.originX + width, test.originY + height};
            const ImagePropertiesInput imageInput{bounds, test.components, pitch * static_cast<int>(sizeof(float))};
            fill_image_properties(sourceProperties, device.source, imageInput);
            const OfxRectI destinationBounds{bounds.x1 - border, bounds.y1 - border, bounds.x2 + border - (destinationLayout == DestinationLayout::Uncovered ? 1 : 0), bounds.y2 + border};
            fill_image_properties(destinationProperties, device.destination, {destinationBounds, test.components, destinationPitch * static_cast<int>(sizeof(float))});
            if (test.fullWidth > 0 && test.fullHeight > 0) {
                sourceProperties.ints[kOfxImagePropRegionOfDefinition] = {0, 0, test.fullWidth, test.fullHeight};
                destinationProperties.ints[kOfxImagePropRegionOfDefinition] = {0, 0, test.fullWidth, test.fullHeight};
            }
            sourceProperties.doubles[kOfxImageEffectPropRenderScale] = {test.scaleX, test.scaleY};
            destinationProperties.doubles[kOfxImageEffectPropRenderScale] = {test.scaleX, test.scaleY};
            const OfxPropertySetHandle sourceHandle =
                reinterpret_cast<OfxPropertySetHandle>(&sourceProperties);
            const OfxPropertySetHandle destinationHandle =
                reinterpret_cast<OfxPropertySetHandle>(&destinationProperties);
            ImageLeaseAudit imageLeaseAudit;
            imageLeaseAudit.acquire(sourceHandle, "source");
            imageLeaseAudit.acquire(destinationHandle, "destination");
            {
                ScopedImageLeaseAudit activeImageLeaseAudit(imageLeaseAudit);
                NarrowEffect effect(reinterpret_cast<OfxImageEffectHandle>(&effectProperties));
                OFX::Image sourceImage(sourceHandle);
                OFX::Image destinationImage(destinationHandle);
                {
                    std::lock_guard<std::mutex> lock(state.pending.m);
                    state.pending.value = PendingParamsState::Valid{parameters, hash_params(parameters)};
                }
                PendingRenderAdmissionResult admitted = admit_pending_render_state(state);
#if defined(JUICER_NOISE_TEST_HOOK)
                if (JuicerAssets::NoiseTest::observation) {
                    const auto earlier = JuicerAssets::NoiseTest::observation->earlier;
                    const auto change = [&](auto& published) {
                        if (!published || earlier == JuicerAssets::NoiseTest::EarlierFailure::None || earlier == JuicerAssets::NoiseTest::EarlierFailure::Missing) {
                            return;
                        }
                        FocusedRenderStateBuildProduct product;
                        std::string diagnostic;
                        const bool print = Spektrafilm::scan_route_is_print(parameters.scanRoute);
                        if (!(print ? build_print_render_state_product(parameters, product, diagnostic) : build_direct_render_state_product(parameters, product, diagnostic))) {
                            throw std::runtime_error("failure fixture preparation: " + diagnostic);
                        }
                        auto copy = std::make_shared<std::remove_const_t<typename std::remove_reference_t<decltype(published)>::element_type>>();
                        copy->recipe = std::move(product.recipe);
                        copy->payload = std::move(product.payload);
                        copy->buildCounter = published->buildCounter;
                        if (earlier == JuicerAssets::NoiseTest::EarlierFailure::Preflight) {
                            copy->recipe.spatialOptics.cameraLensBlur.sigmaUm = 1;
                            copy->recipe.spatialOptics.cameraLensBlur.hash = 1;
                        }
                        if (earlier == JuicerAssets::NoiseTest::EarlierFailure::Focused) {
                            copy->recipe.filmRaw.rgbToRawMethod = Spektrafilm::RgbToRawMethod::Mallett2019;
                        }
                        if (earlier == JuicerAssets::NoiseTest::EarlierFailure::Print) {
                            copy->recipe.print.develop.densityCurves.clear();
                        }
                        published = std::move(copy);
                    };
                    change(admitted.directState);
                    change(admitted.printState);
                }
#endif
                const bool print = Spektrafilm::scan_route_is_print(test.route);
                if (admitted.status != (print ? PendingRenderAdmissionStatus::AdmittedPrint
                                              : PendingRenderAdmissionStatus::AdmittedDirect)) {
                    throw std::runtime_error(std::string(test.name) + ": admission failed: " + admitted.diagnostic);
                }
                if ((print && !admitted.printState) || (!print && !admitted.directState)) {
                    throw std::runtime_error("admitted fixture state unavailable");
                }
                JuicerProcessor processor(effect);
                OFX::RenderArguments args{};
                args.isEnabledCudaRender = true;
                args.pCudaStream = device.stream;
                args.cudaStreamPropertyPresent = true;
                processor.setGPURenderArgs(args);
                processor.setSrcDst({&sourceImage, &destinationImage});
                const OfxRectI renderWindow = emptyWindow
                                                  ? OfxRectI{test.originX, test.originY, test.originX, test.originY}
                                                  : bounds;
                const OfxRectI fullBounds = test.fullWidth > 0 && test.fullHeight > 0
                                                ? OfxRectI{0, 0, test.fullWidth, test.fullHeight}
                                                : bounds;
                const int physicalWidth = test.fullWidth > 0 ? test.fullWidth : width;
                const float pixelSizeUm = 35'000.0f / static_cast<float>(physicalWidth);
                const Spektrafilm::FilmJuicerEffectsGeometry effectsGeometry{
                    {fullBounds.x1, fullBounds.y1, fullBounds.x2 - fullBounds.x1, fullBounds.y2 - fullBounds.y1},
                    static_cast<double>(fullBounds.x1),
                    static_cast<double>(fullBounds.y1),
                    static_cast<double>(fullBounds.x2 - fullBounds.x1),
                    static_cast<double>(fullBounds.y2 - fullBounds.y1),
                    test.scaleX,
                    test.scaleY,
                    1.0};
                std::optional<ScatterHalationFrameDescriptor> halation;
                std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
                const auto& recipe = print ? admitted.printState->recipe : admitted.directState->recipe;
                std::string descriptorDiagnostic;
                if (test.combined && (!JuicerTestSupport::prepare_halation_frame(
                                          recipe.spatialOptics.scatterHalation, pixelSizeUm, halation, descriptorDiagnostic) ||
                                      !halation)) {
                    throw std::runtime_error("halation descriptor: " + descriptorDiagnostic);
                }
                if (parameters.cameraDiffusion.active || parameters.enlargerDiffusion.active) {
                    if (!JuicerTestSupport::prepare_diffusion_frame(
                            recipe.spatialOptics, test.route, pixelSizeUm, Spektrafilm::DiffusionFrameDomain{test.originX, test.originY, width, height}, diffusion, descriptorDiagnostic) ||
                        !diffusion) {
                        throw std::runtime_error("diffusion descriptor: " + descriptorDiagnostic);
                    }
                }
                if (print) {
                    JuicerProcessor::PrintFrameRequest request;
                    request.state = admitted.printState;
#if defined(JUICER_NOISE_TEST_HOOK)
                    if (JuicerAssets::NoiseTest::observation && JuicerAssets::NoiseTest::observation->earlier == JuicerAssets::NoiseTest::EarlierFailure::Missing) {
                        request.state.reset();
                    }
#endif
                    request.diffusionFrameSet = diffusion;
                    request.scatterHalation = halation;
                    request.renderWindow = renderWindow;
                    request.fullFrameExtent = fullBounds;
                    request.sessionSeed = 0x20260923;
                    request.instanceToken = 0x641207 + static_cast<int>(test.route) + (test.combined ? 4 : 0);
                    request.clipToken = 0x5312;
                    request.frameTime = test.time;
                    request.frameRate = 24.0;
                    request.pixelSizeUm = pixelSizeUm;
                    request.effectsGeometry = effectsGeometry;
                    processor.setPrintFrameRequest(request);
                } else {
                    JuicerProcessor::DirectFrameRequest request;
                    request.state = admitted.directState;
#if defined(JUICER_NOISE_TEST_HOOK)
                    if (JuicerAssets::NoiseTest::observation && JuicerAssets::NoiseTest::observation->earlier == JuicerAssets::NoiseTest::EarlierFailure::Missing) {
                        request.state.reset();
                    }
#endif
                    request.diffusionFrameSet = diffusion;
                    request.scatterHalation = halation;
                    request.renderWindow = renderWindow;
                    request.fullFrameExtent = fullBounds;
                    request.sessionSeed = 0x20260923;
                    request.instanceToken = 0x641207 + static_cast<int>(test.route) + (test.combined ? 4 : 0);
                    request.clipToken = 0x5312;
                    request.frameTime = test.time;
                    request.frameRate = 24.0;
                    request.pixelSizeUm = pixelSizeUm;
                    request.effectsGeometry = effectsGeometry;
                    processor.setDirectFrameRequest(request);
                }
                processor.setInstanceState(&state);
                JuicerProcess::Root::FramePreparationToken preparation;
#if defined(JUICER_NOISE_TEST_HOOK)
                const bool testingClosedAdmission = JuicerAssets::NoiseTest::observation && JuicerAssets::NoiseTest::observation->closeBeforeAdmission;
#else
                constexpr bool testingClosedAdmission = false;
#endif
                if (!testingClosedAdmission) {
                    preparation = JuicerProcess::root().begin_frame_preparation();
                    if (!preparation.active()) {
                        throw std::runtime_error("frame preparation guard unavailable");
                    }
                }
#if defined(JUICER_PREPARED_BOUNDARY_TEST) || defined(JUICER_NOISE_TEST_HOOK)
                if (path != ExecutionPath::Processor) {
                    const auto& payload = print ? admitted.printState->payload : admitted.directState->payload;
                    const JuicerCuda::FrameRect nativeBounds{bounds.x1, bounds.y1, bounds.x2, bounds.y2};
                    const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(nativeBounds, nativeBounds, recipe.filmRaw.autoExposureMethod);
                    const JuicerCuda::ExecutionFrame frame{nativeBounds, nativeBounds, nativeBounds, reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<unsigned char*>(device.destination), imageInput.rowBytes, imageInput.rowBytes, test.components, device.stream, diffusion, halation, effectsGeometry, pixelSizeUm, test.time, 24.0, 0x20260923, 0x5312, meter, false, false};
                    JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
                    snapshot.instanceToken.value = 0x641207 + static_cast<int>(test.route) + (test.combined ? 4 : 0);
                    snapshot.frameToken.value = static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(test.time)));
                    snapshot.snapshotId = snapshotId;
                    const FjImage rawSource{reinterpret_cast<std::uintptr_t>(device.source), {bounds.x1, bounds.y1, bounds.x2, bounds.y2}, imageInput.rowBytes, static_cast<std::uint32_t>(test.components), FJ_DEPTH_FLOAT32};
                    FjFrame raw{};
                    raw.source = rawSource;
                    raw.destination = rawSource;
                    raw.destination.address = reinterpret_cast<std::uintptr_t>(device.destination);
                    raw.render_window = rawSource.bounds;
                    raw.full_frame_extent = rawSource.bounds;
                    raw.stream = reinterpret_cast<std::uintptr_t>(device.stream);
                    raw.flags = FJ_FRAME_STREAM_PRESENT;
                    FjCudaContext context{};
                    if (fj_cuda_inspect(JuicerCuda::borrowed_owner(), &raw, &context, nullptr).category != FJ_STATUS_SUCCESS) {
                        throw std::runtime_error("reference frame inspection failed");
                    }
                    snapshot.deviceContextKey = {context.device_id, reinterpret_cast<void*>(context.context)};
                    snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                        payload.uploadCoreHash, recipe.dirCouplers.hash, payload.scannerHash, meter.hash);
                    if (path == ExecutionPath::Boundary) {
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
                        const auto descriptors = JuicerCuda::describe_execution(recipe, payload, frame);
                        std::string diagnostic;
                        FjStatus preparedStatus{};
                        auto preparedOwner = JuicerCudaTest::prepare_state(
                            JuicerProcess::root().assets().borrow_rust_assets(), parameters, preparedStatus, diagnostic);
                        if (preparedStatus.category != FJ_STATUS_SUCCESS || !preparedOwner) {
                            throw std::runtime_error(std::string(test.name) + ": held-state preparation: " + diagnostic);
                        }
                        if (!JuicerCudaTest::execute_boundary(*preparedOwner, recipe, payload, frame, snapshot, descriptors, diagnostic)) {
                            throw std::runtime_error(std::string(test.name) + ": C boundary: " + diagnostic);
                        }
#else
                        throw std::runtime_error("boundary path requires the prepared fixture target");
#endif
                    } else {
                        JuicerCuda::PendingContextLossRecovery recovery;
                        if (print) {
                            JuicerCuda::execute_print({recipe, payload, frame, snapshot}, recovery);
                        } else {
                            JuicerCuda::execute_direct({recipe, payload, frame, snapshot}, recovery);
                        }
                    }
                } else {
                    processor.process();
                }
#else
                (void)path;
                processor.process();
#endif
            }
            imageLeaseAudit.require_complete(test.name);
            if (s_abortObservation && s_abortObservation->cancelAt != 0 &&
                s_abortObservation->calls >= s_abortObservation->cancelAt) {
                return {};
            }
        }
        require_cuda(cudaMemcpyAsync(output.data(), device.destination, destinationBytes, cudaMemcpyDeviceToHost, device.stream),
                     "copy destination");
        require_cuda(cudaStreamSynchronize(device.stream), "synchronize");
        if (emptyWindow) {
            for (float value : output) {
                if (std::bit_cast<std::uint32_t>(value) != std::bit_cast<std::uint32_t>(kCanary)) {
                    throw std::runtime_error("empty processor window wrote output");
                }
            }
            return {};
        }
        std::vector<float> pixels;
        pixels.reserve(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) *
                       static_cast<std::size_t>(test.components));
        for (int y = 0; y < height; ++y) {
            const std::size_t rowOffset = static_cast<std::size_t>(y + border) * static_cast<std::size_t>(destinationPitch);
            for (int x = 0; x < width; ++x) {
                const std::size_t pixelOffset = rowOffset +
                                                static_cast<std::size_t>(x + border) * static_cast<std::size_t>(test.components);
                for (int c = 0; c < test.components; ++c) {
                    const std::size_t index = pixelOffset + static_cast<std::size_t>(c);
                    const float value = output[index];
                    if (!std::isfinite(value)) {
                        throw std::runtime_error(std::string(test.name) + ": nonfinite output");
                    }
                    if (c == 3) {
                        RenderAssertions::require_same_bits(std::string(test.name) + ": alpha", value, input[static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch) + static_cast<std::size_t>(x) * static_cast<std::size_t>(test.components) + static_cast<std::size_t>(c)]);
                    }
                    pixels.push_back(value);
                }
            }
        }
        for (int y = 0; y < destinationHeight; ++y) {
            for (int x = 0; x < destinationPitch; ++x) {
                const bool rendered = y >= border && y < border + height &&
                                      x >= border * test.components && x < (border + width) * test.components;
                if (!rendered) {
                    RenderAssertions::require_same_bits(std::string(test.name) + ": destination outside render window", output[static_cast<std::size_t>(y) * static_cast<std::size_t>(destinationPitch) + static_cast<std::size_t>(x)], kCanary);
                }
            }
        }
        return pixels;
    }

    using RenderAssertions::compare_pixels;

} // namespace

namespace {
    [[maybe_unused]] void check_cutover_callbacks(const Case& test, const std::vector<float>& expected) {
        InstanceState state;
        compare_pixels("same allocation disjoint rows", render_case(test, parameters_for(test), state, false, ExecutionPath::Processor, DestinationLayout::DisjointRows), expected);
        Case uniform = test;
        uniform.uniform = true;
        auto uniformParameters = parameters_for(uniform);
        uniformParameters.glareActive = false;
        const auto initialUniform = render_case(uniform, uniformParameters, state);
        // This is a constant-input invariance test, complementary to the
        // independent accepted pixel fixtures above. Glare would add noise.
        const auto uniform_expected = [&](const Case& frame) {
            std::vector<float> pixels(static_cast<std::size_t>(frame.width * frame.height * frame.components));
            for (std::size_t i = 0; i < pixels.size(); ++i) {
                const auto channel = i % static_cast<std::size_t>(frame.components);
                // Alpha is positional even for uniform RGB in this fixture.
                const auto pixel = i / static_cast<std::size_t>(frame.components);
                const auto x = pixel % static_cast<std::size_t>(frame.width);
                const auto y = pixel / static_cast<std::size_t>(frame.width);
                pixels[i] = channel == 3 ? 0.125f + static_cast<float>((x + y) % 4) * 0.25f : initialUniform[channel];
            }
            return pixels;
        };
        compare_pixels("initial uniform callback", initialUniform, uniform_expected(uniform));
        const auto firstId = state.submissionSnapshotLatch.snapshotId;
        uniform.width += 2;
        uniform.height -= 2;
        uniform.originX = -11;
        uniform.originY = 7;
        uniform.time = -17.25;
        uniform.scaleX = 0.5;
        uniform.scaleY = 1.5;
        const auto changedExpected = uniform_expected(uniform);
        compare_pixels("changed geometry time and scale", render_case(uniform, uniformParameters, state), changedExpected);
        const auto changedId = state.submissionSnapshotLatch.snapshotId;
        if (changedId == firstId || state.submissionSnapshotLatch.frameToken.value != static_cast<std::uint64_t>(std::int64_t{-18})) {
            throw std::runtime_error("changed callback facts reused the previous latch");
        }
        // Each render_case allocates a new supplied stream; the same facts reuse the latch.
        compare_pixels("sequential supplied stream", render_case(uniform, uniformParameters, state), changedExpected);
        if (state.submissionSnapshotLatch.snapshotId != changedId) {
            throw std::runtime_error("unchanged callback facts failed to reuse the latch");
        }
        CUcontext current = nullptr;
        if (cuCtxGetCurrent(&current) != CUDA_SUCCESS) {
            throw std::runtime_error("get callback context");
        }
        std::exception_ptr workerFailure;
        std::thread worker([&] {
            try {
                if (cuCtxSetCurrent(current) != CUDA_SUCCESS) {
                    throw std::runtime_error("set worker callback context");
                }
                compare_pixels("callback on another host thread", render_case(uniform, uniformParameters, state), changedExpected);
            } catch (...) {
                workerFailure = std::current_exception();
            }
            (void)cuCtxSetCurrent(nullptr);
        });
        worker.join();
        if (workerFailure) {
            std::rethrow_exception(workerFailure);
        }
        CUcontext other = nullptr;
        if (cuCtxCreate(&other, nullptr, 0, 0) != CUDA_SUCCESS) {
            throw std::runtime_error("create fixture context");
        }
        try {
            compare_pixels("second context on same device", render_case(uniform, uniformParameters, state), changedExpected);
            if (state.submissionSnapshotLatch.deviceContextKey.contextOpaque != other) {
                throw std::runtime_error("callback retained prior context identity");
            }
            std::string diagnostic;
            JuicerCuda::NativeCall admission(JuicerCuda::borrowed_owner());
            if (!JuicerProcess::root().retire_idle_context(0, other, diagnostic)) {
                throw std::runtime_error(diagnostic);
            }
        } catch (...) {
            (void)cuCtxDestroy(other);
            (void)cuCtxSetCurrent(current);
            throw;
        }
        if (cuCtxDestroy(other) != CUDA_SUCCESS || cuCtxSetCurrent(current) != CUDA_SUCCESS) {
            throw std::runtime_error("release fixture context");
        }
        for (const unsigned checkpoint : {1U, 3U, 4U, 5U}) {
            AbortObservation observation;
            observation.cancelAt = checkpoint;
            s_abortObservation = &observation;
            try {
                (void)render_case(test, parameters_for(test), state);
            } catch (...) {
                s_abortObservation = nullptr;
                throw;
            }
            s_abortObservation = nullptr;
            if (observation.calls != checkpoint || !observation.sameThread) {
                throw std::runtime_error("production abort delivery or callback lifetime changed");
            }
        }
        AbortObservation throwing;
        throwing.throwAt = 4;
        s_abortObservation = &throwing;
        bool fatal = false;
        try {
            (void)render_case(test, parameters_for(test), state);
        } catch (const OFX::Exception::Suite& error) {
            fatal = error.status() == kOfxStatErrFatal;
        } catch (...) {
            s_abortObservation = nullptr;
            throw;
        }
        s_abortObservation = nullptr;
        if (!fatal || throwing.calls != 4 || !throwing.sameThread) {
            throw std::runtime_error("abort suite exception escaped or became successful cancellation");
        }
        compare_pixels("render after cancellation", render_case(test, parameters_for(test), state), expected);
        // This fixture's next case reuses its deterministic instance token with
        // a fresh counter. Match real instance teardown before that reuse.
        if (fj_cuda_retire_instance(JuicerCuda::borrowed_owner(), state.submissionSnapshotLatch.instanceToken.value, nullptr).category != FJ_STATUS_SUCCESS) {
            throw std::runtime_error("retire cutover fixture instance");
        }
        std::cout << test.name << ": production geometry/time/scale, streams, threads, contexts, disjoint alias and abort contracts passed\n";
    }
} // namespace

namespace {
    [[maybe_unused]] void check_physical_effects_pixels(const Case& original) {
        Case test = original;
        test.components = 4;
        // A fresh fixture state restarts snapshot IDs; retire the preceding instance first.
        const std::uint64_t instanceToken = 0x641207 + static_cast<int>(test.route) + (test.combined ? 4 : 0);
        if (fj_cuda_retire_instance(JuicerCuda::borrowed_owner(), instanceToken, nullptr).category != FJ_STATUS_SUCCESS)
            throw std::runtime_error("retire instance before physical fixture");
        InstanceState state;
        auto off = parameters_for(test);
        off.grainControls.active = false;
        off.filmDustAmount = 0;
        off.filmScratchAmount = 0;
        off.gateDustAmount = 0;
        off.gateScratchAmount = 0;
        off.gateWeaveAmount = 0;
        const auto same_bits = [](const std::vector<float>& left, const std::vector<float>& right) {
            return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(), [](float a, float b) {
                       return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
                   });
        };
        const auto baseline = render_case(test, off, state);
        if (!same_bits(render_case(test, off, state), baseline))
            throw std::runtime_error("all-off physical effects is not deterministic identity");
        for (unsigned family = 0; family < 5; ++family) {
            auto enabled = off;
            if (family == 0)
                enabled.filmDustAmount = 10;
            if (family == 1)
                enabled.filmScratchAmount = 10;
            if (family == 2)
                enabled.gateDustAmount = 10;
            if (family == 3)
                enabled.gateScratchAmount = 10;
            if (family == 4)
                enabled.gateWeaveAmount = 10;
            const auto pixels = render_case(test, enabled, state);
            if (same_bits(pixels, baseline))
                throw std::runtime_error("physical effects family did not reach CUDA output: " + std::to_string(family));
            enabled.grainControls.debugView = 3;
            if (!same_bits(render_case(test, enabled, state), pixels))
                throw std::runtime_error("inactive grain debug incorrectly suppresses physical effects");
        }
        auto debug = off;
        debug.grainControls.active = true;
        debug.grainControls.debugView = 3;
        const auto debugOff = render_case(test, debug, state);
        debug.filmDustAmount = 10;
        debug.filmScratchAmount = 10;
        debug.gateDustAmount = 10;
        debug.gateScratchAmount = 10;
        debug.gateWeaveAmount = 10;
        if (!same_bits(render_case(test, debug, state), debugOff))
            throw std::runtime_error("completed grain debug did not suppress physical effects");
        if (fj_cuda_retire_instance(JuicerCuda::borrowed_owner(), instanceToken, nullptr).category != FJ_STATUS_SUCCESS)
            throw std::runtime_error("retire instance after physical fixture");
        std::cout << test.name << " physical families/off/debug/alpha CUDA paths passed\n";
    }

} // namespace

#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
namespace {
    using RawFrameMathBuilder = FjStatus (*)(void*, FjFrameMathInput&, FjErrorBuffer*) noexcept;
    using NativeFrameDescriptorCall = FjStatus (*)(void*, const FjFrameMathInput*, FjNativeFrameDescriptors*, FjErrorBuffer*);

    struct FrameDescriptorObservation {
        NativeFrameDescriptorCall forward = nullptr;
        unsigned callbackCalls = 0;
        unsigned providerCalls = 0;
    };

    thread_local FrameDescriptorObservation* frameDescriptorObservation = nullptr;

    FjStatus observe_frame_descriptors(
        void* scopeUser,
        const FjFrameMathInput* input,
        FjNativeFrameDescriptors* out,
        FjErrorBuffer* error) noexcept {
        if (out)
            *out = {};
        auto* const override = frameDescriptorObservation;
        if (!override || !override->forward)
            return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
        ++override->callbackCalls;
        ++override->providerCalls;
        return override->forward(scopeUser, input, out, error);
    }

    struct ScopedFrameDescriptorObservation {
        explicit ScopedFrameDescriptorObservation(FrameDescriptorObservation* current) noexcept
            : _previous(std::exchange(frameDescriptorObservation, current)) {}
        ~ScopedFrameDescriptorObservation() {
            frameDescriptorObservation = _previous;
        }
        ScopedFrameDescriptorObservation(const ScopedFrameDescriptorObservation&) = delete;
        ScopedFrameDescriptorObservation& operator=(const ScopedFrameDescriptorObservation&) = delete;

    private:
        FrameDescriptorObservation* _previous;
    };

    struct RawFrameJson {
        const nlohmann::json* input = nullptr;
        unsigned facadeCalls = 0;
    };

    FjStatus build_literal_dir_math(void* user, FjFrameMathInput& math, FjErrorBuffer* error) noexcept {
        try {
            auto& context = *static_cast<RawFrameJson*>(user);
            const auto& value = *context.input;
            const auto extent = [](const nlohmann::json& v) -> FjExtent {
                return {v.at(0), v.at(1), v.at(2), v.at(3)};
            };
            const FjDirFrameInput raw{value.at("active").get<std::uint32_t>(),
                                      std::bit_cast<float>(value.at("size").get<std::uint32_t>()),
                                      std::bit_cast<float>(value.at("tail").get<std::uint32_t>()),
                                      std::bit_cast<float>(value.at("weight").get<std::uint32_t>()),
                                      std::bit_cast<float>(value.at("pixel").get<std::uint32_t>()),
                                      extent(value.at("render")),
                                      extent(value.at("full")),
                                      value.at("hash").get<std::uint64_t>()};
            FjDirFrameFailure failure{};
            math.dir = {};
            ++context.facadeCalls;
            return fj_test_dir_frame_prepare(&raw, &math.dir, &failure, error);
        } catch (...) {
            return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        }
    }

    void copy_grain_coefficients(const Spektrafilm::VisualGrainRecipe& source,
                                 FjGrainCoefficients& out) {
        out = {};
        if (!source.active)
            return;
        out.flags = FJ_GRAIN_ACTIVE | (source.sublayersActive ? FJ_GRAIN_SUBLAYERS : 0u);
        out.debug_view = static_cast<std::uint32_t>(source.debugView);
        out.sublayer_count = static_cast<std::size_t>(source.nSubLayers);
        std::copy_n(source.microStructure.data(), 2, out.micro_structure);
        out.clump_temporal_mix = source.clumpTemporalMix;
        out.size_mix_scale = source.sizeMixScale;
        out.amplitude = source.amplitude;
        out.chroma_mix = source.chromaMix;
        out.chroma_shared_weight = source.chromaSharedWeight;
        out.chroma_independent_weight = source.chromaIndependentWeight;
        std::copy_n(source.visualParticleDensityMinCmy.data(), 3, out.particle_density_min_cmy);
        std::copy_n(source.uniformityCmy.data(), 3, out.uniformity_cmy);
        constexpr std::uint32_t axisFlags[]{FJ_GRAIN_AXIS_FINITE_C, FJ_GRAIN_AXIS_FINITE_M, FJ_GRAIN_AXIS_FINITE_Y};
        for (std::size_t channel = 0; channel < 3; ++channel) {
            if (source.grainLayerAxisFinite[channel])
                out.flags |= axisFlags[channel];
            std::copy_n(source.grainLayerAxisBlockPrefixMax[channel].data(), 16, out.layer_axis_block_prefix_max[channel]);
        }
    }

    FjStatus build_literal_grain_math(void* user, FjFrameMathInput& math, FjErrorBuffer* error) noexcept {
        try {
            auto& context = *static_cast<RawFrameJson*>(user);
            const auto& value = *context.input;
            Spektrafilm::VisualGrainRecipe recipe{};
            Spektrafilm::FilmDevelopRecipe film{};
            const auto input = JuicerTestSupport::grain_frame_input(value, recipe, film);
            FjGrainFrameInput raw{};
            std::string diagnostic;
            if (!JuicerGrain::project_input(input, raw, diagnostic))
                return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
            FjGrainFrameFailure failure{};
            math.grain = {};
            copy_grain_coefficients(recipe, math.grain_coefficients);
            ++context.facadeCalls;
            return fj_test_grain_frame_prepare(&raw, &math.grain, &failure, error);
        } catch (...) {
            return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        }
    }

    FjStatus build_literal_effects_math(void* user, FjFrameMathInput& math, FjErrorBuffer* error) noexcept {
        try {
            auto& context = *static_cast<RawFrameJson*>(user);
            const auto& value = *context.input;
            Spektrafilm::FilmJuicerEffectsRecipe recipe{};
            const auto input = JuicerTestSupport::effects_frame_input(value, recipe);
            FjEffectsFrameInput raw{};
            std::string diagnostic;
            if (!JuicerEffects::project_input(input, raw, diagnostic))
                return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
            FjEffectsFrameFailure failure{};
            math.effects = {};
            math.film_dust = raw.film_dust;
            math.film_scratch = raw.film_scratch;
            math.gate_dust = raw.gate_dust;
            math.gate_scratch = raw.gate_scratch;
            ++context.facadeCalls;
            return fj_test_effects_frame_prepare(&raw, &math.effects, &failure, error);
        } catch (...) {
            return {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0};
        }
    }

    enum class FrameObservationKind : std::uint8_t {
        Dir,
        Grain,
        Effects
    };

    struct FrameConstructionResult {
        explicit FrameConstructionResult(FrameObservationKind observed)
            : kind(observed) {}

        FrameObservationKind kind;
        FjStatus status{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        unsigned rawFacadeCalls = 0;
        std::string diagnostic;
        std::optional<Spektrafilm::SpatialDirDescriptor> dir;
        std::optional<Spektrafilm::VisualGrainFrameDescriptor> grain;
        std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor> effects;
    };

    bool capture_held_frame_construction(
        void* user,
        const JuicerCudaTest::PreparedFrameFixture::Inputs& inputs,
        std::string&) {
        auto& result = *static_cast<FrameConstructionResult*>(user);
        result.status = {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        if (result.kind == FrameObservationKind::Dir)
            result.dir = inputs.descriptors.spatialDir;
        else if (result.kind == FrameObservationKind::Grain)
            result.grain = inputs.descriptors.grain;
        else
            result.effects = inputs.descriptors.effects;
        return true;
    }

    FrameConstructionResult prepare_literal_frame_construction(
        FrameObservationKind kind,
        const nlohmann::json& value,
        Spektrafilm::ScanRoute route,
        float pixelSizeUm,
        RawFrameMathBuilder build) {
        FrameConstructionResult result{kind};
        FjFrameMathInput math{};
        std::array<char, 1024> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        RawFrameJson raw{&value};
        result.status = build(&raw, math, &error);
        result.rawFacadeCalls = raw.facadeCalls;
        if (result.status.category != FJ_STATUS_SUCCESS) {
            result.diagnostic.assign(text.data(), std::min(error.length, text.size()));
            return result;
        }
        if (kind == FrameObservationKind::Dir) {
            Spektrafilm::SpatialDirDescriptor descriptor{};
            if (!Spektrafilm::compose_spatial_dir_descriptor(
                    math.dir, pixelSizeUm, Spektrafilm::scan_route_is_print(route) ? "print" : "direct", descriptor)) {
                result.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
                result.diagnostic = "literal DIR native descriptor composition failed";
                return result;
            }
            result.dir = descriptor;
        } else if (kind == FrameObservationKind::Grain) {
            Spektrafilm::VisualGrainFrameDescriptor descriptor{};
            if (!Spektrafilm::compose_visual_grain_frame_descriptor(
                    math.grain_coefficients,
                    static_cast<Spektrafilm::ProfilePolarity>(math.grain.capture_polarity),
                    math.grain,
                    descriptor,
                    result.diagnostic)) {
                result.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
                return result;
            }
            result.grain = descriptor;
        } else {
            Spektrafilm::FilmJuicerEffectsFrameDescriptor descriptor{};
            if (!JuicerEffects::compose_frame_descriptor(
                    math.effects, math.film_dust, math.film_scratch, math.gate_dust, math.gate_scratch, descriptor, result.diagnostic)) {
                result.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
                return result;
            }
            result.effects = descriptor;
        }
        return result;
    }

    struct AbortAfterNativeDescriptor {
        unsigned calls = 0;
        unsigned requestOn = 3;
    };

    std::uint32_t abort_after_native_descriptor(void* user) noexcept {
        auto* const state = static_cast<AbortAfterNativeDescriptor*>(user);
        if (!state)
            return 2u; // Invalid callback state must fail abort transport.
        ++state->calls;
        return state->calls >= state->requestOn ? FJ_ABORT_REQUESTED : FJ_ABORT_CONTINUE;
    }

    struct LegacyCombinedResult {
        FjStatus status{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        std::string diagnostic;
        unsigned abortCalls = 0;
        JuicerCuda::ExecutorTest::DirFrameObservation dir;
        JuicerCuda::ExecutorTest::GrainFrameObservation grain;
        JuicerCuda::ExecutorTest::EffectsFrameObservation effects;
    };

    template <typename PrepareLiteral>
    LegacyCombinedResult sf_temp_bridge_native_combined_call(
        FrameObservationKind kind,
        const ParamSnapshot& params,
        const JuicerCuda::ExecutionFrame& frame,
        JuicerCuda::ResourceManager::InstanceToken instanceToken,
        std::uint64_t snapshotId,
        const JuicerCuda::ResourceManager::DeviceContextKey& contextKey,
        PrepareLiteral&& prepareLiteral) {
        // SF_TEMP_BRIDGE_NATIVE_COMBINED_ABORT_QUERY: retain the old one-query
        // combined C++ caller assertion through I7, then remove this comparison.
        FocusedRenderStateBuildProduct product;
        std::string diagnostic;
        const bool print = Spektrafilm::scan_route_is_print(params.scanRoute);
        if (!(print ? build_print_render_state_product(params, product, diagnostic)
                    : build_direct_render_state_product(params, product, diagnostic)))
            throw std::runtime_error("SF_TEMP_BRIDGE_NATIVE_COMBINED_ABORT_QUERY product: " + diagnostic);
        prepareLiteral(product);
        const auto& recipe = product.recipe;
        const auto& payload = product.payload;
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
        snapshot.instanceToken = instanceToken;
        snapshot.frameToken.value = static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(frame.timeFrames)));
        snapshot.snapshotId = snapshotId;
        snapshot.deviceContextKey = contextKey;
        snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
            payload.uploadCoreHash, recipe.dirCouplers.hash, payload.scannerHash, frame.autoExposureDescriptor.hash);
        AbortAfterNativeDescriptor abortState{0, 1};
        const FjAbortCallback abort{&abort_after_native_descriptor, &abortState};
        JuicerCuda::PendingContextLossRecovery recovery;
        JuicerCuda::ExecutorTest::reset_dir_frame_observation();
        JuicerCuda::ExecutorTest::reset_grain_frame_observation();
        JuicerCuda::ExecutorTest::reset_effects_frame_observation();
        const auto admission = JuicerProcess::root().begin_frame_preparation();
        if (!admission.active())
            throw std::runtime_error("SF_TEMP_BRIDGE_NATIVE_COMBINED_ABORT_QUERY admission unavailable");
        LegacyCombinedResult result;
        try {
            if (print)
                JuicerCuda::execute_print({recipe, payload, frame, snapshot}, recovery, abort);
            else
                JuicerCuda::execute_direct({recipe, payload, frame, snapshot}, recovery, abort);
        } catch (const JuicerCuda::ExecutionFailure& failure) {
            result.status = failure.failure.status;
            result.diagnostic = failure.failure.diagnostic;
        }
        result.abortCalls = abortState.calls;
        result.dir = JuicerCuda::ExecutorTest::dir_frame_observation();
        result.grain = JuicerCuda::ExecutorTest::grain_frame_observation();
        result.effects = JuicerCuda::ExecutorTest::effects_frame_observation();
        if (recovery.pending)
            throw std::runtime_error("SF_TEMP_BRIDGE_NATIVE_COMBINED_ABORT_QUERY unexpectedly entered recovery");
        if (kind == FrameObservationKind::Dir && result.dir.count == 0 &&
            result.status.category != FJ_STATUS_PREPARATION_FAILURE)
            throw std::runtime_error("SF_TEMP_BRIDGE_NATIVE_COMBINED_ABORT_QUERY DIR comparison missed its producer");
        if (kind == FrameObservationKind::Grain && result.grain.count == 0 &&
            result.status.category != FJ_STATUS_PREPARATION_FAILURE)
            throw std::runtime_error("SF_TEMP_BRIDGE_NATIVE_COMBINED_ABORT_QUERY grain comparison missed its producer");
        if (kind == FrameObservationKind::Effects && result.effects.count == 0 &&
            result.status.category != FJ_STATUS_PREPARATION_FAILURE)
            throw std::runtime_error("SF_TEMP_BRIDGE_NATIVE_COMBINED_ABORT_QUERY effects comparison missed its producer");
        return result;
    }

    struct PreparedRenderCompletion {
        unsigned calls = 0;
    };

    FjCallCompletionResult prepared_render_complete(void* user, const FjCallCompletionInput* input) noexcept {
        auto* const completion = static_cast<PreparedRenderCompletion*>(user);
        if (completion)
            ++completion->calls;
        return input ? FjCallCompletionResult{input->outcome, 0}
                     : FjCallCompletionResult{{{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0}, 1};
    }

    struct PreparedSubmissionSelection {
        FjSubmission expected{};
        unsigned calls = 0;
        bool matched = true;
    };

    FjStatus select_prepared_submission(
        void* user,
        const FjCudaContext*,
        const FjSubmission* requested,
        FjSubmission* out,
        FjErrorBuffer*) {
        auto* const selection = static_cast<PreparedSubmissionSelection*>(user);
        if (!selection || !requested || !out)
            return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
        ++selection->calls;
        selection->matched = requested->instance_token == selection->expected.instance_token &&
                             requested->frame_token == selection->expected.frame_token &&
                             requested->upload_core_hash == selection->expected.upload_core_hash &&
                             requested->dir_hash == selection->expected.dir_hash &&
                             requested->scanner_hash == selection->expected.scanner_hash &&
                             requested->auto_exposure_hash == selection->expected.auto_exposure_hash;
        if (!selection->matched)
            return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
        *out = *requested;
        out->submission_id = selection->expected.submission_id;
        return {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
    }

    FjCallCompletionResult render_held_prepared_state(
        const PreparedState& state,
        const FjFrame& frame,
        std::uint64_t instanceToken,
        std::uint64_t submissionId,
        FjAbortCallback abort,
        FrameDescriptorObservation* descriptorObservation,
        bool* recoveryPending,
        std::string& diagnostic) {
        diagnostic.clear();
        std::array<char, 2048> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        FjPreparedStateView view{};
        const FjStatus viewed = state.view(view, error);
        if (viewed.category != FJ_STATUS_SUCCESS) {
            diagnostic.assign(text.data(), error.length);
            return {{viewed, 0}, 0};
        }
        const FjExtent full{frame.full_frame_extent.x1,
                            frame.full_frame_extent.y1,
                            frame.full_frame_extent.x2 - frame.full_frame_extent.x1,
                            frame.full_frame_extent.y2 - frame.full_frame_extent.y1};
        FjPreparedFramePolicy policy{};
        error.length = 0;
        const FjStatus policyStatus = state.frame_policy(full, policy, error);
        if (policyStatus.category != FJ_STATUS_SUCCESS) {
            diagnostic.assign(text.data(), error.length);
            return {{policyStatus, 0}, 0};
        }
        const auto frameRect = JuicerCuda::FrameRect{frame.source.bounds.x1,
                                                     frame.source.bounds.y1,
                                                     frame.source.bounds.x2,
                                                     frame.source.bounds.y2};
        const auto preview = JuicerCuda::make_auto_exposure_preview_descriptor(
            frameRect, frameRect, static_cast<Spektrafilm::AutoExposureMethod>(view.input_policy.meter));
        PreparedSubmissionSelection selection{};
        selection.expected = {instanceToken,
                              static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(frame.time_frames))),
                              submissionId,
                              view.upload_core_hash,
                              view.dir_recipe.hash,
                              view.scanner_hash,
                              preview.hash};
        PreparedRenderCompletion completion{};
        // Set the continuation's stack user only for this synchronous call.
        JuicerCuda::CallScope activeScope(JuicerCuda::borrowed_owner(), view.route, &prepared_render_complete, &completion);
        FjCudaCallAccess access = activeScope.access();
        if (descriptorObservation) {
            descriptorObservation->forward = access.frame_descriptors;
            access.frame_descriptors = &observe_frame_descriptors;
        }
        const FjSubmissionAccess latch{selection.expected.instance_token,
                                       selection.expected.frame_token,
                                       &selection,
                                       &select_prepared_submission};
        FjCallCompletionResult result{};
        auto admission = JuicerProcess::root().begin_frame_preparation();
        if (!admission.active()) {
            diagnostic = "prepared frame render admission unavailable";
            return {{{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, 0}, 1};
        }
        {
            ScopedFrameDescriptorObservation overrideScope(descriptorObservation);
            result = state.render(JuicerProcess::root().assets().borrow_rust_assets(),
                                  access,
                                  frame,
                                  policy,
                                  latch,
                                  abort,
                                  error);
        }
        if (completion.calls != 1 || selection.calls > 1 || !selection.matched) {
            diagnostic = activeScope.diagnostic().empty()
                             ? "prepared frame render completion or submission selection count changed"
                             : activeScope.diagnostic();
            return {{{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0}, 1};
        }
        if (recoveryPending)
            *recoveryPending = activeScope.recovery().pending;
        diagnostic = activeScope.diagnostic();
        if (diagnostic.empty() && error.length != 0)
            diagnostic.assign(text.data(), error.length);
        return result;
    }

    FjFrame prepared_raw_frame(const JuicerCuda::ExecutionFrame& source) {
        const auto rect = [](const JuicerCuda::FrameRect& value) {
            return FjRect{value.x1, value.y1, value.x2, value.y2};
        };
        FjFrame out{};
        out.source = {reinterpret_cast<std::uintptr_t>(source.source), rect(source.sourceBounds), source.sourceRowBytes, static_cast<std::uint32_t>(source.components), FJ_DEPTH_FLOAT32};
        out.destination = {reinterpret_cast<std::uintptr_t>(source.destination), rect(source.sourceBounds), source.destinationRowBytes, static_cast<std::uint32_t>(source.components), FJ_DEPTH_FLOAT32};
        out.render_window = rect(source.renderWindow);
        out.full_frame_extent = rect(source.fullFrameExtent);
        out.stream = reinterpret_cast<std::uintptr_t>(source.stream);
        out.flags = FJ_FRAME_STREAM_PRESENT;
        out.pixel_size_um = source.pixelSizeUm;
        out.time_frames = source.timeFrames;
        out.frame_rate = source.frameRate;
        out.session_seed = source.sessionSeed;
        out.clip_token = static_cast<std::uint64_t>(source.clipToken);
        const auto& geometry = source.effectsGeometry;
        out.effects_geometry = {{geometry.pixelDefinition.x, geometry.pixelDefinition.y, geometry.pixelDefinition.width, geometry.pixelDefinition.height},
                                geometry.canonicalX,
                                geometry.canonicalY,
                                geometry.canonicalWidth,
                                geometry.canonicalHeight,
                                geometry.scaleX,
                                geometry.scaleY,
                                geometry.pixelAspectRatio};
        return out;
    }

    struct WorkspacePreparedFrameContext {
        JuicerCuda::ResourceManager::DeviceContextKey deviceContextKey{};
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot{};
        void* stream = nullptr;
        JuicerProcess::Root::PreparedCudaFrame* frame = nullptr;
        JuicerCuda::Failure* failure = nullptr;
        bool invalidateGrainExtent = false;
        bool invalidateDirRoles = false;
        bool invalidateGateWidth = false;
        bool omitGrain = false;
        bool omitDir = false;
        bool dirWasActive = false;
        bool grainWasValid = false;
        bool gateWasActive = false;
        bool cameraDiffusionWasPresent = false;
    };

    bool prepare_workspace_frame_from_state(
        void* user,
        const JuicerCudaTest::PreparedFrameFixture::Inputs& inputs,
        std::string&) {
        auto& context = *static_cast<WorkspacePreparedFrameContext*>(user);
        if (!context.frame || !context.failure)
            return false;
        context.dirWasActive = inputs.descriptors.spatialDir.hash != 0;
        context.grainWasValid = inputs.preparation.visualGrainDescriptor.has_value() &&
                                inputs.preparation.visualGrainDescriptor->hash != 0;
        context.gateWasActive = inputs.preparation.effectsDescriptor.has_value() &&
                                inputs.preparation.effectsDescriptor->gateTransmittanceActive &&
                                inputs.preparation.effectsDescriptor->hash != 0;
        context.cameraDiffusionWasPresent = inputs.descriptors.diffusion.has_value() &&
                                            inputs.descriptors.diffusion->camera.has_value();
        auto request = inputs.preparation;
        std::optional<Spektrafilm::SpatialDirDescriptor> dir;
        std::optional<Spektrafilm::VisualGrainFrameDescriptor> grain;
        std::optional<Spektrafilm::FilmJuicerEffectsFrameDescriptor> effects;
        if (context.invalidateDirRoles && request.spatialDirDescriptor) {
            dir = *request.spatialDirDescriptor;
            dir->scratchTier = Spektrafilm::DirScratchTier::Tier1F;
            dir->planeRoles = {3, 3, 3, 0};
            request.spatialDirDescriptor = &*dir;
        }
        if (context.invalidateGrainExtent && request.visualGrainDescriptor) {
            grain = *request.visualGrainDescriptor;
            ++grain->renderExtent.width;
            request.visualGrainDescriptor = *grain;
        }
        if (context.invalidateGateWidth && request.effectsDescriptor) {
            effects = *request.effectsDescriptor;
            effects->gateWidth = 0;
            request.effectsDescriptor = *effects;
        }
        if (context.omitGrain)
            request.visualGrainDescriptor.reset();
        if (context.omitDir)
            request.spatialDirDescriptor = nullptr;
        context.snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
            inputs.state.upload_core_hash,
            inputs.state.dir_recipe.hash,
            inputs.state.scanner_hash,
            0);
        *context.frame = JuicerProcess::root().prepare_cuda_frame(
            context.deviceContextKey,
            context.snapshot,
            request,
            {},
            context.stream,
            *context.failure,
            inputs.hostPreparation);
        return true;
    }

    void run_workspace_transitions() {
        using Frame = JuicerProcess::Root::PreparedCudaFrame;
        using namespace JuicerCuda::ResourceManager;
        const auto require = [](bool ok, const std::string& message) {
            if (!ok) {
                throw std::runtime_error(message);
            }
        };
        int deviceId = 0;
        CUcontext context = nullptr;
        require_cuda(cudaGetDevice(&deviceId), "workspace device");
        require(cuCtxGetCurrent(&context) == CUDA_SUCCESS && context, "workspace context");
        const DeviceContextKey key{deviceId, context};
        DeviceFrame device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "workspace stream");
        std::uint64_t next = 1;
        for (const auto route : {Spektrafilm::ScanRoute::NegativeDirectScan, Spektrafilm::ScanRoute::NegativePrintScan}) {
            const bool print = Spektrafilm::scan_route_is_print(route);
            ParamSnapshot params;
            params.scanRoute = route;
            params.grainControls.active = true;
            params.scatterHalationControls = {};
            PreparedState state = JuicerTestSupport::prepare_product(params);
            std::string diagnostic;
            JuicerCudaTest::PreparedFrameFixture::FrameFacts facts;
            facts.renderExtent = {0, 0, 64, 48};
            facts.fullFrameExtent = facts.renderExtent;
            facts.pixelDefinition = facts.fullFrameExtent;
            facts.pixelSizeUm = 10.0f;
            facts.filmFormatLongEdgeMm = 35.0f;
            facts.canonicalWidth = 64.0;
            facts.canonicalHeight = 48.0;
            facts.scaleX = facts.scaleY = 1.0;
            facts.frameRate = 24.0;
            facts.sessionSeed = 17;
            facts.clipToken = 23;
            for (bool competing : {true, false}) {
                SubmissionSnapshot snapshot;
                snapshot.instanceToken.value = 0x53344438;
                snapshot.frameToken.value = next;
                snapshot.snapshotId = next++;
                snapshot.deviceContextKey = key;
                JuicerCuda::Failure error;
                Frame frame;
                WorkspacePreparedFrameContext preparation{key, snapshot, device.stream, &frame, &error, competing, !competing, false};
                JuicerCuda::ExecutorTest::reset_workspace_transitions();
                require(JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                            state, facts, &preparation, &prepare_workspace_frame_from_state, diagnostic),
                        diagnostic);
                require(preparation.dirWasActive, "workspace DIR must be active");
                require(preparation.grainWasValid, "workspace grain frame descriptor must be valid");
                // Root preparation is inside the actual prepared-state fixture callback.
                const std::string routeKey = Spektrafilm::scan_route_key(route);
                if (competing) {
                    require(!frame.active() && !frame.workspace_lease().active(), "competing failure published frame/marker");
                    require(error.diagnostic == "ResourceDescriptorMismatch phase=workspace field=grain_extent route=" + routeKey,
                            "competing failure precedence: " + error.diagnostic);
                    require(std::string(frame.failure_stage_tag()) == "derive_cuda_frame_workspace_request" &&
                                std::string(frame.failure_prefix()) == "CUDA frame workspace derivation failed",
                            "competing failure envelope");
                } else {
                    require(!frame.active() && !frame.workspace_lease().active(), "malformed roles published frame/marker");
                    require(error.diagnostic == "invalid frame scratch request route=" + routeKey, "malformed role diagnostic: " + error.diagnostic);
                    require(std::string(frame.failure_stage_tag()) == "derive_cuda_frame_workspace_request" &&
                                std::string(frame.failure_prefix()) == "CUDA frame workspace derivation failed",
                            "malformed role envelope");
                }
                require(error.status.category == FJ_STATUS_PREPARATION_FAILURE && error.status.api == FJ_API_NONE && error.status.native_code == 0,
                        "workspace failure status changed");
                const auto observed = JuicerCuda::ExecutorTest::workspace_transitions();
                require(!observed.requested && !observed.admitted && observed.resourcesAcquired == 0 && observed.candidateCount == 0 &&
                            observed.finishes == 0 && !observed.recoveryPending,
                        "malformed construction acquired or published workspace");
                const auto primary = error;
                frame.abort();
                require(error.diagnostic == primary.diagnostic && error.status.category == primary.status.category, "abort lost primary failure");
                std::cout << "replacement R2 route=" << routeKey << " competing=" << competing << " status={3,0,0} diagnostic=" << error.diagnostic
                          << " stage=derive_cuda_frame_workspace_request prefix=CUDA frame workspace derivation failed" << '\n';
            }
            for (unsigned carrier : {0u, 1u, 2u}) {
                params.grainControls.active = false;
                params.dirCouplers.active = false;
                params.gateDustAmount = 1.0;
                params.cameraDiffusion.active = carrier == 2;
                require(Spektrafilm::build_scatter_halation_controls(
                            ScatterHalationRawControls{carrier != 0, 1.0, 1.0, 1.0, 1.0}, params.scatterHalationControls, diagnostic),
                        diagnostic);
                PreparedState state = JuicerTestSupport::prepare_product(params);
                auto gateFacts = facts;
                gateFacts.pixelSizeUm = 35'000.0f / 64;
                gateFacts.filmFormatLongEdgeMm = 35.0f;
                SubmissionSnapshot snapshot;
                snapshot.instanceToken.value = 0x53344438;
                snapshot.frameToken.value = next;
                snapshot.snapshotId = next++;
                snapshot.deviceContextKey = key;
                JuicerCuda::ExecutorTest::reset_workspace_transitions();
                JuicerCuda::Failure gateFailure;
                Frame frame;
                WorkspacePreparedFrameContext preparation{key, snapshot, device.stream, &frame, &gateFailure};
                preparation.invalidateGateWidth = true;
                preparation.omitGrain = true;
                preparation.omitDir = true;
                require(JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                            state, gateFacts, &preparation, &prepare_workspace_frame_from_state, diagnostic),
                        diagnostic);
                require(preparation.gateWasActive, "gate fixture not active");
                require(preparation.cameraDiffusionWasPresent == (carrier == 2),
                        "camera diffusion descriptor presence changed");
                const auto marker = frame.workspace_lease();
                require(frame.active() && marker.active(), "attachment failure moved ahead of checkpoint: " + gateFailure.diagnostic);
                require(!frame.scanner_workspace(marker).active && !frame.spatial_dir_scratch(marker).active, "failed construction exposed a workspace view");
                auto halationView = frame.scatter_halation_resources();
                require((halationView.descriptor != nullptr) == (carrier != 0), "attachment failure changed halation activation");
                if (carrier != 0) {
                    require(halationView.fullFrameWidth == 64 && halationView.fullFrameHeight == 48,
                            "attachment failure erased prepared halation extent");
                    require(halationView.carrierSource == (carrier == 2 ? JuicerCuda::ScatterHalationCarrierSource::CameraDiffusionStagePlanes
                                                                        : JuicerCuda::ScatterHalationCarrierSource::DedicatedPreparedPlanes),
                            "attachment failure changed halation carrier");
                    if (carrier == 2) {
                        const auto preparedDiffusion = frame.diffusion_resources();
                        require(preparedDiffusion.active && !halationView.currentCarrier.redSensitive && !halationView.currentCarrier.greenSensitive && !halationView.currentCarrier.blueSensitive,
                                "shared halation carrier was bound before diffusion");
                        const auto& planes = preparedDiffusion.execution.stagePlanes;
                        halationView.currentCarrier = {planes.redSensitive, planes.greenSensitive, planes.blueSensitive, planes.rowStrideFloats};
                        frame.mark_diffusion_work_enqueued();
                    }
                    const std::size_t planeBytes = halationView.currentCarrier.rowStrideFloats * halationView.fullFrameHeight * sizeof(float);
                    require_cuda(cudaMemsetAsync(halationView.currentCarrier.redSensitive, 0, planeBytes, device.stream), "attachment halation red carrier");
                    require_cuda(cudaMemsetAsync(halationView.currentCarrier.greenSensitive, 0, planeBytes, device.stream), "attachment halation green carrier");
                    require_cuda(cudaMemsetAsync(halationView.currentCarrier.blueSensitive, 0, planeBytes, device.stream), "attachment halation blue carrier");
                    require_cuda(JuicerCuda::launch_scatter_halation(halationView, device.stream).status, "attachment halation launch");
                }
                require(!frame.checkpoint_large_scratch_transition(marker, device.stream, print ? "print_large_scratch_transition" : "direct_large_scratch_transition", gateFailure), "malformed attachment passed checkpoint");
                require(gateFailure.status.category == FJ_STATUS_PREPARATION_FAILURE && gateFailure.status.api == FJ_API_NONE && gateFailure.status.native_code == 0 &&
                            gateFailure.diagnostic == "invalid frame scratch request",
                        "attachment checkpoint envelope changed");
                const auto observed = JuicerCuda::ExecutorTest::workspace_transitions();
                require(observed.resourcesAcquired == 1 && !observed.requested && !observed.admitted && observed.candidateCount == 0, "attachment failure fabricated a completed layout");
                frame.abort();
                require(gateFailure.diagnostic == "invalid frame scratch request" && gateFailure.status.category == FJ_STATUS_PREPARATION_FAILURE &&
                            gateFailure.status.api == FJ_API_NONE && gateFailure.status.native_code == 0,
                        "abort lost attachment checkpoint failure");
                require(!frame.scatter_halation_resources().descriptor && !frame.diffusion_resources().active, "aborted attachment failure exposed prepared optics");
                std::cout << "preserved attachment checkpoint route=" << Spektrafilm::scan_route_key(route) << " carrier=" << carrier << " status={3,0,0} diagnostic=" << gateFailure.diagnostic << '\n';
            }
        }
        const std::array cases{Case{"negative-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                               Case{"negative-print", Spektrafilm::ScanRoute::NegativePrintScan, 1, 4, 0, 0},
                               Case{"positive-direct", Spektrafilm::ScanRoute::PositiveDirectScan, 2, 3, 11, -3},
                               Case{"positive-print", Spektrafilm::ScanRoute::PositivePrintScan, 0, 4, 0, 0}};
        std::ifstream fixture(JUICER_PROCESSOR_REFERENCE_PATH);
        require(static_cast<bool>(fixture), "workspace processor fixture unavailable");
        unsigned pixels = 0;
        for (const auto& test : cases) {
            std::string name;
            std::size_t count = 0;
            fixture >> name >> count;
            require(fixture && name == test.name && count > 0, "workspace fixture membership");
            std::vector<float> expected(count);
            for (float& value : expected) {
                fixture >> value;
            }
            require(static_cast<bool>(fixture), "workspace fixture truncated");
            nlohmann::json words = nlohmann::json::array();
            for (float value : expected) {
                words.push_back(std::bit_cast<std::uint32_t>(value));
            }
            words = PrintIlluminantFixtures::apply(std::string(JUICER_PROCESSOR_REFERENCE_PATH) + "." + test.name, words);
            for (std::size_t i = 0; i < expected.size(); ++i) {
                expected[i] = std::bit_cast<float>(words.at(i).get<std::uint32_t>());
            }
            InstanceState state;
            JuicerCuda::ExecutorTest::reset_workspace_transitions();
            compare_pixels(test.name, render_case(test, parameters_for(test), state, false, ExecutionPath::Direct, DestinationLayout::Matching, next++), expected);
            require(JuicerCuda::ExecutorTest::workspaceHandoff.has_value(), "actual schedule handoff not observed");
            ++pixels;
            std::cout << "workspace actual schedule route=" << Spektrafilm::scan_route_key(test.route)
                      << " handoff=" << static_cast<unsigned>(*JuicerCuda::ExecutorTest::workspaceHandoff) << '\n';
        }
        unsigned injectedFailures = 0;
        std::vector<float> latestPixels;
        const auto run = [&](const Case& test, const ParamSnapshot& params, unsigned reject, JuicerCuda::ExecutorTest::WorkspaceRejection kind, bool failureExpected) {
            using namespace JuicerCuda::ExecutorTest;
            reset_workspace_transitions();
            require(arm_workspace_rejection(reject, kind), "workspace seam configuration rejected");
            require(!arm_workspace_rejection(7, kind) && !arm_workspace_rejection(1, static_cast<WorkspaceRejection>(255)),
                    "invalid seam configuration replaced armed input");
            InstanceState state;
            bool failed = false;
            JuicerCuda::Failure failure;
            try {
                const auto actual = render_case(test, params, state, false, ExecutionPath::Direct, DestinationLayout::Matching, next++);
                require(!actual.empty() && std::all_of(actual.begin(), actual.end(), [](float value) {
                    return std::isfinite(value);
                }),
                        "workspace actual pixels unavailable");
                latestPixels = actual;
                ++pixels;
            } catch (const JuicerCuda::ExecutionFailure& error) {
                failure = error.failure;
                failed = true;
                ++injectedFailures;
            }
            require(failed == failureExpected, std::string(test.name) + ": unexpected workspace failure: " + failure.diagnostic);
            const auto observed = workspace_transitions();
            if (failed) {
                const auto status = kind == WorkspaceRejection::Capacity       ? FjStatus{FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}
                                    : kind == WorkspaceRejection::InvalidValue ? JuicerCuda::runtime_failure_status(cudaErrorInvalidValue)
                                                                               : JuicerCuda::runtime_failure_status(cudaErrorContextIsDestroyed);
                const std::string text = kind == WorkspaceRejection::Capacity       ? "workspace candidate capacity rejection"
                                         : kind == WorkspaceRejection::InvalidValue ? "workspace candidate invalid-value rejection"
                                                                                    : "workspace candidate context-loss rejection";
                require(failure.status.category == status.category && failure.status.api == status.api && failure.status.native_code == status.native_code && failure.diagnostic == text,
                        "workspace failure envelope or primary failure changed");
                require(observed.classified.category == status.category && observed.recoveryPending == (kind == WorkspaceRejection::ContextLoss), "workspace typed recovery classification changed");
                require(observed.candidateCount == (kind == WorkspaceRejection::Capacity ? 6u : 1u) && !observed.admitted &&
                            observed.finishes == 0 && observed.aborts == 1 && !observed.captureEnqueued && !observed.scannerEnqueued,
                        "failed workspace published or advanced candidates");
                require(observed.postFrame && *observed.postFrame == *observed.candidates[observed.candidateCount - 1].layout,
                        "failed candidate lost post-frame shedding layout");
            } else {
                require(observed.finishes == 1 && observed.aborts == 0 && observed.scannerEnqueued && workspaceHandoff.has_value(), "workspace successful schedule/finish missing");
                if (observed.admitted) {
                    require(observed.bindingsObserved && observed.scannerMapping && observed.grainMapping && observed.postFrame.has_value(), "workspace plane association or retention changed");
                    require(JuicerCuda::ResourceManager::scratch_request_descriptor_is_valid(observed.admitted->allocation()), "admitted projection invalid");
                    require(*observed.admitted == *observed.postFrame, "post-frame retention differs from admitted build candidate");
                }
                require(observed.captureEnqueued == (*workspaceHandoff == WorkspaceHandoff::MaterializedDensity), "capture result did not follow actual development");
            }
            for (unsigned i = 0; i < observed.candidateCount; ++i) {
                const auto& candidate = observed.candidates[i];
                require(candidate.layout.has_value() && candidate.backendCalls >= 1, "candidate did not reach validated backend");
                require(candidate.succeeded == (!failureExpected && i == reject), "outer candidate decision changed across backend retries");
                if (!candidate.succeeded) {
                    require(candidate.result.category == (kind == WorkspaceRejection::Capacity       ? FJ_STATUS_ALLOCATION_FAILURE
                                                          : kind == WorkspaceRejection::InvalidValue ? FJ_STATUS_CUDA_FAILURE
                                                                                                     : FJ_STATUS_CONTEXT_LOSS),
                            "candidate typed rejection changed");
                }
                std::cout << test.name << " candidate=" << i << " backend_calls=" << candidate.backendCalls << " success=" << candidate.succeeded
                          << " generation=" << candidate.layout->allocation().generation << '\n';
            }
            std::cout << "workspace cohort route=" << Spektrafilm::scan_route_key(test.route) << " case=" << test.name
                      << " reject=" << reject << " failed=" << failed << " source_cache=" << observed.sourceCacheChannels
                      << " final_cache=" << observed.finalCacheChannels << '\n';
            return observed;
        };
        for (unsigned route = 0; route < 4; ++route) {
            const Case test{"workspace-route", static_cast<Spektrafilm::ScanRoute>(route), 0, 4, 0, 0};
            auto params = parameters_for(test);
            params.grainControls.active = false;
            params.dirCouplers.active = false;
            params.glareActive = false;
            params.scannerUnsharpMask = {0.0, 0.0};
            const auto noWork = run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
            require(!noWork.requested && !noWork.admitted && noWork.candidateCount == 0, "no-family schedule requested scratch");
            params.dirCouplers.active = true;
            params.scannerUnsharpMask = {0.5, 0.5};
            run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
            params.grainControls.active = true;
            params.grainControls.sublayersActive = true;
            params.grainControls.chromaSharedWeight = 1.0f;
            Case strict = test;
            strict.name = "strict-yvv";
            strict.width = 512;
            strict.height = 48;
            if (route < 2) {
                std::vector<float> baseline;
                for (unsigned reject = 0; reject <= 6; ++reject) {
                    const auto observed = run(strict, params, reject, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, reject == 6);
                    const std::array raw{3, 3, 3, 3, 3, 1};
                    const std::array temp{3, 3, 3, 2, 1, 1};
                    const std::array cache{3, 2, 0, 0, 0, 0};
                    for (unsigned i = 0; i < observed.candidateCount; ++i) {
                        const auto& allocation = observed.candidates[i].layout->allocation();
                        require(allocation.spatialDirPlaneRoles.rawCorrectionPlanes == raw[i] && allocation.spatialDirPlaneRoles.filteredCorrectionPlanes == 3 &&
                                    allocation.spatialDirPlaneRoles.filterTempPlanes == temp[i] && allocation.spatialDirPlaneRoles.cachedLogRawPlanes == cache[i] &&
                                    allocation.spatialDirTargetPlaneRoles.rawCorrectionPlanes == raw[i] && allocation.spatialDirTargetPlaneRoles.filterTempPlanes == temp[i] &&
                                    allocation.spatialDirTargetPlaneRoles.cachedLogRawPlanes == cache[i],
                                "strict-YVV candidate order/target changed");
                    }
                    if (reject < 6) {
                        if (reject == 0) {
                            baseline = latestPixels;
                        } else {
                            require(latestPixels.size() == baseline.size(), "fallback pixel count changed");
                            for (std::size_t i = 0; i < baseline.size(); ++i) {
                                RenderAssertions::require_same_bits("strict-YVV candidate output", latestPixels[i], baseline[i]);
                            }
                        }
                        require(observed.admitted->grain_storage() == (reject == 5 ? GrainStorage::Dedicated : GrainStorage::DirRaw), "fallback grain binding changed");
                        require(observed.sourceCacheChannels == (reject == 0 ? 3u : reject == 1 ? 2u
                                                                                                : 0u),
                                "admitted source-cache count changed");
                    }
                }
                run(strict, params, 1, JuicerCuda::ExecutorTest::WorkspaceRejection::InvalidValue, true);
                run(strict, params, 1, JuicerCuda::ExecutorTest::WorkspaceRejection::ContextLoss, true);
                for (unsigned shape = 0; shape < 4; ++shape) {
                    params.grainControls.sublayersActive = shape >= 2;
                    params.grainControls.chromaSharedWeight = shape % 2 ? 1.0f : 0.0f;
                    params.grainControls.chromaIndependentWeight = shape % 2 ? 0.0f : 1.0f;
                    run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
                }
                params.grainControls.debugView = 1;
                run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
                params.grainControls.debugView = 0;
                params.cameraDiffusion.active = true;
                const auto diffusion = run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
                require(diffusion.admitted->grain_storage() == GrainStorage::Dedicated, "diffusion used DIR raw grain");
            } else {
                const auto positive = run(strict, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
                require(positive.admitted->scanner_storage() == ScannerStorage::Dedicated && positive.admitted->grain_storage() == GrainStorage::Dedicated,
                        "positive route aliased corrections");
                require(positive.sourceCacheChannels == 0 && positive.finalCacheChannels == 3 && positive.finalDevelopment.has_value(), "later cache staging missing");
            }
            params.grainControls.active = false;
            params.cameraDiffusion.active = false;
            params.filmDustAmount = 1.0;
            params.filmScratchAmount = 1.0;
            params.gateDustAmount = 1.0;
            params.gateScratchAmount = 1.0;
            params.gateWeaveAmount = 1.0;
            run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
        }
        for (bool camera : {false, true}) {
            Case test{"halation-carrier", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0, true};
            auto params = parameters_for(test);
            params.cameraDiffusion.active = camera;
            params.enlargerDiffusion.active = true;
            run(test, params, 0, JuicerCuda::ExecutorTest::WorkspaceRejection::Capacity, false);
        }
        JuicerCuda::ExecutorTest::reset_workspace_transitions();
        std::cout << "WorkspaceTransitions replacement: genuine_pixels=" << pixels << " construction_failures=4 checkpoint_failures=6 injected_failures=" << injectedFailures << '\n';
    }

    enum class EffectsCallerPath : std::uint8_t {
        Pixels,
        StopBeforePreparation,
        BeforeBuilder
    };

    void run_effects_frame_cases(const std::filesystem::path& manifestPath, const std::string& preset) {
        using Json = nlohmann::json;
        const auto manifest = Json::parse(std::ifstream(manifestPath));
        const auto captured = load_shared_fixture(manifestPath.parent_path() / manifest.at("presets").at(preset).at("native").get<std::string>(), "manifest.json");
        if (captured.at("preset") != preset || captured.at("cases").size() != 347 || captured.at("products").size() != 72)
            throw std::runtime_error("effects caller independent context/membership changed");
        const auto scalar = [](const Json& v) {
            return std::bit_cast<float>(v.get<std::uint32_t>());
        };
        const auto wide = [](const Json& v) {
            return std::bit_cast<double>(v.get<std::uint64_t>());
        };
        const auto extent = [](const Json& v) -> Spektrafilm::FilmJuicerEffectsFrameExtent {
            return {v.at(0), v.at(1), v.at(2), v.at(3)};
        };
        const auto rect = [](const Spektrafilm::FilmJuicerEffectsFrameExtent& v) -> JuicerCuda::FrameRect {
            return {v.x, v.y, v.x + v.width, v.y + v.height};
        };
        const auto fields = [](const Spektrafilm::FilmJuicerEffectsFrameExtent& v) -> Json {
            return {v.x, v.y, v.width, v.height};
        };
        DeviceFrame device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "effects caller stream");
        constexpr std::size_t kSamples = std::size_t{96} * 71 * 4;
        constexpr std::size_t kBytes = kSamples * sizeof(float);
        require_cuda(cudaMalloc(&device.source, kBytes), "effects caller input");
        require_cuda(cudaMalloc(&device.destination, kBytes), "effects caller output");
        const std::vector<float> source(kSamples, 0.18f), canary(kSamples, kCanary);
        require_cuda(cudaMemcpy(device.source, source.data(), kBytes, cudaMemcpyHostToDevice), "effects caller initialized input");
        int deviceId = 0;
        CUcontext context = nullptr;
        require_cuda(cudaGetDevice(&deviceId), "effects caller device");
        if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context)
            throw std::runtime_error("effects caller exact context");
        unsigned runs = 0, pixels = 0, constructions = 0, failures = 0;
        for (unsigned route = 0; route < 4; ++route) {
            const auto run = [&](const Json& row, EffectsCallerPath path, bool fullFallback = false) {
                const bool genuine = row.contains("spec");
                if (!genuine && path == EffectsCallerPath::Pixels)
                    throw std::runtime_error("literal effects construction cannot execute pixels");
                const auto& v = row.at("input");
                const auto render = extent(v.at("render_extent")), full = extent(v.at("pixel_definition"));
                ParamSnapshot params;
                params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
                params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
                params.cameraFilmFormatLongEdgeMm = 36.0f;
                params.dirCouplers.active = false;
                params.grainControls.active = false;
                if (genuine) {
                    const auto& spec = row.at("spec");
                    params.grainControls.active = spec.at("grain_active");
                    params.grainControls.debugView = spec.at("debug");
                    params.filmDustAmount = scalar(spec.at("amounts")[0]);
                    params.filmScratchAmount = scalar(spec.at("amounts")[1]);
                    params.gateDustAmount = scalar(spec.at("amounts")[2]);
                    params.gateScratchAmount = scalar(spec.at("amounts")[3]);
                    params.gateWeaveAmount = wide(spec.at("weave"));
                }
                PreparedState state = JuicerTestSupport::prepare_product(params);
                std::array<char, 1024> stateErrorText{};
                FjErrorBuffer stateError{stateErrorText.data(), stateErrorText.size(), 0};
                FjPreparedStateView stateView{};
                if (state.view(stateView, stateError).category != FJ_STATUS_SUCCESS)
                    throw std::runtime_error(std::string(stateErrorText.data(), stateError.length));
                const auto bounds = rect(full);
                const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(
                    bounds, bounds, static_cast<Spektrafilm::AutoExposureMethod>(stateView.input_policy.meter));
                const std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
                const std::optional<ScatterHalationFrameDescriptor> halation;
                Spektrafilm::FilmJuicerEffectsRecipe ignored;
                const auto geometry = JuicerTestSupport::effects_frame_input(v, ignored).geometry;
                const auto rowBytes = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(full.width) * 4 * sizeof(float));
                const JuicerCuda::ExecutionFrame frame{
                    bounds, rect(render), fullFallback ? JuicerCuda::FrameRect{} : bounds, reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<unsigned char*>(device.destination), rowBytes, rowBytes, path == EffectsCallerPath::BeforeBuilder ? 2 : 4, device.stream, diffusion, halation, geometry, scalar(v.at("pixel_size_um")), wide(v.at("frame_time")), wide(v.at("frame_rate")), v.at("session_seed").get<std::uint64_t>(), v.at("clip_token").get<std::uintptr_t>(), meter, false, false};
                ++runs;
                unsigned abortCalls = 0;
                require_cuda(cudaMemcpy(device.destination, canary.data(), kBytes, cudaMemcpyHostToDevice), "effects caller canary");
                JuicerCuda::ExecutorTest::reset_effects_frame_observation();
                JuicerCuda::ExecutorTest::FailureObservation events;
                JuicerCuda::ExecutorTest::ResourceFailures resources;
                const bool success = !v.at("active").get<bool>() || row.at("success").get<bool>();
                const bool early = path == EffectsCallerPath::BeforeBuilder || !success;
                const bool observeNoResourceWork = early ||
                                                   (path == EffectsCallerPath::StopBeforePreparation && success);
                if (observeNoResourceWork) {
                    resources.lutPosition = 1;
                    JuicerCuda::ExecutorTest::observation = &events;
                    JuicerCuda::ExecutorTest::resourceFailures = &resources;
                }
                FjStatus result{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
                std::string failureText;
                bool recoveryPending = false;
                FrameDescriptorObservation frameMath{};
                JuicerCuda::ExecutorTest::EffectsFrameObservation observed{};
                LegacyCombinedResult legacy{};
                if (genuine) {
                    AbortAfterNativeDescriptor stagedAbort{};
                    const FjAbortCallback actualAbort{&abort_after_native_descriptor, &stagedAbort};
                    const FjAbortCallback selectedAbort = path == EffectsCallerPath::Pixels ||
                                                                  (!success && path != EffectsCallerPath::BeforeBuilder)
                                                              ? FjAbortCallback{}
                                                              : actualAbort;
                    const FjCallCompletionResult rendered = render_held_prepared_state(
                        state, prepared_raw_frame(frame), 0x645700 + route, runs, selectedAbort, &frameMath, &recoveryPending, failureText);
                    result = rendered.outcome.status;
                    abortCalls = selectedAbort.query == &abort_after_native_descriptor ? stagedAbort.calls : 0;
                    observed = JuicerCuda::ExecutorTest::effects_frame_observation();
                } else {
                    const auto raw = prepare_literal_frame_construction(
                        FrameObservationKind::Effects, v, static_cast<Spektrafilm::ScanRoute>(route), scalar(v.at("pixel_size_um")), &build_literal_effects_math);
                    if (raw.rawFacadeCalls != 1 || raw.status.category != (success ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE) ||
                        (success && (!raw.effects || JuicerTestSupport::effects_frame_fields(*raw.effects) != row.at("native"))) ||
                        (!success && raw.diagnostic != row.at("diagnostic").get<std::string>()))
                        throw std::runtime_error(v.at("id").get<std::string>() + ": raw effects facade/native composer witness changed");
                    legacy = sf_temp_bridge_native_combined_call(
                        FrameObservationKind::Effects, params, frame, JuicerCuda::ResourceManager::InstanceToken{static_cast<std::uint64_t>(0x645700 + route)}, runs, {deviceId, context}, [&](FocusedRenderStateBuildProduct& product) {
                            JuicerTestSupport::effects_frame_input(v, product.recipe.filmJuicerEffects);
                            product.recipe.filmRaw.filmFormatLongEdgeMm = scalar(v.at("film_format_long_edge_mm"));
                        });
                    result = legacy.status;
                    failureText = legacy.diagnostic;
                    abortCalls = legacy.abortCalls;
                    observed = legacy.effects;
                }
                JuicerCuda::ExecutorTest::observation = nullptr;
                JuicerCuda::ExecutorTest::resourceFailures = nullptr;
                const auto name = v.at("id").get<std::string>();
                const bool descriptorVisited = path != EffectsCallerPath::BeforeBuilder &&
                                               (!genuine || path == EffectsCallerPath::Pixels || !success ||
                                                (path == EffectsCallerPath::StopBeforePreparation && success));
                if (genuine && (frameMath.callbackCalls != (descriptorVisited ? 1u : 0u) ||
                                frameMath.providerCalls != (descriptorVisited ? 1u : 0u)))
                    throw std::runtime_error(name + ": actual prepared-state callback/provider visit count changed");
                const auto geometryFields = [&](const Spektrafilm::FilmJuicerEffectsGeometry& g) -> Json {
                    return {fields(g.pixelDefinition), std::bit_cast<std::uint64_t>(g.canonicalX), std::bit_cast<std::uint64_t>(g.canonicalY), std::bit_cast<std::uint64_t>(g.canonicalWidth), std::bit_cast<std::uint64_t>(g.canonicalHeight), std::bit_cast<std::uint64_t>(g.scaleX), std::bit_cast<std::uint64_t>(g.scaleY), std::bit_cast<std::uint64_t>(g.pixelAspectRatio)};
                };
                if (genuine && path == EffectsCallerPath::StopBeforePreparation && success) {
                    JuicerCudaTest::PreparedFrameFixture::FrameFacts witnessFacts;
                    witnessFacts.renderExtent = {render.x, render.y, render.width, render.height};
                    witnessFacts.fullFrameExtent = {full.x, full.y, full.width, full.height};
                    witnessFacts.pixelDefinition = witnessFacts.fullFrameExtent;
                    witnessFacts.pixelSizeUm = scalar(v.at("pixel_size_um"));
                    witnessFacts.filmFormatLongEdgeMm = stateView.input_policy.film_format_long_edge_mm;
                    witnessFacts.canonicalX = geometry.canonicalX;
                    witnessFacts.canonicalY = geometry.canonicalY;
                    witnessFacts.canonicalWidth = geometry.canonicalWidth;
                    witnessFacts.canonicalHeight = geometry.canonicalHeight;
                    witnessFacts.scaleX = geometry.scaleX;
                    witnessFacts.scaleY = geometry.scaleY;
                    witnessFacts.pixelAspectRatio = geometry.pixelAspectRatio;
                    witnessFacts.frameTime = wide(v.at("frame_time"));
                    witnessFacts.frameRate = wide(v.at("frame_rate"));
                    witnessFacts.sessionSeed = v.at("session_seed");
                    witnessFacts.clipToken = v.at("clip_token");
                    FrameConstructionResult witness{FrameObservationKind::Effects};
                    std::string witnessDiagnostic;
                    if (!JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                            state, witnessFacts, &witness, &capture_held_frame_construction, witnessDiagnostic)) {
                        std::string message = name;
                        message += ": held-state construction witness: ";
                        message += witnessDiagnostic;
                        throw std::runtime_error(message);
                    }
                    if (witness.status.category != FJ_STATUS_SUCCESS || !witness.effects ||
                        JuicerTestSupport::effects_frame_fields(*witness.effects) != row.at("native"))
                        throw std::runtime_error(name + ": held effects construction witness fields/hash changed");
                }
                if (observed.count != (descriptorVisited ? 1u : 0u))
                    throw std::runtime_error(name + ": effects helper invocation count");
                if (descriptorVisited) {
                    if (observed.route != params.scanRoute || fields(observed.renderExtent) != v.at("render_extent") || fields(observed.fullFrameExtent) != fields(full) || geometryFields(observed.geometry) != geometryFields(geometry) || std::bit_cast<std::uint32_t>(observed.filmFormatLongEdgeMm) != v.at("film_format_long_edge_mm") || std::bit_cast<std::uint32_t>(observed.pixelSizeUm) != v.at("pixel_size_um") || std::bit_cast<std::uint64_t>(observed.frameTime) != v.at("frame_time") || std::bit_cast<std::uint64_t>(observed.frameRate) != v.at("frame_rate") || observed.sessionSeed != v.at("session_seed") || observed.clipToken != v.at("clip_token") || observed.success != success)
                        throw std::runtime_error(name + ": current callback effects facts");
                    const bool present = success && v.at("active").get<bool>();
                    if (observed.descriptor.has_value() != present || (present && JuicerTestSupport::effects_frame_fields(*observed.descriptor) != row.at("native")))
                        throw std::runtime_error(name + ": complete effects policies/geometry/identity or failed/inactive omission");
                }
                const auto expectedStatus = early ? FJ_STATUS_PREPARATION_FAILURE : path == EffectsCallerPath::Pixels ? FJ_STATUS_SUCCESS
                                                                                                                      : FJ_STATUS_CANCELLED;
                const unsigned expectedAbortCalls = genuine && path == EffectsCallerPath::StopBeforePreparation && success
                                                        ? 3u
                                                    : !early && path != EffectsCallerPath::Pixels ? 1u
                                                                                                  : 0u;
                if (result.category != expectedStatus || result.api != 0 || result.native_code != 0 || recoveryPending || abortCalls != expectedAbortCalls)
                    throw std::runtime_error(std::string(name).append(": effects caller stage/status: ").append(failureText));
                if (early) {
                    const auto expectedText = path == EffectsCallerPath::BeforeBuilder
                                                  ? route % 2 ? "UnsupportedPrintComponentCountForPhase4C" : "UnsupportedDirectComponentCountForPhase3C"
                                                  : row.at("diagnostic").get<std::string>();
                    if (failureText != expectedText || events.count != 0 || !events.armed || resources.lutAttempts != 0 || resources.stbnAttempts != 0 || resources.wangAttempts != 0 || resources.injections != 0)
                        throw std::runtime_error(name + ": failed effects reached preparation/upload/launch");
                    ++failures;
                } else if (path == EffectsCallerPath::StopBeforePreparation &&
                           (events.count != 0 || resources.lutAttempts != 0 || resources.stbnAttempts != 0 ||
                            resources.wangAttempts != 0 || resources.injections != 0)) {
                    throw std::runtime_error(name + ": cancellation entered resource preparation");
                }
                std::vector<float> actual(kSamples);
                require_cuda(cudaMemcpyAsync(actual.data(), device.destination, kBytes, cudaMemcpyDeviceToHost, device.stream), "effects caller readback");
                require_cuda(cudaStreamSynchronize(device.stream), "effects caller completion");
                if (path == EffectsCallerPath::Pixels) {
                    if (!genuine || early || actual == canary)
                        throw std::runtime_error("genuine effects pixel execution missing");
                    const auto count = static_cast<std::size_t>(full.width) * static_cast<std::size_t>(full.height) * 4;
                    for (std::size_t i = 0; i < count; ++i) {
                        if (i % 4 == 3 ? std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(0.18f) : !std::isfinite(actual[i]) || actual[i] < 0.0f || actual[i] > 1.0f)
                            throw std::runtime_error("effects genuine pixel/alpha contract");
                    }
                    ++pixels;
                } else {
                    if (actual != canary)
                        throw std::runtime_error("effects construction-only execution wrote pixels");
                    if (!early)
                        ++constructions;
                }
                std::cout << "effects caller route=" << route << " name=" << name << " pixels=" << (path == EffectsCallerPath::Pixels) << " fallback=" << fullFallback << " builder_calls=" << observed.count << " success=" << success << " passed\n";
            };
            for (const auto& row : captured.at("products")) {
                if (row.at("route") != route)
                    continue;
                const auto& spec = row.at("spec");
                const bool pixel = spec.at("pixels");
                const bool witness = spec.at("id") == "off";
                if (witness && fj_test_effects_frame_fault(1).category != 0)
                    throw std::runtime_error("arm inactive effects delegation witness");
                run(row, pixel ? EffectsCallerPath::Pixels : EffectsCallerPath::StopBeforePreparation);
                if (witness) {
                    FjEffectsFrameInput active{};
                    active.active = 1;
                    FjEffectsFrameMath output{};
                    FjEffectsFrameFailure failure{};
                    if (fj_test_effects_frame_prepare(&active, &output, &failure, nullptr).category != 8 || failure.reason != 4)
                        throw std::runtime_error("inactive effects helper consumed Rust mathematics");
                }
                if (spec.at("id") == "mixed") {
                    run(row, EffectsCallerPath::BeforeBuilder);
                    run(row, EffectsCallerPath::StopBeforePreparation, true);
                }
            }
            const std::array names{
                "mixed", "active-zero-work", "active-zero-work-recipe-zero", "inactive", "inactive-bad-identity", "film_dust", "film_scratch", "gate_dust", "gate_scratch", "weave-1", "weave-nan", "geometry-roi", "geometry-negative", "weave-roi", "film-time--1.75", "film_dust-ignored-frame_rate", "gate_dust-ignored-frame_time", "gate_dust-ignored-frame_rate", "gate_dust-ignored-pixel_size_um", "weave-frame_rate-0", "film_format_long_edge_mm-zero"};
            for (const auto& row : captured.at("cases")) {
                const auto name = row.at("input").at("id").get<std::string>();
                if (std::find(names.begin(), names.end(), name) != names.end())
                    run(row, EffectsCallerPath::StopBeforePreparation);
            }
            if (fj_cuda_retire_instance(JuicerCuda::borrowed_owner(), 0x645700 + route, nullptr).category != 0)
                throw std::runtime_error("effects caller membership retirement");
        }
        if (runs != 164 || pixels != 32 || constructions != 112 || failures != 20)
            throw std::runtime_error("effects caller exact pixel/failure/stop membership");
        std::cout << "effects caller runs=" << runs << " genuine pixels=" << pixels << " construction-only=" << constructions << " failures=" << failures << " passed\n";
    }

    enum class GrainCallerPath : std::uint8_t {
        Pixels,
        StopBeforePreparation,
        BeforeBuilder
    };

    void run_grain_frame_cases(const std::filesystem::path& manifestPath, const std::string& preset) {
        using Json = nlohmann::json;
        const auto manifest = Json::parse(std::ifstream(manifestPath));
        const auto captured = Json::parse(std::ifstream(manifestPath.parent_path() / manifest.at("presets").at(preset).at("native").get<std::string>()));
        if (captured.at("preset") != preset || captured.at("cases").size() != 153 || captured.at("products").size() != 44)
            throw std::runtime_error("grain caller independent context/membership changed");
        const auto scalar = [](const Json& v) {
            return std::bit_cast<float>(v.get<std::uint32_t>());
        };
        const auto wide = [](const Json& v) {
            return std::bit_cast<double>(v.get<std::uint64_t>());
        };
        const auto extent = [](const Json& v) -> Spektrafilm::VisualGrainFrameExtent {
            return {v.at(0), v.at(1), v.at(2), v.at(3)};
        };
        const auto rect = [](const Spektrafilm::VisualGrainFrameExtent& v) -> JuicerCuda::FrameRect {
            return {v.x, v.y, v.x + v.width, v.y + v.height};
        };
        const auto fields = [](const Spektrafilm::VisualGrainFrameExtent& v) -> Json {
            return {v.x, v.y, v.width, v.height};
        };
        DeviceFrame device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "grain caller stream");
        constexpr std::size_t kSamples = std::size_t{96} * 71 * 4;
        constexpr std::size_t kBytes = kSamples * sizeof(float);
        require_cuda(cudaMalloc(&device.source, kBytes), "grain caller input");
        require_cuda(cudaMalloc(&device.destination, kBytes), "grain caller output");
        const std::vector<float> source(kSamples, 0.18f), canary(kSamples, kCanary);
        require_cuda(cudaMemcpy(device.source, source.data(), kBytes, cudaMemcpyHostToDevice), "grain caller initialized input");
        int deviceId = 0;
        CUcontext context = nullptr;
        require_cuda(cudaGetDevice(&deviceId), "grain caller device");
        if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context)
            throw std::runtime_error("grain caller exact context");
        unsigned runs = 0, pixels = 0, constructions = 0, failures = 0;
        const auto& base = captured.at("cases")[0].at("input");
        for (unsigned route = 0; route < 4; ++route) {
            const auto run = [&](const Json& row, GrainCallerPath path, bool fullFallback = false) {
                const bool genuine = row.contains("spec");
                if (!genuine && path == GrainCallerPath::Pixels)
                    throw std::runtime_error("literal grain construction cannot execute pixels");
                const auto& v = genuine ? base : row.at("input");
                const auto render = extent(v.at("render_extent")), full = extent(v.at("full_frame_extent"));
                ParamSnapshot params;
                params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
                params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
                if (genuine) {
                    const auto& spec = row.at("spec");
                    params.grainControls.active = spec.at("active");
                    params.grainControls.sublayersActive = spec.at("layers");
                    params.grainControls.debugView = spec.at("debug");
                    params.grainControls.chromaMix = scalar(spec.at("chroma"));
                    params.grainControls.amplitude = scalar(spec.value("amplitude", Json(1065353216u)));
                    params.grainControls.particleAreaUm2 = scalar(spec.value("area", Json(1051428127u)));
                    params.cameraExposureCompensationEv = spec.value("ev", 0.0);
                } else {
                    params.dirCouplers.active = false;
                }
                PreparedState state = JuicerTestSupport::prepare_product(params);
                const auto bounds = rect(full);
                std::array<char, 1024> stateErrorText{};
                FjErrorBuffer stateError{stateErrorText.data(), stateErrorText.size(), 0};
                FjPreparedStateView stateView{};
                if (state.view(stateView, stateError).category != FJ_STATUS_SUCCESS)
                    throw std::runtime_error(std::string(stateErrorText.data(), stateError.length));
                const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(
                    bounds, bounds, static_cast<Spektrafilm::AutoExposureMethod>(stateView.input_policy.meter));
                const std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
                const std::optional<ScatterHalationFrameDescriptor> halation;
                const Spektrafilm::FilmJuicerEffectsGeometry geometry{};
                const auto rowBytes = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(full.width) * 4 * sizeof(float));
                const JuicerCuda::ExecutionFrame frame{
                    bounds, rect(render), fullFallback ? JuicerCuda::FrameRect{} : bounds, reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<unsigned char*>(device.destination), rowBytes, rowBytes, path == GrainCallerPath::BeforeBuilder ? 2 : 4, device.stream, diffusion, halation, geometry, scalar(v.at("pixel_size_um")), wide(v.at("frame_time")), wide(v.at("frame_rate")), v.at("session_seed").get<std::uint64_t>(), v.at("clip_token").get<std::uintptr_t>(), meter, false, false};
                ++runs;
                unsigned abortCalls = 0;
                require_cuda(cudaMemcpy(device.destination, canary.data(), kBytes, cudaMemcpyHostToDevice), "grain caller canary");
                JuicerCuda::ExecutorTest::reset_grain_frame_observation();
                JuicerCuda::ExecutorTest::FailureObservation events;
                JuicerCuda::ExecutorTest::ResourceFailures resources;
                const bool success = row.at("success");
                const bool early = path == GrainCallerPath::BeforeBuilder || !success;
                const bool observeNoResourceWork = early ||
                                                   (path == GrainCallerPath::StopBeforePreparation && success);
                if (observeNoResourceWork) {
                    resources.lutPosition = 1;
                    JuicerCuda::ExecutorTest::observation = &events;
                    JuicerCuda::ExecutorTest::resourceFailures = &resources;
                }
                FjStatus result{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
                std::string failureText;
                bool recoveryPending = false;
                FrameDescriptorObservation frameMath{};
                JuicerCuda::ExecutorTest::GrainFrameObservation observed{};
                if (genuine) {
                    AbortAfterNativeDescriptor stagedAbort{};
                    const FjAbortCallback actualAbort{&abort_after_native_descriptor, &stagedAbort};
                    const FjAbortCallback selectedAbort = path == GrainCallerPath::Pixels ||
                                                                  (!success && path != GrainCallerPath::BeforeBuilder)
                                                              ? FjAbortCallback{}
                                                              : actualAbort;
                    const FjCallCompletionResult rendered = render_held_prepared_state(
                        state, prepared_raw_frame(frame), 0x645600 + route, runs, selectedAbort, &frameMath, &recoveryPending, failureText);
                    result = rendered.outcome.status;
                    abortCalls = selectedAbort.query == &abort_after_native_descriptor ? stagedAbort.calls : 0;
                    observed = JuicerCuda::ExecutorTest::grain_frame_observation();
                } else {
                    const auto raw = prepare_literal_frame_construction(
                        FrameObservationKind::Grain, v, static_cast<Spektrafilm::ScanRoute>(route), scalar(v.at("pixel_size_um")), &build_literal_grain_math);
                    if (raw.rawFacadeCalls != 1 || raw.status.category != (success ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE) ||
                        (success && (!raw.grain || JuicerTestSupport::grain_frame_fields(*raw.grain) != row.at("native"))) ||
                        (!success && raw.diagnostic != row.at("diagnostic").get<std::string>()))
                        throw std::runtime_error(v.at("name").get<std::string>() + ": raw grain facade/native composer witness changed");
                    const auto legacy = sf_temp_bridge_native_combined_call(
                        FrameObservationKind::Grain, params, frame, JuicerCuda::ResourceManager::InstanceToken{static_cast<std::uint64_t>(0x645600 + route)}, runs, {deviceId, context}, [&](FocusedRenderStateBuildProduct& product) {
                            JuicerTestSupport::grain_frame_input(v, product.recipe.visualGrain, product.recipe.filmDevelop);
                        });
                    result = legacy.status;
                    failureText = legacy.diagnostic;
                    abortCalls = legacy.abortCalls;
                    observed = legacy.grain;
                }
                JuicerCuda::ExecutorTest::observation = nullptr;
                JuicerCuda::ExecutorTest::resourceFailures = nullptr;
                const auto name = genuine ? row.at("spec").dump() : v.at("name").get<std::string>();
                const bool descriptorVisited = path != GrainCallerPath::BeforeBuilder &&
                                               (!genuine || path == GrainCallerPath::Pixels || !success ||
                                                (path == GrainCallerPath::StopBeforePreparation && success));
                if (genuine && (frameMath.callbackCalls != (descriptorVisited ? 1u : 0u) ||
                                frameMath.providerCalls != (descriptorVisited ? 1u : 0u)))
                    throw std::runtime_error(name + ": actual prepared-state callback/provider visit count changed");
                const auto verifyGrainWitness = [&](const JuicerCuda::ExecutorTest::GrainFrameObservation& witness) {
                    if (witness.count != 1 || witness.route != params.scanRoute ||
                        fields(witness.renderExtent) != v.at("render_extent") ||
                        fields(witness.fullFrameExtent) != v.at("full_frame_extent") ||
                        std::bit_cast<std::uint32_t>(witness.pixelSizeUm) != v.at("pixel_size_um") ||
                        std::bit_cast<std::uint64_t>(witness.frameTime) != v.at("frame_time") ||
                        std::bit_cast<std::uint64_t>(witness.frameRate) != v.at("frame_rate") ||
                        witness.sessionSeed != v.at("session_seed") || witness.clipToken != v.at("clip_token") ||
                        witness.success != success)
                        throw std::runtime_error(name + ": current callback grain facts");
                    const bool present = success && row.at("native").at("active").get<bool>();
                    if (witness.descriptor.has_value() != present ||
                        (present && JuicerTestSupport::grain_frame_fields(*witness.descriptor) != row.at("native")))
                        throw std::runtime_error(name + ": complete successful grain fields/hash or absent failed/inactive descriptor");
                };
                if (genuine && path == GrainCallerPath::StopBeforePreparation && success) {
                    JuicerCuda::ExecutorTest::reset_grain_frame_observation();
                    if (genuine) {
                        JuicerCudaTest::PreparedFrameFixture::FrameFacts witnessFacts;
                        witnessFacts.renderExtent = {render.x, render.y, render.width, render.height};
                        witnessFacts.fullFrameExtent = {full.x, full.y, full.width, full.height};
                        witnessFacts.pixelDefinition = witnessFacts.fullFrameExtent;
                        witnessFacts.pixelSizeUm = scalar(v.at("pixel_size_um"));
                        witnessFacts.frameTime = wide(v.at("frame_time"));
                        witnessFacts.frameRate = wide(v.at("frame_rate"));
                        witnessFacts.sessionSeed = v.at("session_seed");
                        witnessFacts.clipToken = v.at("clip_token");
                        FrameConstructionResult witness{FrameObservationKind::Grain};
                        std::string witnessDiagnostic;
                        if (!JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                                state, witnessFacts, &witness, &capture_held_frame_construction, witnessDiagnostic)) {
                            std::string message = name;
                            message += ": held-state construction witness: ";
                            message += witnessDiagnostic;
                            throw std::runtime_error(message);
                        }
                        if (witness.status.category != FJ_STATUS_SUCCESS || !witness.grain ||
                            JuicerTestSupport::grain_frame_fields(*witness.grain) != row.at("native"))
                            throw std::runtime_error(name + ": held grain construction witness fields/hash changed");
                    }
                }
                if (observed.count != (descriptorVisited ? 1u : 0u))
                    throw std::runtime_error(name + ": grain helper invocation count");
                if (descriptorVisited)
                    verifyGrainWitness(observed);
                const auto expectedStatus = early ? FJ_STATUS_PREPARATION_FAILURE : path == GrainCallerPath::Pixels ? FJ_STATUS_SUCCESS
                                                                                                                    : FJ_STATUS_CANCELLED;
                const unsigned expectedAbortCalls = genuine && path == GrainCallerPath::StopBeforePreparation && success
                                                        ? 3u
                                                    : !early && path != GrainCallerPath::Pixels ? 1u
                                                                                                : 0u;
                if (result.category != expectedStatus || result.api != 0 || result.native_code != 0 || recoveryPending || abortCalls != expectedAbortCalls)
                    throw std::runtime_error(std::string(name).append(": grain caller stage/status: ").append(failureText));
                if (early) {
                    const auto expectedText = path == GrainCallerPath::BeforeBuilder
                                                  ? route % 2 ? "UnsupportedPrintComponentCountForPhase4C" : "UnsupportedDirectComponentCountForPhase3C"
                                                  : row.at("diagnostic").get<std::string>();
                    if (failureText != expectedText || events.count != 0 || !events.armed || resources.lutAttempts != 0 || resources.stbnAttempts != 0 || resources.wangAttempts != 0 || resources.injections != 0)
                        throw std::runtime_error(name + ": failed grain reached preparation/upload/launch");
                    ++failures;
                } else if (path == GrainCallerPath::StopBeforePreparation &&
                           (events.count != 0 || resources.lutAttempts != 0 || resources.stbnAttempts != 0 ||
                            resources.wangAttempts != 0 || resources.injections != 0)) {
                    throw std::runtime_error(name + ": cancellation entered resource preparation");
                }
                std::vector<float> actual(kSamples);
                require_cuda(cudaMemcpyAsync(actual.data(), device.destination, kBytes, cudaMemcpyDeviceToHost, device.stream), "grain caller readback");
                require_cuda(cudaStreamSynchronize(device.stream), "grain caller completion");
                if (path == GrainCallerPath::Pixels) {
                    if (!genuine || early || actual == canary)
                        throw std::runtime_error("genuine grain pixel execution missing");
                    const auto count = static_cast<std::size_t>(full.width) * static_cast<std::size_t>(full.height) * 4;
                    for (std::size_t i = 0; i < count; ++i) {
                        if (i % 4 == 3 ? std::bit_cast<std::uint32_t>(actual[i]) != std::bit_cast<std::uint32_t>(0.18f) : !std::isfinite(actual[i]) || actual[i] < 0.0f || actual[i] > 1.0f)
                            throw std::runtime_error("grain genuine pixel/alpha contract");
                    }
                    ++pixels;
                } else {
                    if (actual != canary)
                        throw std::runtime_error("grain construction-only execution wrote pixels");
                    if (!early)
                        ++constructions;
                }
                std::cout << "grain caller route=" << route << " name=" << name << " pixels=" << (path == GrainCallerPath::Pixels) << " fallback=" << fullFallback << " builder_calls=" << observed.count << " success=" << success << " passed\n";
            };
            for (const auto& row : captured.at("products")) {
                if (row.at("route") != route)
                    continue;
                const auto& spec = row.at("spec");
                const bool pixel = !spec.at("active").get<bool>() || (spec.at("debug") == 0 && !spec.contains("amplitude"));
                if (!spec.at("active").get<bool>() && fj_test_grain_frame_fault(1).category != 0)
                    throw std::runtime_error("arm inactive delegation witness");
                run(row, pixel ? GrainCallerPath::Pixels : GrainCallerPath::StopBeforePreparation);
                if (!spec.at("active").get<bool>()) {
                    FjGrainFrameInput active{};
                    active.active = 1;
                    FjGrainFrameMath output{};
                    FjGrainFrameFailure failure{};
                    if (fj_test_grain_frame_prepare(&active, &output, &failure, nullptr).category != 8 || failure.reason != 3)
                        throw std::runtime_error("inactive executor consumed Rust frame mathematics");
                }
                if (spec.at("active").get<bool>() && !spec.at("layers").get<bool>() && !spec.contains("amplitude")) {
                    run(row, GrainCallerPath::BeforeBuilder);
                    run(row, GrainCallerPath::StopBeforePreparation, true);
                }
            }
            const std::array names{
                "time--1.75", "time-1.5", "fps-23.976", "fps-0.2", "negative-origin", "partial-unblurred", "sigma-0.4", "sigma-0.4000001", "new-session", "new-clip", "mix", "S4D6-R1", "pixel-half", "pixel-eight", "geometry-96x71", "geometry-offset-96x71", "layers-film_layers_hash", "model-0", "frame-frame_time"};
            for (const auto& row : captured.at("cases")) {
                const auto& v = row.at("input");
                const auto name = v.at("name").get<std::string>();
                const auto logical = name.starts_with("positive/") ? name.substr(9) : name;
                if (v.at("capture_polarity") == route / 2 && v.at("film_polarity") == route / 2 && std::find(names.begin(), names.end(), logical) != names.end())
                    run(row, GrainCallerPath::StopBeforePreparation);
            }
            if (fj_cuda_retire_instance(JuicerCuda::borrowed_owner(), 0x645600 + route, nullptr).category != 0)
                throw std::runtime_error("grain caller membership retirement");
        }
        if (runs != 122 || pixels != 16 || constructions != 92 || failures != 14)
            throw std::runtime_error("grain caller exact pixel/failure/stop membership");
        std::cout << "grain caller runs=" << runs << " genuine pixels=" << pixels << " construction-only=" << constructions << " failures=" << failures << " passed\n";
    }

    void run_dir_frame_cases(const std::filesystem::path& manifestPath, const std::string& preset) {
        using Json = nlohmann::json;
        const auto manifest = Json::parse(std::ifstream(manifestPath));
        const auto captured = load_shared_fixture(manifestPath.parent_path() / manifest.at("presets").at(preset).at("native").get<std::string>(), "manifest.json");
        if (captured.at("preset") != preset || captured.at("cases").size() != 74 || captured.at("products").size() != 24) {
            throw std::runtime_error("DIR caller capture membership changed");
        }
        const auto extent = [](const Json& value) -> Spektrafilm::DirFrameExtent {
            return {value.at(0), value.at(1), value.at(2), value.at(3)};
        };
        const auto rect = [](Spektrafilm::DirFrameExtent value) -> JuicerCuda::FrameRect {
            return {value.x, value.y, value.x + value.width, value.y + value.height};
        };
        const auto fields = [](Spektrafilm::DirFrameExtent value) -> Json {
            return {value.x, value.y, value.width, value.height};
        };
        const auto scalar = [](const Json& value) {
            return std::bit_cast<float>(value.get<std::uint32_t>());
        };
        DeviceFrame device;
        require_cuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking), "DIR caller stream");
        constexpr std::size_t kSamples = std::size_t{96} * 71 * 4;
        constexpr std::size_t kBytes = kSamples * sizeof(float);
        require_cuda(cudaMalloc(&device.source, kBytes), "DIR caller source");
        require_cuda(cudaMalloc(&device.destination, kBytes), "DIR caller destination");
        const std::vector<float> source(kSamples, 0.18f), canary(kSamples, kCanary);
        require_cuda(cudaMemcpy(device.source, source.data(), kBytes, cudaMemcpyHostToDevice), "DIR caller input");
        int deviceId = 0;
        CUcontext context = nullptr;
        require_cuda(cudaGetDevice(&deviceId), "DIR caller device");
        if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context) {
            throw std::runtime_error("DIR caller context unavailable");
        }
        unsigned runs = 0;
        for (unsigned route = 0; route < 4; ++route) {
            ParamSnapshot params;
            params.scanRoute = static_cast<Spektrafilm::ScanRoute>(route);
            params.filmProfileKey = route < 2 ? "kodak_portra_400" : "fujifilm_provia_100f";
            params.grainControls.active = false;
            params.dirCouplers.diffusionSizeUm = 20.0f;
            params.dirCouplers.diffusionTailUm = 200.0f;
            params.dirCouplers.diffusionTailWeight = 0.03f;
            PreparedState state = JuicerTestSupport::prepare_product(params);
            std::array<char, 1024> stateErrorText{};
            FjErrorBuffer stateError{stateErrorText.data(), stateErrorText.size(), 0};
            FjPreparedStateView stateView{};
            if (state.view(stateView, stateError).category != FJ_STATUS_SUCCESS)
                throw std::runtime_error(std::string(stateErrorText.data(), stateError.length));
            const auto run = [&](const Json& row, bool genuine, bool beforeBuilder, bool executePixels, bool fullFallback) {
                const auto& input = row.at("input");
                const auto name = input.at("name").get<std::string>() + "/route-" + std::to_string(route);
                if (genuine && (row.at("recipe_hash") != stateView.recipe_hash || input.at("hash") != stateView.dir_recipe.hash)) {
                    throw std::runtime_error(name + ": genuine recipe identity changed");
                }
                const auto render = extent(input.at("render")), full = extent(input.at("full"));
                const auto sourceDomain = full.width > 0 && full.height > 0 ? full : Spektrafilm::DirFrameExtent{0, 0, 64, 48};
                const auto sourceBounds = rect(sourceDomain);
                const auto effectiveFull = fullFallback || full.width <= 0 || full.height <= 0 ? sourceDomain : full;
                const auto fullBounds = fullFallback ? JuicerCuda::FrameRect{} : rect(full);
                const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(
                    sourceBounds, sourceBounds, static_cast<Spektrafilm::AutoExposureMethod>(stateView.input_policy.meter));
                const std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
                const std::optional<ScatterHalationFrameDescriptor> halation;
                const Spektrafilm::FilmJuicerEffectsGeometry geometry{};
                const auto rowBytes = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(sourceDomain.width) * 4 * sizeof(float));
                const JuicerCuda::ExecutionFrame frame{sourceBounds, rect(render), fullBounds, reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<const unsigned char*>(device.source), reinterpret_cast<unsigned char*>(device.destination), rowBytes, rowBytes, beforeBuilder ? 2 : 4, device.stream, diffusion, halation, geometry, scalar(input.at("pixel")), 37.0, 24.0, 0x20260923, 0x5312, meter, false, false};
                ++runs;
                unsigned abortCalls = 0;
                require_cuda(cudaMemcpy(device.destination, canary.data(), kBytes, cudaMemcpyHostToDevice), "DIR caller canary");
                JuicerCuda::ExecutorTest::reset_dir_frame_observation();
                JuicerCuda::ExecutorTest::FailureObservation failureEvents;
                JuicerCuda::ExecutorTest::ResourceFailures resources;
                const bool success = row.at("success");
                const bool earlyFailure = beforeBuilder || !success;
                const bool observeNoResourceWork = earlyFailure || (!executePixels && !beforeBuilder && success);
                if (observeNoResourceWork) {
                    resources.lutPosition = 1;
                    JuicerCuda::ExecutorTest::observation = &failureEvents;
                    JuicerCuda::ExecutorTest::resourceFailures = &resources;
                }
                FjStatus status{FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
                std::string failureText;
                bool recoveryPending = false;
                FrameDescriptorObservation frameMath{};
                auto expected = row.at("native");
                if (!expected.at("label").is_null())
                    expected["label"] = route % 2 ? "print" : "direct";
                JuicerCuda::ExecutorTest::DirFrameObservation observed{};
                if (genuine) {
                    AbortAfterNativeDescriptor stagedAbort{};
                    const FjAbortCallback actualAbort{&abort_after_native_descriptor, &stagedAbort};
                    const FjAbortCallback selectedAbort = executePixels || (!success && !beforeBuilder)
                                                              ? FjAbortCallback{}
                                                              : actualAbort;
                    const FjCallCompletionResult rendered = render_held_prepared_state(
                        state, prepared_raw_frame(frame), 0x641300 + route, runs, selectedAbort, &frameMath, &recoveryPending, failureText);
                    status = rendered.outcome.status;
                    abortCalls = selectedAbort.query == &abort_after_native_descriptor ? stagedAbort.calls : 0;
                    observed = JuicerCuda::ExecutorTest::dir_frame_observation();
                } else {
                    const auto raw = prepare_literal_frame_construction(
                        FrameObservationKind::Dir, input, static_cast<Spektrafilm::ScanRoute>(route), scalar(input.at("pixel")), &build_literal_dir_math);
                    if (raw.rawFacadeCalls != 1 || raw.status.category != (success ? FJ_STATUS_SUCCESS : FJ_STATUS_PREPARATION_FAILURE) ||
                        (success && (!raw.dir || JuicerTestSupport::dir_frame_fields(*raw.dir) != expected)) ||
                        (!success && raw.diagnostic != row.at("diagnostic").get<std::string>()))
                        throw std::runtime_error(name + ": raw DIR facade/native composer witness changed");
                    const auto legacy = sf_temp_bridge_native_combined_call(
                        FrameObservationKind::Dir, params, frame, JuicerCuda::ResourceManager::InstanceToken{static_cast<std::uint64_t>(0x641300 + route)}, runs, {deviceId, context}, [&](FocusedRenderStateBuildProduct& product) {
                            auto& dir = product.recipe.dirCouplers;
                            dir.active = input.at("active");
                            dir.diffusionSizeUm = scalar(input.at("size"));
                            dir.diffusionTailUm = scalar(input.at("tail"));
                            dir.diffusionTailWeight = scalar(input.at("weight"));
                            dir.hash = input.at("hash");
                        });
                    status = legacy.status;
                    failureText = legacy.diagnostic;
                    abortCalls = legacy.abortCalls;
                    observed = legacy.dir;
                }
                JuicerCuda::ExecutorTest::observation = nullptr;
                JuicerCuda::ExecutorTest::resourceFailures = nullptr;
                const bool descriptorVisited = !beforeBuilder &&
                                               (!genuine || executePixels || !success || (!executePixels && success));
                if (genuine && (frameMath.callbackCalls != (descriptorVisited ? 1u : 0u) ||
                                frameMath.providerCalls != (descriptorVisited ? 1u : 0u)))
                    throw std::runtime_error(name + ": actual prepared-state callback/provider visit count changed");
                const auto verifyDirWitness = [&](const JuicerCuda::ExecutorTest::DirFrameObservation& witness,
                                                  bool expectedSuccess) {
                    if (witness.count != 1 || witness.route != params.scanRoute ||
                        witness.pixelSizeBits != input.at("pixel") ||
                        fields(witness.renderExtent) != input.at("render") ||
                        fields(witness.fullFrameExtent) != fields(effectiveFull) ||
                        witness.success != expectedSuccess ||
                        JuicerTestSupport::dir_frame_fields(witness.descriptor) != expected)
                        throw std::runtime_error(name + ": actual builder facts, fields or final hash changed");
                };
                if (genuine && !beforeBuilder && !executePixels && success) {
                    JuicerCudaTest::PreparedFrameFixture::FrameFacts witnessFacts;
                    witnessFacts.renderExtent = {render.x, render.y, render.width, render.height};
                    witnessFacts.fullFrameExtent = {effectiveFull.x, effectiveFull.y, effectiveFull.width, effectiveFull.height};
                    witnessFacts.pixelDefinition = witnessFacts.fullFrameExtent;
                    witnessFacts.pixelSizeUm = scalar(input.at("pixel"));
                    witnessFacts.frameTime = 37.0;
                    witnessFacts.frameRate = 24.0;
                    witnessFacts.sessionSeed = 0x20260923;
                    witnessFacts.clipToken = 0x5312;
                    FrameConstructionResult witness{FrameObservationKind::Dir};
                    std::string witnessDiagnostic;
                    if (!JuicerCudaTest::PreparedFrameFixture::with_prepared_frame_inputs(
                            state, witnessFacts, &witness, &capture_held_frame_construction, witnessDiagnostic)) {
                        std::string message = name;
                        message += ": held-state construction witness: ";
                        message += witnessDiagnostic;
                        throw std::runtime_error(message);
                    }
                    if (witness.status.category != FJ_STATUS_SUCCESS || !witness.dir ||
                        JuicerTestSupport::dir_frame_fields(*witness.dir) != expected)
                        throw std::runtime_error(name + ": held DIR construction witness fields/hash changed");
                }
                if (observed.count != (descriptorVisited ? 1u : 0u)) {
                    throw std::runtime_error(name + ": actual builder facts, fields or final hash changed");
                }
                if (descriptorVisited)
                    verifyDirWitness(observed, success);
                const auto expectedStatus = earlyFailure ? FJ_STATUS_PREPARATION_FAILURE : executePixels ? FJ_STATUS_SUCCESS
                                                                                                         : FJ_STATUS_CANCELLED;
                if (status.category != expectedStatus || status.api != FJ_API_NONE || status.native_code != 0 || recoveryPending ||
                    abortCalls != (!earlyFailure && !executePixels ? (genuine && success ? 3u : 1u) : 0u)) {
                    std::string message = name;
                    message += ": caller did not stop at the expected stage: ";
                    message += failureText;
                    throw std::runtime_error(message);
                }
                if (earlyFailure) {
                    const std::string expectedText = beforeBuilder
                                                         ? route % 2 ? "UnsupportedPrintComponentCountForPhase4C" : "UnsupportedDirectComponentCountForPhase3C"
                                                     : route % 2 ? "ResourceDescriptorMismatch phase=4C field=spatial_dir_descriptor"
                                                                 : "ResourceDescriptorMismatch phase=3D-3 field=spatial_dir_descriptor";
                    if (failureText != expectedText || failureEvents.count != 0 || !failureEvents.armed ||
                        resources.lutAttempts != 0 || resources.stbnAttempts != 0 || resources.wangAttempts != 0 || resources.injections != 0) {
                        throw std::runtime_error(name + ": failed mathematics reached resource preparation or launch");
                    }
                } else if (!executePixels && !beforeBuilder &&
                           (failureEvents.count != 0 || resources.lutAttempts != 0 || resources.stbnAttempts != 0 ||
                            resources.wangAttempts != 0 || resources.injections != 0)) {
                    throw std::runtime_error(name + ": cancellation entered resource preparation");
                }
                if (!executePixels) {
                    std::vector<float> actual(kSamples);
                    require_cuda(cudaMemcpyAsync(actual.data(), device.destination, kBytes, cudaMemcpyDeviceToHost, device.stream), "DIR caller output canary");
                    require_cuda(cudaStreamSynchronize(device.stream), "DIR caller completion");
                    if (actual != canary) {
                        throw std::runtime_error(name + ": stopped caller wrote pixels");
                    }
                } else {
                    require_cuda(cudaStreamSynchronize(device.stream), "DIR caller rendered completion");
                }
                std::cout << "DIR caller " << name << " fallback=" << fullFallback << " before=" << beforeBuilder << " pixels=" << executePixels << " passed\n";
            };
            for (const auto& row : captured.at("products")) {
                if (row.at("route") == route) {
                    const bool executePixels = row.at("input").at("name") == std::to_string(route) + "/pixel-8";
                    run(row, true, false, executePixels, false);
                    if (executePixels) {
                        run(row, true, true, false, false);
                        run(row, true, false, false, true);
                    }
                }
            }
            for (const char* name : {"inactive", "size-zero", "size-nan", "core-word-1", "core-word-1042983595", "fir", "core-word-1077936128", "tail-mixed", "tail-weight-one", "tail-zero", "pixel-change", "negative-origin", "pixel-zero", "offset", "tail-radius", "core-inf", "active-hash-zero"}) {
                const auto found = std::find_if(captured.at("cases").begin(), captured.at("cases").end(), [&](const auto& row) {
                    return row.at("input").at("name") == name;
                });
                if (found == captured.at("cases").end()) {
                    throw std::runtime_error("DIR caller literal missing");
                }
                run(*found, false, false, false, false);
            }
        }
        if (runs != 100) {
            throw std::runtime_error("DIR caller exact membership changed");
        }
    }

    void check_failure_order(const Case& test,
                             JuicerCuda::ExecutorTest::Delivery delivery,
                             bool deferredDirFailure,
                             FjStatus failureStatus,
                             const char* wording) {
        const bool contextLoss = failureStatus.category == FJ_STATUS_CONTEXT_LOSS;
        using JuicerCuda::ExecutorTest::Event;
        JuicerCuda::ExecutorTest::FailureObservation observation;
        observation.delivery = delivery;
        observation.status = failureStatus;
        observation.diagnostic = wording;
        observation.scanResult = deferredDirFailure
                                     ? JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::DeferredDirFailure
                                     : JuicerProcess::Root::PreparedCudaFrame::ScanErrorStageResult::Failed;
        observation.stage = Spektrafilm::scan_route_is_print(test.route)
                                ? "print_scan_error_stage"
                                : "direct_scan_error_stage";
        JuicerCuda::ExecutorTest::observation = &observation;
        InstanceState state;
        bool fatal = false;
        try {
            (void)render_case(test, parameters_for(test), state);
        } catch (const OFX::Exception::Suite& error) {
            fatal = error.status() == kOfxStatErrFatal;
            observation.record(Event::FatalMapped);
        } catch (...) {
            JuicerCuda::ExecutorTest::observation = nullptr;
            throw;
        }
        JuicerCuda::ExecutorTest::observation = nullptr;
        std::vector<Event> expected{Event::Injected, Event::Classified, Event::FrameAborted};
        if (deferredDirFailure) {
            expected.push_back(Event::Message);
        }
        expected.insert(expected.end(), {Event::RecoveryStarted, Event::RecoveryEnded, Event::FatalMapped});
        if (!fatal || observation.overflow || observation.count != expected.size() ||
            !std::equal(expected.begin(), expected.end(), observation.events.begin()) ||
            !observation.failureStageMatches || !observation.failureStatusMatches ||
            observation.nativeRecoveryPending != contextLoss ||
            observation.adapterRecoveryPending != contextLoss ||
            (deferredDirFailure && !observation.messageMatches) ||
            state.submissionSnapshotLatchValid == contextLoss) {
            throw std::runtime_error(std::string(test.name) + ": executor failure order or host mapping changed");
        }
        std::cout << test.name << " dir=" << deferredDirFailure << " context_loss=" << contextLoss
                  << " delivery=" << static_cast<int>(delivery)
                  << " classify/abort/message/recovery/fatal order passed\n";
    }

    void run_failure_order_cases() {
        const std::array<Case, 2> cases{{{"failure-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                                         {"failure-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}}};
        for (const Case& test : cases) {
            for (const auto delivery : {JuicerCuda::ExecutorTest::Delivery::Success,
                                        JuicerCuda::ExecutorTest::Delivery::Failure,
                                        JuicerCuda::ExecutorTest::Delivery::Throw}) {
                check_failure_order(test, delivery, true, JuicerCuda::runtime_failure_status(cudaErrorContextIsDestroyed), "renamed receiver arithmetic failure 100%");
            }
            for (const auto status : {JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED),
                                      JuicerCuda::runtime_failure_status(cudaErrorDeviceUninitialized),
                                      JuicerCuda::runtime_failure_status(cudaErrorInvalidValue),
                                      FjStatus{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}}) {
                for (const char* wording : {"test component=dir 100% device lost out of memory", "renamed arithmetic failure 100%", ""}) {
                    check_failure_order(test, JuicerCuda::ExecutorTest::Delivery::Success, true, status, wording);
                }
            }
            check_failure_order(test, JuicerCuda::ExecutorTest::Delivery::Success, false, JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED), "test component=scanner 100% renamed diagnostic");
            check_failure_order(test, JuicerCuda::ExecutorTest::Delivery::Success, false, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "non-DIR failure mentioning component=dir 100%");
            const std::string longDiagnostic = std::string(8192, 'x') + " 100% complete diagnostic";
            check_failure_order(test, JuicerCuda::ExecutorTest::Delivery::Success, true, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, longDiagnostic.c_str());
        }
    }

    void check_resource_failure(const Case& test,
                                const ParamSnapshot& parameters,
                                InstanceState& state,
                                JuicerCuda::ExecutorTest::ResourceFailures& failures,
                                const JuicerCuda::Failure& expectedFailure,
                                const char* stage,
                                bool preparationFailure = false) {
        using JuicerCuda::ExecutorTest::Event;
        JuicerAssets::NoiseTest::Observation noise;
        if (parameters.grainControls.active) {
            JuicerAssets::NoiseTest::observation = &noise;
        }
        JuicerCuda::ExecutorTest::FailureObservation observation;
        observation.armed = false;
        observation.status = expectedFailure.status;
        observation.diagnostic = expectedFailure.diagnostic.c_str();
        observation.stage = stage;
        JuicerCuda::ExecutorTest::observation = &observation;
        JuicerCuda::ExecutorTest::resourceFailures = &failures;
        bool fatal = false;
        try {
            (void)render_case(test, parameters, state);
        } catch (const OFX::Exception::Suite& error) {
            fatal = error.status() == kOfxStatErrFatal;
            observation.record(Event::FatalMapped);
        } catch (...) {
            JuicerCuda::ExecutorTest::observation = nullptr;
            JuicerCuda::ExecutorTest::resourceFailures = nullptr;
            throw;
        }
        JuicerCuda::ExecutorTest::observation = nullptr;
        JuicerCuda::ExecutorTest::resourceFailures = nullptr;
        JuicerAssets::NoiseTest::observation = nullptr;
        if (parameters.grainControls.active && (noise.operations != std::array<unsigned, 3>{1, 1, 1} || noise.gateViolation || !noise.releaseAfterRecovery)) {
            throw std::runtime_error("noise source did not survive native upload failure and recovery");
        }
        const bool contextLoss = expectedFailure.status.category == FJ_STATUS_CONTEXT_LOSS;
        std::array expectedEvents{Event::Injected, Event::Classified, Event::FrameAborted, Event::RecoveryStarted, Event::RecoveryEnded, Event::FatalMapped};
        if (preparationFailure) {
            // Root aborts failed preparation before returning it to the executor.
            std::swap(expectedEvents[1], expectedEvents[2]);
        }
        if (!fatal || observation.overflow || observation.count != expectedEvents.size() ||
            !std::equal(expectedEvents.begin(), expectedEvents.end(), observation.events.begin()) ||
            !observation.failureStageMatches || !observation.failureStatusMatches ||
            !observation.failureDiagnosticMatches ||
            observation.nativeRecoveryPending != contextLoss ||
            observation.adapterRecoveryPending != contextLoss ||
            state.submissionSnapshotLatchValid == contextLoss) {
            throw std::runtime_error(std::string(test.name) + ": resource failure propagation changed; category=" +
                                     std::to_string(observation.classifiedStatus.category) + " api=" +
                                     std::to_string(observation.classifiedStatus.api) + " code=" +
                                     std::to_string(observation.classifiedStatus.native_code) + " events=" +
                                     std::to_string(observation.count) + " stage_match=" +
                                     std::to_string(observation.failureStageMatches) + " diagnostic_match=" +
                                     std::to_string(observation.failureDiagnosticMatches));
        }
        std::cout << test.name << " stage=" << stage << " category=" << expectedFailure.status.category
                  << " api=" << expectedFailure.status.api << " code=" << expectedFailure.status.native_code
                  << " injections=" << failures.injections
                  << (preparationFailure ? " abort/classify" : " classify/abort")
                  << "/recovery/fatal order passed\n";
    }

    void run_scanner_lut_upload_failure_cases() {
        const std::array cases{Case{"lut-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                               Case{"lut-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0},
                               Case{"lut-positive-direct", Spektrafilm::ScanRoute::PositiveDirectScan, 0, 3, 0, 0},
                               Case{"lut-positive-print", Spektrafilm::ScanRoute::PositivePrintScan, 0, 4, 0, 0}};
        for (const auto& test : cases) {
            InstanceState state;
            auto parameters = parameters_for(test);
            parameters.grainControls.active = false;
            for (unsigned position = 1; position <= 6; ++position) {
                JuicerCuda::ExecutorTest::ResourceFailures failures;
                failures.lutPosition = position;
                const std::string diagnostic = std::string("scanner LUT upload fixture failure route=") + Spektrafilm::scan_route_metadata(test.route).key;
                check_resource_failure(test, parameters, state, failures, {{FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, diagnostic}, "prepare_focused_route_resources", true);
                if (failures.lutAttempts != position || failures.injections != 1 || fj_test_scanner_lut_live_owners() != 0)
                    throw std::runtime_error("scanner LUT upload short circuit or host-owner cleanup changed");
            }
            (void)render_case(test, parameters, state);
        }
    }
    void run_defect_fence_failure_cases() {
        const std::array<Case, 2> cases{{{"defect-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                                         {"defect-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}}};
        for (const Case& test : cases) {
            InstanceState state;
            for (const bool failCreate : {true, false}) {
                for (const auto code : {cudaErrorContextIsDestroyed, cudaErrorInvalidResourceHandle, cudaErrorMemoryAllocation}) {
                    auto parameters = parameters_for(test);
                    parameters.filmDustAmount = 1.0f;
                    (void)render_case(test, parameters, state);
                    parameters.filmDustAmount = 0.0f;
                    parameters.gateDustAmount = 1.0f;
                    JuicerCuda::ExecutorTest::ResourceFailures failures;
                    (failCreate ? failures.eventCreate : failures.eventRecord) = code;
                    check_resource_failure(test, parameters, state, failures, {JuicerCuda::runtime_failure_status(code), "defect attachment retirement fence failed"}, Spektrafilm::scan_route_is_print(test.route) ? "print_focused_workspace_stage" : "direct_focused_workspace_stage");
                    if (failures.injections != 1) {
                        throw std::runtime_error("defect fence failure did not reach the resource owner");
                    }
                }
            }
        }
    }

    void run_noise_lifetime_cases() {
        const std::array cases{Case{"noise-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                               Case{"noise-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}};
        using JuicerAssets::NoiseTest::EarlierFailure;
        using JuicerAssets::NoiseTest::Observation;
        for (const Case& test : {cases[0], cases[1]}) {
            InstanceState state;
            const auto run = [&](Observation& observed, const ParamSnapshot& parameters, bool fatalExpected, unsigned cancelAt = 0, DestinationLayout layout = DestinationLayout::Matching, ExecutionPath path = ExecutionPath::Processor) {
                JuicerAssets::NoiseTest::observation = &observed;
                AbortObservation abort;
                abort.cancelAt = cancelAt;
                s_abortObservation = cancelAt ? &abort : nullptr;
                bool fatal = false;
                std::string failureDiagnostic;
                try {
                    std::uint64_t snapshotId = 1;
                    if (path == ExecutionPath::Direct) {
                        snapshotId = state.submissionSnapshotIdNext.fetch_add(1, std::memory_order_relaxed);
                        state.submissionSnapshotLatchValid = false;
                    }
                    (void)render_case(test, parameters, state, false, path, layout, snapshotId);
                } catch (const OFX::Exception::Suite& error) {
                    fatal = error.status() == kOfxStatErrFatal;
                } catch (const JuicerCuda::ExecutionFailure& failure) {
                    if (path != ExecutionPath::Direct) {
                        throw;
                    }
                    fatal = failure.failure.status.category != FJ_STATUS_SUCCESS;
                    observed.result = failure.failure.status;
                    failureDiagnostic = failure.failure.diagnostic;
                } catch (...) {
                    JuicerAssets::NoiseTest::observation = nullptr;
                    s_abortObservation = nullptr;
                    throw;
                }
                JuicerAssets::NoiseTest::observation = nullptr;
                s_abortObservation = nullptr;
                if (fatal != fatalExpected || observed.gateViolation ||
                    (observed.noiseStep && !observed.noiseStepGated) || fj_test_live_noise_owners() != 0) {
                    throw std::runtime_error("production noise gate/lifetime/failure contract changed: path=" + std::to_string(static_cast<unsigned>(path)) +
                                             " source=" + std::to_string(observed.sourceFailure) + " earlier=" + std::to_string(static_cast<unsigned>(observed.earlier)) +
                                             " fatal=" + std::to_string(fatal) + " expected=" + std::to_string(fatalExpected) +
                                             " gate=" + std::to_string(observed.gateViolation) + " step=" + std::to_string(observed.noiseStep) +
                                             " owners=" + std::to_string(fj_test_live_noise_owners()) + " status=" + std::to_string(observed.result.category) + " " + failureDiagnostic);
                }
            };
            auto active = parameters_for(test);
            active.grainControls.active = true;
            Observation disabled;
            run(disabled, parameters_for(test), false);
            if (disabled.operations != std::array<unsigned, 3>{}) {
                throw std::runtime_error("disabled grain acquired noise");
            }
            JuicerProcess::root().assets().release_cached_payloads();
            for (bool cold : {true, false}) {
                (void)cold;
                Observation success;
                run(success, active, false);
                if (success.operations != std::array<unsigned, 3>{1, 1, 1} || !success.noiseStep || !success.releaseAfterRecovery) {
                    throw std::runtime_error("cold/warm caller source lifetime changed");
                }
            }
            Observation directRoot;
            run(directRoot, active, false, 0, DestinationLayout::Matching, ExecutionPath::Direct);
            if (directRoot.operations != std::array<unsigned, 3>{1, 1, 1} || directRoot.noiseStep) {
                throw std::runtime_error("direct Root source fallback changed");
            }
            Observation directDisabled;
            run(directDisabled, parameters_for(test), false, 0, DestinationLayout::Matching, ExecutionPath::Direct);
            if (directDisabled.operations != std::array<unsigned, 3>{}) {
                throw std::runtime_error("inactive Root fallback acquired noise");
            }
            Observation directFailure;
            directFailure.sourceFailure = FJ_STATUS_ALLOCATION_FAILURE;
            run(directFailure, active, true, 0, DestinationLayout::Matching, ExecutionPath::Direct);
            if (directFailure.result.category != FJ_STATUS_ALLOCATION_FAILURE || directFailure.operations != std::array<unsigned, 3>{1, 1, 1}) {
                throw std::runtime_error("Root fallback source error or abort lifetime changed");
            }
            for (auto earlier : {EarlierFailure::Preflight, EarlierFailure::Focused, EarlierFailure::Print}) {
                if (earlier == EarlierFailure::Print && !Spektrafilm::scan_route_is_print(test.route)) {
                    continue;
                }
                Observation failure;
                failure.earlier = earlier;
                failure.sourceFailure = FJ_STATUS_INTERNAL_FAILURE;
                run(failure, active, true);
                if (failure.noiseStep || failure.operations != std::array<unsigned, 3>{1, 1, 1} || failure.exceptionDrops != 1 || (earlier != EarlierFailure::Preflight && !failure.exceptionReleasedAfterRecovery)) {
                    throw std::runtime_error("earlier failure lost precedence to source error: earlier=" + std::to_string(static_cast<unsigned>(earlier)) +
                                             " step=" + std::to_string(failure.noiseStep) + " operations=" + std::to_string(failure.operations[0]) + "/" +
                                             std::to_string(failure.operations[1]) + "/" + std::to_string(failure.operations[2]) +
                                             " drops=" + std::to_string(failure.exceptionDrops) + " recovery=" + std::to_string(failure.exceptionReleasedAfterRecovery));
                }
            }
            for (auto category : {FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE}) {
                Observation failure;
                failure.sourceFailure = category;
                run(failure, active, true);
                if (!failure.noiseStep || failure.result.category != category || (category == FJ_STATUS_INTERNAL_FAILURE && (failure.exceptionDrops != 1 || !failure.exceptionReleasedAfterRecovery))) {
                    throw std::runtime_error("typed deferred source error changed");
                }
            }
            Observation malformed;
            malformed.corrupt = true;
            run(malformed, active, true);
            if (!malformed.noiseStep || malformed.result.category != FJ_STATUS_UNSUPPORTED_INPUT) {
                throw std::runtime_error("failed source construction was not deferred");
            }
            Observation inspection;
            inspection.sourceFailure = FJ_STATUS_PREPARATION_FAILURE;
            run(inspection, active, true, 0, DestinationLayout::Uncovered);
            if (inspection.noiseStep || inspection.operations != std::array<unsigned, 3>{1, 1, 1}) {
                throw std::runtime_error("inspection/source ordering changed");
            }
            Observation missing;
            missing.earlier = EarlierFailure::Missing;
            run(missing, active, true);
            if (missing.operations != std::array<unsigned, 3>{}) {
                throw std::runtime_error("missing recipe acquired source");
            }
            for (unsigned checkpoint : {1u, 2u, 3u, 4u, 5u}) {
                Observation cancelled;
                run(cancelled, active, false, checkpoint);
                if (cancelled.operations != std::array<unsigned, 3>{1, 1, 1}) {
                    throw std::runtime_error("cancelled source was not released");
                }
            }
            {
                JuicerCuda::NativeCall existing(JuicerCuda::borrowed_owner());
                Observation reentry;
                JuicerAssets::NoiseTest::observation = &reentry;
                bool rejected = false;
                try {
                    (void)render_case(test, active, state);
                } catch (const JuicerCuda::ExecutionFailure& failure) {
                    rejected = failure.failure.status.category == FJ_STATUS_UNSUPPORTED_INPUT;
                } catch (...) {
                    JuicerAssets::NoiseTest::observation = nullptr;
                    throw;
                }
                JuicerAssets::NoiseTest::observation = nullptr;
                if (!rejected || reentry.operations != std::array<unsigned, 3>{}) {
                    throw std::runtime_error("native reentry reached Rust noise source");
                }
            }
            if (fj_cuda_retire_instance(JuicerCuda::borrowed_owner(), state.submissionSnapshotLatch.instanceToken.value, nullptr).category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("noise fixture retirement failed");
            }
        }
        {
            InstanceState state;
            auto active = parameters_for(cases[1]);
            active.grainControls.active = true;
            Observation closed;
            closed.closeBeforeAdmission = true;
            JuicerAssets::NoiseTest::observation = &closed;
            bool rejected = false;
            try {
                (void)render_case(cases[1], active, state);
            } catch (const JuicerCuda::ExecutionFailure& failure) {
                rejected = failure.failure.status.category == FJ_STATUS_PREPARATION_FAILURE;
            } catch (...) {
                JuicerAssets::NoiseTest::observation = nullptr;
                throw;
            }
            JuicerAssets::NoiseTest::observation = nullptr;
            if (!rejected || closed.operations != std::array<unsigned, 3>{1, 1, 1} || closed.noiseStep || closed.gateViolation || fj_test_live_noise_owners() != 0) {
                throw std::runtime_error("final native admission/source destruction changed");
            }
            Observation sourceError;
            sourceError.sourceFailure = FJ_STATUS_INTERNAL_FAILURE;
            sourceError.closeBeforeAdmission = true;
            JuicerAssets::NoiseTest::observation = &sourceError;
            rejected = false;
            try {
                (void)render_case(cases[1], active, state);
            } catch (const JuicerCuda::ExecutionFailure& failure) {
                rejected = failure.failure.status.category == FJ_STATUS_PREPARATION_FAILURE;
            } catch (...) {
                JuicerAssets::NoiseTest::observation = nullptr;
                throw;
            }
            JuicerAssets::NoiseTest::observation = nullptr;
            if (!rejected || sourceError.noiseStep || sourceError.exceptionDrops != 1 || sourceError.gateViolation) {
                throw std::runtime_error("closed native admission/held source exception ordering changed");
            }
        }
        std::puts("PASS production direct/print noise: calling-thread acquire/view/release outside gate, continuous projection gate, deferred typed failures, earlier failure precedence, missing/disabled, cancellation and reentry");
    }

    void run_grain_upload_failure_cases() {
        const std::array<Case, 2> cases{{{"grain-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                                         {"grain-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 4, 0, 0}}};
        for (const Case& test : cases) {
            for (const char* wording : {"renamed upload component=dir diagnostic", "device lost out of memory 100%"}) {
                const JuicerCuda::Failure capacity{{FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, wording};
                const JuicerCuda::Failure contextLoss{JuicerCuda::runtime_failure_status(cudaErrorContextIsDestroyed), wording};
                const JuicerCuda::Failure driverLoss{JuicerCuda::driver_failure_status(CUDA_ERROR_CONTEXT_IS_DESTROYED), wording};
                const JuicerCuda::Failure ordinary{JuicerCuda::runtime_failure_status(cudaErrorInvalidValue), wording};
                struct UploadCase {
                    std::optional<JuicerCuda::Failure> stbn;
                    std::optional<JuicerCuda::Failure> wang;
                    FjStatus expectedStatus{};
                };
                const std::array<UploadCase, 8> uploads{{{capacity, contextLoss, contextLoss.status},
                                                         {contextLoss, ordinary, contextLoss.status},
                                                         {capacity, std::nullopt, capacity.status},
                                                         {contextLoss, std::nullopt, contextLoss.status},
                                                         {std::nullopt, driverLoss, driverLoss.status},
                                                         {capacity, ordinary, capacity.status},
                                                         {ordinary, capacity, ordinary.status},
                                                         {contextLoss, driverLoss, contextLoss.status}}};
                for (const auto& upload : uploads) {
                    InstanceState state;
                    auto parameters = parameters_for(test);
                    parameters.grainControls.active = true;
                    JuicerCuda::ExecutorTest::ResourceFailures failures;
                    failures.stbn = upload.stbn;
                    failures.wang = upload.wang;
                    if (failures.stbn) {
                        failures.stbn->diagnostic = std::string("STBN: ") + wording;
                    }
                    if (failures.wang) {
                        failures.wang->diagnostic = std::string("Wang: ") + wording;
                    }
                    std::string diagnostic = std::string(upload.stbn ? "STBN: " : "Wang: ") + wording;
                    if (upload.stbn && upload.wang) {
                        diagnostic += " | Wang upload failed: Wang: ";
                        diagnostic += wording;
                    }
                    diagnostic += " route=";
                    diagnostic += Spektrafilm::scan_route_metadata(test.route).key;
                    check_resource_failure(test, parameters, state, failures, {upload.expectedStatus, diagnostic}, "ensure_grain_static_assets_uploaded", true);
                    if (failures.stbnAttempts != 1 || failures.wangAttempts != 1 ||
                        failures.injections != static_cast<int>(upload.stbn.has_value()) + static_cast<int>(upload.wang.has_value())) {
                        throw std::runtime_error("grain upload attempts or failure order changed");
                    }
                }
            }
        }
    }
} // namespace
#endif

#if defined(JUICER_ACCEPTED_CAPTURE_PATH)
namespace {
    void run_accepted_clean_captures() {
        std::ifstream fixture(JUICER_ACCEPTED_CAPTURE_PATH);
        if (!fixture) {
            throw std::runtime_error("accepted CUDA capture fixture unavailable");
        }
        const auto captures = nlohmann::json::parse(fixture);
        const auto decode = [](const nlohmann::json& bits) {
            std::vector<float> pixels;
            pixels.reserve(bits.size());
            for (const auto& value : bits) {
                pixels.push_back(std::bit_cast<float>(value.get<std::uint32_t>()));
            }
            return pixels;
        };
        const auto input = decode(captures.at("input_bits"));
        for (const auto& capture : captures.at("cases")) {
            const auto& settings = capture.at("settings");
            const std::string name = capture.at("name").get<std::string>();
            Case test{};
            test.name = name.c_str();
            test.route = static_cast<Spektrafilm::ScanRoute>(settings.at("route").get<int>());
            test.components = 3;
            test.sourcePixels = &input;
            test.fullWidth = settings.at("width").get<int>();
            test.fullHeight = settings.at("height").get<int>();
            auto parameters = parameters_for(test);
            parameters.filmProfileKey = settings.at("film").get<std::string>();
            parameters.printProfileKey = settings.at("print").get<std::string>();
            parameters.printGammaFactor = settings.at("print_gamma").get<double>();
            parameters.inputColorSpace = Spectral::inputColorSpaceToIndex(Spectral::InputColorSpace::ITU_R_BT2020);
            parameters.inputCctfDecoding = 0;
            parameters.outputColorSpace = OutputEncoding::toIndex(OutputEncoding::ColorSpace::ITU_R_BT2020);
            parameters.outputCctfEncoding = 0;
            parameters.dirCouplers.active = false;
            parameters.grainControls.active = false;
            parameters.glareActive = false;
            parameters.glarePercent = 0.03;
            InstanceState state;
            const auto pixels = render_case(test, parameters, state);
            compare_pixels(name, pixels, decode(capture.at("output_bits")));
            std::cerr << name << ": exact stored CUDA samples within retained pixel bounds\n";
        }
    }
} // namespace
#endif

int main(int argc, char** argv) {
    JuicerCuda::Owner cudaOwner;
    try {
        cudaOwner.create(JuicerProcess::data_directory());
        NarrowHost host;
        require_cuda(cudaSetDevice(0), "select device");
        require_cuda(cudaFree(nullptr), "initialize CUDA");
        JuicerProcess::root().ensure_bootstrap();
#if defined(JUICER_ACCEPTED_CAPTURE_PATH)
        if (argc == 2 && std::string(argv[1]) == "--accepted-clean-captures") {
            run_accepted_clean_captures();
            if (cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("accepted capture owner cleanup failed");
            }
            return 0;
        }
#endif
#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
        if (argc == 2 && std::string(argv[1]) == "--workspace-transitions") {
            run_workspace_transitions();
            if (cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("workspace transitions owner cleanup failed");
            }
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--effects-frame") {
            run_effects_frame_cases(argv[2], argv[3]);
            if (cudaOwner.close().category != FJ_STATUS_SUCCESS)
                throw std::runtime_error("effects caller owner cleanup failed");
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--grain-frame") {
            run_grain_frame_cases(argv[2], argv[3]);
            if (cudaOwner.close().category != FJ_STATUS_SUCCESS)
                throw std::runtime_error("grain caller owner cleanup failed");
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--dir-frame") {
            run_dir_frame_cases(argv[2], argv[3]);
            if (cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("DIR caller owner cleanup failed");
            }
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--noise-lifetime") {
            run_noise_lifetime_cases();
            if (cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("noise lifetime owner cleanup failed");
            }
            return 0;
        }
        if (argc == 3 && std::string(argv[1]) == "--resource-failure") {
            const std::string resourceFailure = argv[2];
            if (resourceFailure == "DefectFence") {
                run_defect_fence_failure_cases();
            } else if (resourceFailure == "ScannerLut") {
                run_scanner_lut_upload_failure_cases();
            } else if (resourceFailure == "GrainUpload") {
                run_grain_upload_failure_cases();
            } else {
                throw std::runtime_error("unknown resource failure case: " + resourceFailure);
            }
            const auto closed = fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
            if (closed.category != FJ_STATUS_SUCCESS || cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("terminal cleanup after native failure failed");
            }
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--failure-order") {
            run_failure_order_cases();
            const auto closed = fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
            if (closed.category != FJ_STATUS_SUCCESS || cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                throw std::runtime_error("terminal cleanup after native failure failed");
            }
            return 0;
        }
#endif
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        const bool stateContract = argc == 3 && std::string(argv[1]) == "--state-contract";
        const bool stateLifetime = argc == 3 && std::string(argv[1]) == "--state-lifetime";
        if (argc != 3 || (!stateContract && !stateLifetime && std::string(argv[1]) != "--prepared-case")) {
            throw std::runtime_error("usage: JuicerPreparedBoundaryProbe --prepared-case <name>");
        }
        const std::string preparedCase = argv[2];
        bool preparedCaseFound = false;
        constexpr bool emit = false;
#else
        const bool cutoverContract = argc == 2 && std::string(argv[1]) == "--cutover-contract";
        const bool sequentialOwners = argc == 2 && std::string(argv[1]) == "--sequential-owners";
        const bool emit = argc == 2 && std::string(argv[1]) == "--emit-reference";
        if (argc != 1 && !emit && !sequentialOwners && !cutoverContract) {
            throw std::runtime_error("usage: JuicerProcessorReferenceProbe [--emit-reference]");
        }
#endif
        const Case cases[]{{"negative-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 0, 3, 0, 0},
                           {"negative-print", Spektrafilm::ScanRoute::NegativePrintScan, 1, 4, 0, 0},
                           {"positive-direct", Spektrafilm::ScanRoute::PositiveDirectScan, 2, 3, 11, -3},
                           {"positive-print", Spektrafilm::ScanRoute::PositivePrintScan, 0, 4, 0, 0},
                           {"combined-print", Spektrafilm::ScanRoute::NegativePrintScan, 0, 3, 0, 0, true},
                           {"glare-plus-zero", Spektrafilm::ScanRoute::NegativePrintScan, 0, 3, 0, 0, false, true, false},
                           {"glare-minus-zero", Spektrafilm::ScanRoute::NegativePrintScan, 0, 3, 0, 0, false, true, true}
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
                           ,
                           {"mallett-direct", Spektrafilm::ScanRoute::NegativeDirectScan, 1, 4, 0, 0},
                           {"arctic-print", Spektrafilm::ScanRoute::NegativePrintScan, 2, 4, 0, 0}
#endif
        };
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        if (stateLifetime) {
            for (std::size_t index = 0; index < 2; ++index) {
                InstanceState unused;
                for (unsigned lifetimeCase = 0; lifetimeCase < static_cast<unsigned>(JuicerCudaTest::StateLifetimeCase::Count); ++lifetimeCase) {
                    (void)render_case(cases[index], parameters_for(cases[index]), unused, false, ExecutionPath::StateLifetime, DestinationLayout::Matching, 1, static_cast<JuicerCudaTest::StateLifetimeCase>(lifetimeCase));
                }
                if (cudaOwner.close().category != FJ_STATUS_SUCCESS)
                    throw std::runtime_error("held-state lifetime owner close");
                if (index == 0) {
                    cudaOwner.create(JuicerProcess::data_directory());
                    JuicerProcess::root().ensure_bootstrap();
                }
            }
            return 0;
        }
#endif
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        constexpr bool numericalFixture = false;
#else
        const bool numericalFixture = !emit && !sequentialOwners && !cutoverContract;
#endif
        std::ifstream fixture;
        if (numericalFixture) {
            fixture.open(JUICER_PROCESSOR_REFERENCE_PATH);
            if (!fixture) {
                throw std::runtime_error("processor reference fixture unavailable");
            }
        }
#if !defined(JUICER_PREPARED_BOUNDARY_TEST)
        void* initialContext = nullptr;
        std::uint64_t previousEpoch = 0;
        if (sequentialOwners) {
            CUcontext current = nullptr;
            if (cuCtxGetCurrent(&current) != CUDA_SUCCESS) {
                throw std::runtime_error("missing sequential-owner CUDA context");
            }
            initialContext = current;
        }
#endif
        std::vector<float> freshPlus;
        std::vector<float> freshMinus;
        for (const Case& test : cases) {
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
            if (preparedCase != test.name) {
                if (numericalFixture) {
                    fixture.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
                continue;
            }
            preparedCaseFound = true;
            // Each CTest case starts a fresh process and owner. C crosses the
            // boundary first, so no other caller can hide its table uploads.
            std::cerr << "cold C case: " << test.name << '\n';
            InstanceState boundaryState;
            const auto boundary = render_case(test, parameters_for(test), boundaryState, false, stateContract ? ExecutionPath::StateContract : ExecutionPath::Boundary);
            const auto warmBoundary = render_case(test, parameters_for(test), boundaryState, false, stateContract ? ExecutionPath::State : ExecutionPath::Boundary);
            const auto direct = [&] {
                InstanceState directState;
                return render_case(test, parameters_for(test), directState, false, ExecutionPath::Direct);
            }();
#endif
            std::cerr << "processor case: " << test.name << '\n';
            const std::vector<float> pixels = [&] {
                InstanceState state;
                auto rendered = render_case(test, parameters_for(test), state);
#if !defined(JUICER_PREPARED_BOUNDARY_TEST)
                if (sequentialOwners) {
                    const auto& snapshot = state.submissionSnapshotLatch;
                    JuicerCuda::ResourceManager::RegistryContextSnapshot admitted;
                    if (snapshot.deviceContextKey.contextOpaque != initialContext ||
                        !JuicerCuda::ResourceManager::registry_begin_submission(snapshot.deviceContextKey, admitted)) {
                        throw std::runtime_error("sequential owner changed or failed to admit the live context");
                    }
                    const bool ended = JuicerCuda::ResourceManager::registry_note_submission_end(snapshot.deviceContextKey);
                    if (!ended || admitted.contextEpoch <= previousEpoch) {
                        throw std::runtime_error("owner recreation did not advance the native epoch");
                    }
                    std::cerr << "owner transition case=" << test.name << " previous_epoch=" << previousEpoch << " admitted_epoch=" << admitted.contextEpoch << '\n';
                    previousEpoch = admitted.contextEpoch;
                }
#endif
                return rendered;
            }();
            if (test.signedZeroGlare) {
                (test.negativeZero ? freshMinus : freshPlus) = pixels;
            }
            if (emit) {
                std::cout << test.name << ' ' << pixels.size();
                for (float pixel : pixels) {
                    std::cout << ' ' << std::setprecision(std::numeric_limits<float>::max_digits10) << pixel;
                }
                std::cout << '\n';
            } else {
                std::vector<float> expected = pixels;
                if (numericalFixture) {
                    std::string name;
                    std::size_t count = 0;
                    fixture >> name >> count;
                    if (!fixture || name != test.name || count != pixels.size()) {
                        throw std::runtime_error(std::string(test.name) + ": fixture row mismatch");
                    }
                    expected.resize(count);
                    for (float& pixel : expected) {
                        fixture >> pixel;
                    }
                    if (!fixture) {
                        throw std::runtime_error(std::string(test.name) + ": incomplete fixture row");
                    }
                    nlohmann::json originalWords = nlohmann::json::array();
                    for (float value : expected)
                        originalWords.push_back(std::bit_cast<std::uint32_t>(value));
                    const auto selectedWords = PrintIlluminantFixtures::apply(
                        std::string(JUICER_PROCESSOR_REFERENCE_PATH) + "." + test.name, originalWords);
                    if (selectedWords.size() != expected.size())
                        throw std::runtime_error(std::string(test.name) + ": selected fixture shape mismatch");
                    for (std::size_t i = 0; i < expected.size(); ++i)
                        expected[i] = std::bit_cast<float>(selectedWords.at(i).get<std::uint32_t>());
                    compare_pixels(test.name, pixels, expected);
                }
#if !defined(JUICER_PREPARED_BOUNDARY_TEST)
                if (cutoverContract && (std::string_view(test.name) == "negative-direct" || std::string_view(test.name) == "negative-print")) {
                    check_cutover_callbacks(test, expected);
                }
                if (cutoverContract && !test.signedZeroGlare)
                    check_physical_effects_pixels(test);
                if (sequentialOwners && std::string_view(test.name) == "negative-direct") {
                    Case uniform = test;
                    uniform.uniform = true;
                    std::vector<float> uniformExpected(expected.size());
                    for (std::size_t sample = 0; sample < uniformExpected.size(); ++sample) {
                        uniformExpected[sample] = expected[sample % 3];
                    }
                    // At this fixture's physical scale the DIR FIR has no nonzero
                    // off-center float samples. Normalized diffusion preserves a
                    // constant exposure. Both controls use the immutable first pixel.
                    {
                        InstanceState control;
                        compare_pixels("uniform direct control", render_case(uniform, parameters_for(uniform), control), uniformExpected);
                    }
                    if (cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                        throw std::runtime_error("direct diffusion predecessor close failed");
                    }
                    cudaOwner.create(JuicerProcess::data_directory());
                    JuicerProcess::root().ensure_bootstrap();
                    {
                        auto parameters = parameters_for(uniform);
                        parameters.cameraDiffusion.active = true;
                        InstanceState diffusionState;
                        compare_pixels("uniform direct diffusion", render_case(uniform, parameters, diffusionState), uniformExpected);
                        JuicerCuda::ResourceManager::RegistryContextSnapshot admitted;
                        const auto& key = diffusionState.submissionSnapshotLatch.deviceContextKey;
                        if (key.contextOpaque != initialContext || !JuicerCuda::ResourceManager::registry_begin_submission(key, admitted)) {
                            throw std::runtime_error("direct diffusion context admission failed");
                        }
                        const bool ended = JuicerCuda::ResourceManager::registry_note_submission_end(key);
                        if (!ended || admitted.contextEpoch <= previousEpoch) {
                            throw std::runtime_error("direct diffusion did not use a later admitted epoch");
                        }
                        previousEpoch = admitted.contextEpoch;
                        std::cerr << "uniform direct diffusion matches immutable constant sample at epoch=" << previousEpoch << '\n';
                    }
                }
                if (sequentialOwners && !test.combined) {
                    InstanceState originState;
                    compare_pixels(std::string(test.name) + ": independent destination origin", render_case(test, parameters_for(test), originState, false, ExecutionPath::Processor, DestinationLayout::Offset), expected);
                    InstanceState rejectedState;
                    bool fatal = false;
                    try {
                        (void)render_case(test, parameters_for(test), rejectedState, false, ExecutionPath::Processor, DestinationLayout::Uncovered);
                    } catch (const OFX::Exception::Suite& error) {
                        fatal = error.status() == kOfxStatErrFatal;
                    }
                    if (!fatal || rejectedState.submissionSnapshotLatchValid) {
                        throw std::runtime_error("uncovered destination was not rejected before submission latch publication");
                    }
                }
#endif
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
                RenderAssertions::compare_bits(std::string(test.name) + ": processor/direct", pixels, direct);
                RenderAssertions::compare_bits(std::string(test.name) + ": direct/cold C", direct, boundary);
                RenderAssertions::compare_bits(std::string(test.name) + ": cold/warm C", boundary, warmBoundary);
                std::cerr << test.name << ": processor/direct/cold C/warm C bit-exact\n";
                if (stateContract) {
                    auto grainParameters = parameters_for(test);
                    grainParameters.grainControls.active = true;
                    InstanceState grainState;
                    (void)render_case(test, grainParameters, grainState, false, ExecutionPath::Boundary, DestinationLayout::Matching, 2);
                }
#endif
            }
#if !defined(JUICER_PREPARED_BOUNDARY_TEST)
            if (sequentialOwners) {
                if (cudaOwner.close().category != FJ_STATUS_SUCCESS) {
                    throw std::runtime_error("sequential native owner failed to close");
                }
                cudaOwner.create(JuicerProcess::data_directory());
                JuicerProcess::root().ensure_bootstrap();
                CUcontext current = nullptr;
                if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != initialContext) {
                    throw std::runtime_error("owner close/recreate replaced the live CUDA context");
                }
            }
#endif
        }
#if defined(JUICER_PREPARED_BOUNDARY_TEST)
        if (!preparedCaseFound) {
            throw std::runtime_error("unknown prepared-boundary case: " + preparedCase);
        }
#else
        if (!emit) {
            if (std::equal(freshPlus.begin(), freshPlus.end(), freshMinus.begin())) {
                throw std::runtime_error("signed-zero fresh print renders did not differ");
            }
            InstanceState transition;
            compare_pixels("glare plus to minus initial", render_case(cases[5], parameters_for(cases[5]), transition), freshPlus);
            compare_pixels("glare plus to minus", render_case(cases[6], parameters_for(cases[6]), transition), freshMinus);
            compare_pixels("glare minus to plus", render_case(cases[5], parameters_for(cases[5]), transition), freshPlus);
        }
        InstanceState emptyWindowState;
        (void)render_case(cases[0], parameters_for(cases[0]), emptyWindowState, true);
#endif
        if (cudaOwner.close().category != FJ_STATUS_SUCCESS) {
            throw std::runtime_error("final native owner close failed");
        }
        return 0;
    } catch (const std::exception& error) {
        (void)std::fprintf(stderr, "%s\n", error.what());
        try {
            fj_cuda_shutdown(JuicerCuda::borrowed_owner(), nullptr);
        } catch (...) {
            (void)std::fputs("shutdown failed after processor test error\n", stderr);
        }
        return 1;
    }
}
