#include "processor_test_hooks.h"

#include <algorithm>
#include <bit>
#include <cstring>

#if defined(JUICER_EXECUTOR_FAILURE_TEST_HOOK)
namespace JuicerCuda::ExecutorTest {
    void FailureObservation::record(Event event) noexcept {
        if (count < events.size()) {
            events[count++] = event;
        } else {
            overflow = true;
        }
    }
    void ResourceFailures::record_injection() noexcept {
        if (injections++ == 0 && observation) {
            observation->record(Event::Injected);
        }
    }
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
    thread_local void (*recoveryStartHook)(void*) noexcept = nullptr;
    thread_local void* recoveryStartHookUser = nullptr;

    void observe_recovery_start(bool pending) noexcept {
        if (observation) {
            observation->adapterRecoveryPending = pending;
            observation->record(Event::RecoveryStarted);
        }
        if (recoveryStartHook) {
            auto* hook = recoveryStartHook;
            recoveryStartHook = nullptr;
            hook(recoveryStartHookUser);
        }
    }
    void observe_recovery_end() noexcept {
        if (observation) {
            observation->record(Event::RecoveryEnded);
        }
    }
    void observe_context_retirement(bool accepted, FjStatus invalidation, bool cleared) noexcept {
        if (observation) {
            observation->retirementObserved = true;
            observation->retirementAccepted = accepted;
            observation->invalidationStatus = invalidation;
            observation->invalidationCleared = cleared;
        }
    }
    void observe_adapter_completion(FjRenderOutcome outcome, bool fatal, std::string_view diagnostic) noexcept {
        if (observation) {
            observation->completionObserved = true;
            observation->completionOutcome = outcome;
            observation->completionFatal = fatal;
            observation->completionDiagnosticMatches = observation->diagnostic && diagnostic == observation->diagnostic;
        }
    }
} // namespace JuicerCuda::ExecutorTest
#endif
