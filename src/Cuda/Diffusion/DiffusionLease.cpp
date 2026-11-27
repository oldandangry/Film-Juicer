#include "Cuda/Diffusion/JuicerCudaDiffusionResources.h"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <new>
#include <utility>

#include <cuda.h>
#include <cudaTypedefs.h>

#include "Logging.h"
#include "Cuda/Diffusion/DiffusionInternals.h"
#include "Cuda/JuicerCudaResourceOperations.h"

namespace JuicerCuda::Diffusion {

    static bool checked_add(
        std::uint64_t left,
        std::uint64_t right,
        std::uint64_t& out) noexcept {
        if (right > std::numeric_limits<std::uint64_t>::max() - left) {
            return false;
        }
        out = left + right;
        return true;
    }

    static bool checked_multiply(
        std::uint64_t left,
        std::uint64_t right,
        std::uint64_t& out) noexcept {
        if (left != 0 &&
            right > std::numeric_limits<std::uint64_t>::max() / left) {
            return false;
        }
        out = left * right;
        return true;
    }

    bool acquire_host_lease(
        DiffusionContextResources& resources,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        std::lock_guard<std::mutex> lock(resources.metadataMutex);
        if (!resources.acceptingPreparations) {
            outError.diagnostic =
                "ExactAdmissionFailure component=diffusion class=context_retiring requested_new_bytes=0";
            return false;
        }
        if (resources.hostLeaseActive) {
            outError.diagnostic =
                "ResourceDescriptorMismatch component=diffusion field=serialized_host_lease";
            return false;
        }
        resources.hostLeaseActive = true;
        return true;
    }

    void release_host_lease(
        DiffusionContextResources& resources) noexcept {
        try {
            std::lock_guard<std::mutex> lock(resources.metadataMutex);
            resources.hostLeaseActive = false;
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    static const Spektrafilm::DiffusionStageFrameDescriptor* stage_for_key(
        const Spektrafilm::DiffusionFrameSetDescriptor& frameSet,
        const Spektrafilm::DiffusionExecutionDescriptor& descriptor,
        std::size_t keyIndex) noexcept {
        for (std::size_t stageIndex = 0;
             stageIndex < descriptor.stageCount;
             ++stageIndex) {
            if (descriptor.stages[stageIndex].spectrumKeyIndex != keyIndex) {
                continue;
            }
            if (descriptor.stages[stageIndex].stage ==
                Spektrafilm::DiffusionLinearStage::CameraFilmLinear) {
                return frameSet.camera ? &*frameSet.camera : nullptr;
            }
            return frameSet.enlarger ? &*frameSet.enlarger : nullptr;
        }
        return nullptr;
    }

    PreparedDiffusionLease::~PreparedDiffusionLease() noexcept {
        if (_active && _owner) {
            Failure ignored;
            (void)release_diffusion_resources(
                *_owner,
                *this,
                _streamOpaque,
                true,
                ignored);
        }
    }

    PreparedDiffusionLease::PreparedDiffusionLease(
        PreparedDiffusionLease&& other) noexcept {
        *this = std::move(other);
    }

    PreparedDiffusionLease& PreparedDiffusionLease::operator=(
        PreparedDiffusionLease&& other) noexcept {
        if (this != &other) {
            if (_active && _owner) {
                Failure ignored;
                (void)release_diffusion_resources(
                    *_owner,
                    *this,
                    _streamOpaque,
                    true,
                    ignored);
            }
            _owner = std::exchange(other._owner, nullptr);
            _descriptor = other._descriptor;
            _view = other._view;
            _spectrumIndices = other._spectrumIndices;
            _spectrumCount = std::exchange(other._spectrumCount, 0);
            _workspaceIndex = other._workspaceIndex;
            _streamOpaque = std::exchange(other._streamOpaque, nullptr);
            _active = std::exchange(other._active, false);
            _workEnqueued = std::exchange(other._workEnqueued, false);
        }
        return *this;
    }

    bool PreparedDiffusionLease::active() const noexcept {
        return _active;
    }

    DiffusionPreparedView PreparedDiffusionLease::view() const noexcept {
        return _active ? _view : DiffusionPreparedView{};
    }

    void PreparedDiffusionLease::reset() noexcept {
        _owner = nullptr;
        _descriptor = {};
        _view = {};
        _spectrumIndices = {};
        _spectrumCount = 0;
        _workspaceIndex = 0;
        _streamOpaque = nullptr;
        _active = false;
        _workEnqueued = false;
    }

    void PreparedDiffusionLease::refresh_view() noexcept {
        _view = {};
        if (!_owner || !_active) {
            return;
        }
        _view.executionDescriptor = _descriptor;
        _view.spectrumCount = _spectrumCount;
        _view.active = true;
        for (std::size_t index = 0; index < _spectrumCount; ++index) {
            const DiffusionSpectrumEntry& entry =
                _owner->spectra[_spectrumIndices[index]];
            _view.spectra[index] = {entry.key, entry.spectra};
        }
        const DiffusionWorkspaceSlot& workspace =
            _owner->workspaces[_workspaceIndex];
        _view.execution.planKey = workspace.planKey;
        _view.execution.execution = {
            static_cast<float*>(workspace.transform.get()),
            workspace.r2cPlan.get(),
            workspace.c2rPlan.get()};
        _view.execution.stagePlanes = workspace.stagePlanes;
    }

    bool prepare_diffusion_resources(
        DiffusionContextResources& resources,
        const ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t contextEpoch,
        const std::shared_ptr<DeviceAllocationLedger>& ledger,
        const Spektrafilm::DiffusionFrameSetDescriptor& frameSet,
        const Spektrafilm::DiffusionExecutionDescriptor& descriptor,
        void* cudaStreamOpaque,
        PreparedDiffusionLease& outLease,
        Failure& outError) {
        outError = {};
        if (outLease.active() || contextKey.deviceId < 0 ||
            !contextKey.contextOpaque || contextEpoch == 0 || !ledger ||
            !cudaStreamOpaque) {
            outError.diagnostic =
                "ResourceDescriptorMismatch component=diffusion field=prepare_identity";
            return false;
        }
        if (!current_owner_matches(contextKey, outError)) {
            return false;
        }
        if (descriptor.contextEpoch != contextEpoch ||
            descriptor.frameSetHash != frameSet.hash) {
            outError.diagnostic =
                "ResourceDescriptorMismatch component=diffusion field=prepare_association";
            return false;
        }
        std::array<Spektrafilm::DiffusionPsfComponents, 2> components{};
        std::array<int, 2> radii{};
        for (std::size_t keyIndex = 0;
             keyIndex < descriptor.uniqueSpectrumCount;
             ++keyIndex) {
            const auto& stage =
                *stage_for_key(frameSet, descriptor, keyIndex);
            components[keyIndex] = stage.sample.components;
            radii[keyIndex] = stage.sample.radiusPixels;
        }

        if (!acquire_host_lease(resources, outError)) {
            return false;
        }
        const auto releaseClaim = [](DiffusionContextResources* owner) noexcept {
            release_host_lease(*owner);
        };
        std::unique_ptr<DiffusionContextResources, decltype(releaseClaim)> claim(&resources, releaseClaim);

        const cudaStream_t stream =
            reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        std::array<std::size_t, 2> spectrumIndices{};
        std::array<bool, 2> missingSpectrum{};
        std::array<bool, 2> dependentBuilding{};
        std::size_t workspaceIndex = kInvalidIndex;
        bool rebuildWorkspace = false;
        DeviceByteReservation aggregate;

        if (!reap_completed(resources, outError)) {
            return false;
        }
        for (std::size_t index = 0; index < resources.workspaces.size(); ++index) {
            if (resources.workspaces[index].state == WorkspaceSlotState::Vacant) {
                workspaceIndex = index;
                break;
            }
        }
        if (workspaceIndex == kInvalidIndex) {
            const std::size_t retiringIndex =
                choose_retiring_workspace(resources);
            if (retiringIndex == kInvalidIndex) {
                outError.diagnostic =
                    "ExactAdmissionFailure component=diffusion class=execution_slot_unrecoverable requested_new_bytes=0 slots=2";
                return false;
            }

            DiffusionWorkspaceSlot& retiring =
                resources.workspaces[retiringIndex];
            const cudaError_t waitResult = cudaEventSynchronize(
                retiring.completionEvent.get());
            if (waitResult != cudaSuccess) {
                retiring.state = WorkspaceSlotState::FailedQuarantined;
                set_failure(outError, runtime_failure_status(waitResult), "cudaEventSynchronize(diffusion admission) failed");
                return false;
            }
            if (!complete_workspace_retirement(resources, retiringIndex, outError) ||
                !reap_retired_spectra(resources, outError)) {
                return false;
            }
            workspaceIndex = retiringIndex;
        }

        DiffusionWorkspaceSlot& workspace =
            resources.workspaces[workspaceIndex];
        workspace.releaseSequence = 0;
        const std::uint64_t requiredPlaneElements =
            descriptor.stagePlaneBytes /
            (4u * static_cast<std::uint64_t>(sizeof(float)));
        rebuildWorkspace =
            !same_plan_key(workspace.planKey, descriptor.planKey) ||
            workspace.r2cPlan.get() == 0 || workspace.c2rPlan.get() == 0 ||
            !static_cast<float*>(workspace.transform.get()) ||
            workspace.transformCapacityBytes <
                descriptor.layout.transformBytes ||
            workspace.stagePlaneCapacityElements < requiredPlaneElements;

        std::size_t leasedSpectrumCount = 0;
        const auto rollbackSpectrumLeases = [&](DiffusionContextResources* owner) noexcept {
            for (std::size_t index = 0; index < leasedSpectrumCount; ++index) {
                auto& entry = owner->spectra[spectrumIndices[index]];
                entry.leased = false;
                if (missingSpectrum[index]) {
                    entry = DiffusionSpectrumEntry{};
                }
            }
        };
        std::unique_ptr<DiffusionContextResources, decltype(rollbackSpectrumLeases)> spectrumClaims(&resources, rollbackSpectrumLeases);
        for (std::size_t keyIndex = 0;
             keyIndex < descriptor.uniqueSpectrumCount;
             ++keyIndex) {
            std::size_t found = kInvalidIndex;
            for (std::size_t entryIndex = 0;
                 entryIndex < resources.spectra.size();
                 ++entryIndex) {
                const DiffusionSpectrumEntry& entry = resources.spectra[entryIndex];
                if ((entry.state == SpectrumEntryState::Ready ||
                     entry.state == SpectrumEntryState::Building) &&
                    same_spectrum_key(entry.key, descriptor.spectrumKeys[keyIndex])) {
                    if (entry.leased) {
                        set_failure(outError, outError.status, "ResourceDescriptorMismatch component=diffusion field=serialized_spectrum_lease");
                        return false;
                    }
                    found = entryIndex;
                    dependentBuilding[keyIndex] = entry.state == SpectrumEntryState::Building;
                    break;
                }
            }
            if (found == kInvalidIndex) {
                Failure evictionError{{FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}};
                found = choose_missing_spectrum_slot(resources, evictionError);
                if (found == kInvalidIndex) {
                    if (evictionError.status.category == FJ_STATUS_SUCCESS) {
                        set_failure(outError, outError.status, "ExactAdmissionFailure component=diffusion class=spectrum requested_new_bytes=0 slots=4");
                    } else {
                        outError = std::move(evictionError);
                    }
                    return false;
                }
                DiffusionSpectrumEntry& entry = resources.spectra[found];
                entry.key = descriptor.spectrumKeys[keyIndex];
                entry.state = SpectrumEntryState::Building;
                missingSpectrum[keyIndex] = true;
            }
            resources.spectra[found].leased = true;
            spectrumIndices[keyIndex] = found;
            ++leasedSpectrumCount;
        }

        std::uint64_t prospectiveBytes = 0;
        for (std::size_t keyIndex = 0;
             keyIndex < descriptor.uniqueSpectrumCount;
             ++keyIndex) {
            if (!missingSpectrum[keyIndex]) {
                continue;
            }
            std::uint64_t packageBytes = 0;
            if (!checked_multiply(
                    descriptor.layout.transformBytes,
                    3,
                    packageBytes) ||
                !checked_add(
                    prospectiveBytes,
                    packageBytes,
                    prospectiveBytes)) {
                outError.diagnostic =
                    "ExactAdmissionFailure component=diffusion field=spectrum_upper_bound";
                return false;
            }
        }
        if (rebuildWorkspace &&
            (!checked_add(
                 prospectiveBytes,
                 descriptor.layout.transformBytes,
                 prospectiveBytes) ||
             !checked_add(
                 prospectiveBytes,
                 descriptor.reservedSharedWorkBytes,
                 prospectiveBytes) ||
             !checked_add(
                 prospectiveBytes,
                 descriptor.planAllowanceBytes,
                 prospectiveBytes) ||
             !checked_add(
                 prospectiveBytes,
                 descriptor.stagePlaneBytes,
                 prospectiveBytes))) {
            outError.diagnostic =
                "ExactAdmissionFailure component=diffusion field=workspace_upper_bound";
            return false;
        }
        if (prospectiveBytes > 0) {
            Failure ledgerError;
            if (!ledger->reserve(
                    DeviceReservationRequest{
                        .contextKey = contextKey,
                        .contextEpoch = contextEpoch,
                        .bytes = prospectiveBytes},
                    aggregate,
                    ledgerError)) {
                outError = std::move(ledgerError);
                return false;
            }
        }
        workspace.state = WorkspaceSlotState::Building;

        PreparedDiffusionLease lease;
        lease._owner = &resources;
        lease._descriptor = descriptor;
        lease._spectrumIndices = spectrumIndices;
        lease._spectrumCount = descriptor.uniqueSpectrumCount;
        lease._workspaceIndex = workspaceIndex;
        lease._streamOpaque = cudaStreamOpaque;
        lease._active = true;

        (void)spectrumClaims.release();
        (void)claim.release();

        const auto failPreparation = [&] {
            const auto waitResult = lease._workEnqueued ? ResourceOperations::synchronize_stream(stream) : cudaSuccess;
            record_cleanup_context_loss(outError, runtime_failure_status(waitResult));
            workspace.state = WorkspaceSlotState::FailedQuarantined;
            if (waitResult != cudaSuccess) {
                workspace.completion = WorkspaceCompletion::Unfenced;
            }
            for (std::size_t index = 0; index < lease._spectrumCount; ++index) {
                auto& entry = resources.spectra[spectrumIndices[index]];
                entry.leased = false;
                if (missingSpectrum[index] || lease._workEnqueued) {
                    entry.state = SpectrumEntryState::FailedQuarantined;
                }
                if (waitResult != cudaSuccess) {
                    entry.pendingWorkspaceMask |= static_cast<std::uint8_t>(1u << workspaceIndex);
                }
            }
            lease.reset();
            if (waitResult == cudaSuccess) {
                Failure cleanupError{{FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}};
                if (!complete_workspace_retirement(resources, workspaceIndex, cleanupError)) {
                    retain_first_failure(outError, std::move(cleanupError));
                }
                Failure spectrumError;
                if (!reap_retired_spectra(resources, spectrumError)) {
                    retain_first_failure(outError, std::move(spectrumError));
                }
            }
            release_host_lease(resources);
            return false;
        };

        try {
            for (std::size_t keyIndex = 0;
                 keyIndex < descriptor.uniqueSpectrumCount;
                 ++keyIndex) {
                if (!dependentBuilding[keyIndex]) {
                    continue;
                }
                const DiffusionSpectrumEntry& entry =
                    resources.spectra[spectrumIndices[keyIndex]];
                if (entry.state != SpectrumEntryState::Building ||
                    !entry.buildEventRecorded) {
                    outError.diagnostic =
                        "MissingRequiredResource component=diffusion field=same_key_build_event";
                    return failPreparation();
                }
                const cudaError_t waitResult = cudaStreamWaitEvent(
                    stream,
                    entry.buildEvent.get(),
                    0);
                if (waitResult != cudaSuccess) {
                    set_failure(outError, runtime_failure_status(waitResult), "cudaStreamWaitEvent(diffusion same key) failed");
                    return failPreparation();
                }
                lease._workEnqueued = true;
            }

            if (rebuildWorkspace) {
                Failure cleanupError;
                const bool workspaceCleared = destroy_workspace_contents(workspace, cleanupError);
                if (!workspaceCleared ||
                    !build_workspace(
                        workspace,
                        frameSet,
                        descriptor,
                        aggregate,
                        stream,
                        outError)) {
                    if (!workspaceCleared) {
                        outError = std::move(cleanupError);
                    }
                    return failPreparation();
                }
            } else {
                const cufftResult r2cResult =
                    cufftSetStream(workspace.r2cPlan.get(), stream);
                const cufftResult c2rResult =
                    cufftSetStream(workspace.c2rPlan.get(), stream);
                if (r2cResult != CUFFT_SUCCESS || c2rResult != CUFFT_SUCCESS) {
                    set_failure(outError, cufft_failure_status(r2cResult != CUFFT_SUCCESS ? r2cResult : c2rResult), "cufftSetStream(diffusion reuse) failed");
                    return failPreparation();
                }
                workspace.stagePlanes.rowStrideFloats =
                    static_cast<std::size_t>(frameSet.fullFrame.width);
            }

            for (std::size_t keyIndex = 0;
                 keyIndex < descriptor.uniqueSpectrumCount;
                 ++keyIndex) {
                if (!missingSpectrum[keyIndex]) {
                    continue;
                }
                DiffusionSpectrumEntry& entry =
                    resources.spectra[spectrumIndices[keyIndex]];
                lease._workEnqueued = true;
                if (!build_spectrum(
                        entry,
                        workspace,
                        descriptor,
                        components[keyIndex],
                        radii[keyIndex],
                        aggregate,
                        stream,
                        outError)) {
                    return failPreparation();
                }
            }

        } catch (const std::bad_alloc&) {
            if (outError.status.category == FJ_STATUS_PREPARATION_FAILURE && outError.status.api == FJ_API_NONE) {
                set_failure(outError, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "diffusion preparation allocation failed");
            }
            return failPreparation();
        } catch (...) {
            JuicerLogging::discard_current_exception();
            if (outError.status.category == FJ_STATUS_PREPARATION_FAILURE && outError.status.api == FJ_API_NONE) {
                set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "diffusion preparation failed");
            }
            return failPreparation();
        }

        workspace.state = WorkspaceSlotState::Leased;
        lease.refresh_view();
        outLease = std::move(lease);
        return true;
    }

    void mark_diffusion_work_enqueued(PreparedDiffusionLease& lease) noexcept {
        if (lease._active) {
            lease._workEnqueued = true;
        }
    }

    bool release_diffusion_resources(
        DiffusionContextResources& resources,
        PreparedDiffusionLease& lease,
        void* cudaStreamOpaque,
        bool dependentFailure,
        Failure& outError) noexcept {
        outError = {{FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}};
        if (!lease._active || lease._owner != &resources) {
            return true;
        }
        auto* const stream = reinterpret_cast<cudaStream_t>(cudaStreamOpaque ? cudaStreamOpaque : lease._streamOpaque);
        const auto workspaceIndex = lease._workspaceIndex;
        auto& workspace = resources.workspaces[workspaceIndex];
        if (lease._workEnqueued) {
            if (!workspace.completionEvent) {
                cudaEvent_t event = nullptr;
                const auto result = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
                workspace.completionEvent = RetiredEvent(event);
                if (result != cudaSuccess || !event) {
                    set_failure(outError, runtime_failure_status(result), "cudaEventCreate(diffusion release) failed");
                }
            }
            if (outError.status.category == FJ_STATUS_SUCCESS) {
                const auto result = ResourceOperations::record_event(workspace.completionEvent.get(), stream);
                if (result == cudaSuccess) {
                    workspace.completion = WorkspaceCompletion::Recorded;
                } else {
                    set_failure(outError, runtime_failure_status(result), "cudaEventRecord(diffusion release) failed");
                }
            }
            if (workspace.completion != WorkspaceCompletion::Recorded) {
                const auto waitResult = ResourceOperations::synchronize_stream(stream);
                record_cleanup_context_loss(outError, runtime_failure_status(waitResult));
                if (waitResult != cudaSuccess) {
                    workspace.completion = WorkspaceCompletion::Unfenced;
                }
            }
        }

        for (std::size_t index = 0; index < lease._spectrumCount; ++index) {
            const auto spectrumIndex = lease._spectrumIndices[index];
            auto& entry = resources.spectra[spectrumIndex];
            entry.leased = false;
            if (workspace.completion != WorkspaceCompletion::Complete) {
                entry.pendingWorkspaceMask |= static_cast<std::uint8_t>(1u << workspaceIndex);
            }
            if (dependentFailure || workspace.completion == WorkspaceCompletion::Unfenced) {
                entry.state = SpectrumEntryState::FailedQuarantined;
            } else if (!retained_spectrum_slot(spectrumIndex)) {
                entry.state = SpectrumEntryState::Retiring;
            }
            entry.releaseSequence = resources.nextReleaseSequence;
            if (resources.nextReleaseSequence < std::numeric_limits<std::uint64_t>::max()) {
                ++resources.nextReleaseSequence;
            }
        }
        workspace.state = dependentFailure || workspace.completion == WorkspaceCompletion::Unfenced
                              ? WorkspaceSlotState::FailedQuarantined
                              : WorkspaceSlotState::Retiring;
        workspace.releaseSequence = resources.nextReleaseSequence;
        if (resources.nextReleaseSequence < std::numeric_limits<std::uint64_t>::max()) {
            ++resources.nextReleaseSequence;
        }

        lease.reset();
        bool released = workspace.completion != WorkspaceCompletion::Unfenced;
        if (workspace.completion == WorkspaceCompletion::Complete) {
            Failure cleanupError{{FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}};
            released = complete_workspace_retirement(resources, workspaceIndex, cleanupError);
            if (!released) {
                retain_first_failure(outError, std::move(cleanupError));
            }
            Failure spectrumError;
            if (!reap_retired_spectra(resources, spectrumError)) {
                retain_first_failure(outError, std::move(spectrumError));
                released = false;
            }
        }
        release_host_lease(resources);
        return released && !context_loss(outError);
    }

} // namespace JuicerCuda::Diffusion
