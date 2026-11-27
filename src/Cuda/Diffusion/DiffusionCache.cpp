#include "Cuda/Diffusion/JuicerCudaDiffusionResources.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <optional>
#include <utility>

#include <cuda.h>
#include <cudaTypedefs.h>

#include "Logging.h"
#include "Cuda/Diffusion/DiffusionInternals.h"
#include "Cuda/JuicerCudaResourceOperations.h"

namespace JuicerCuda::Diffusion {

    namespace {
        constexpr std::size_t kRetainedSpectrumSlots = 2;
        constexpr std::size_t kRetainedWorkspaceSlots = 1;

    } // namespace

    // Diffusion selects the Runtime's CUDA 4.0 entry point after checking
    // the runtime device. Driver inspection also runs before Runtime setup.
    struct CurrentContextDispatch {
        PFN_cuCtxGetCurrent_v4000 function = nullptr;
        cudaError_t lookupResult = cudaSuccess;
        cudaDriverEntryPointQueryResult queryResult =
            cudaDriverEntryPointSymbolNotFound;
    };

    static const CurrentContextDispatch& current_context_dispatch() {
        static const CurrentContextDispatch dispatch = [] {
            CurrentContextDispatch result{};
            void* functionPointer = nullptr;
            result.lookupResult = cudaGetDriverEntryPointByVersion(
                "cuCtxGetCurrent",
                &functionPointer,
                4000,
                cudaEnableDefault,
                &result.queryResult);
            result.function = reinterpret_cast<PFN_cuCtxGetCurrent_v4000>(
                functionPointer);
            return result;
        }();
        return dispatch;
    }

    bool current_owner_matches(
        const ResourceManager::DeviceContextKey& contextKey,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        int currentDevice = -1;
        const cudaError_t deviceResult = cudaGetDevice(&currentDevice);
        if (deviceResult != cudaSuccess) {
            set_failure(outError, runtime_failure_status(deviceResult), "cudaGetDevice(diffusion owner) failed");
            return false;
        }
        const CurrentContextDispatch& dispatch = current_context_dispatch();
        if (dispatch.lookupResult != cudaSuccess || !dispatch.function) {
            std::array<char, 192> diagnostic{};
            std::snprintf(diagnostic.data(), diagnostic.size(), "cudaGetDriverEntryPointByVersion(cuCtxGetCurrent) failed runtime_code=%d query_status=%d", static_cast<int>(dispatch.lookupResult), static_cast<int>(dispatch.queryResult));
            set_failure(outError, runtime_failure_status(dispatch.lookupResult), diagnostic.data());
            return false;
        }
        CUcontext currentContext = nullptr;
        const CUresult contextResult = dispatch.function(&currentContext);
        if (contextResult != CUDA_SUCCESS) {
            set_failure(outError, driver_failure_status(contextResult), "cuCtxGetCurrent(diffusion owner) failed");
            return false;
        }
        if (currentDevice != contextKey.deviceId ||
            reinterpret_cast<void*>(currentContext) !=
                contextKey.contextOpaque) {
            outError.diagnostic =
                "ResourceDescriptorMismatch component=diffusion field=context_owner";
            return false;
        }
        return true;
    }

    bool retained_spectrum_slot(std::size_t index) noexcept {
        return index < kRetainedSpectrumSlots;
    }

    bool retained_workspace_slot(std::size_t index) noexcept {
        return index < kRetainedWorkspaceSlots;
    }

    static bool release_reservation_after_free(
        DeviceByteReservation& reservation,
        Failure& outError) noexcept try {
        if (!reservation.active()) {
            return true;
        }
        if (reservation.state() == DeviceReservationState::Committed &&
            !reservation.mark_retiring(outError.diagnostic)) {
            return false;
        }
        return reservation.release_after_physical_free(true, outError.diagnostic);
    } catch (...) {
        set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "diffusion plan allowance release failed");
        return false;
    }

    static bool destroy_plan_pair(
        DiffusionWorkspaceSlot& slot,
        Failure& outError) noexcept {
        outError = {{FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}};
        const auto destroy = [&](CufftPlan& plan, const char* diagnostic) {
            const auto result = plan.destroy_after_completion();
            if (result != CUFFT_SUCCESS) {
                Failure failure;
                set_failure(failure, cufft_failure_status(result), diagnostic);
                retain_first_failure(outError, std::move(failure));
            }
        };
        destroy(slot.r2cPlan, "cufftDestroy(diffusion R2C) failed");
        destroy(slot.c2rPlan, "cufftDestroy(diffusion C2R) failed");
        if (outError.status.category != FJ_STATUS_SUCCESS) {
            return false;
        }
        Failure releaseError;
        if (!release_reservation_after_free(slot.planAllowanceReservation, releaseError)) {
            outError = std::move(releaseError);
            return false;
        }
        return true;
    }

    bool destroy_workspace_contents(
        DiffusionWorkspaceSlot& slot,
        Failure& outError) noexcept {
        outError = {{FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}};
        const auto free = [&](TrackedDeviceBuffer& buffer) {
            Failure failure;
            if (buffer.free_after_completion(failure)) {
                return true;
            }
            retain_first_failure(outError, std::move(failure));
            return false;
        };
        for (std::size_t index = 0; index < slot.stagePlaneBuffers.size(); ++index) {
            if (free(slot.stagePlaneBuffers[index])) {
                assign_stage_plane_pointer(slot, index, nullptr);
            }
        }
        free(slot.transform);
        Failure planError;
        if (!destroy_plan_pair(slot, planError)) {
            retain_first_failure(outError, std::move(planError));
        }
        if (slot.r2cPlan.get() == 0 && slot.c2rPlan.get() == 0) {
            free(slot.workArea);
        }
        if (outError.status.category != FJ_STATUS_SUCCESS) {
            return false;
        }
        slot.planKey = {};
        slot.transformCapacityBytes = 0;
        slot.stagePlaneCapacityElements = 0;
        slot.releaseSequence = 0;
        slot.stagePlanes = {};
        return true;
    }

    bool destroy_spectrum_contents(
        DiffusionSpectrumEntry& entry,
        Failure& outError) noexcept {
        outError = {{FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}};
        for (std::size_t index = 0; index < entry.spectrumBuffers.size(); ++index) {
            Failure failure;
            if (entry.spectrumBuffers[index].free_after_completion(failure)) {
                assign_spectrum_pointer(entry, index, nullptr);
            } else {
                retain_first_failure(outError, std::move(failure));
            }
        }
        entry.buildEventRecorded = false;
        if (const auto result = entry.buildEvent.destroy(); result != cudaSuccess) {
            Failure failure;
            set_failure(failure, runtime_failure_status(result), "cudaEventDestroy(diffusion spectrum) failed");
            retain_first_failure(outError, std::move(failure));
        }
        if (outError.status.category != FJ_STATUS_SUCCESS) {
            return false;
        }
        entry = DiffusionSpectrumEntry{};
        return true;
    }

    static void clear_workspace_use_bit(
        DiffusionContextResources& resources,
        std::size_t workspaceIndex) noexcept {
        const std::uint8_t bit =
            static_cast<std::uint8_t>(1u << workspaceIndex);
        for (DiffusionSpectrumEntry& entry : resources.spectra) {
            entry.pendingWorkspaceMask = static_cast<std::uint8_t>(
                entry.pendingWorkspaceMask & ~bit);
        }
    }

    bool complete_workspace_retirement(
        DiffusionContextResources& resources,
        std::size_t workspaceIndex,
        Failure& outError) noexcept {
        DiffusionWorkspaceSlot& slot = resources.workspaces[workspaceIndex];
        slot.completion = WorkspaceCompletion::Complete;
        clear_workspace_use_bit(resources, workspaceIndex);
        if ((slot.state == WorkspaceSlotState::FailedQuarantined ||
             !retained_workspace_slot(workspaceIndex)) &&
            !destroy_workspace_contents(slot, outError)) {
            slot.state = WorkspaceSlotState::FailedQuarantined;
            return false;
        }
        slot.releaseSequence = 0;
        slot.state = WorkspaceSlotState::Vacant;
        return true;
    }

    bool reap_retired_spectra(
        DiffusionContextResources& resources,
        Failure& outError) noexcept {
        bool reaped = true;
        for (DiffusionSpectrumEntry& entry : resources.spectra) {
            if ((entry.state != SpectrumEntryState::Retiring &&
                 entry.state != SpectrumEntryState::FailedQuarantined) ||
                entry.leased || entry.pendingWorkspaceMask != 0) {
                continue;
            }
            // The producing workspace is a dependency as well as every
            // consuming workspace. The build event only establishes readiness.
            Failure failure;
            if (!destroy_spectrum_contents(entry, failure)) {
                entry.state = SpectrumEntryState::FailedQuarantined;
                if (reaped) {
                    outError = std::move(failure);
                } else {
                    retain_first_failure(outError, std::move(failure));
                }
                reaped = false;
            }
        }
        return reaped;
    }

    bool reap_completed(
        DiffusionContextResources& resources,
        Failure& outError) {
        for (DiffusionSpectrumEntry& entry : resources.spectra) {
            if (entry.state != SpectrumEntryState::Building ||
                !entry.buildEventRecorded) {
                continue;
            }
            const auto result = cudaEventQuery(entry.buildEvent.get());
            if (result == cudaSuccess) {
                entry.buildEventRecorded = false;
                entry.state = SpectrumEntryState::Ready;
                const auto destroyResult = entry.buildEvent.destroy();
                const auto status = runtime_failure_status(destroyResult);
                if (status.category == FJ_STATUS_CONTEXT_LOSS) {
                    entry.state = SpectrumEntryState::FailedQuarantined;
                    set_failure(outError, status, "cudaEventDestroy(diffusion build completion) failed");
                    return false;
                }
            } else if (result != cudaErrorNotReady) {
                entry.state = SpectrumEntryState::FailedQuarantined;
                if (runtime_failure_status(result).category == FJ_STATUS_CONTEXT_LOSS) {
                    set_failure(outError, runtime_failure_status(result), "cudaEventQuery(diffusion spectrum) failed");
                    return false;
                }
            }
        }

        for (std::size_t index = 0; index < resources.workspaces.size();
             ++index) {
            DiffusionWorkspaceSlot& slot = resources.workspaces[index];
            if ((slot.state != WorkspaceSlotState::Retiring &&
                 slot.state != WorkspaceSlotState::FailedQuarantined) ||
                slot.completion == WorkspaceCompletion::Unfenced) {
                continue;
            }
            const auto result = slot.completion == WorkspaceCompletion::Complete
                                    ? cudaSuccess
                                    : ResourceOperations::query_event(slot.completionEvent.get());
            if (result == cudaSuccess) {
                if (!complete_workspace_retirement(resources, index, outError)) {
                    return false;
                }
            } else if (result != cudaErrorNotReady) {
                slot.state = WorkspaceSlotState::FailedQuarantined;
                set_failure(outError, runtime_failure_status(result), "cudaEventQuery(diffusion workspace) failed");
                return false;
            }
        }

        return reap_retired_spectra(resources, outError);
    }

    std::size_t choose_retiring_workspace(
        const DiffusionContextResources& resources) noexcept {
        std::size_t selected = kInvalidIndex;
        for (std::size_t index = 0; index < resources.workspaces.size();
             ++index) {
            const DiffusionWorkspaceSlot& slot = resources.workspaces[index];
            if ((slot.state != WorkspaceSlotState::Retiring &&
                 slot.state != WorkspaceSlotState::FailedQuarantined) ||
                slot.completion != WorkspaceCompletion::Recorded) {
                continue;
            }
            if (selected == kInvalidIndex ||
                slot.releaseSequence <
                    resources.workspaces[selected].releaseSequence ||
                (slot.releaseSequence ==
                     resources.workspaces[selected].releaseSequence &&
                 index < selected)) {
                selected = index;
            }
        }
        return selected;
    }

    static bool has_diffusion_residency(
        const DiffusionContextResources& resources) noexcept {
        for (const DiffusionWorkspaceSlot& slot : resources.workspaces) {
            if (slot.state != WorkspaceSlotState::Vacant ||
                slot.r2cPlan.get() != 0 || slot.c2rPlan.get() != 0 || slot.workArea.get() ||
                slot.transform.get() || slot.stagePlanes.redSensitive ||
                slot.stagePlanes.greenSensitive ||
                slot.stagePlanes.blueSensitive ||
                slot.stagePlanes.auxiliary) {
                return true;
            }
        }
        for (const DiffusionSpectrumEntry& entry : resources.spectra) {
            if (entry.state != SpectrumEntryState::Vacant ||
                entry.spectra.red || entry.spectra.green ||
                entry.spectra.blue) {
                return true;
            }
        }
        return false;
    }

    bool same_spectrum_key(
        const Spektrafilm::DiffusionSpectrumKey& left,
        const Spektrafilm::DiffusionSpectrumKey& right) noexcept {
        return left.hash != 0 && left.hash == right.hash &&
               left.sampleHash == right.sampleHash &&
               left.extent == right.extent;
    }

    bool same_plan_key(
        const Spektrafilm::DiffusionPlanKey& left,
        const Spektrafilm::DiffusionPlanKey& right) noexcept {
        return left.hash != 0 && left.hash == right.hash &&
               left.extent == right.extent;
    }

    std::size_t choose_missing_spectrum_slot(
        DiffusionContextResources& resources,
        Failure& outError) {
        for (std::size_t index = 0; index < kRetainedSpectrumSlots; ++index) {
            if (resources.spectra[index].state ==
                SpectrumEntryState::Vacant) {
                return index;
            }
        }

        std::size_t eviction = kInvalidIndex;
        for (std::size_t index = 0; index < kRetainedSpectrumSlots; ++index) {
            const DiffusionSpectrumEntry& entry = resources.spectra[index];
            if (entry.state != SpectrumEntryState::Ready ||
                entry.leased || entry.pendingWorkspaceMask != 0) {
                continue;
            }
            if (eviction == kInvalidIndex ||
                entry.releaseSequence <
                    resources.spectra[eviction].releaseSequence ||
                (entry.releaseSequence ==
                     resources.spectra[eviction].releaseSequence &&
                 entry.key.hash < resources.spectra[eviction].key.hash)) {
                eviction = index;
            }
        }
        if (eviction != kInvalidIndex) {
            if (destroy_spectrum_contents(resources.spectra[eviction], outError)) {
                return eviction;
            }
            resources.spectra[eviction].state = SpectrumEntryState::FailedQuarantined;
            if (context_loss(outError)) {
                return kInvalidIndex;
            }
        }

        for (std::size_t index = kRetainedSpectrumSlots;
             index < resources.spectra.size();
             ++index) {
            if (resources.spectra[index].state ==
                SpectrumEntryState::Vacant) {
                return index;
            }
        }
        return kInvalidIndex;
    }

    bool trim_inactive_diffusion_resources(
        DiffusionContextResources& resources,
        const ResourceManager::DeviceContextKey& contextKey,
        Failure& outError) noexcept {
        bool hostLeaseClaimed = false;
        try {
            outError = {};
            if (!acquire_host_lease(resources, outError)) {
                return false;
            }
            hostLeaseClaimed = true;
            if (!has_diffusion_residency(resources)) {
                release_host_lease(resources);
                return true;
            }
            if (!current_owner_matches(contextKey, outError)) {
                release_host_lease(resources);
                return false;
            }
            if (!reap_completed(resources, outError)) {
                release_host_lease(resources);
                return false;
            }

            outError = {{FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}};
            for (DiffusionWorkspaceSlot& workspace : resources.workspaces) {
                if (workspace.state != WorkspaceSlotState::Vacant) {
                    continue;
                }
                Failure localError;
                if (!destroy_workspace_contents(workspace, localError)) {
                    workspace.state = WorkspaceSlotState::FailedQuarantined;
                    retain_first_failure(outError, std::move(localError));
                }
            }
            for (DiffusionSpectrumEntry& entry : resources.spectra) {
                if (entry.state != SpectrumEntryState::Ready ||
                    entry.leased || entry.pendingWorkspaceMask != 0) {
                    continue;
                }
                Failure localError;
                if (!destroy_spectrum_contents(entry, localError)) {
                    entry.state = SpectrumEntryState::FailedQuarantined;
                    retain_first_failure(outError, std::move(localError));
                }
            }
            release_host_lease(resources);
            return outError.status.category == FJ_STATUS_SUCCESS;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            if (hostLeaseClaimed) {
                release_host_lease(resources);
            }
            try {
                outError.diagnostic = "inactive diffusion residency trim failed";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
    }

    bool drain_diffusion_resources(
        DiffusionContextResources& resources,
        Failure& outError) noexcept {
        bool hostLeaseClaimed = false;
        bool drained = false;
        try {
            outError = {};
            outError.status = {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
            {
                std::lock_guard<std::mutex> lock(resources.metadataMutex);
                resources.acceptingPreparations = false;
                if (resources.hostLeaseActive) {
                    set_failure(outError, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "ResourceDescriptorMismatch component=diffusion field=drain_with_active_host_lease");
                    return false;
                }
                resources.hostLeaseActive = true;
                hostLeaseClaimed = true;
            }
            drained = true;
            std::optional<cudaError_t> contextWait;
            for (std::size_t index = 0; index < resources.workspaces.size(); ++index) {
                auto& workspace = resources.workspaces[index];
                cudaError_t waitResult = cudaSuccess;
                if (workspace.completion == WorkspaceCompletion::Unfenced) {
                    // The serving owner has validated the current context. A
                    // borrowed per-thread stream is meaningful only on its
                    // producer thread, so retirement needs a context barrier.
                    if (!contextWait) {
                        contextWait = ResourceOperations::synchronize_context();
                    }
                    waitResult = *contextWait;
                } else if (workspace.completion == WorkspaceCompletion::Recorded) {
                    waitResult = cudaEventSynchronize(workspace.completionEvent.get());
                }
                Failure failure;
                if (waitResult != cudaSuccess) {
                    drained = false;
                    set_failure(failure, runtime_failure_status(waitResult), "diffusion workspace completion wait failed");
                    retain_first_failure(outError, std::move(failure));
                    continue;
                }
                workspace.completion = WorkspaceCompletion::Complete;
                clear_workspace_use_bit(resources, index);
                if (!destroy_workspace_contents(workspace, failure)) {
                    drained = false;
                    workspace.state = WorkspaceSlotState::FailedQuarantined;
                    retain_first_failure(outError, std::move(failure));
                    continue;
                }
                if (const auto result = workspace.completionEvent.destroy(); result != cudaSuccess) {
                    drained = false;
                    workspace.state = WorkspaceSlotState::FailedQuarantined;
                    set_failure(failure, runtime_failure_status(result), "cudaEventDestroy(diffusion workspace) failed");
                    retain_first_failure(outError, std::move(failure));
                    continue;
                }
                workspace = DiffusionWorkspaceSlot{};
            }

            for (DiffusionSpectrumEntry& entry : resources.spectra) {
                if (entry.pendingWorkspaceMask != 0) {
                    drained = false;
                    continue;
                }
                Failure failure;
                if (!destroy_spectrum_contents(entry, failure)) {
                    drained = false;
                    entry.state = SpectrumEntryState::FailedQuarantined;
                    retain_first_failure(outError, std::move(failure));
                }
            }
        } catch (const std::bad_alloc&) {
            drained = false;
            if (outError.status.category == FJ_STATUS_SUCCESS) {
                set_failure(outError, {FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, "diffusion resource drain allocation failed");
            }
        } catch (...) {
            drained = false;
            if (outError.status.category == FJ_STATUS_SUCCESS) {
                set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "diffusion resource drain bookkeeping failed");
            }
        }
        if (hostLeaseClaimed) {
            release_host_lease(resources);
        }
        return drained;
    }

    void invalidate_diffusion_resources_after_proven_context_loss(
        DiffusionContextResources& resources) noexcept {
        try {
            std::lock_guard<std::mutex> lock(resources.metadataMutex);
            resources.acceptingPreparations = false;
            resources.hostLeaseActive = false;
            for (DiffusionSpectrumEntry& entry : resources.spectra) {
                (void)entry.buildEvent.release();
                for (auto& buffer : entry.spectrumBuffers) {
                    buffer.forget_after_context_loss();
                }
                entry = DiffusionSpectrumEntry{};
            }
            for (DiffusionWorkspaceSlot& workspace : resources.workspaces) {
                (void)workspace.completionEvent.release();
                workspace.r2cPlan.forget_after_context_loss();
                workspace.c2rPlan.forget_after_context_loss();
                workspace.workArea.forget_after_context_loss();
                workspace.transform.forget_after_context_loss();
                for (auto& buffer : workspace.stagePlaneBuffers) {
                    buffer.forget_after_context_loss();
                }
                workspace = DiffusionWorkspaceSlot{};
            }
            resources.nextReleaseSequence = 1;
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }


} // namespace JuicerCuda::Diffusion
