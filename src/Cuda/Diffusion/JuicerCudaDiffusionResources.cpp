
#include "Cuda/Diffusion/JuicerCudaDiffusionResources.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

#include <cuda.h>
#include <cudaTypedefs.h>

#include "Logging.h"

namespace JuicerCuda::Diffusion {
    namespace {

        constexpr std::size_t kInvalidIndex =
            std::numeric_limits<std::size_t>::max();
        constexpr std::size_t kRetainedSpectrumSlots = 2;
        constexpr std::size_t kRetainedWorkspaceSlots = 1;

        bool checked_add(
            std::uint64_t left,
            std::uint64_t right,
            std::uint64_t& out) noexcept {
            if (right > std::numeric_limits<std::uint64_t>::max() - left) {
                return false;
            }
            out = left + right;
            return true;
        }

        bool checked_multiply(
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

        const char* cuda_message(cudaError_t result) noexcept {
            const char* message = cudaGetErrorString(result);
            return message ? message : "unknown CUDA error";
        }

        struct CurrentContextDispatch {
            PFN_cuCtxGetCurrent_v4000 function = nullptr;
            cudaError_t lookupResult = cudaSuccess;
            cudaDriverEntryPointQueryResult queryResult =
                cudaDriverEntryPointSymbolNotFound;
        };

        const CurrentContextDispatch& current_context_dispatch() {
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
            std::string& outError) {
            int currentDevice = -1;
            const cudaError_t deviceResult = cudaGetDevice(&currentDevice);
            if (deviceResult != cudaSuccess) {
                outError = std::string("cudaGetDevice(diffusion owner) failed: ") +
                           cuda_message(deviceResult);
                return false;
            }
            const CurrentContextDispatch& dispatch = current_context_dispatch();
            if (dispatch.lookupResult != cudaSuccess || !dispatch.function) {
                outError =
                    "cudaGetDriverEntryPointByVersion(cuCtxGetCurrent) failed runtime_code=" +
                    std::to_string(static_cast<int>(dispatch.lookupResult)) +
                    " query_status=" +
                    std::to_string(static_cast<int>(dispatch.queryResult));
                return false;
            }
            CUcontext currentContext = nullptr;
            const CUresult contextResult = dispatch.function(&currentContext);
            if (contextResult != CUDA_SUCCESS) {
                outError = "cuCtxGetCurrent(diffusion owner) failed code=" +
                           std::to_string(static_cast<int>(contextResult));
                return false;
            }
            if (currentDevice != contextKey.deviceId ||
                reinterpret_cast<void*>(currentContext) !=
                    contextKey.contextOpaque) {
                outError =
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

        bool acquire_host_lease(
            DiffusionContextResources& resources,
            std::string& outError) {
            std::lock_guard<std::mutex> lock(resources.metadataMutex);
            if (!resources.acceptingPreparations) {
                outError =
                    "ExactAdmissionFailure component=diffusion class=context_retiring requested_new_bytes=0";
                return false;
            }
            if (resources.hostLeaseActive) {
                outError =
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

        bool allocate_from_aggregate(
            DeviceByteReservation& aggregate,
            std::uint64_t bytes,
            void*& outPointer,
            DeviceByteReservation& outReservation,
            std::string& outError) {
            outPointer = nullptr;
            if (bytes == 0 ||
                bytes > static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max())) {
                outError = "ExactAdmissionFailure component=diffusion field=allocation_bytes";
                return false;
            }
            DeviceByteReservation reservation;
            if (!aggregate.split(bytes, reservation, outError)) {
                return false;
            }
            void* pointer = nullptr;
            const cudaError_t allocationResult = cudaMalloc(
                &pointer,
                static_cast<std::size_t>(bytes));
            if (allocationResult != cudaSuccess || !pointer) {
                outError = std::string("cudaMalloc(diffusion) failed: ") +
                           cuda_message(allocationResult);
                return false;
            }
            if (!reservation.commit(bytes, outError)) {
                (void)cudaFree(pointer);
                return false;
            }
            outPointer = pointer;
            outReservation = std::move(reservation);
            return true;
        }

        bool commit_allowance_from_aggregate(
            DeviceByteReservation& aggregate,
            std::uint64_t bytes,
            DeviceByteReservation& outReservation,
            std::string& outError) {
            if (bytes == 0) {
                outError =
                    "UnsupportedCudaExecutionProfile component=diffusion field=plan_allowance";
                return false;
            }
            DeviceByteReservation reservation;
            if (!aggregate.split(bytes, reservation, outError) ||
                !reservation.commit(bytes, outError)) {
                return false;
            }
            outReservation = std::move(reservation);
            return true;
        }

        bool release_reservation_after_free(
            DeviceByteReservation& reservation,
            bool physicalReleaseSucceeded,
            std::string& outError) {
            if (!reservation.active()) {
                return physicalReleaseSucceeded;
            }
            if (reservation.state() == DeviceReservationState::Committed &&
                !reservation.mark_retiring(outError)) {
                return false;
            }
            return reservation.release_after_physical_free(
                physicalReleaseSucceeded,
                outError);
        }

        bool free_device_allocation(
            void*& pointer,
            DeviceByteReservation& reservation,
            std::string& outError) {
            if (!pointer) {
                return release_reservation_after_free(
                    reservation,
                    true,
                    outError);
            }
            const cudaError_t freeResult = cudaFree(pointer);
            if (freeResult != cudaSuccess) {
                std::string ignored;
                (void)release_reservation_after_free(
                    reservation,
                    false,
                    ignored);
                outError = std::string("cudaFree(diffusion) failed: ") +
                           cuda_message(freeResult);
                return false;
            }
            if (!release_reservation_after_free(reservation, true, outError)) {
                return false;
            }
            pointer = nullptr;
            return true;
        }

        bool destroy_plan_pair(
            DiffusionWorkspaceSlot& slot,
            std::string& outError) {
            bool destroyed = true;
            if (slot.r2cPlan != 0) {
                const cufftResult result = cufftDestroy(slot.r2cPlan);
                if (result == CUFFT_SUCCESS) {
                    slot.r2cPlan = 0;
                } else {
                    destroyed = false;
                    outError = "cufftDestroy(diffusion R2C) failed code=" +
                               std::to_string(static_cast<int>(result));
                }
            }
            if (slot.c2rPlan != 0) {
                const cufftResult result = cufftDestroy(slot.c2rPlan);
                if (result == CUFFT_SUCCESS) {
                    slot.c2rPlan = 0;
                } else {
                    destroyed = false;
                    if (outError.empty()) {
                        outError =
                            "cufftDestroy(diffusion C2R) failed code=" +
                            std::to_string(static_cast<int>(result));
                    }
                }
            }
            if (slot.r2cPlan == 0 && slot.c2rPlan == 0 &&
                !release_reservation_after_free(
                    slot.planAllowanceReservation,
                    true,
                    outError)) {
                destroyed = false;
            }
            return destroyed;
        }

        std::array<void*, 4> stage_plane_pointers(
            DiffusionWorkspaceSlot& slot) noexcept {
            return {slot.stagePlanes.redSensitive,
                    slot.stagePlanes.greenSensitive,
                    slot.stagePlanes.blueSensitive,
                    slot.stagePlanes.auxiliary};
        }

        void assign_stage_plane_pointer(
            DiffusionWorkspaceSlot& slot,
            std::size_t index,
            void* pointer) noexcept {
            float* value = static_cast<float*>(pointer);
            if (index == 0) {
                slot.stagePlanes.redSensitive = value;
            } else if (index == 1) {
                slot.stagePlanes.greenSensitive = value;
            } else if (index == 2) {
                slot.stagePlanes.blueSensitive = value;
            } else {
                slot.stagePlanes.auxiliary = value;
            }
        }

        bool destroy_workspace_contents(
            DiffusionWorkspaceSlot& slot,
            std::string& outError) {
            bool released = true;
            auto planePointers = stage_plane_pointers(slot);
            for (std::size_t index = 0; index < planePointers.size(); ++index) {
                void* pointer = planePointers[index];
                std::string localError;
                if (!free_device_allocation(
                        pointer,
                        slot.stagePlaneReservations[index],
                        localError)) {
                    released = false;
                    if (outError.empty()) {
                        outError = localError;
                    }
                } else {
                    assign_stage_plane_pointer(slot, index, nullptr);
                }
            }
            void* transform = slot.transformBuffer;
            std::string localError;
            if (!free_device_allocation(
                    transform,
                    slot.transformReservation,
                    localError)) {
                released = false;
                if (outError.empty()) {
                    outError = localError;
                }
            } else {
                slot.transformBuffer = nullptr;
            }
            void* work = slot.workArea;
            if (!free_device_allocation(
                    work,
                    slot.workAreaReservation,
                    localError)) {
                released = false;
                if (outError.empty()) {
                    outError = localError;
                }
            } else {
                slot.workArea = nullptr;
            }
            if (!destroy_plan_pair(slot, localError)) {
                released = false;
                if (outError.empty()) {
                    outError = localError;
                }
            }
            if (released) {
                slot.planKey = {};
                slot.transformCapacityBytes = 0;
                slot.stagePlaneCapacityElements = 0;
                slot.releaseSequence = 0;
                slot.stagePlanes = {};
                slot.completionUnknown = false;
            }
            return released;
        }

        std::array<void*, 3> spectrum_pointers(
            DiffusionSpectrumEntry& entry) noexcept {
            return {entry.spectra.red, entry.spectra.green, entry.spectra.blue};
        }

        void assign_spectrum_pointer(
            DiffusionSpectrumEntry& entry,
            std::size_t index,
            void* pointer) noexcept {
            cufftComplex* value = static_cast<cufftComplex*>(pointer);
            if (index == 0) {
                entry.spectra.red = value;
            } else if (index == 1) {
                entry.spectra.green = value;
            } else {
                entry.spectra.blue = value;
            }
        }

        bool destroy_spectrum_contents(
            DiffusionSpectrumEntry& entry,
            std::string& outError) {
            bool released = true;
            auto pointers = spectrum_pointers(entry);
            for (std::size_t index = 0; index < pointers.size(); ++index) {
                void* pointer = pointers[index];
                std::string localError;
                if (!free_device_allocation(
                        pointer,
                        entry.spectrumReservations[index],
                        localError)) {
                    released = false;
                    if (outError.empty()) {
                        outError = localError;
                    }
                } else {
                    assign_spectrum_pointer(entry, index, nullptr);
                }
            }
            if (entry.buildEventOpaque) {
                const cudaError_t eventResult = cudaEventDestroy(
                    reinterpret_cast<cudaEvent_t>(entry.buildEventOpaque));
                if (eventResult == cudaSuccess) {
                    entry.buildEventOpaque = nullptr;
                } else {
                    released = false;
                    if (outError.empty()) {
                        outError =
                            std::string("cudaEventDestroy(diffusion spectrum) failed: ") +
                            cuda_message(eventResult);
                    }
                }
            }
            if (released) {
                entry = DiffusionSpectrumEntry{};
            }
            return released;
        }

        bool event_complete(void* eventOpaque, cudaError_t& outResult) noexcept {
            if (!eventOpaque) {
                outResult = cudaSuccess;
                return true;
            }
            outResult = cudaEventQuery(
                reinterpret_cast<cudaEvent_t>(eventOpaque));
            return outResult == cudaSuccess;
        }

        void clear_workspace_use_bit(
            DiffusionContextResources& resources,
            std::size_t workspaceIndex) noexcept {
            const std::uint8_t bit =
                static_cast<std::uint8_t>(1u << workspaceIndex);
            for (DiffusionSpectrumEntry& entry : resources.spectra) {
                entry.pendingWorkspaceMask = static_cast<std::uint8_t>(
                    entry.pendingWorkspaceMask & ~bit);
            }
        }

        void complete_workspace_retirement(
            DiffusionContextResources& resources,
            std::size_t workspaceIndex) {
            DiffusionWorkspaceSlot& slot = resources.workspaces[workspaceIndex];
            const bool discard =
                slot.state == WorkspaceSlotState::FailedQuarantined ||
                !retained_workspace_slot(workspaceIndex);
            clear_workspace_use_bit(resources, workspaceIndex);
            if (discard) {
                std::string ignored;
                if (destroy_workspace_contents(slot, ignored)) {
                    slot.state = WorkspaceSlotState::Vacant;
                } else {
                    slot.state = WorkspaceSlotState::FailedQuarantined;
                }
            } else {
                slot.releaseSequence = 0;
                slot.state = WorkspaceSlotState::Vacant;
            }
        }

        void reap_retired_spectra(
            DiffusionContextResources& resources) {
            for (DiffusionSpectrumEntry& entry : resources.spectra) {
                if ((entry.state != SpectrumEntryState::Retiring &&
                     entry.state != SpectrumEntryState::FailedQuarantined) ||
                    entry.leased || entry.pendingWorkspaceMask != 0) {
                    continue;
                }
                cudaError_t buildResult = cudaSuccess;
                if (entry.buildEventOpaque &&
                    !event_complete(entry.buildEventOpaque, buildResult)) {
                    continue;
                }
                std::string ignored;
                if (!destroy_spectrum_contents(entry, ignored)) {
                    entry.state = SpectrumEntryState::FailedQuarantined;
                }
            }
        }

        void reap_completed(
            DiffusionContextResources& resources) {
            for (DiffusionSpectrumEntry& entry : resources.spectra) {
                if (entry.state != SpectrumEntryState::Building ||
                    !entry.buildEventOpaque) {
                    continue;
                }
                cudaError_t result = cudaSuccess;
                if (event_complete(entry.buildEventOpaque, result)) {
                    (void)cudaEventDestroy(
                        reinterpret_cast<cudaEvent_t>(entry.buildEventOpaque));
                    entry.buildEventOpaque = nullptr;
                    entry.state = SpectrumEntryState::Ready;
                } else if (result != cudaErrorNotReady) {
                    entry.state = SpectrumEntryState::FailedQuarantined;
                }
            }

            for (std::size_t index = 0; index < resources.workspaces.size();
                 ++index) {
                DiffusionWorkspaceSlot& slot = resources.workspaces[index];
                if ((slot.state != WorkspaceSlotState::Retiring &&
                     slot.state != WorkspaceSlotState::FailedQuarantined) ||
                    slot.completionUnknown || !slot.completionEventOpaque) {
                    continue;
                }
                cudaError_t result = cudaSuccess;
                if (event_complete(slot.completionEventOpaque, result)) {
                    complete_workspace_retirement(resources, index);
                } else if (result != cudaErrorNotReady) {
                    slot.state = WorkspaceSlotState::FailedQuarantined;
                }
            }

            reap_retired_spectra(resources);
        }

        std::size_t choose_retiring_workspace(
            const DiffusionContextResources& resources) noexcept {
            std::size_t selected = kInvalidIndex;
            for (std::size_t index = 0; index < resources.workspaces.size();
                 ++index) {
                const DiffusionWorkspaceSlot& slot = resources.workspaces[index];
                if ((slot.state != WorkspaceSlotState::Retiring &&
                     slot.state != WorkspaceSlotState::FailedQuarantined) ||
                    slot.completionUnknown || !slot.completionEventOpaque) {
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

        bool has_diffusion_residency(
            const DiffusionContextResources& resources) noexcept {
            for (const DiffusionWorkspaceSlot& slot : resources.workspaces) {
                if (slot.state != WorkspaceSlotState::Vacant ||
                    slot.r2cPlan != 0 || slot.c2rPlan != 0 || slot.workArea ||
                    slot.transformBuffer || slot.stagePlanes.redSensitive ||
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
            DiffusionContextResources& resources) {
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
                std::string ignored;
                if (destroy_spectrum_contents(
                        resources.spectra[eviction],
                        ignored)) {
                    return eviction;
                }
                resources.spectra[eviction].state =
                    SpectrumEntryState::FailedQuarantined;
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

        const Spektrafilm::DiffusionStageFrameDescriptor* stage_for_key(
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

        bool validate_prepare_request(
            const Spektrafilm::DiffusionFrameSetDescriptor& frameSet,
            const Spektrafilm::DiffusionExecutionDescriptor& descriptor,
            std::uint64_t contextEpoch,
            std::string& outError) {
            if (descriptor.hash == 0 || descriptor.frameSetHash != frameSet.hash ||
                descriptor.contextEpoch != contextEpoch ||
                descriptor.stageCount == 0 || descriptor.stageCount > 2 ||
                descriptor.uniqueSpectrumCount == 0 ||
                descriptor.uniqueSpectrumCount > 2 ||
                descriptor.planKey.hash == 0 ||
                descriptor.layout.transformBytes == 0 ||
                descriptor.stagePlaneBytes == 0 ||
                descriptor.stagePlaneBytes % 4u != 0) {
                outError =
                    "ResourceDescriptorMismatch component=diffusion field=execution_descriptor";
                return false;
            }
            for (std::size_t keyIndex = 0;
                 keyIndex < descriptor.uniqueSpectrumCount;
                 ++keyIndex) {
                const auto* stage = stage_for_key(
                    frameSet,
                    descriptor,
                    keyIndex);
                if (!stage || stage->sample.hash != descriptor.spectrumKeys[keyIndex].sampleHash ||
                    stage->sample.radiusPixels <= 0) {
                    outError =
                        "ResourceDescriptorMismatch component=diffusion field=spectrum_key";
                    return false;
                }
            }
            return true;
        }

        bool make_plan_pair(
            DiffusionWorkspaceSlot& slot,
            const Spektrafilm::DiffusionExecutionDescriptor& descriptor,
            DeviceByteReservation& aggregate,
            cudaStream_t stream,
            std::string& outError) {
            if (!commit_allowance_from_aggregate(
                    aggregate,
                    descriptor.planAllowanceBytes,
                    slot.planAllowanceReservation,
                    outError)) {
                return false;
            }
            cufftResult result = cufftCreate(&slot.r2cPlan);
            if (result != CUFFT_SUCCESS) {
                outError = "cufftCreate(diffusion R2C) failed code=" +
                           std::to_string(static_cast<int>(result));
                return false;
            }
            result = cufftSetAutoAllocation(slot.r2cPlan, 0);
            if (result != CUFFT_SUCCESS) {
                outError = "cufftSetAutoAllocation(diffusion R2C) failed code=" +
                           std::to_string(static_cast<int>(result));
                return false;
            }
            result = cufftCreate(&slot.c2rPlan);
            if (result != CUFFT_SUCCESS) {
                outError = "cufftCreate(diffusion C2R) failed code=" +
                           std::to_string(static_cast<int>(result));
                return false;
            }
            result = cufftSetAutoAllocation(slot.c2rPlan, 0);
            if (result != CUFFT_SUCCESS) {
                outError = "cufftSetAutoAllocation(diffusion C2R) failed code=" +
                           std::to_string(static_cast<int>(result));
                return false;
            }

            int dimensions[2]{descriptor.layout.height, descriptor.layout.width};
            int r2cInputEmbed[2]{descriptor.layout.height,
                                 descriptor.layout.physicalRealRowFloats};
            int r2cOutputEmbed[2]{descriptor.layout.height,
                                  descriptor.layout.complexWidth};
            int c2rInputEmbed[2]{descriptor.layout.height,
                                 descriptor.layout.complexWidth};
            int c2rOutputEmbed[2]{descriptor.layout.height,
                                  descriptor.layout.physicalRealRowFloats};
            std::size_t actualR2cWork = 0;
            std::size_t actualC2rWork = 0;
            result = cufftMakePlanMany(
                slot.r2cPlan,
                2,
                dimensions,
                r2cInputEmbed,
                1,
                descriptor.layout.realDistance,
                r2cOutputEmbed,
                1,
                descriptor.layout.complexDistance,
                CUFFT_R2C,
                1,
                &actualR2cWork);
            if (result != CUFFT_SUCCESS) {
                outError = "cufftMakePlanMany(diffusion R2C) failed code=" +
                           std::to_string(static_cast<int>(result));
                return false;
            }
            result = cufftMakePlanMany(
                slot.c2rPlan,
                2,
                dimensions,
                c2rInputEmbed,
                1,
                descriptor.layout.complexDistance,
                c2rOutputEmbed,
                1,
                descriptor.layout.realDistance,
                CUFFT_C2R,
                1,
                &actualC2rWork);
            if (result != CUFFT_SUCCESS) {
                outError = "cufftMakePlanMany(diffusion C2R) failed code=" +
                           std::to_string(static_cast<int>(result));
                return false;
            }
            const std::uint64_t actualSharedWorkBytes = std::max<std::uint64_t>(
                static_cast<std::uint64_t>(actualR2cWork),
                static_cast<std::uint64_t>(actualC2rWork));
            if (actualSharedWorkBytes > descriptor.reservedSharedWorkBytes) {
                outError =
                    "InvalidDiffusionExecutionDescriptor field=actual_shared_work_bytes";
                return false;
            }

            if (actualSharedWorkBytes > 0) {
                void* work = nullptr;
                if (!allocate_from_aggregate(
                        aggregate,
                        actualSharedWorkBytes,
                        work,
                        slot.workAreaReservation,
                        outError)) {
                    return false;
                }
                slot.workArea = work;
            }
            result = cufftSetWorkArea(slot.r2cPlan, slot.workArea);
            if (result != CUFFT_SUCCESS) {
                outError = "cufftSetWorkArea(diffusion R2C) failed code=" +
                           std::to_string(static_cast<int>(result));
                return false;
            }
            result = cufftSetWorkArea(slot.c2rPlan, slot.workArea);
            if (result != CUFFT_SUCCESS) {
                outError = "cufftSetWorkArea(diffusion C2R) failed code=" +
                           std::to_string(static_cast<int>(result));
                return false;
            }
            result = cufftSetStream(slot.r2cPlan, stream);
            if (result != CUFFT_SUCCESS) {
                outError = "cufftSetStream(diffusion R2C) failed code=" +
                           std::to_string(static_cast<int>(result));
                return false;
            }
            result = cufftSetStream(slot.c2rPlan, stream);
            if (result != CUFFT_SUCCESS) {
                outError = "cufftSetStream(diffusion C2R) failed code=" +
                           std::to_string(static_cast<int>(result));
                return false;
            }
            slot.planKey = descriptor.planKey;
            return true;
        }

        bool build_workspace(
            DiffusionWorkspaceSlot& slot,
            const Spektrafilm::DiffusionFrameSetDescriptor& frameSet,
            const Spektrafilm::DiffusionExecutionDescriptor& descriptor,
            DeviceByteReservation& aggregate,
            cudaStream_t stream,
            std::string& outError) {
            if (!make_plan_pair(
                    slot,
                    descriptor,
                    aggregate,
                    stream,
                    outError)) {
                return false;
            }
            void* transform = nullptr;
            if (!allocate_from_aggregate(
                    aggregate,
                    descriptor.layout.transformBytes,
                    transform,
                    slot.transformReservation,
                    outError)) {
                return false;
            }
            slot.transformBuffer = static_cast<float*>(transform);
            slot.transformCapacityBytes = descriptor.layout.transformBytes;

            if (descriptor.stagePlaneBytes % 4u != 0) {
                outError =
                    "ResourceDescriptorMismatch component=diffusion field=stage_plane_bytes";
                return false;
            }
            const std::uint64_t onePlaneBytes = descriptor.stagePlaneBytes / 4u;
            for (std::size_t index = 0; index < 4; ++index) {
                void* plane = nullptr;
                if (!allocate_from_aggregate(
                        aggregate,
                        onePlaneBytes,
                        plane,
                        slot.stagePlaneReservations[index],
                        outError)) {
                    return false;
                }
                assign_stage_plane_pointer(slot, index, plane);
            }
            slot.stagePlaneCapacityElements =
                onePlaneBytes / static_cast<std::uint64_t>(sizeof(float));
            slot.stagePlanes.rowStrideFloats =
                static_cast<std::size_t>(frameSet.fullFrame.width);

            if (!slot.completionEventOpaque) {
                cudaEvent_t event = nullptr;
                const cudaError_t eventResult = cudaEventCreateWithFlags(
                    &event,
                    cudaEventDisableTiming);
                if (eventResult != cudaSuccess || !event) {
                    outError =
                        std::string("cudaEventCreate(diffusion workspace) failed: ") +
                        cuda_message(eventResult);
                    return false;
                }
                slot.completionEventOpaque = reinterpret_cast<void*>(event);
            }
            return true;
        }

        bool build_spectrum(
            DiffusionSpectrumEntry& entry,
            DiffusionWorkspaceSlot& workspace,
            const Spektrafilm::DiffusionExecutionDescriptor& descriptor,
            const Spektrafilm::DiffusionPsfComponents& components,
            int radiusPixels,
            DeviceByteReservation& aggregate,
            cudaStream_t stream,
            std::string& outError) {
            for (std::size_t index = 0; index < 3; ++index) {
                void* spectrum = nullptr;
                if (!allocate_from_aggregate(
                        aggregate,
                        descriptor.layout.transformBytes,
                        spectrum,
                        entry.spectrumReservations[index],
                        outError)) {
                    return false;
                }
                assign_spectrum_pointer(entry, index, spectrum);
            }

            cudaEvent_t buildEvent = nullptr;
            const cudaError_t eventResult = cudaEventCreateWithFlags(
                &buildEvent,
                cudaEventDisableTiming);
            if (eventResult != cudaSuccess || !buildEvent) {
                outError =
                    std::string("cudaEventCreate(diffusion spectrum) failed: ") +
                    cuda_message(eventResult);
                return false;
            }
            entry.buildEventOpaque = reinterpret_cast<void*>(buildEvent);

            SpectrumBuildRequest request{};
            request.layout = descriptor.layout;
            request.components = components;
            request.radiusPixels = radiusPixels;
            request.transformBuffer = workspace.transformBuffer;
            request.normalizationScratch =
                reinterpret_cast<double*>(entry.spectra.red);
            request.normalizationScratchBytes =
                static_cast<std::size_t>(descriptor.layout.transformBytes);
            request.r2cPlan = workspace.r2cPlan;
            request.destination = entry.spectra;
            request.stream = stream;
            const LaunchResult result = build_spectrum_package(request);
            if (!result.ok()) {
                outError = std::string("diffusion spectrum build failed stage=") +
                           (result.stage ? result.stage : "unknown") +
                           " code=" + std::to_string(result.code);
                return false;
            }
            const cudaError_t recordResult = cudaEventRecord(buildEvent, stream);
            if (recordResult != cudaSuccess) {
                outError =
                    std::string("cudaEventRecord(diffusion spectrum) failed: ") +
                    cuda_message(recordResult);
                return false;
            }
            return true;
        }

    } // namespace

    PreparedDiffusionLease::~PreparedDiffusionLease() noexcept {
        if (_active && _owner) {
            std::string ignored;
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
                std::string ignored;
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
            workspace.transformBuffer,
            workspace.r2cPlan,
            workspace.c2rPlan};
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
        std::string& outError) {
        outError.clear();
        if (outLease.active() || contextKey.deviceId < 0 ||
            !contextKey.contextOpaque || contextEpoch == 0 || !ledger ||
            !cudaStreamOpaque) {
            outError =
                "ResourceDescriptorMismatch component=diffusion field=prepare_identity";
            return false;
        }
        if (!current_owner_matches(contextKey, outError) ||
            !validate_prepare_request(
                frameSet,
                descriptor,
                contextEpoch,
                outError)) {
            return false;
        }
        std::array<Spektrafilm::DiffusionPsfComponents, 2> components{};
        std::array<int, 2> radii{};
        for (std::size_t keyIndex = 0;
             keyIndex < descriptor.uniqueSpectrumCount;
             ++keyIndex) {
            const auto* stage = stage_for_key(frameSet, descriptor, keyIndex);
            if (!stage || !Spektrafilm::expand_diffusion_psf_components(
                              stage->sample,
                              components[keyIndex],
                              outError)) {
                if (outError.empty()) {
                    outError =
                        "ResourceDescriptorMismatch component=diffusion field=psf_components";
                }
                return false;
            }
            radii[keyIndex] = stage->sample.radiusPixels;
        }

        if (!acquire_host_lease(resources, outError)) {
            return false;
        }
        const auto failClaimedPreparation = [&resources]() {
            release_host_lease(resources);
            return false;
        };

        const cudaStream_t stream =
            reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        std::array<std::size_t, 2> spectrumIndices{};
        std::array<bool, 2> missingSpectrum{};
        std::array<bool, 2> dependentBuilding{};
        std::size_t workspaceIndex = kInvalidIndex;
        bool rebuildWorkspace = false;
        DeviceByteReservation aggregate;

        for (;;) {
            reap_completed(resources);
            for (std::size_t index = 0;
                 index < resources.workspaces.size();
                 ++index) {
                if (resources.workspaces[index].state ==
                    WorkspaceSlotState::Vacant) {
                    workspaceIndex = index;
                    break;
                }
            }
            if (workspaceIndex != kInvalidIndex) {
                break;
            }

            const std::size_t retiringIndex =
                choose_retiring_workspace(resources);
            if (retiringIndex == kInvalidIndex) {
                outError =
                    "ExactAdmissionFailure component=diffusion class=execution_slot_unrecoverable requested_new_bytes=0 slots=2";
                return failClaimedPreparation();
            }

            DiffusionWorkspaceSlot& retiring =
                resources.workspaces[retiringIndex];
            const cudaError_t waitResult = cudaEventSynchronize(
                reinterpret_cast<cudaEvent_t>(
                    retiring.completionEventOpaque));
            if (waitResult != cudaSuccess) {
                retiring.state = WorkspaceSlotState::FailedQuarantined;
                outError =
                    std::string("cudaEventSynchronize(diffusion admission) failed: ") +
                    cuda_message(waitResult);
                return failClaimedPreparation();
            }
            complete_workspace_retirement(resources, retiringIndex);
            reap_retired_spectra(resources);
        }

        DiffusionWorkspaceSlot& workspace =
            resources.workspaces[workspaceIndex];
        workspace.releaseSequence = 0;
        const std::uint64_t requiredPlaneElements =
            descriptor.stagePlaneBytes /
            (4u * static_cast<std::uint64_t>(sizeof(float)));
        rebuildWorkspace =
            !same_plan_key(workspace.planKey, descriptor.planKey) ||
            workspace.r2cPlan == 0 || workspace.c2rPlan == 0 ||
            !workspace.transformBuffer ||
            workspace.transformCapacityBytes <
                descriptor.layout.transformBytes ||
            workspace.stagePlaneCapacityElements < requiredPlaneElements;

        for (std::size_t keyIndex = 0;
             keyIndex < descriptor.uniqueSpectrumCount;
             ++keyIndex) {
            std::size_t found = kInvalidIndex;
            for (std::size_t entryIndex = 0;
                 entryIndex < resources.spectra.size();
                 ++entryIndex) {
                const DiffusionSpectrumEntry& entry =
                    resources.spectra[entryIndex];
                if ((entry.state == SpectrumEntryState::Ready ||
                     entry.state == SpectrumEntryState::Building) &&
                    same_spectrum_key(
                        entry.key,
                        descriptor.spectrumKeys[keyIndex])) {
                    if (entry.leased) {
                        outError =
                            "ResourceDescriptorMismatch component=diffusion field=serialized_spectrum_lease";
                        for (std::size_t rollback = 0;
                             rollback < keyIndex;
                             ++rollback) {
                            DiffusionSpectrumEntry& prior =
                                resources.spectra[spectrumIndices[rollback]];
                            prior.leased = false;
                            if (missingSpectrum[rollback]) {
                                prior = DiffusionSpectrumEntry{};
                            }
                        }
                        return failClaimedPreparation();
                    }
                    found = entryIndex;
                    dependentBuilding[keyIndex] =
                        entry.state == SpectrumEntryState::Building;
                    break;
                }
            }
            if (found == kInvalidIndex) {
                found = choose_missing_spectrum_slot(resources);
                if (found == kInvalidIndex) {
                    outError =
                        "ExactAdmissionFailure component=diffusion class=spectrum requested_new_bytes=0 slots=4";
                    for (std::size_t rollback = 0;
                         rollback < keyIndex;
                         ++rollback) {
                        DiffusionSpectrumEntry& entry =
                            resources.spectra[spectrumIndices[rollback]];
                        entry.leased = false;
                        if (missingSpectrum[rollback]) {
                            entry = DiffusionSpectrumEntry{};
                        }
                    }
                    return failClaimedPreparation();
                }
                DiffusionSpectrumEntry& entry = resources.spectra[found];
                entry.key = descriptor.spectrumKeys[keyIndex];
                entry.leased = true;
                entry.state = SpectrumEntryState::Building;
                missingSpectrum[keyIndex] = true;
            } else {
                resources.spectra[found].leased = true;
            }
            spectrumIndices[keyIndex] = found;
        }

        const auto rollbackSpectrumLeases = [&] {
            for (std::size_t keyIndex = 0;
                 keyIndex < descriptor.uniqueSpectrumCount;
                 ++keyIndex) {
                DiffusionSpectrumEntry& entry =
                    resources.spectra[spectrumIndices[keyIndex]];
                entry.leased = false;
                if (missingSpectrum[keyIndex]) {
                    entry = DiffusionSpectrumEntry{};
                }
            }
        };

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
                outError =
                    "ExactAdmissionFailure component=diffusion field=spectrum_upper_bound";
                rollbackSpectrumLeases();
                return failClaimedPreparation();
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
            outError =
                "ExactAdmissionFailure component=diffusion field=workspace_upper_bound";
            rollbackSpectrumLeases();
            return failClaimedPreparation();
        }
        if (prospectiveBytes > 0) {
            std::string ledgerError;
            if (!ledger->reserve(
                    DeviceReservationRequest{
                        .contextKey = contextKey,
                        .contextEpoch = contextEpoch,
                        .bytes = prospectiveBytes},
                    aggregate,
                    ledgerError)) {
                outError =
                    "ExactAdmissionFailure component=diffusion class=aggregate_reservation requested_new_bytes=" +
                    std::to_string(prospectiveBytes) + " detail=" + ledgerError;
                rollbackSpectrumLeases();
                return failClaimedPreparation();
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

        const auto failPreparation = [&] {
            const bool completionCertain =
                !lease._workEnqueued ||
                cudaStreamSynchronize(stream) == cudaSuccess;
            DiffusionWorkspaceSlot& failedWorkspace =
                resources.workspaces[workspaceIndex];
            failedWorkspace.completionUnknown = !completionCertain;
            failedWorkspace.state = WorkspaceSlotState::FailedQuarantined;
            for (std::size_t keyIndex = 0;
                 keyIndex < descriptor.uniqueSpectrumCount;
                 ++keyIndex) {
                DiffusionSpectrumEntry& entry =
                    resources.spectra[spectrumIndices[keyIndex]];
                entry.leased = false;
                if (missingSpectrum[keyIndex] || lease._workEnqueued) {
                    entry.state =
                        SpectrumEntryState::FailedQuarantined;
                }
            }
            if (completionCertain) {
                std::string ignored;
                if (destroy_workspace_contents(failedWorkspace, ignored)) {
                    failedWorkspace.state = WorkspaceSlotState::Vacant;
                }
                for (std::size_t keyIndex = 0;
                     keyIndex < descriptor.uniqueSpectrumCount;
                     ++keyIndex) {
                    DiffusionSpectrumEntry& entry =
                        resources.spectra[spectrumIndices[keyIndex]];
                    if (entry.state ==
                            SpectrumEntryState::FailedQuarantined &&
                        !entry.leased &&
                        entry.pendingWorkspaceMask == 0) {
                        (void)destroy_spectrum_contents(entry, ignored);
                    }
                }
            }
            lease.reset();
            release_host_lease(resources);
            return false;
        };

        for (std::size_t keyIndex = 0;
             keyIndex < descriptor.uniqueSpectrumCount;
             ++keyIndex) {
            if (!dependentBuilding[keyIndex]) {
                continue;
            }
            const DiffusionSpectrumEntry& entry =
                resources.spectra[spectrumIndices[keyIndex]];
            if (entry.state != SpectrumEntryState::Building ||
                !entry.buildEventOpaque) {
                outError =
                    "MissingRequiredResource component=diffusion field=same_key_build_event";
                return failPreparation();
            }
            const cudaError_t waitResult = cudaStreamWaitEvent(
                stream,
                reinterpret_cast<cudaEvent_t>(entry.buildEventOpaque),
                0);
            if (waitResult != cudaSuccess) {
                outError =
                    std::string("cudaStreamWaitEvent(diffusion same key) failed: ") +
                    cuda_message(waitResult);
                return failPreparation();
            }
            lease._workEnqueued = true;
        }

        if (rebuildWorkspace) {
            std::string cleanupError;
            if (!destroy_workspace_contents(workspace, cleanupError) ||
                !build_workspace(
                    workspace,
                    frameSet,
                    descriptor,
                    aggregate,
                    stream,
                    outError)) {
                if (outError.empty()) {
                    outError = cleanupError;
                }
                return failPreparation();
            }
        } else {
            const cufftResult r2cResult =
                cufftSetStream(workspace.r2cPlan, stream);
            const cufftResult c2rResult =
                cufftSetStream(workspace.c2rPlan, stream);
            if (r2cResult != CUFFT_SUCCESS || c2rResult != CUFFT_SUCCESS) {
                outError = "cufftSetStream(diffusion reuse) failed";
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
        std::string& outError) noexcept {
        const bool ownsHostLease =
            lease._active && lease._owner == &resources;
        try {
            outError.clear();
            if (!ownsHostLease) {
                return true;
            }
            const cudaStream_t stream = cudaStreamOpaque
                                            ? reinterpret_cast<cudaStream_t>(
                                                  cudaStreamOpaque)
                                            : reinterpret_cast<cudaStream_t>(
                                                  lease._streamOpaque);
            lease._streamOpaque = reinterpret_cast<void*>(stream);

            bool completionRecorded = false;
            if (lease._workEnqueued) {
                DiffusionWorkspaceSlot& workspace =
                    resources.workspaces[lease._workspaceIndex];
                if (!workspace.completionEventOpaque) {
                    cudaEvent_t event = nullptr;
                    const cudaError_t createResult = cudaEventCreateWithFlags(
                        &event,
                        cudaEventDisableTiming);
                    if (createResult == cudaSuccess && event) {
                        workspace.completionEventOpaque =
                            reinterpret_cast<void*>(event);
                    }
                }
                if (workspace.completionEventOpaque) {
                    const cudaError_t recordResult = cudaEventRecord(
                        reinterpret_cast<cudaEvent_t>(
                            workspace.completionEventOpaque),
                        stream);
                    completionRecorded = recordResult == cudaSuccess;
                    if (!completionRecorded) {
                        outError =
                            std::string("cudaEventRecord(diffusion release) failed: ") +
                            cuda_message(recordResult);
                    }
                } else {
                    outError =
                        "cudaEventCreate(diffusion release) failed";
                }
            }

            bool completionCertain = !lease._workEnqueued || completionRecorded;
            if (lease._workEnqueued && !completionRecorded) {
                const cudaError_t syncResult = cudaStreamSynchronize(stream);
                completionCertain = syncResult == cudaSuccess;
                if (!completionCertain && outError.empty()) {
                    outError =
                        std::string("cudaStreamSynchronize(diffusion release) failed: ") +
                        cuda_message(syncResult);
                }
            }

            bool cleanupWorkspace = false;
            std::array<bool, 2> cleanupSpectrum{};
            DiffusionWorkspaceSlot& workspace =
                resources.workspaces[lease._workspaceIndex];
            for (std::size_t index = 0; index < lease._spectrumCount; ++index) {
                const std::size_t spectrumIndex =
                    lease._spectrumIndices[index];
                DiffusionSpectrumEntry& entry =
                    resources.spectra[spectrumIndex];
                entry.leased = false;
                if (dependentFailure) {
                    entry.state = SpectrumEntryState::FailedQuarantined;
                }
                if (completionRecorded) {
                    entry.pendingWorkspaceMask = static_cast<std::uint8_t>(
                        entry.pendingWorkspaceMask |
                        (1u << lease._workspaceIndex));
                }
                if (!retained_spectrum_slot(spectrumIndex)) {
                    entry.state = SpectrumEntryState::Retiring;
                }
                if (completionCertain && !completionRecorded &&
                    (entry.state == SpectrumEntryState::Retiring ||
                     entry.state ==
                         SpectrumEntryState::FailedQuarantined)) {
                    cleanupSpectrum[index] = true;
                }
                entry.releaseSequence = resources.nextReleaseSequence;
                if (resources.nextReleaseSequence <
                    std::numeric_limits<std::uint64_t>::max()) {
                    ++resources.nextReleaseSequence;
                }
            }

            if (completionRecorded) {
                workspace.releaseSequence = resources.nextReleaseSequence;
                if (resources.nextReleaseSequence <
                    std::numeric_limits<std::uint64_t>::max()) {
                    ++resources.nextReleaseSequence;
                }
                workspace.state =
                    dependentFailure
                        ? WorkspaceSlotState::FailedQuarantined
                        : WorkspaceSlotState::Retiring;
            } else if (completionCertain) {
                workspace.state =
                    retained_workspace_slot(lease._workspaceIndex) &&
                            !dependentFailure
                        ? WorkspaceSlotState::Vacant
                        : WorkspaceSlotState::Retiring;
                cleanupWorkspace =
                    !retained_workspace_slot(lease._workspaceIndex) ||
                    dependentFailure;
            } else {
                workspace.completionUnknown = true;
                workspace.state = WorkspaceSlotState::FailedQuarantined;
                for (std::size_t index = 0;
                     index < lease._spectrumCount;
                     ++index) {
                    DiffusionSpectrumEntry& entry =
                        resources.spectra[lease._spectrumIndices[index]];
                    entry.state = SpectrumEntryState::FailedQuarantined;
                }
            }

            if (cleanupWorkspace || cleanupSpectrum[0] || cleanupSpectrum[1]) {
                std::string cleanupError;
                if (cleanupWorkspace) {
                    if (destroy_workspace_contents(workspace, cleanupError)) {
                        workspace.state = WorkspaceSlotState::Vacant;
                    } else {
                        workspace.state =
                            WorkspaceSlotState::FailedQuarantined;
                    }
                }
                for (std::size_t index = 0; index < lease._spectrumCount; ++index) {
                    if (!cleanupSpectrum[index]) {
                        continue;
                    }
                    DiffusionSpectrumEntry& entry =
                        resources.spectra[lease._spectrumIndices[index]];
                    if (!destroy_spectrum_contents(entry, cleanupError)) {
                        entry.state = SpectrumEntryState::FailedQuarantined;
                    }
                }
                if (!cleanupError.empty() && outError.empty()) {
                    outError = cleanupError;
                }
            }

            const bool safeRelease = completionCertain;
            lease.reset();
            release_host_lease(resources);
            return safeRelease;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            if (ownsHostLease) {
                lease.reset();
                release_host_lease(resources);
            }
            try {
                outError = "diffusion resource release bookkeeping failed";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
    }

    bool trim_inactive_diffusion_resources(
        DiffusionContextResources& resources,
        const ResourceManager::DeviceContextKey& contextKey,
        std::string& outError) noexcept {
        bool hostLeaseClaimed = false;
        try {
            outError.clear();
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
            reap_completed(resources);

            bool trimmed = true;
            for (DiffusionWorkspaceSlot& workspace : resources.workspaces) {
                if (workspace.state != WorkspaceSlotState::Vacant) {
                    continue;
                }
                std::string localError;
                if (!destroy_workspace_contents(workspace, localError)) {
                    workspace.state =
                        WorkspaceSlotState::FailedQuarantined;
                    trimmed = false;
                    if (outError.empty()) {
                        outError = localError;
                    }
                }
            }
            for (DiffusionSpectrumEntry& entry : resources.spectra) {
                if (entry.state != SpectrumEntryState::Ready ||
                    entry.leased || entry.pendingWorkspaceMask != 0) {
                    continue;
                }
                std::string localError;
                if (!destroy_spectrum_contents(entry, localError)) {
                    entry.state = SpectrumEntryState::FailedQuarantined;
                    trimmed = false;
                    if (outError.empty()) {
                        outError = localError;
                    }
                }
            }
            release_host_lease(resources);
            return trimmed;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            if (hostLeaseClaimed) {
                release_host_lease(resources);
            }
            try {
                outError = "inactive diffusion residency trim failed";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
    }

    bool drain_diffusion_resources(
        DiffusionContextResources& resources,
        std::string& outError) noexcept {
        bool hostLeaseClaimed = false;
        try {
            outError.clear();
            {
                std::lock_guard<std::mutex> lock(resources.metadataMutex);
                resources.acceptingPreparations = false;
                if (resources.hostLeaseActive) {
                    outError =
                        "ResourceDescriptorMismatch component=diffusion field=drain_with_active_host_lease";
                    return false;
                }
                resources.hostLeaseActive = true;
                hostLeaseClaimed = true;
            }
            bool drained = true;
            for (DiffusionWorkspaceSlot& workspace : resources.workspaces) {
                if (workspace.completionUnknown) {
                    drained = false;
                    if (outError.empty()) {
                        outError =
                            "diffusion workspace completion unknown; proven context loss required";
                    }
                    continue;
                }
                if (workspace.completionEventOpaque) {
                    const cudaError_t syncResult = cudaEventSynchronize(
                        reinterpret_cast<cudaEvent_t>(
                            workspace.completionEventOpaque));
                    if (syncResult != cudaSuccess) {
                        drained = false;
                        if (outError.empty()) {
                            outError =
                                std::string("cudaEventSynchronize(diffusion workspace) failed: ") +
                                cuda_message(syncResult);
                        }
                        continue;
                    }
                }
                std::string localError;
                if (!destroy_workspace_contents(workspace, localError)) {
                    drained = false;
                    workspace.state = WorkspaceSlotState::FailedQuarantined;
                    if (outError.empty()) {
                        outError = localError;
                    }
                    continue;
                }
                if (workspace.completionEventOpaque) {
                    const cudaError_t destroyResult = cudaEventDestroy(
                        reinterpret_cast<cudaEvent_t>(
                            workspace.completionEventOpaque));
                    if (destroyResult != cudaSuccess) {
                        drained = false;
                        if (outError.empty()) {
                            outError =
                                std::string("cudaEventDestroy(diffusion workspace) failed: ") +
                                cuda_message(destroyResult);
                        }
                        continue;
                    }
                    workspace.completionEventOpaque = nullptr;
                }
                workspace = DiffusionWorkspaceSlot{};
            }

            for (DiffusionSpectrumEntry& entry : resources.spectra) {
                if (entry.buildEventOpaque) {
                    const cudaError_t syncResult = cudaEventSynchronize(
                        reinterpret_cast<cudaEvent_t>(entry.buildEventOpaque));
                    if (syncResult != cudaSuccess) {
                        drained = false;
                        if (outError.empty()) {
                            outError =
                                std::string("cudaEventSynchronize(diffusion spectrum) failed: ") +
                                cuda_message(syncResult);
                        }
                        continue;
                    }
                }
                std::string localError;
                if (!destroy_spectrum_contents(entry, localError)) {
                    drained = false;
                    entry.state = SpectrumEntryState::FailedQuarantined;
                    if (outError.empty()) {
                        outError = localError;
                    }
                }
            }
            release_host_lease(resources);
            return drained;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            if (hostLeaseClaimed) {
                release_host_lease(resources);
            }
            try {
                outError = "diffusion resource drain bookkeeping failed";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
    }

    void invalidate_diffusion_resources_after_proven_context_loss(
        DiffusionContextResources& resources) noexcept {
        try {
            std::lock_guard<std::mutex> lock(resources.metadataMutex);
            resources.acceptingPreparations = false;
            resources.hostLeaseActive = false;
            for (DiffusionSpectrumEntry& entry : resources.spectra) {
                entry = DiffusionSpectrumEntry{};
            }
            for (DiffusionWorkspaceSlot& workspace : resources.workspaces) {
                workspace = DiffusionWorkspaceSlot{};
            }
            resources.nextReleaseSequence = 1;
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

} // namespace JuicerCuda::Diffusion
