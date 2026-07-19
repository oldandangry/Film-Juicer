#if defined(JUICER_ENABLE_CUDA) && !defined(__APPLE__)

#include "Cuda/Diffusion/JuicerCudaDiffusionResources.h"

#include <algorithm>
#include <array>
#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
#include <atomic>
#endif
#include <limits>
#include <utility>

#include <cuda.h>
#include <cudaTypedefs.h>

#include "Logging.h"

namespace JuicerCuda::Diffusion {
    namespace {

        constexpr std::size_t kInvalidIndex =
            std::numeric_limits<std::size_t>::max();

#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
        std::atomic<DiffusionLifecycleFaultPoint> gLifecycleFault{
            DiffusionLifecycleFaultPoint::None};

        bool consume_lifecycle_fault(
            DiffusionLifecycleFaultPoint point) noexcept {
            DiffusionLifecycleFaultPoint expected = point;
            return gLifecycleFault.compare_exchange_strong(
                expected,
                DiffusionLifecycleFaultPoint::None,
                std::memory_order_relaxed);
        }
#endif

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

        DeviceAllocationIdentity make_identity(
            const ResourceManager::DeviceContextKey& contextKey,
            std::uint64_t contextEpoch,
            DeviceAllocationClass allocationClass,
            const char* label) {
            DeviceAllocationIdentity identity{};
            identity.deviceId = contextKey.deviceId;
            identity.contextKey = contextKey;
            identity.contextEpoch = contextEpoch;
            identity.allocationClass = allocationClass;
            identity.diagnosticIdentity = label ? label : "diffusion allocation";
            return identity;
        }

        struct DiffusionAllocationOwner {
            ResourceManager::DeviceContextKey contextKey{};
            std::uint64_t contextEpoch = 0;
        };

        bool allocate_from_aggregate(
            DeviceByteReservation& aggregate,
            const DeviceAllocationIdentity& identity,
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
            if (!aggregate.split(identity, bytes, reservation, outError)) {
                return false;
            }
#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
            if (consume_lifecycle_fault(
                    DiffusionLifecycleFaultPoint::DeviceAllocation)) {
                outError =
                    "cudaMalloc(diffusion) injected failure code=2";
                return false;
            }
#endif
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
            const DeviceAllocationIdentity& identity,
            std::uint64_t bytes,
            DeviceByteReservation& outReservation,
            std::string& outError) {
            if (bytes == 0) {
                outError =
                    "UnsupportedCudaExecutionProfile component=diffusion field=plan_allowance";
                return false;
            }
            DeviceByteReservation reservation;
            if (!aggregate.split(identity, bytes, reservation, outError) ||
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
                slot.workAreaCapacityBytes = 0;
                slot.transformCapacityBytes = 0;
                slot.stagePlaneCapacityElements = 0;
                slot.stagePlanes = {};
                slot.failureApi = FailureApi::None;
                slot.failureCode = 0;
                slot.failureStage = nullptr;
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
                const bool retainedRole = entry.retainedRole;
                entry = DiffusionSpectrumEntry{};
                entry.retainedRole = retainedRole;
            }
            return released;
        }

        bool event_complete(void* eventOpaque, cudaError_t& outResult) noexcept {
            if (!eventOpaque) {
                outResult = cudaSuccess;
                return true;
            }
#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
            if (consume_lifecycle_fault(
                    DiffusionLifecycleFaultPoint::AsyncCompletionQuery)) {
                outResult = cudaErrorUnknown;
                return false;
            }
#endif
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

        void reap_completed_locked(
            DiffusionContextResources& resources) {
            for (DiffusionSpectrumEntry& entry : resources.spectra) {
                if (entry.state != SpectrumEntryState::Building ||
                    !entry.buildEventPublished || !entry.buildEventOpaque) {
                    continue;
                }
                cudaError_t result = cudaSuccess;
                if (event_complete(entry.buildEventOpaque, result)) {
                    (void)cudaEventDestroy(
                        reinterpret_cast<cudaEvent_t>(entry.buildEventOpaque));
                    entry.buildEventOpaque = nullptr;
                    entry.buildEventPublished = false;
                    entry.expandedComponents = {};
                    entry.state = SpectrumEntryState::Ready;
                } else if (result != cudaErrorNotReady) {
                    entry.failureApi = FailureApi::Cuda;
                    entry.failureCode = static_cast<int>(result);
                    entry.failureStage = "cudaEventQuery_spectrum_build";
                    entry.state = SpectrumEntryState::FailedQuarantined;
                }
            }

            for (std::size_t index = 0; index < resources.workspaces.size();
                 ++index) {
                DiffusionWorkspaceSlot& slot = resources.workspaces[index];
                if (slot.state != WorkspaceSlotState::Retiring ||
                    !slot.completionEventOpaque) {
                    continue;
                }
                cudaError_t result = cudaSuccess;
                if (event_complete(slot.completionEventOpaque, result)) {
                    clear_workspace_use_bit(resources, index);
                    if (slot.failureApi != FailureApi::None) {
                        std::string ignored;
                        if (destroy_workspace_contents(slot, ignored)) {
                            slot.state = WorkspaceSlotState::Vacant;
                        } else {
                            slot.state =
                                WorkspaceSlotState::FailedQuarantined;
                        }
                    } else if (slot.retainedRole) {
                        slot.state = WorkspaceSlotState::Vacant;
                    } else {
                        std::string ignored;
                        if (destroy_workspace_contents(slot, ignored)) {
                            slot.state = WorkspaceSlotState::Vacant;
                        } else {
                            slot.state =
                                WorkspaceSlotState::FailedQuarantined;
                        }
                    }
                } else if (result != cudaErrorNotReady) {
                    slot.failureApi = FailureApi::Cuda;
                    slot.failureCode = static_cast<int>(result);
                    slot.failureStage = "cudaEventQuery_workspace_completion";
                    slot.state = WorkspaceSlotState::FailedQuarantined;
                }
            }

            for (DiffusionSpectrumEntry& entry : resources.spectra) {
                if ((entry.state != SpectrumEntryState::Retiring &&
                     entry.state != SpectrumEntryState::FailedQuarantined) ||
                    entry.leaseCount != 0 || entry.pendingWorkspaceMask != 0) {
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

        bool same_spectrum_key(
            const Spektrafilm::DiffusionSpectrumKey& left,
            const Spektrafilm::DiffusionSpectrumKey& right) noexcept {
            return left.hash != 0 && left.hash == right.hash &&
                   left.sampleHash == right.sampleHash &&
                   left.extent == right.extent &&
                   left.layoutSchema == right.layoutSchema;
        }

        bool same_plan_key(
            const Spektrafilm::DiffusionPlanKey& left,
            const Spektrafilm::DiffusionPlanKey& right) noexcept {
            return left.hash != 0 && left.hash == right.hash &&
                   left.profileDigest == right.profileDigest &&
                   left.extent == right.extent &&
                   left.layoutSchema == right.layoutSchema;
        }

        std::size_t choose_missing_spectrum_slot_locked(
            DiffusionContextResources& resources) {
            for (std::size_t index = 0; index < 2; ++index) {
                if (resources.spectra[index].state ==
                    SpectrumEntryState::Vacant) {
                    return index;
                }
            }

            std::size_t eviction = kInvalidIndex;
            for (std::size_t index = 0; index < 2; ++index) {
                const DiffusionSpectrumEntry& entry = resources.spectra[index];
                if (entry.state != SpectrumEntryState::Ready ||
                    entry.leaseCount != 0 || entry.pendingWorkspaceMask != 0) {
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

            for (std::size_t index = 2; index < resources.spectra.size();
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
            const auto& profile =
                Spektrafilm::supported_diffusion_execution_profile();
            if (descriptor.hash == 0 || descriptor.frameSetHash != frameSet.hash ||
                descriptor.contextEpoch != contextEpoch ||
                descriptor.stageCount == 0 || descriptor.stageCount > 2 ||
                descriptor.uniqueSpectrumCount == 0 ||
                descriptor.uniqueSpectrumCount > 2 ||
                descriptor.profileKeyDigest != profile.profileKeyDigest ||
                descriptor.profileDigest != profile.profileDigest ||
                descriptor.planKey.hash == 0 ||
                descriptor.transformBufferBytes !=
                    descriptor.layout.transformBytes ||
                descriptor.stagePlaneBytes != frameSet.workspaceBytes) {
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
                    stage->radiusPixels <= 0) {
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
            const DiffusionAllocationOwner& owner,
            cudaStream_t stream,
            std::string& outError) {
            if (!commit_allowance_from_aggregate(
                    aggregate,
                    make_identity(
                        owner.contextKey,
                        owner.contextEpoch,
                        DeviceAllocationClass::CufftPlanAllowance,
                        "diffusion cuFFT plan pair allowance"),
                    descriptor.acceptedPlanAllowanceBytes,
                    slot.planAllowanceReservation,
                    outError)) {
                return false;
            }

#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
            if (consume_lifecycle_fault(
                    DiffusionLifecycleFaultPoint::PlanCreate)) {
                outError =
                    "cufftCreate(diffusion R2C) injected failure code=2";
                return false;
            }
#endif
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
#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
            if (consume_lifecycle_fault(
                    DiffusionLifecycleFaultPoint::PlanQuery)) {
                outError =
                    "cufftMakePlanMany(diffusion R2C) injected failure code=5";
                return false;
            }
#endif
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
            if (!Spektrafilm::diffusion_plan_work_fits(
                    descriptor,
                    static_cast<std::uint64_t>(actualR2cWork),
                    static_cast<std::uint64_t>(actualC2rWork),
                    outError)) {
                return false;
            }

            const std::uint64_t sharedWork = std::max<std::uint64_t>(
                static_cast<std::uint64_t>(actualR2cWork),
                static_cast<std::uint64_t>(actualC2rWork));
            if (sharedWork > 0) {
                void* work = nullptr;
                if (!allocate_from_aggregate(
                        aggregate,
                        make_identity(
                            owner.contextKey,
                            owner.contextEpoch,
                            DeviceAllocationClass::DiffusionTransformWorkArea,
                            "diffusion shared cuFFT work area"),
                        sharedWork,
                        work,
                        slot.workAreaReservation,
                        outError)) {
                    return false;
                }
                slot.workArea = work;
                slot.workAreaCapacityBytes = sharedWork;
            }
#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
            if (consume_lifecycle_fault(
                    DiffusionLifecycleFaultPoint::PlanBind)) {
                outError =
                    "cufftSetWorkArea(diffusion R2C) injected failure code=1";
                return false;
            }
#endif
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
            const Spektrafilm::DiffusionExecutionDescriptor& descriptor,
            DeviceByteReservation& aggregate,
            const DiffusionAllocationOwner& owner,
            cudaStream_t stream,
            std::string& outError) {
            if (!make_plan_pair(
                    slot,
                    descriptor,
                    aggregate,
                    owner,
                    stream,
                    outError)) {
                return false;
            }
            void* transform = nullptr;
            if (!allocate_from_aggregate(
                    aggregate,
                    make_identity(
                        owner.contextKey,
                        owner.contextEpoch,
                        DeviceAllocationClass::DiffusionTransformWorkArea,
                        "diffusion transform buffer"),
                    descriptor.transformBufferBytes,
                    transform,
                    slot.transformReservation,
                    outError)) {
                return false;
            }
            slot.transformBuffer = static_cast<float*>(transform);
            slot.transformCapacityBytes = descriptor.transformBufferBytes;

            if (descriptor.stagePlaneBytes % 4u != 0) {
                outError =
                    "ResourceDescriptorMismatch component=diffusion field=stage_plane_bytes";
                return false;
            }
            const std::uint64_t onePlaneBytes = descriptor.stagePlaneBytes / 4u;
            constexpr const char* kPlaneLabels[]{
                "diffusion red-sensitive stage plane",
                "diffusion green-sensitive stage plane",
                "diffusion blue-sensitive stage plane",
                "diffusion sequential auxiliary stage plane"};
            for (std::size_t index = 0; index < 4; ++index) {
                void* plane = nullptr;
                if (!allocate_from_aggregate(
                        aggregate,
                        make_identity(
                            owner.contextKey,
                            owner.contextEpoch,
                            DeviceAllocationClass::DiffusionStagePlane,
                            kPlaneLabels[index]),
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
                static_cast<std::size_t>(descriptor.fullFrame.width);

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
            DeviceByteReservation& aggregate,
            const DiffusionAllocationOwner& owner,
            cudaStream_t stream,
            std::string& outError) {
            constexpr const char* kSpectrumLabels[]{
                "diffusion red spectrum",
                "diffusion green spectrum",
                "diffusion blue spectrum"};
            for (std::size_t index = 0; index < 3; ++index) {
                void* spectrum = nullptr;
                if (!allocate_from_aggregate(
                        aggregate,
                        make_identity(
                            owner.contextKey,
                            owner.contextEpoch,
                            DeviceAllocationClass::DiffusionSpectrum,
                            kSpectrumLabels[index]),
                        descriptor.transformBufferBytes,
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
            request.components = entry.expandedComponents;
            request.radiusPixels = entry.radiusPixels;
            request.transformBuffer = workspace.transformBuffer;
            request.normalizationScratch =
                reinterpret_cast<double*>(entry.spectra.red);
            request.normalizationScratchBytes =
                static_cast<std::size_t>(descriptor.transformBufferBytes);
            request.r2cPlan = workspace.r2cPlan;
            request.destination = entry.spectra;
            request.stream = stream;
#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
            if (consume_lifecycle_fault(
                    DiffusionLifecycleFaultPoint::SpectrumBuild)) {
                entry.failureApi = FailureApi::Cuda;
                entry.failureCode = static_cast<int>(cudaErrorLaunchFailure);
                entry.failureStage = "build_spectrum_package_injected";
                outError =
                    "diffusion spectrum build injected failure code=719";
                return false;
            }
#endif
            const LaunchResult result = build_spectrum_package(request);
            if (!result.ok()) {
                entry.failureApi = result.api;
                entry.failureCode = result.code;
                entry.failureStage = result.stage;
                outError = std::string("diffusion spectrum build failed stage=") +
                           (result.stage ? result.stage : "unknown") +
                           " code=" + std::to_string(result.code);
                return false;
            }
#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
            if (consume_lifecycle_fault(
                    DiffusionLifecycleFaultPoint::SpectrumBuildEventRecord)) {
                entry.failureApi = FailureApi::Cuda;
                entry.failureCode = static_cast<int>(cudaErrorUnknown);
                entry.failureStage =
                    "cudaEventRecord_spectrum_build_injected";
                outError =
                    "cudaEventRecord(diffusion spectrum) injected failure code=999";
                return false;
            }
#endif
            const cudaError_t recordResult = cudaEventRecord(buildEvent, stream);
            if (recordResult != cudaSuccess) {
                entry.failureApi = FailureApi::Cuda;
                entry.failureCode = static_cast<int>(recordResult);
                entry.failureStage = "cudaEventRecord_spectrum_build";
                outError =
                    std::string("cudaEventRecord(diffusion spectrum) failed: ") +
                    cuda_message(recordResult);
                return false;
            }
            return true;
        }

    } // namespace

#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
    void set_diffusion_lifecycle_fault(
        DiffusionLifecycleFaultPoint point) noexcept {
        gLifecycleFault.store(point, std::memory_order_relaxed);
    }
#endif

    DiffusionContextResources::DiffusionContextResources() noexcept {
        spectra[0].retainedRole = true;
        spectra[1].retainedRole = true;
        workspaces[0].retainedRole = true;
    }

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

    bool PreparedDiffusionLease::work_enqueued() const noexcept {
        return _active && _workEnqueued;
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

    bool query_observed_diffusion_execution_profile(
        Spektrafilm::DiffusionExecutionProfileKey& out,
        std::string& outError) noexcept {
        try {
            out = {};
            outError.clear();
            int runtimeVersion = 0;
            int driverVersion = 0;
            int cufftVersion = 0;
            int deviceId = -1;
            cudaDeviceProp properties{};
            const cudaError_t runtimeResult =
                cudaRuntimeGetVersion(&runtimeVersion);
            if (runtimeResult != cudaSuccess) {
                outError = std::string("cudaRuntimeGetVersion(diffusion) failed: ") +
                           cuda_message(runtimeResult);
                return false;
            }
            const cudaError_t driverResult = cudaDriverGetVersion(&driverVersion);
            if (driverResult != cudaSuccess) {
                outError = std::string("cudaDriverGetVersion(diffusion) failed: ") +
                           cuda_message(driverResult);
                return false;
            }
            const cufftResult cufftResultValue = cufftGetVersion(&cufftVersion);
            if (cufftResultValue != CUFFT_SUCCESS) {
                outError = "cufftGetVersion(diffusion) failed code=" +
                           std::to_string(static_cast<int>(cufftResultValue));
                return false;
            }
            const cudaError_t deviceResult = cudaGetDevice(&deviceId);
            if (deviceResult != cudaSuccess) {
                outError = std::string("cudaGetDevice(diffusion profile) failed: ") +
                           cuda_message(deviceResult);
                return false;
            }
            const cudaError_t propertiesResult =
                cudaGetDeviceProperties(&properties, deviceId);
            if (propertiesResult != cudaSuccess) {
                outError =
                    std::string("cudaGetDeviceProperties(diffusion) failed: ") +
                    cuda_message(propertiesResult);
                return false;
            }
            out = {CUDART_VERSION,
                   runtimeVersion,
                   driverVersion,
                   cufftVersion,
                   properties.major,
                   properties.minor,
                   Spektrafilm::kDiffusionPrecisionSchemaVersion,
                   Spektrafilm::kDiffusionPlanLayoutSchemaVersion,
                   Spektrafilm::kDiffusionCandidateTableVersion,
                   Spektrafilm::kDiffusionSelectionPolicyVersion};
            return true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            try {
                outError = "diffusion execution profile query failed";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
    }

    bool prepare_diffusion_resources(
        DiffusionContextResources& resources,
        const ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t contextEpoch,
        const std::shared_ptr<DeviceAllocationLedger>& ledger,
        const Spektrafilm::DiffusionExecutionProfileKey& observedProfile,
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
        if (!Spektrafilm::diffusion_execution_profile_matches(
                observedProfile,
                outError)) {
            if (outError.empty()) {
                outError =
                    "UnsupportedCudaExecutionProfile component=diffusion";
            }
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
            radii[keyIndex] = stage->radiusPixels;
        }

        const cudaStream_t stream =
            reinterpret_cast<cudaStream_t>(cudaStreamOpaque);
        const DiffusionAllocationOwner allocationOwner{
            contextKey,
            contextEpoch};
        std::array<std::size_t, 2> spectrumIndices{};
        std::array<bool, 2> missingSpectrum{};
        std::array<bool, 2> dependentBuilding{};
        std::size_t workspaceIndex = kInvalidIndex;
        bool rebuildWorkspace = false;
        DeviceByteReservation aggregate;

        {
            std::unique_lock<std::mutex> lock(resources.metadataMutex);
            reap_completed_locked(resources);
            for (std::size_t index = 0; index < resources.workspaces.size();
                 ++index) {
                if (resources.workspaces[index].state ==
                    WorkspaceSlotState::Vacant) {
                    workspaceIndex = index;
                    break;
                }
            }
            if (workspaceIndex == kInvalidIndex) {
                outError =
                    "ExactAdmissionFailure component=diffusion class=execution_slot requested_new_bytes=0 slots=2";
                return false;
            }

            DiffusionWorkspaceSlot& workspace =
                resources.workspaces[workspaceIndex];
            const std::uint64_t requiredPlaneElements =
                descriptor.stagePlaneBytes /
                (4u * static_cast<std::uint64_t>(sizeof(float)));
            rebuildWorkspace =
                !same_plan_key(workspace.planKey, descriptor.planKey) ||
                workspace.r2cPlan == 0 || workspace.c2rPlan == 0 ||
                !workspace.transformBuffer ||
                workspace.transformCapacityBytes <
                    descriptor.transformBufferBytes ||
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
                        found = entryIndex;
                        dependentBuilding[keyIndex] =
                            entry.state == SpectrumEntryState::Building;
                        break;
                    }
                }
                if (found == kInvalidIndex) {
                    found = choose_missing_spectrum_slot_locked(resources);
                    if (found == kInvalidIndex) {
                        outError =
                            "ExactAdmissionFailure component=diffusion class=spectrum requested_new_bytes=0 slots=4";
                        for (std::size_t rollback = 0;
                             rollback < keyIndex;
                             ++rollback) {
                            DiffusionSpectrumEntry& entry =
                                resources.spectra[spectrumIndices[rollback]];
                            if (entry.leaseCount > 0) {
                                --entry.leaseCount;
                            }
                            if (missingSpectrum[rollback]) {
                                const bool retainedRole = entry.retainedRole;
                                entry = DiffusionSpectrumEntry{};
                                entry.retainedRole = retainedRole;
                            }
                        }
                        return false;
                    }
                    DiffusionSpectrumEntry& entry = resources.spectra[found];
                    entry.key = descriptor.spectrumKeys[keyIndex];
                    entry.expandedComponents = components[keyIndex];
                    entry.radiusPixels = radii[keyIndex];
                    entry.leaseCount = 1;
                    entry.state = SpectrumEntryState::Building;
                    entry.buildEventPublished = false;
                    missingSpectrum[keyIndex] = true;
                } else {
                    ++resources.spectra[found].leaseCount;
                }
                spectrumIndices[keyIndex] = found;
            }

            const auto rollbackSpectrumLeases = [&] {
                for (std::size_t keyIndex = 0;
                     keyIndex < descriptor.uniqueSpectrumCount;
                     ++keyIndex) {
                    DiffusionSpectrumEntry& entry =
                        resources.spectra[spectrumIndices[keyIndex]];
                    if (entry.leaseCount > 0) {
                        --entry.leaseCount;
                    }
                    if (missingSpectrum[keyIndex]) {
                        const bool retainedRole = entry.retainedRole;
                        entry = DiffusionSpectrumEntry{};
                        entry.retainedRole = retainedRole;
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
                        descriptor.transformBufferBytes,
                        3,
                        packageBytes) ||
                    !checked_add(
                        prospectiveBytes,
                        packageBytes,
                        prospectiveBytes)) {
                    outError =
                        "ExactAdmissionFailure component=diffusion field=spectrum_upper_bound";
                    rollbackSpectrumLeases();
                    return false;
                }
            }
            if (rebuildWorkspace &&
                (!checked_add(
                     prospectiveBytes,
                     descriptor.transformBufferBytes,
                     prospectiveBytes) ||
                 !checked_add(
                     prospectiveBytes,
                     descriptor.acceptedSharedWorkBytes,
                     prospectiveBytes) ||
                 !checked_add(
                     prospectiveBytes,
                     descriptor.acceptedPlanAllowanceBytes,
                     prospectiveBytes) ||
                 !checked_add(
                     prospectiveBytes,
                     descriptor.stagePlaneBytes,
                     prospectiveBytes))) {
                outError =
                    "ExactAdmissionFailure component=diffusion field=workspace_upper_bound";
                rollbackSpectrumLeases();
                return false;
            }
#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
            if (prospectiveBytes > 0 &&
                consume_lifecycle_fault(
                    DiffusionLifecycleFaultPoint::AggregateReservation)) {
                outError =
                    "ExactAdmissionFailure component=diffusion class=aggregate_reservation requested_new_bytes=" +
                    std::to_string(prospectiveBytes) + " injected=1";
                rollbackSpectrumLeases();
                return false;
            }
#endif
            if (prospectiveBytes > 0) {
                std::string ledgerError;
                if (!ledger->reserve(
                        make_identity(
                            contextKey,
                            contextEpoch,
                            DeviceAllocationClass::DiffusionTransformWorkArea,
                            "diffusion aggregate preparation"),
                        prospectiveBytes,
                        aggregate,
                        ledgerError)) {
                    outError =
                        "ExactAdmissionFailure component=diffusion class=aggregate_reservation requested_new_bytes=" +
                        std::to_string(prospectiveBytes) + " detail=" + ledgerError;
                    rollbackSpectrumLeases();
                    return false;
                }
            }
            workspace.state = WorkspaceSlotState::Building;
        }

        PreparedDiffusionLease lease;
        lease._owner = &resources;
        lease._descriptor = descriptor;
        lease._spectrumIndices = spectrumIndices;
        lease._spectrumCount = descriptor.uniqueSpectrumCount;
        lease._workspaceIndex = workspaceIndex;
        lease._streamOpaque = cudaStreamOpaque;
        lease._active = true;

        auto failPreparation = [&](const char* stage) {
            const bool completionCertain =
                !lease._workEnqueued ||
                cudaStreamSynchronize(stream) == cudaSuccess;
            {
                std::lock_guard<std::mutex> lock(resources.metadataMutex);
                DiffusionWorkspaceSlot& workspace =
                    resources.workspaces[workspaceIndex];
                workspace.failureApi = FailureApi::Cuda;
                workspace.failureStage = stage;
                workspace.completionUnknown = !completionCertain;
                workspace.state = WorkspaceSlotState::FailedQuarantined;
                for (std::size_t keyIndex = 0;
                     keyIndex < descriptor.uniqueSpectrumCount;
                     ++keyIndex) {
                    DiffusionSpectrumEntry& entry =
                        resources.spectra[spectrumIndices[keyIndex]];
                    if (entry.leaseCount > 0) {
                        --entry.leaseCount;
                    }
                    if (missingSpectrum[keyIndex] || lease._workEnqueued) {
                        entry.failureApi = FailureApi::Cuda;
                        entry.failureStage = stage;
                        entry.state =
                            SpectrumEntryState::FailedQuarantined;
                    }
                }
                resources.buildPublication.notify_all();
            }
            if (completionCertain) {
                std::lock_guard<std::mutex> lock(resources.metadataMutex);
                DiffusionWorkspaceSlot& workspace =
                    resources.workspaces[workspaceIndex];
                std::string ignored;
                if (destroy_workspace_contents(workspace, ignored)) {
                    workspace.state = WorkspaceSlotState::Vacant;
                }
                for (std::size_t keyIndex = 0;
                     keyIndex < descriptor.uniqueSpectrumCount;
                     ++keyIndex) {
                    DiffusionSpectrumEntry& entry =
                        resources.spectra[spectrumIndices[keyIndex]];
                    if (entry.state ==
                            SpectrumEntryState::FailedQuarantined &&
                        entry.leaseCount == 0 &&
                        entry.pendingWorkspaceMask == 0) {
                        (void)destroy_spectrum_contents(entry, ignored);
                    }
                }
            }
            lease.reset();
            return false;
        };

        for (std::size_t keyIndex = 0;
             keyIndex < descriptor.uniqueSpectrumCount;
             ++keyIndex) {
            if (!dependentBuilding[keyIndex]) {
                continue;
            }
            void* buildEventOpaque = nullptr;
            {
                std::unique_lock<std::mutex> lock(resources.metadataMutex);
                resources.buildPublication.wait(lock, [&] {
                    const DiffusionSpectrumEntry& entry =
                        resources.spectra[spectrumIndices[keyIndex]];
                    return entry.buildEventPublished ||
                           entry.state ==
                               SpectrumEntryState::FailedQuarantined;
                });
                const DiffusionSpectrumEntry& entry =
                    resources.spectra[spectrumIndices[keyIndex]];
                if (entry.state == SpectrumEntryState::FailedQuarantined ||
                    !entry.buildEventOpaque) {
                    outError =
                        "MissingRequiredResource component=diffusion field=same_key_build_event";
                    return failPreparation("same_key_build_failed");
                }
                buildEventOpaque = entry.buildEventOpaque;
            }
            const cudaError_t waitResult = cudaStreamWaitEvent(
                stream,
                reinterpret_cast<cudaEvent_t>(buildEventOpaque),
                0);
            if (waitResult != cudaSuccess) {
                outError =
                    std::string("cudaStreamWaitEvent(diffusion same key) failed: ") +
                    cuda_message(waitResult);
                return failPreparation("cudaStreamWaitEvent_same_key");
            }
            lease._workEnqueued = true;
        }

        DiffusionWorkspaceSlot& workspace =
            resources.workspaces[workspaceIndex];
        if (rebuildWorkspace) {
            std::string cleanupError;
            if (!destroy_workspace_contents(workspace, cleanupError) ||
                !build_workspace(
                    workspace,
                    descriptor,
                    aggregate,
                    allocationOwner,
                    stream,
                    outError)) {
                if (outError.empty()) {
                    outError = cleanupError;
                }
                return failPreparation("build_workspace");
            }
        } else {
            const cufftResult r2cResult =
                cufftSetStream(workspace.r2cPlan, stream);
            const cufftResult c2rResult =
                cufftSetStream(workspace.c2rPlan, stream);
            if (r2cResult != CUFFT_SUCCESS || c2rResult != CUFFT_SUCCESS) {
                outError = "cufftSetStream(diffusion reuse) failed";
                return failPreparation("bind_reused_workspace_stream");
            }
            workspace.stagePlanes.rowStrideFloats =
                static_cast<std::size_t>(descriptor.fullFrame.width);
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
                    aggregate,
                    allocationOwner,
                    stream,
                    outError)) {
                return failPreparation("build_spectrum");
            }
            {
                std::lock_guard<std::mutex> lock(resources.metadataMutex);
                entry.buildEventPublished = true;
                resources.buildPublication.notify_all();
            }
        }

        {
            std::lock_guard<std::mutex> lock(resources.metadataMutex);
            workspace.state = WorkspaceSlotState::Leased;
            lease.refresh_view();
        }
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
        try {
            outError.clear();
            if (!lease._active || lease._owner != &resources) {
                return true;
            }
            const cudaStream_t stream = cudaStreamOpaque
                                            ? reinterpret_cast<cudaStream_t>(
                                                  cudaStreamOpaque)
                                            : reinterpret_cast<cudaStream_t>(
                                                  lease._streamOpaque);
            lease._streamOpaque = reinterpret_cast<void*>(stream);

            bool completionRecorded = false;
#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
            const bool injectReleaseRecordAndSync =
                consume_lifecycle_fault(
                    DiffusionLifecycleFaultPoint::ReleaseEventRecordAndStreamSync);
            const bool injectReleaseRecord = injectReleaseRecordAndSync ||
                                             consume_lifecycle_fault(
                                                 DiffusionLifecycleFaultPoint::ReleaseEventRecord);
#endif
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
#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
                    const cudaError_t recordResult = injectReleaseRecord
                                                         ? cudaErrorUnknown
                                                         : cudaEventRecord(
                                                               reinterpret_cast<cudaEvent_t>(
                                                                   workspace.completionEventOpaque),
                                                               stream);
#else
                    const cudaError_t recordResult = cudaEventRecord(
                        reinterpret_cast<cudaEvent_t>(
                            workspace.completionEventOpaque),
                        stream);
#endif
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
#if defined(JUICER_DIFFUSION_LIFECYCLE_TEST_FAULTS)
                const cudaError_t syncResult = injectReleaseRecordAndSync
                                                   ? cudaErrorUnknown
                                                   : cudaStreamSynchronize(stream);
#else
                const cudaError_t syncResult = cudaStreamSynchronize(stream);
#endif
                completionCertain = syncResult == cudaSuccess;
                if (!completionCertain && outError.empty()) {
                    outError =
                        std::string("cudaStreamSynchronize(diffusion release) failed: ") +
                        cuda_message(syncResult);
                }
            }

            bool cleanupWorkspace = false;
            std::array<bool, 2> cleanupSpectrum{};
            {
                std::lock_guard<std::mutex> lock(resources.metadataMutex);
                DiffusionWorkspaceSlot& workspace =
                    resources.workspaces[lease._workspaceIndex];
                if (dependentFailure) {
                    workspace.failureApi = FailureApi::Cuda;
                    workspace.failureStage = "dependent_frame_failure";
                }
                for (std::size_t index = 0; index < lease._spectrumCount; ++index) {
                    DiffusionSpectrumEntry& entry =
                        resources.spectra[lease._spectrumIndices[index]];
                    if (entry.leaseCount > 0) {
                        --entry.leaseCount;
                    }
                    if (dependentFailure) {
                        entry.failureApi = FailureApi::Cuda;
                        entry.failureStage = "dependent_frame_failure";
                        entry.state = SpectrumEntryState::FailedQuarantined;
                    }
                    if (completionRecorded) {
                        entry.pendingWorkspaceMask = static_cast<std::uint8_t>(
                            entry.pendingWorkspaceMask |
                            (1u << lease._workspaceIndex));
                    }
                    if (!entry.retainedRole && entry.leaseCount == 0) {
                        entry.state = SpectrumEntryState::Retiring;
                    }
                    if (completionCertain && !completionRecorded &&
                        entry.leaseCount == 0 &&
                        (entry.state == SpectrumEntryState::Retiring ||
                         entry.state ==
                             SpectrumEntryState::FailedQuarantined)) {
                        cleanupSpectrum[index] = true;
                    }
                    if (entry.leaseCount == 0) {
                        entry.releaseSequence = resources.nextReleaseSequence;
                        if (resources.nextReleaseSequence <
                            std::numeric_limits<std::uint64_t>::max()) {
                            ++resources.nextReleaseSequence;
                        }
                    }
                }

                if (completionRecorded) {
                    workspace.state = WorkspaceSlotState::Retiring;
                } else if (completionCertain) {
                    workspace.state =
                        workspace.retainedRole && !dependentFailure
                            ? WorkspaceSlotState::Vacant
                            : WorkspaceSlotState::Retiring;
                    cleanupWorkspace =
                        !workspace.retainedRole || dependentFailure;
                } else {
                    workspace.failureApi = FailureApi::Cuda;
                    workspace.failureStage =
                        "diffusion_release_completion_unknown";
                    workspace.completionUnknown = true;
                    workspace.state = WorkspaceSlotState::FailedQuarantined;
                    for (std::size_t index = 0; index < lease._spectrumCount;
                         ++index) {
                        DiffusionSpectrumEntry& entry =
                            resources.spectra[lease._spectrumIndices[index]];
                        entry.failureApi = FailureApi::Cuda;
                        entry.failureStage =
                            "diffusion_release_completion_unknown";
                        entry.state = SpectrumEntryState::FailedQuarantined;
                    }
                }
            }

            if (cleanupWorkspace || cleanupSpectrum[0] || cleanupSpectrum[1]) {
                std::lock_guard<std::mutex> lock(resources.metadataMutex);
                std::string cleanupError;
                if (cleanupWorkspace) {
                    DiffusionWorkspaceSlot& workspace =
                        resources.workspaces[lease._workspaceIndex];
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
            return safeRelease;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            try {
                outError = "diffusion resource release bookkeeping failed";
            } catch (...) {
                JuicerLogging::discard_current_exception();
            }
            return false;
        }
    }

    bool drain_diffusion_resources(
        DiffusionContextResources& resources,
        std::string& outError) noexcept {
        try {
            outError.clear();
            std::lock_guard<std::mutex> lock(resources.metadataMutex);
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
                const bool retainedRole = workspace.retainedRole;
                workspace = DiffusionWorkspaceSlot{};
                workspace.retainedRole = retainedRole;
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
            return drained;
        } catch (...) {
            JuicerLogging::discard_current_exception();
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
            for (std::size_t index = 0; index < resources.spectra.size();
                 ++index) {
                const bool retainedRole = index < 2;
                resources.spectra[index] = DiffusionSpectrumEntry{};
                resources.spectra[index].retainedRole = retainedRole;
            }
            for (std::size_t index = 0; index < resources.workspaces.size();
                 ++index) {
                const bool retainedRole = index == 0;
                resources.workspaces[index] = DiffusionWorkspaceSlot{};
                resources.workspaces[index].retainedRole = retainedRole;
            }
            resources.nextReleaseSequence = 1;
            resources.buildPublication.notify_all();
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

} // namespace JuicerCuda::Diffusion

#endif
