#include "Cuda/Diffusion/JuicerCudaDiffusionResources.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <limits>
#include <utility>

#include <cuda.h>
#include <cudaTypedefs.h>

#include "Logging.h"
#include "Cuda/JuicerCudaResourceOperations.h"
#include "Cuda/Diffusion/DiffusionInternals.h"

namespace JuicerCuda::Diffusion {

    CufftPlan& CufftPlan::operator=(CufftPlan&& other) noexcept {
        if (this != &other) {
            assert(_handle == 0);
            _handle = std::exchange(other._handle, 0);
        }
        return *this;
    }

    cufftResult CufftPlan::create() noexcept {
        assert(_handle == 0);
        return cufftCreate(&_handle);
    }

    cufftResult CufftPlan::destroy_after_completion() noexcept {
        if (_handle == 0) {
            return CUFFT_SUCCESS;
        }
        const auto result = ResourceOperations::destroy_plan(_handle);
        if (result == CUFFT_SUCCESS) {
            _handle = 0;
        }
        return result;
    }

    static bool allocate_from_aggregate(
        DeviceByteReservation& aggregate,
        std::uint64_t bytes,
        TrackedDeviceBuffer& outBuffer,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (bytes == 0 ||
            bytes > static_cast<std::uint64_t>(
                        std::numeric_limits<std::size_t>::max())) {
            outError.diagnostic = "ExactAdmissionFailure component=diffusion field=allocation_bytes";
            return false;
        }
        DeviceByteReservation reservation;
        if (!aggregate.split(bytes, reservation, outError.diagnostic) ||
            !reservation.commit(bytes, outError.diagnostic)) {
            return false;
        }
        void* pointer = nullptr;
        const auto allocationResult = cudaMalloc(&pointer, static_cast<std::size_t>(bytes));
        outBuffer = TrackedDeviceBuffer(pointer, std::move(reservation));
        if (allocationResult != cudaSuccess || !pointer) {
            set_failure(outError, runtime_failure_status(allocationResult), "cudaMalloc(diffusion) failed");
            return false;
        }
        return true;
    }

    static bool commit_allowance_from_aggregate(
        DeviceByteReservation& aggregate,
        std::uint64_t bytes,
        DeviceByteReservation& outReservation,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (bytes == 0) {
            outError.diagnostic =
                "UnsupportedCudaExecutionProfile component=diffusion field=plan_allowance";
            return false;
        }
        DeviceByteReservation reservation;
        if (!aggregate.split(bytes, reservation, outError.diagnostic) ||
            !reservation.commit(bytes, outError.diagnostic)) {
            return false;
        }
        outReservation = std::move(reservation);
        return true;
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

    static bool make_plan_pair(
        DiffusionWorkspaceSlot& slot,
        const Spektrafilm::DiffusionExecutionDescriptor& descriptor,
        DeviceByteReservation& aggregate,
        cudaStream_t stream,
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!commit_allowance_from_aggregate(
                aggregate,
                descriptor.planAllowanceBytes,
                slot.planAllowanceReservation,
                outError)) {
            return false;
        }
        cufftResult result = slot.r2cPlan.create();
        if (result != CUFFT_SUCCESS) {
            set_failure(outError, cufft_failure_status(result), "cufftCreate(diffusion R2C) failed");
            return false;
        }
        result = cufftSetAutoAllocation(slot.r2cPlan.get(), 0);
        if (result != CUFFT_SUCCESS) {
            set_failure(outError, cufft_failure_status(result), "cufftSetAutoAllocation(diffusion R2C) failed");
            return false;
        }
        result = slot.c2rPlan.create();
        if (result != CUFFT_SUCCESS) {
            set_failure(outError, cufft_failure_status(result), "cufftCreate(diffusion C2R) failed");
            return false;
        }
        result = cufftSetAutoAllocation(slot.c2rPlan.get(), 0);
        if (result != CUFFT_SUCCESS) {
            set_failure(outError, cufft_failure_status(result), "cufftSetAutoAllocation(diffusion C2R) failed");
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
            slot.r2cPlan.get(),
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
            set_failure(outError, cufft_failure_status(result), "cufftMakePlanMany(diffusion R2C) failed");
            return false;
        }
        result = cufftMakePlanMany(
            slot.c2rPlan.get(),
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
            set_failure(outError, cufft_failure_status(result), "cufftMakePlanMany(diffusion C2R) failed");
            return false;
        }
        const std::uint64_t actualSharedWorkBytes = std::max<std::uint64_t>(
            static_cast<std::uint64_t>(actualR2cWork),
            static_cast<std::uint64_t>(actualC2rWork));
        if (actualSharedWorkBytes > descriptor.reservedSharedWorkBytes) {
            outError.diagnostic =
                "InvalidDiffusionExecutionDescriptor field=actual_shared_work_bytes";
            return false;
        }

        if (actualSharedWorkBytes > 0) {
            if (!allocate_from_aggregate(
                    aggregate,
                    actualSharedWorkBytes,
                    slot.workArea,
                    outError)) {
                return false;
            }
        }
        result = cufftSetWorkArea(slot.r2cPlan.get(), slot.workArea.get());
        if (result != CUFFT_SUCCESS) {
            set_failure(outError, cufft_failure_status(result), "cufftSetWorkArea(diffusion R2C) failed");
            return false;
        }
        result = cufftSetWorkArea(slot.c2rPlan.get(), slot.workArea.get());
        if (result != CUFFT_SUCCESS) {
            set_failure(outError, cufft_failure_status(result), "cufftSetWorkArea(diffusion C2R) failed");
            return false;
        }
        result = cufftSetStream(slot.r2cPlan.get(), stream);
        if (result != CUFFT_SUCCESS) {
            set_failure(outError, cufft_failure_status(result), "cufftSetStream(diffusion R2C) failed");
            return false;
        }
        result = cufftSetStream(slot.c2rPlan.get(), stream);
        if (result != CUFFT_SUCCESS) {
            set_failure(outError, cufft_failure_status(result), "cufftSetStream(diffusion C2R) failed");
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
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!make_plan_pair(
                slot,
                descriptor,
                aggregate,
                stream,
                outError)) {
            return false;
        }
        if (!allocate_from_aggregate(
                aggregate,
                descriptor.layout.transformBytes,
                slot.transform,
                outError)) {
            return false;
        }
        slot.transformCapacityBytes = descriptor.layout.transformBytes;

        const std::uint64_t onePlaneBytes = descriptor.stagePlaneBytes / 4u;
        for (std::size_t index = 0; index < 4; ++index) {
            if (!allocate_from_aggregate(
                    aggregate,
                    onePlaneBytes,
                    slot.stagePlaneBuffers[index],
                    outError)) {
                return false;
            }
            assign_stage_plane_pointer(slot, index, slot.stagePlaneBuffers[index].get());
        }
        slot.stagePlaneCapacityElements =
            onePlaneBytes / static_cast<std::uint64_t>(sizeof(float));
        slot.stagePlanes.rowStrideFloats =
            static_cast<std::size_t>(frameSet.fullFrame.width);

        if (!slot.completionEvent) {
            cudaEvent_t event = nullptr;
            const cudaError_t eventResult = cudaEventCreateWithFlags(
                &event,
                cudaEventDisableTiming);
            if (eventResult != cudaSuccess || !event) {
                set_failure(outError, runtime_failure_status(eventResult), "cudaEventCreate(diffusion workspace) failed");
                return false;
            }
            slot.completionEvent = RetiredEvent(event);
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
        Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        for (std::size_t index = 0; index < 3; ++index) {
            if (!allocate_from_aggregate(
                    aggregate,
                    descriptor.layout.transformBytes,
                    entry.spectrumBuffers[index],
                    outError)) {
                return false;
            }
            assign_spectrum_pointer(entry, index, entry.spectrumBuffers[index].get());
        }

        cudaEvent_t buildEvent = nullptr;
        const cudaError_t eventResult = cudaEventCreateWithFlags(
            &buildEvent,
            cudaEventDisableTiming);
        if (eventResult != cudaSuccess || !buildEvent) {
            set_failure(outError, runtime_failure_status(eventResult), "cudaEventCreate(diffusion spectrum) failed");
            return false;
        }
        entry.buildEvent = RetiredEvent(buildEvent);

        SpectrumBuildRequest request{};
        request.layout = descriptor.layout;
        request.components = components;
        request.radiusPixels = radiusPixels;
        request.transformBuffer = static_cast<float*>(workspace.transform.get());
        request.normalizationScratch =
            reinterpret_cast<double*>(entry.spectra.red);
        request.normalizationScratchBytes =
            static_cast<std::size_t>(descriptor.layout.transformBytes);
        request.r2cPlan = workspace.r2cPlan.get();
        request.destination = entry.spectra;
        request.stream = stream;
        auto result = build_spectrum_package(request);
        if (!result.ok()) {
            outError = std::move(result.failure);
            return false;
        }
        const cudaError_t recordResult = ResourceOperations::record_event(buildEvent, stream);
        if (recordResult != cudaSuccess) {
            set_failure(outError, runtime_failure_status(recordResult), "cudaEventRecord(diffusion spectrum) failed");
            return false;
        }
        entry.buildEventRecorded = true;
        return true;
    }


} // namespace JuicerCuda::Diffusion
