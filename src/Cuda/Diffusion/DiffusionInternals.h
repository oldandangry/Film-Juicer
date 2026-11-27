#pragma once

#include <limits>

#include "Cuda/Diffusion/JuicerCudaDiffusionResources.h"

namespace JuicerCuda::Diffusion {

    inline constexpr auto kInvalidIndex = std::numeric_limits<std::size_t>::max();

    void assign_stage_plane_pointer(
        DiffusionWorkspaceSlot& slot,
        std::size_t index,
        void* pointer) noexcept;

    void assign_spectrum_pointer(
        DiffusionSpectrumEntry& entry,
        std::size_t index,
        void* pointer) noexcept;

    bool build_workspace(
        DiffusionWorkspaceSlot& slot,
        const Spektrafilm::DiffusionFrameSetDescriptor& frameSet,
        const Spektrafilm::DiffusionExecutionDescriptor& descriptor,
        DeviceByteReservation& aggregate,
        cudaStream_t stream,
        Failure& outError);

    bool build_spectrum(
        DiffusionSpectrumEntry& entry,
        DiffusionWorkspaceSlot& workspace,
        const Spektrafilm::DiffusionExecutionDescriptor& descriptor,
        const Spektrafilm::DiffusionPsfComponents& components,
        int radiusPixels,
        DeviceByteReservation& aggregate,
        cudaStream_t stream,
        Failure& outError);

    bool current_owner_matches(
        const ResourceManager::DeviceContextKey& contextKey,
        Failure& outError);

    bool retained_spectrum_slot(std::size_t index) noexcept;

    bool retained_workspace_slot(std::size_t index) noexcept;

    bool destroy_workspace_contents(
        DiffusionWorkspaceSlot& slot,
        Failure& outError) noexcept;

    bool destroy_spectrum_contents(
        DiffusionSpectrumEntry& entry,
        Failure& outError) noexcept;

    bool complete_workspace_retirement(
        DiffusionContextResources& resources,
        std::size_t workspaceIndex,
        Failure& outError) noexcept;

    bool reap_retired_spectra(
        DiffusionContextResources& resources,
        Failure& outError) noexcept;

    bool reap_completed(
        DiffusionContextResources& resources,
        Failure& outError);

    std::size_t choose_retiring_workspace(
        const DiffusionContextResources& resources) noexcept;

    bool same_spectrum_key(
        const Spektrafilm::DiffusionSpectrumKey& left,
        const Spektrafilm::DiffusionSpectrumKey& right) noexcept;

    bool same_plan_key(
        const Spektrafilm::DiffusionPlanKey& left,
        const Spektrafilm::DiffusionPlanKey& right) noexcept;

    std::size_t choose_missing_spectrum_slot(
        DiffusionContextResources& resources,
        Failure& outError);

    bool acquire_host_lease(
        DiffusionContextResources& resources,
        Failure& outError);

    void release_host_lease(
        DiffusionContextResources& resources) noexcept;

} // namespace JuicerCuda::Diffusion
