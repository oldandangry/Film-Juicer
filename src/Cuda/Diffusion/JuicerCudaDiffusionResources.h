#pragma once


#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "Cuda/Diffusion/JuicerCudaDiffusion.h"
#include "Cuda/JuicerCudaDeviceLedger.h"
#include "DiffusionExecution.h"

namespace JuicerCuda::Diffusion {

    enum class SpectrumEntryState : std::uint8_t {
        Vacant,
        Building,
        Ready,
        FailedQuarantined,
        Retiring
    };

    enum class WorkspaceSlotState : std::uint8_t {
        Vacant,
        Building,
        Leased,
        Retiring,
        FailedQuarantined
    };

    struct DiffusionSpectrumEntry {
        Spektrafilm::DiffusionSpectrumKey key{};
        SpectrumPackageView spectra{};
        std::array<DeviceByteReservation, 3> spectrumReservations{};
        void* buildEventOpaque = nullptr;
        std::uint64_t releaseSequence = 0;
        std::uint8_t pendingWorkspaceMask = 0;
        SpectrumEntryState state = SpectrumEntryState::Vacant;
        bool leased = false;
    };

    struct DiffusionWorkspaceSlot {
        Spektrafilm::DiffusionPlanKey planKey{};
        cufftHandle r2cPlan = 0;
        cufftHandle c2rPlan = 0;
        DeviceByteReservation planAllowanceReservation;
        void* workArea = nullptr;
        DeviceByteReservation workAreaReservation;
        float* transformBuffer = nullptr;
        DeviceByteReservation transformReservation;
        StagePlaneSet stagePlanes{};
        std::array<DeviceByteReservation, 4> stagePlaneReservations{};
        std::uint64_t transformCapacityBytes = 0;
        std::uint64_t stagePlaneCapacityElements = 0;
        std::uint64_t releaseSequence = 0;
        void* completionEventOpaque = nullptr;
        WorkspaceSlotState state = WorkspaceSlotState::Vacant;
        bool completionUnknown = false;
    };

    struct DiffusionContextResources {
        // Guards admission flags; the active host lease exclusively owns slots.
        std::mutex metadataMutex;
        std::array<DiffusionSpectrumEntry, 4> spectra{};
        std::array<DiffusionWorkspaceSlot, 2> workspaces{};
        std::uint64_t nextReleaseSequence = 1;
        bool acceptingPreparations = true;
        bool hostLeaseActive = false;

        DiffusionContextResources() noexcept = default;
        DiffusionContextResources(const DiffusionContextResources&) = delete;
        DiffusionContextResources& operator=(const DiffusionContextResources&) = delete;
    };

    struct PreparedSpectrumView {
        Spektrafilm::DiffusionSpectrumKey key{};
        SpectrumPackageView spectra{};
    };

    struct PreparedExecutionView {
        Spektrafilm::DiffusionPlanKey planKey{};
        ExecutionWorkspaceView execution{};
        StagePlaneSet stagePlanes{};
    };

    struct DiffusionPreparedView {
        Spektrafilm::DiffusionExecutionDescriptor executionDescriptor{};
        std::array<PreparedSpectrumView, 2> spectra{};
        PreparedExecutionView execution{};
        std::size_t spectrumCount = 0;
        bool active = false;
    };

    class PreparedDiffusionLease final {
    public:
        PreparedDiffusionLease() noexcept = default;
        ~PreparedDiffusionLease() noexcept;

        PreparedDiffusionLease(const PreparedDiffusionLease&) = delete;
        PreparedDiffusionLease& operator=(const PreparedDiffusionLease&) = delete;
        PreparedDiffusionLease(PreparedDiffusionLease&& other) noexcept;
        PreparedDiffusionLease& operator=(PreparedDiffusionLease&& other) noexcept;

        [[nodiscard]] bool active() const noexcept;
        [[nodiscard]] DiffusionPreparedView view() const noexcept;

    private:
        friend bool prepare_diffusion_resources(
            DiffusionContextResources&,
            const ResourceManager::DeviceContextKey&,
            std::uint64_t,
            const std::shared_ptr<DeviceAllocationLedger>&,
            const Spektrafilm::DiffusionFrameSetDescriptor&,
            const Spektrafilm::DiffusionExecutionDescriptor&,
            void*,
            PreparedDiffusionLease&,
            std::string&);
        friend bool release_diffusion_resources(
            DiffusionContextResources&,
            PreparedDiffusionLease&,
            void*,
            bool,
            std::string&) noexcept;
        friend void mark_diffusion_work_enqueued(PreparedDiffusionLease&) noexcept;

        void reset() noexcept;
        void refresh_view() noexcept;

        DiffusionContextResources* _owner = nullptr;
        Spektrafilm::DiffusionExecutionDescriptor _descriptor{};
        DiffusionPreparedView _view{};
        std::array<std::size_t, 2> _spectrumIndices{};
        std::size_t _spectrumCount = 0;
        std::size_t _workspaceIndex = 0;
        void* _streamOpaque = nullptr;
        bool _active = false;
        bool _workEnqueued = false;
    };

    bool prepare_diffusion_resources(
        DiffusionContextResources& resources,
        const ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t contextEpoch,
        const std::shared_ptr<DeviceAllocationLedger>& ledger,
        const Spektrafilm::DiffusionFrameSetDescriptor& frameSet,
        const Spektrafilm::DiffusionExecutionDescriptor& descriptor,
        void* cudaStreamOpaque,
        PreparedDiffusionLease& outLease,
        std::string& outError);

    void mark_diffusion_work_enqueued(PreparedDiffusionLease& lease) noexcept;

    bool release_diffusion_resources(
        DiffusionContextResources& resources,
        PreparedDiffusionLease& lease,
        void* cudaStreamOpaque,
        bool dependentFailure,
        std::string& outError) noexcept;

    bool trim_inactive_diffusion_resources(
        DiffusionContextResources& resources,
        const ResourceManager::DeviceContextKey& contextKey,
        std::string& outError) noexcept;

    bool drain_diffusion_resources(
        DiffusionContextResources& resources,
        std::string& outError) noexcept;

    void invalidate_diffusion_resources_after_proven_context_loss(
        DiffusionContextResources& resources) noexcept;

} // namespace JuicerCuda::Diffusion
