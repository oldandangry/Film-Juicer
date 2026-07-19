// Cuda/JuicerCudaDeviceLedger.h
//
// Physical-device byte accounting for Film-Juicer-owned CUDA allocations. This
// module owns accounting records only; allocation owners retain all CUDA resources.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"

namespace JuicerCuda {

    enum class DeviceAllocationClass : std::uint8_t {
        DurableStatic = 0,
        RetainedScratch = 1,
        FrameScratch = 2,
        SharedGaussian = 3,
        DiffusionSpectrum = 4,
        DiffusionTransformWorkArea = 5,
        DiffusionStagePlane = 6,
        CufftPlanAllowance = 7,
        Count = 8
    };

    enum class DeviceReservationState : std::uint8_t {
        Reserved = 0,
        Committed = 1,
        Retiring = 2,
        Released = 3,
        Count = 4
    };

    constexpr std::size_t kDeviceAllocationClassCount =
        static_cast<std::size_t>(DeviceAllocationClass::Count);
    constexpr std::size_t kDeviceReservationStateCount =
        static_cast<std::size_t>(DeviceReservationState::Count);

    struct DeviceAllocationIdentity {
        int deviceId = -1;
        ResourceManager::DeviceContextKey contextKey{};
        std::uint64_t contextEpoch = 0;
        DeviceAllocationClass allocationClass = DeviceAllocationClass::DurableStatic;
        std::string diagnosticIdentity;
    };

    struct DeviceLedgerSnapshot {
        int deviceId = -1;
        std::uint64_t deviceBudgetBytes = 0;
        std::uint64_t capBytes = 0;
        std::uint64_t reservedBytes = 0;
        std::uint64_t committedBytes = 0;
        std::uint64_t retiringBytes = 0;
        std::uint64_t chargedBytes = 0;
        std::uint64_t recordCount = 0;
        std::uint64_t abandonedRecordCount = 0;
        std::array<std::array<std::uint64_t, kDeviceAllocationClassCount>,
                   kDeviceReservationStateCount>
            recordCountsByStateAndClass{};
    };

    class DeviceAllocationLedger;

    class DeviceByteReservation final {
    public:
        DeviceByteReservation() noexcept = default;
        ~DeviceByteReservation() noexcept;

        DeviceByteReservation(const DeviceByteReservation&) = delete;
        DeviceByteReservation& operator=(const DeviceByteReservation&) = delete;
        DeviceByteReservation(DeviceByteReservation&& other) noexcept;
        DeviceByteReservation& operator=(DeviceByteReservation&& other) noexcept;

        [[nodiscard]] bool active() const noexcept;
        [[nodiscard]] std::uint64_t bytes() const noexcept;
        [[nodiscard]] DeviceReservationState state() const noexcept;

        bool split(
            const DeviceAllocationIdentity& childIdentity,
            std::uint64_t childBytes,
            DeviceByteReservation& outChild,
            std::string& outError);
        bool commit(std::uint64_t actualBytes, std::string& outError);
        bool mark_retiring(std::string& outError);
        bool release_after_physical_free(bool physicalFreeSucceeded, std::string& outError);
        bool rollback_reserved(std::string& outError);

    private:
        friend class DeviceAllocationLedger;

        DeviceByteReservation(
            std::shared_ptr<DeviceAllocationLedger> ledger,
            std::uint64_t recordId) noexcept;
        void abandon_or_rollback() noexcept;

        std::shared_ptr<DeviceAllocationLedger> _ledger;
        std::uint64_t _recordId = 0;
    };

    class DeviceAllocationLedger final : public std::enable_shared_from_this<DeviceAllocationLedger> {
    public:
        static std::shared_ptr<DeviceAllocationLedger> create(
            int deviceId,
            std::string& outError);

        DeviceAllocationLedger(const DeviceAllocationLedger&) = delete;
        DeviceAllocationLedger& operator=(const DeviceAllocationLedger&) = delete;

        bool bind_or_validate_cap(
            std::uint64_t deviceBudgetBytes,
            std::uint64_t softTargetBytes,
            std::string& outError);
        bool reserve(
            const DeviceAllocationIdentity& identity,
            std::uint64_t bytes,
            DeviceByteReservation& outReservation,
            std::string& outError);
        [[nodiscard]] DeviceLedgerSnapshot snapshot() const;
        [[nodiscard]] std::uint64_t record_count_for_context(
            const ResourceManager::DeviceContextKey& contextKey,
            std::uint64_t contextEpoch) const noexcept;
        bool release_context_after_proven_loss(
            const ResourceManager::DeviceContextKey& contextKey,
            std::uint64_t contextEpoch,
            std::uint64_t& outReleasedBytes,
            std::string& outError) noexcept;

    private:
        friend class DeviceByteReservation;

        struct Record {
            DeviceAllocationIdentity identity{};
            std::uint64_t bytes = 0;
            DeviceReservationState state = DeviceReservationState::Reserved;
            bool abandoned = false;
        };

        struct SplitResult {
            std::uint64_t childRecordId = 0;
            bool parentReleased = false;
        };

        struct CommitRecordRequest {
            std::uint64_t recordId = 0;
            std::uint64_t actualBytes = 0;
        };

        explicit DeviceAllocationLedger(int deviceId);

        bool split_reserved(
            std::uint64_t parentRecordId,
            const DeviceAllocationIdentity& childIdentity,
            std::uint64_t childBytes,
            SplitResult& outResult,
            std::string& outError);
        bool commit_record(
            const CommitRecordRequest& request,
            std::string& outError);
        bool mark_record_retiring(std::uint64_t recordId, std::string& outError);
        bool release_record_after_free(
            std::uint64_t recordId,
            bool physicalFreeSucceeded,
            std::string& outError);
        bool rollback_reserved_record(std::uint64_t recordId, std::string& outError);
        void abandon_or_rollback_record(std::uint64_t recordId) noexcept;
        [[nodiscard]] std::uint64_t record_bytes(std::uint64_t recordId) const noexcept;
        [[nodiscard]] DeviceReservationState record_state(std::uint64_t recordId) const noexcept;

        int _deviceId = -1;
        mutable std::mutex _mutex;
        std::uint64_t _deviceBudgetBytes = 0;
        std::uint64_t _capBytes = 0;
        std::uint64_t _reservedBytes = 0;
        std::uint64_t _committedBytes = 0;
        std::uint64_t _retiringBytes = 0;
        std::uint64_t _nextRecordId = 1;
        std::unordered_map<std::uint64_t, Record> _records;
    };

} // namespace JuicerCuda
