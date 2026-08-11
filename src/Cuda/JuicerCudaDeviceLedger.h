// Cuda/JuicerCudaDeviceLedger.h
//
// Physical-device byte accounting for Film-Juicer-owned CUDA allocations. This
// module owns accounting records only; allocation owners retain all CUDA resources.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "Cuda/ResourceManager/JuicerCudaResourceCore.h"

namespace JuicerCuda {

    enum class DeviceReservationState : std::uint8_t {
        Reserved = 0,
        Committed = 1,
        Retiring = 2,
        Released = 3
    };

    struct DeviceLedgerSnapshot {
        std::uint64_t capBytes = 0;
        std::uint64_t reservedBytes = 0;
        std::uint64_t committedBytes = 0;
        std::uint64_t retiringBytes = 0;
        std::uint64_t chargedBytes = 0;
        std::uint64_t recordCount = 0;
    };

    struct DeviceLedgerBudget {
        int deviceId = -1;
        std::uint64_t bytes = 0;
    };

    struct DeviceReservationRequest {
        ResourceManager::DeviceContextKey contextKey{};
        std::uint64_t contextEpoch = 0;
        std::uint64_t bytes = 0;
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
            std::uint64_t childBytes,
            DeviceByteReservation& outChild,
            std::string& outError);
        bool commit(std::uint64_t actualBytes, std::string& outError);
        bool mark_retiring(std::string& outError);
        bool release_after_physical_free(bool physicalFreeSucceeded, std::string& outError);

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
            DeviceLedgerBudget budget,
            std::string& outError);

        DeviceAllocationLedger(const DeviceAllocationLedger&) = delete;
        DeviceAllocationLedger& operator=(const DeviceAllocationLedger&) = delete;

        bool bind_or_validate_cap(
            std::uint64_t softTargetBytes,
            std::string& outError);
        [[nodiscard]] std::uint64_t device_budget_bytes() const noexcept;
        bool reserve(
            const DeviceReservationRequest& request,
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
            ResourceManager::DeviceContextKey contextKey{};
            std::uint64_t contextEpoch = 0;
            std::uint64_t bytes = 0;
            DeviceReservationState state = DeviceReservationState::Reserved;
            bool abandoned = false;
        };

        struct SplitResult {
            std::uint64_t childRecordId = 0;
            bool parentReleased = false;
        };

        struct SplitRecordRequest {
            std::uint64_t parentRecordId = 0;
            std::uint64_t childBytes = 0;
        };

        struct CommitRecordRequest {
            std::uint64_t recordId = 0;
            std::uint64_t actualBytes = 0;
        };

        explicit DeviceAllocationLedger(DeviceLedgerBudget budget);

        bool split_reserved(
            const SplitRecordRequest& request,
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
        void abandon_or_rollback_record(std::uint64_t recordId) noexcept;
        [[nodiscard]] std::uint64_t record_bytes(std::uint64_t recordId) const noexcept;
        [[nodiscard]] DeviceReservationState record_state(std::uint64_t recordId) const noexcept;

        const int _deviceId;
        mutable std::mutex _mutex;
        const std::uint64_t _deviceBudgetBytes;
        std::uint64_t _capBytes = 0;
        std::uint64_t _reservedBytes = 0;
        std::uint64_t _committedBytes = 0;
        std::uint64_t _retiringBytes = 0;
        std::uint64_t _nextRecordId = 1;
        std::unordered_map<std::uint64_t, Record> _records;
    };

} // namespace JuicerCuda
