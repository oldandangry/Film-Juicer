// Cuda/JuicerCudaDeviceLedger.cpp

#include "Cuda/JuicerCudaDeviceLedger.h"

#include <exception>
#include <limits>
#include <new>
#include <utility>

namespace JuicerCuda {
    namespace {

        bool fail(std::string& outError, const char* code) {
            outError = code;
            return false;
        }

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

        bool context_identity_is_valid(
            const ResourceManager::DeviceContextKey& contextKey,
            std::uint64_t contextEpoch,
            int ledgerDeviceId) noexcept {
            return contextKey.deviceId == ledgerDeviceId &&
                   contextKey.contextOpaque != nullptr && contextEpoch != 0;
        }

    } // namespace

    DeviceByteReservation::DeviceByteReservation(
        std::shared_ptr<DeviceAllocationLedger> ledger,
        std::uint64_t recordId) noexcept
        : _ledger(std::move(ledger)),
          _recordId(recordId) {
    }

    DeviceByteReservation::~DeviceByteReservation() noexcept {
        abandon_or_rollback();
    }

    DeviceByteReservation::DeviceByteReservation(DeviceByteReservation&& other) noexcept
        : _ledger(std::move(other._ledger)),
          _recordId(std::exchange(other._recordId, 0)) {
    }

    DeviceByteReservation& DeviceByteReservation::operator=(
        DeviceByteReservation&& other) noexcept {
        if (this != &other) {
            abandon_or_rollback();
            _ledger = std::move(other._ledger);
            _recordId = std::exchange(other._recordId, 0);
        }
        return *this;
    }

    bool DeviceByteReservation::active() const noexcept {
        return _ledger != nullptr && _recordId != 0;
    }

    std::uint64_t DeviceByteReservation::bytes() const noexcept {
        return active() ? _ledger->record_bytes(_recordId) : 0;
    }

    DeviceReservationState DeviceByteReservation::state() const noexcept {
        return active() ? _ledger->record_state(_recordId) : DeviceReservationState::Released;
    }

    bool DeviceByteReservation::split(
        std::uint64_t childBytes,
        DeviceByteReservation& outChild,
        std::string& outError) {
        outError.clear();
        if (!active()) {
            return fail(outError, "reservation_not_active");
        }
        if (outChild.active()) {
            return fail(outError, "child_reservation_not_empty");
        }
        DeviceAllocationLedger::SplitResult result{};
        if (!_ledger->split_reserved(
                DeviceAllocationLedger::SplitRecordRequest{
                    .parentRecordId = _recordId,
                    .childBytes = childBytes},
                result,
                outError)) {
            return false;
        }
        outChild = DeviceByteReservation(_ledger, result.childRecordId);
        if (result.parentReleased) {
            _recordId = 0;
            _ledger.reset();
        }
        return true;
    }

    bool DeviceByteReservation::commit(
        std::uint64_t actualBytes,
        std::string& outError) {
        outError.clear();
        return active() ? _ledger->commit_record(
                              DeviceAllocationLedger::CommitRecordRequest{
                                  _recordId,
                                  actualBytes},
                              outError)
                        : fail(outError, "reservation_not_active");
    }

    bool DeviceByteReservation::mark_retiring(std::string& outError) {
        outError.clear();
        return active() ? _ledger->mark_record_retiring(_recordId, outError)
                        : fail(outError, "reservation_not_active");
    }

    bool DeviceByteReservation::release_after_physical_free(
        bool physicalFreeSucceeded,
        std::string& outError) {
        outError.clear();
        if (!active()) {
            return fail(outError, "reservation_not_active");
        }
        if (!_ledger->release_record_after_free(
                _recordId,
                physicalFreeSucceeded,
                outError)) {
            return false;
        }
        _recordId = 0;
        _ledger.reset();
        return true;
    }

    void DeviceByteReservation::abandon_or_rollback() noexcept {
        if (_ledger != nullptr && _recordId != 0) {
            _ledger->abandon_or_rollback_record(_recordId);
        }
        _recordId = 0;
        _ledger.reset();
    }

    std::shared_ptr<DeviceAllocationLedger> DeviceAllocationLedger::create(
        DeviceLedgerBudget budget,
        std::string& outError) {
        outError.clear();
        if (budget.deviceId < 0) {
            fail(outError, "invalid_device_id");
            return nullptr;
        }
        if (budget.bytes <= 1) {
            fail(outError, "invalid_device_budget");
            return nullptr;
        }
        try {
            return std::shared_ptr<DeviceAllocationLedger>(
                new DeviceAllocationLedger(budget));
        } catch (...) {
            fail(outError, "ledger_allocation_failed");
            return nullptr;
        }
    }

    DeviceAllocationLedger::DeviceAllocationLedger(DeviceLedgerBudget budget)
        : _deviceId(budget.deviceId),
          _deviceBudgetBytes(budget.bytes) {
    }

    bool DeviceAllocationLedger::bind_or_validate_cap(
        std::uint64_t softTargetBytes,
        std::string& outError) {
        outError.clear();
        if (softTargetBytes == 0 || softTargetBytes > _deviceBudgetBytes) {
            return fail(outError, "invalid_device_cap");
        }
        std::scoped_lock lock(_mutex);
        if (_capBytes == softTargetBytes) {
            return true;
        }
        if (_reservedBytes != 0 || _committedBytes != 0 ||
            _retiringBytes != 0 || !_records.empty()) {
            return fail(outError, "device_cap_change_while_charged");
        }
        _capBytes = softTargetBytes;
        return true;
    }

    std::uint64_t DeviceAllocationLedger::device_budget_bytes() const noexcept {
        return _deviceBudgetBytes;
    }

    bool DeviceAllocationLedger::reserve(
        const DeviceReservationRequest& request,
        DeviceByteReservation& outReservation,
        std::string& outError) {
        outError.clear();
        if (outReservation.active()) {
            return fail(outError, "output_reservation_not_empty");
        }
        if (request.bytes == 0) {
            return fail(outError, "zero_byte_reservation");
        }
        if (!context_identity_is_valid(
                request.contextKey,
                request.contextEpoch,
                _deviceId)) {
            return fail(outError, "invalid_allocation_identity");
        }

        std::shared_ptr<DeviceAllocationLedger> self;
        try {
            self = shared_from_this();
        } catch (const std::bad_weak_ptr&) {
            return fail(outError, "ledger_not_shared_owned");
        }

        std::scoped_lock lock(_mutex);
        if (_capBytes == 0 || _deviceBudgetBytes == 0) {
            return fail(outError, "device_cap_not_bound");
        }
        std::uint64_t charged = 0;
        if (!checked_add(_reservedBytes, _committedBytes, charged) ||
            !checked_add(charged, _retiringBytes, charged) ||
            !checked_add(charged, request.bytes, charged)) {
            return fail(outError, "device_charge_overflow");
        }
        if (charged >= _capBytes) {
            return fail(outError, "device_cap_exceeded");
        }
        if (_nextRecordId == 0) {
            return fail(outError, "record_id_exhausted");
        }
        const std::uint64_t recordId = _nextRecordId;
        Record record{};
        try {
            record.contextKey = request.contextKey;
            record.contextEpoch = request.contextEpoch;
            record.bytes = request.bytes;
            const auto inserted = _records.emplace(recordId, record);
            if (!inserted.second) {
                return fail(outError, "record_id_collision");
            }
        } catch (const std::bad_alloc&) {
            return fail(outError, "record_allocation_failed");
        }
        _nextRecordId += 1;
        _reservedBytes += request.bytes;
        outReservation = DeviceByteReservation(std::move(self), recordId);
        return true;
    }

    bool DeviceAllocationLedger::split_reserved(
        const SplitRecordRequest& request,
        SplitResult& outResult,
        std::string& outError) {
        outResult = SplitResult{};
        if (request.childBytes == 0) {
            return fail(outError, "zero_byte_split");
        }
        std::scoped_lock lock(_mutex);
        auto parentIt = _records.find(request.parentRecordId);
        if (parentIt == _records.end() ||
            parentIt->second.state != DeviceReservationState::Reserved ||
            parentIt->second.abandoned) {
            return fail(outError, "parent_not_reserved");
        }
        Record& parent = parentIt->second;
        if (request.childBytes > parent.bytes) {
            return fail(outError, "split_exceeds_parent");
        }
        if (request.childBytes == parent.bytes) {
            outResult.childRecordId = request.parentRecordId;
            outResult.parentReleased = true;
            return true;
        }
        if (_nextRecordId == 0) {
            return fail(outError, "record_id_exhausted");
        }
        const std::uint64_t childRecordId = _nextRecordId;
        Record child{};
        child.contextKey = parent.contextKey;
        child.contextEpoch = parent.contextEpoch;
        child.bytes = request.childBytes;
        try {
            const auto inserted = _records.emplace(childRecordId, child);
            if (!inserted.second) {
                return fail(outError, "record_id_collision");
            }
        } catch (const std::bad_alloc&) {
            return fail(outError, "record_allocation_failed");
        }
        _nextRecordId += 1;
        parent.bytes -= request.childBytes;
        outResult.childRecordId = childRecordId;
        return true;
    }

    bool DeviceAllocationLedger::commit_record(
        const CommitRecordRequest& request,
        std::string& outError) {
        if (request.actualBytes == 0) {
            return fail(outError, "zero_byte_commit");
        }
        std::scoped_lock lock(_mutex);
        auto it = _records.find(request.recordId);
        if (it == _records.end() ||
            it->second.state != DeviceReservationState::Reserved ||
            it->second.abandoned) {
            return fail(outError, "record_not_reserved");
        }
        Record& record = it->second;
        if (request.actualBytes > record.bytes) {
            return fail(outError, "commit_exceeds_reservation");
        }
        _reservedBytes -= record.bytes;
        _committedBytes += request.actualBytes;
        record.bytes = request.actualBytes;
        record.state = DeviceReservationState::Committed;
        return true;
    }

    bool DeviceAllocationLedger::mark_record_retiring(
        std::uint64_t recordId,
        std::string& outError) {
        std::scoped_lock lock(_mutex);
        auto it = _records.find(recordId);
        if (it == _records.end() ||
            it->second.state != DeviceReservationState::Committed ||
            it->second.abandoned) {
            return fail(outError, "record_not_committed");
        }
        Record& record = it->second;
        _committedBytes -= record.bytes;
        _retiringBytes += record.bytes;
        record.state = DeviceReservationState::Retiring;
        return true;
    }

    bool DeviceAllocationLedger::release_record_after_free(
        std::uint64_t recordId,
        bool physicalFreeSucceeded,
        std::string& outError) {
        std::scoped_lock lock(_mutex);
        auto it = _records.find(recordId);
        if (it == _records.end() || it->second.abandoned ||
            (it->second.state != DeviceReservationState::Committed &&
             it->second.state != DeviceReservationState::Retiring)) {
            return fail(outError, "record_not_physically_owned");
        }
        if (!physicalFreeSucceeded) {
            return fail(outError, "physical_free_failed");
        }
        if (it->second.state == DeviceReservationState::Committed) {
            _committedBytes -= it->second.bytes;
        } else {
            _retiringBytes -= it->second.bytes;
        }
        _records.erase(it);
        return true;
    }

    void DeviceAllocationLedger::abandon_or_rollback_record(
        std::uint64_t recordId) noexcept {
        try {
            std::scoped_lock lock(_mutex);
            auto it = _records.find(recordId);
            if (it == _records.end()) {
                return;
            }
            if (it->second.state == DeviceReservationState::Reserved) {
                _reservedBytes -= it->second.bytes;
                _records.erase(it);
            } else {
                it->second.abandoned = true;
            }
        } catch (...) {
            std::terminate();
        }
    }

    std::uint64_t DeviceAllocationLedger::record_bytes(
        std::uint64_t recordId) const noexcept {
        try {
            std::scoped_lock lock(_mutex);
            const auto it = _records.find(recordId);
            return it != _records.end() ? it->second.bytes : 0;
        } catch (...) {
            return 0;
        }
    }

    DeviceReservationState DeviceAllocationLedger::record_state(
        std::uint64_t recordId) const noexcept {
        try {
            std::scoped_lock lock(_mutex);
            const auto it = _records.find(recordId);
            return it != _records.end() ? it->second.state
                                        : DeviceReservationState::Released;
        } catch (...) {
            return DeviceReservationState::Released;
        }
    }

    DeviceLedgerSnapshot DeviceAllocationLedger::snapshot() const {
        std::scoped_lock lock(_mutex);
        DeviceLedgerSnapshot result{};
        result.capBytes = _capBytes;
        result.reservedBytes = _reservedBytes;
        result.committedBytes = _committedBytes;
        result.retiringBytes = _retiringBytes;
        result.chargedBytes = _reservedBytes + _committedBytes + _retiringBytes;
        result.recordCount = static_cast<std::uint64_t>(_records.size());
        return result;
    }

    std::uint64_t DeviceAllocationLedger::record_count_for_context(
        const ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t contextEpoch) const noexcept {
        std::scoped_lock lock(_mutex);
        std::uint64_t count = 0;
        for (const auto& [recordId, record] : _records) {
            (void)recordId;
            if (record.contextKey == contextKey &&
                record.contextEpoch == contextEpoch) {
                count += 1;
            }
        }
        return count;
    }

    bool DeviceAllocationLedger::release_context_after_proven_loss(
        const ResourceManager::DeviceContextKey& contextKey,
        std::uint64_t contextEpoch,
        std::uint64_t& outReleasedBytes,
        std::string& outError) noexcept {
        outReleasedBytes = 0;
        outError.clear();
        try {
            if (contextKey.deviceId != _deviceId ||
                contextKey.contextOpaque == nullptr || contextEpoch == 0) {
                outError = "invalid_proven_context_loss_identity";
                return false;
            }
            std::scoped_lock lock(_mutex);
            for (auto it = _records.begin(); it != _records.end();) {
                Record& record = it->second;
                if (!(record.contextKey == contextKey) ||
                    record.contextEpoch != contextEpoch) {
                    ++it;
                    continue;
                }
                switch (record.state) {
                    case DeviceReservationState::Reserved:
                        _reservedBytes -= record.bytes;
                        break;
                    case DeviceReservationState::Committed:
                        _committedBytes -= record.bytes;
                        break;
                    case DeviceReservationState::Retiring:
                        _retiringBytes -= record.bytes;
                        break;
                    case DeviceReservationState::Released:
                    default:
                        outError = "invalid_proven_context_loss_record_state";
                        return false;
                }
                if (record.bytes >
                    std::numeric_limits<std::uint64_t>::max() -
                        outReleasedBytes) {
                    outError = "proven_context_loss_release_overflow";
                    return false;
                }
                outReleasedBytes += record.bytes;
                it = _records.erase(it);
            }
            return true;
        } catch (...) {
            try {
                outError = "proven_context_loss_release_failed";
            } catch (...) {
                std::terminate();
            }
            return false;
        }
    }

} // namespace JuicerCuda
