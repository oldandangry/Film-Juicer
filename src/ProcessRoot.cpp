#include "ProcessRoot.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "Logging.h"
#include "Scanner.h"

#include <cuda_runtime.h>

#include "Cuda/JuicerCudaResourceOperations.h"
#include "Cuda/JuicerCudaFailure.h"
#include "Cuda/JuicerCudaResources.h"
#include "Cuda/JuicerCudaExecutor.h"
#include "Cuda/ResourceManager/JuicerCudaResourceManager.h"
#include "Cuda/Scan/JuicerCudaScannerBinding.h"

namespace JuicerProcess::detail {
    namespace {
        enum class GrainStaticMembershipResult : std::uint8_t {
            Applied,
            Idempotent,
            StaleSnapshot,
            ConflictingEqualSnapshot,
            InvalidInput
        };
        bool grain_static_has_active_instance(
            const GrainStaticInstanceMap& instances) noexcept {
            for (const auto& item : instances) {
                if (item.second.active) {
                    return true;
                }
            }
            return false;
        }
        GrainStaticMembershipResult apply_grain_static_membership(
            GrainStaticInstanceMap& instances,
            std::uint64_t instanceToken,
            std::uint64_t snapshotId,
            bool active,
            GrainStaticMembershipChange& outChange) {
            outChange = GrainStaticMembershipChange{};
            if (instanceToken == 0 || snapshotId == 0) {
                return GrainStaticMembershipResult::InvalidInput;
            }
            const auto current = instances.find(instanceToken);
            if (current != instances.end()) {
                const GrainStaticInstanceState& state = current->second;
                if (snapshotId < state.latestSnapshotId) {
                    return GrainStaticMembershipResult::StaleSnapshot;
                }
                if (snapshotId == state.latestSnapshotId) {
                    return active == state.active
                               ? GrainStaticMembershipResult::Idempotent
                               : GrainStaticMembershipResult::ConflictingEqualSnapshot;
                }
                outChange.previous = state;
                outChange.hadPrevious = true;
            }
            instances[instanceToken] = GrainStaticInstanceState{snapshotId, active};
            outChange.instanceToken = instanceToken;
            outChange.appliedSnapshotId = snapshotId;
            outChange.applied = true;
            return GrainStaticMembershipResult::Applied;
        }
        bool rollback_grain_static_membership(
            GrainStaticInstanceMap& instances,
            const GrainStaticMembershipChange& change) {
            if (!change.applied || change.instanceToken == 0 ||
                change.appliedSnapshotId == 0) {
                return false;
            }
            const auto current = instances.find(change.instanceToken);
            if (current == instances.end() ||
                current->second.latestSnapshotId != change.appliedSnapshotId) {
                return false;
            }
            if (change.hadPrevious) {
                current->second = change.previous;
            } else {
                instances.erase(current);
            }
            return true;
        }
    } // namespace
} // namespace JuicerProcess::detail

namespace JuicerProcess {

    using JuicerCuda::ResourceManager::WorkspaceLayout;

    namespace {

        bool query_cuda_device_total_bytes(
            int deviceId,
            std::uint64_t& outDeviceBudgetBytes,
            JuicerCuda::Failure& outError) {
            outDeviceBudgetBytes = 0;
            outError = {};
            if (deviceId < 0) {
                outError.diagnostic = "CUDA device-total query requires a valid device id";
                return false;
            }

            int previousDevice = -1;
            cudaError_t status = cudaGetDevice(&previousDevice);
            if (status != cudaSuccess) {
                outError.status = JuicerCuda::runtime_failure_status(status);
                outError.diagnostic = std::string("cudaGetDevice failed during device-total query: ") +
                                      cudaGetErrorString(status);
                return false;
            }

            const bool switchedDevice = previousDevice != deviceId;
            if (switchedDevice) {
                status = cudaSetDevice(deviceId);
                if (status != cudaSuccess) {
                    outError.status = JuicerCuda::runtime_failure_status(status);
                    outError.diagnostic = std::string("cudaSetDevice failed during device-total query: ") +
                                          cudaGetErrorString(status);
                    return false;
                }
            }

            std::size_t freeBytes = 0;
            std::size_t totalBytes = 0;
            const cudaError_t queryStatus = cudaMemGetInfo(&freeBytes, &totalBytes);
            const cudaError_t restoreStatus = switchedDevice
                                                  ? cudaSetDevice(previousDevice)
                                                  : cudaSuccess;
            if (queryStatus != cudaSuccess) {
                outError.status = JuicerCuda::runtime_failure_status(queryStatus);
                outError.diagnostic = std::string("cudaMemGetInfo failed during device-total query: ") +
                                      cudaGetErrorString(queryStatus);
                return false;
            }
            if (restoreStatus != cudaSuccess) {
                outError.status = JuicerCuda::runtime_failure_status(restoreStatus);
                outError.diagnostic = std::string("cudaSetDevice failed restoring device after device-total query: ") +
                                      cudaGetErrorString(restoreStatus);
                return false;
            }

            outDeviceBudgetBytes = static_cast<std::uint64_t>(
                std::min<std::size_t>(
                    totalBytes,
                    static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())));
            if (outDeviceBudgetBytes <= 1) {
                outError.diagnostic = "CUDA device-total query returned an invalid budget";
                outDeviceBudgetBytes = 0;
                return false;
            }
            return true;
        }

    } // namespace

    Root::FramePreparationToken::FramePreparationToken(Root* root) noexcept
        : _root(root) {
    }

    Root::FramePreparationToken::~FramePreparationToken() {
        reset();
    }

    Root::FramePreparationToken::FramePreparationToken(FramePreparationToken&& other) noexcept
        : _root(std::exchange(other._root, nullptr)) {
    }

    Root::FramePreparationToken& Root::FramePreparationToken::operator=(FramePreparationToken&& other) noexcept {
        if (this != &other) {
            reset();
            _root = std::exchange(other._root, nullptr);
        }
        return *this;
    }

    bool Root::FramePreparationToken::active() const noexcept {
        return _root != nullptr;
    }

    void Root::FramePreparationToken::reset() noexcept {
        Root* root = _root;
        _root = nullptr;
        if (root) {
            root->finish_frame_preparation();
        }
    }
    Root::Root() = default;

    Root::~Root() = default;

    bool Root::shutdown(std::mutex& nativeCallMutex, JuicerCuda::Failure& outError) {
        if (!wait_for_frame_preparation()) {
            JuicerCuda::set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "CUDA shutdown frame admission drain failed");
            return false;
        }
        // Outer preparation guards may still need the native gate to finish.
        // Admission is closed and those guards are drained before taking it.
        std::lock_guard<std::mutex> nativeCallLock(nativeCallMutex);
        if (!retire_known_contexts(outError)) {
            return false;
        }
        if (!release_cuda_context_resource_owners()) {
            JuicerCuda::set_failure(outError, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "CUDA shutdown could not release context owners");
            return false;
        }
        return true;
    }

    void Root::retire_grain_static_instance(std::uint64_t instanceToken) {
        for (;;) {
            CudaResourceOwner retiredOwner;
            {
                std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
                const auto member = std::find_if(_cudaContextResources.begin(), _cudaContextResources.end(), [instanceToken](const auto& item) {
                    return item.second.grainInstances.contains(instanceToken);
                });
                if (member == _cudaContextResources.end()) {
                    return;
                }
                auto& entry = member->second;
                entry.grainInstances.erase(instanceToken);
                if (!detail::grain_static_has_active_instance(entry.grainInstances)) {
                    retiredOwner = std::move(entry.grainOwner);
                }
            }
            // Release outside the map lock. Re-find the next membership after
            // relocking; no iterator or CUDA destruction crosses that lock.
        }
    }

    Root::FramePreparationToken Root::begin_frame_preparation() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            if (!_acceptFramePreparation) {
                return FramePreparationToken{};
            }
            ++_activeFramePreparations;
            return FramePreparationToken(this);
        } catch (...) {
            return FramePreparationToken{};
        }
    }

    bool Root::retire_context(const FjCudaRetirement& identity, JuicerCuda::Failure& failure) {
        const JuicerCuda::ResourceManager::DeviceContextKey key{
            identity.context.device_id, reinterpret_cast<void*>(identity.context.context)};
        return retire_cuda_context(key, identity.submission_id != 0, failure, &identity);
    }

    bool Root::retire_cuda_context(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        bool contextReset,
        JuicerCuda::Failure& outError,
        const FjCudaRetirement* identity) {
        outError = {};
        if (deviceContextKey.deviceId < 0 ||
            !deviceContextKey.contextOpaque) {
            JuicerCuda::set_failure(outError, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "invalid CUDA context retirement key");
            return false;
        }
        JuicerCuda::ResourceManager::RegistryContextSnapshot contextSnapshot{};
        if (!JuicerCuda::ResourceManager::registry_begin_owner_retire(
                deviceContextKey,
                identity,
                contextSnapshot)) {
            JuicerCuda::set_failure(outError, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "context owner-retire rejected");
            return false;
        }
        if (contextSnapshot.contextEpoch == 0) {
            JuicerCuda::set_failure(outError, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "context owner-retire epoch invalid");
            return false;
        }

        if (contextSnapshot.inspectionOnly) {
            // An inspection-only epoch has no resources to drain, even if a
            // different context on this device created a ledger.
            return JuicerCuda::ResourceManager::registry_retire(deviceContextKey);
        }

        std::vector<CudaResourceOwner> owners;
        const auto epoch = contextSnapshot.contextEpoch;
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger> deviceLedger;
        {
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            const auto ledger =
                _cudaDeviceLedgers.find(deviceContextKey.deviceId);
            if (ledger != _cudaDeviceLedgers.end()) {
                deviceLedger = ledger->second;
            }
            for (const auto& [key, entry] : _cudaContextResources) {
                if (key.deviceContextKey == deviceContextKey && key.contextEpoch == epoch) {
                    if (entry.frameOwner) {
                        owners.push_back(entry.frameOwner);
                    }
                    if (entry.grainOwner) {
                        owners.push_back(entry.grainOwner);
                    }
                }
            }
        }

        outError.status = {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
        bool provenContextLoss = false;
        for (const CudaResourceOwner& owner : owners) {
            JuicerCuda::Failure ownerError;
            if (!JuicerCuda::drain_for_context_retire(*owner, ownerError)) {
                JuicerCuda::retain_first_failure(outError, std::move(ownerError));
            }
        }
        if (deviceLedger) {
            JuicerCuda::Failure deferredError;
            if (!JuicerCuda::drain_deferred_resources(deviceContextKey, epoch, deferredError)) {
                JuicerCuda::retain_first_failure(outError, std::move(deferredError));
            }
        }
        if (outError.status.category != FJ_STATUS_SUCCESS) {
            if (!contextReset || !deviceLedger) {
                if (outError.diagnostic.empty()) {
                    JuicerCuda::set_failure(outError, outError.status, "CUDA context owner drain failed");
                }
                return false;
            }
            provenContextLoss = true;
            std::uint64_t releasedBytes = 0;
            if (!deviceLedger->release_context_after_proven_loss(
                    deviceContextKey, epoch, releasedBytes, outError.diagnostic)) {
                return false;
            }
            for (const CudaResourceOwner& owner : owners) {
                if (owner) {
                    JuicerCuda::invalidate_resources_after_proven_context_loss(
                        *owner);
                }
            }
            JuicerCuda::invalidate_deferred_resources_after_proven_context_loss(deviceContextKey, epoch);
            outError = {};
        }
        if (deviceLedger && deviceLedger->record_count_for_context(deviceContextKey, epoch) != 0) {
            std::array<char, 160> diagnostic{};
            std::snprintf(diagnostic.data(), diagnostic.size(), "CUDA context retirement retained ledger records epoch=%llu", static_cast<unsigned long long>(epoch));
            JuicerCuda::set_failure(outError, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, diagnostic.data());
            return false;
        }

        if (!JuicerCuda::purge_pinned_upload_staging_for_context(
                deviceContextKey.deviceId, deviceContextKey.contextOpaque, epoch, provenContextLoss ? JuicerCuda::PinnedUploadPurgeDisposition::ProvenContextLoss : JuicerCuda::PinnedUploadPurgeDisposition::NormalRetire, outError)) {
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            for (auto it = _cudaContextResources.begin();
                 it != _cudaContextResources.end();) {
                if (it->first.deviceContextKey == deviceContextKey && it->first.contextEpoch == epoch) {
                    it = _cudaContextResources.erase(it);
                } else {
                    ++it;
                }
            }
        }

        if (!JuicerCuda::ResourceManager::registry_retire(
                deviceContextKey)) {
            JuicerCuda::set_failure(outError, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "context registry retire rejected");
            return false;
        }
        owners.clear();
        return true;
    }

    void Root::CudaResourcesDeleter::operator()(JuicerCuda::Resources* resources) const noexcept {
        JuicerCuda::destroy(resources);
    }

    bool Root::resolve_context_cuda_resources(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::uint64_t contextEpoch,
        const std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& deviceLedger,
        CudaResourceOwner& outResourceOwner,
        JuicerCuda::Resources*& outResources,
        JuicerCuda::Failure& outError) {
        outResourceOwner.reset();
        outResources = nullptr;
        outError = {};
        if (deviceContextKey.deviceId < 0 || !deviceContextKey.contextOpaque ||
            !deviceLedger) {
            outError.diagnostic = "invalid CUDA context key";
            return false;
        }
        if (contextEpoch == 0) {
            outError.diagnostic = "invalid CUDA context epoch";
            return false;
        }

        const ContextCudaResourceKey resourceKey{deviceContextKey, contextEpoch};
        CudaResourceOwner candidateOwner;
        {
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            CudaContextResourceEntry& entry = _cudaContextResources[resourceKey];
            CudaResourceOwner& resourceOwner = entry.frameOwner;
            if (!resourceOwner) {
                candidateOwner = CudaResourceOwner(
                    JuicerCuda::create(
                        deviceContextKey,
                        contextEpoch,
                        deviceLedger,
                        outError),
                    CudaResourcesDeleter{});
                if (!candidateOwner) {
                    JTRACE("CUDA", "FATAL: failed to allocate CUDA resources");
                    if (!entry.grainOwner && entry.grainInstances.empty()) {
                        _cudaContextResources.erase(resourceKey);
                    }
                    if (outError.diagnostic.empty()) {
                        outError.diagnostic = "failed to allocate CUDA resources";
                    }
                    return false;
                }
                if (!(candidateOwner->ownerContextKey == deviceContextKey) ||
                    candidateOwner->contextEpoch != contextEpoch ||
                    candidateOwner->deviceLedger != deviceLedger) {
                    if (!entry.grainOwner && entry.grainInstances.empty()) {
                        _cudaContextResources.erase(resourceKey);
                    }
                    outError.diagnostic = "CUDA resources resolved for a different context";
                    return false;
                }
                resourceOwner = std::move(candidateOwner);
            }

            outResourceOwner = resourceOwner;
        }

        outResources = outResourceOwner.get();
        if (!outResources) {
            outError.diagnostic = "CUDA resources missing after allocation";
            return false;
        }
        return true;
    }

    bool Root::resolve_cuda_device_ledger(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& outDeviceLedger,
        JuicerCuda::Failure& outError) {
        outDeviceLedger.reset();
        outError = {};
        if (deviceContextKey.deviceId < 0) {
            outError.diagnostic =
                "UnsupportedCudaExecutionProfile component=device_ledger field=device_id";
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            const auto ledgerIt = _cudaDeviceLedgers.find(deviceContextKey.deviceId);
            if (ledgerIt != _cudaDeviceLedgers.end() && ledgerIt->second) {
                outDeviceLedger = ledgerIt->second;
                return true;
            }
        }

        std::uint64_t deviceBudgetBytes = 0;
        if (!query_cuda_device_total_bytes(
                deviceContextKey.deviceId,
                deviceBudgetBytes,
                outError)) {
            return false;
        }
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger> candidate =
            JuicerCuda::DeviceAllocationLedger::create(
                JuicerCuda::DeviceLedgerBudget{
                    .deviceId = deviceContextKey.deviceId,
                    .bytes = deviceBudgetBytes},
                outError.diagnostic);
        if (!candidate) {
            if (outError.diagnostic.empty()) {
                outError.diagnostic =
                    "UnsupportedCudaExecutionProfile component=device_ledger field=create";
            }
            return false;
        }

        {
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            auto [ledgerIt, inserted] =
                _cudaDeviceLedgers.try_emplace(deviceContextKey.deviceId, candidate);
            if (!inserted && !ledgerIt->second) {
                ledgerIt->second = std::move(candidate);
            }
            outDeviceLedger = ledgerIt->second;
        }
        return outDeviceLedger != nullptr;
    }

    bool Root::resolve_cuda_frame_resources(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::uint64_t contextEpoch,
        const JuicerCuda::ResourceManager::ResolvedMemoryBudget& memoryBudget,
        const std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& deviceLedger,
        CudaResourceOwner& outResourceOwner,
        JuicerCuda::Resources*& outResources,
        JuicerCuda::Failure& outError) {
        outResourceOwner.reset();
        outResources = nullptr;
        outError = {};
        if (!deviceLedger ||
            memoryBudget.deviceId != deviceContextKey.deviceId ||
            memoryBudget.deviceBudgetBytes == 0 ||
            memoryBudget.allocationCapBytes == 0 ||
            deviceLedger->device_budget_bytes() != memoryBudget.deviceBudgetBytes) {
            outError.diagnostic =
                "UnsupportedCudaExecutionProfile component=device_ledger field=memory_budget";
            return false;
        }

        if (!deviceLedger->bind_or_validate_cap(
                memoryBudget.allocationCapBytes,
                outError.diagnostic)) {
            const JuicerCuda::DeviceLedgerSnapshot ledgerSnapshot =
                deviceLedger->snapshot();
            outError.diagnostic =
                "UnsupportedCudaExecutionProfile component=device_ledger field=cap" +
                std::string(" device=") + std::to_string(deviceContextKey.deviceId) +
                " requested_cap=" +
                std::to_string(memoryBudget.allocationCapBytes) +
                " current_cap=" + std::to_string(ledgerSnapshot.capBytes) +
                " reserved=" + std::to_string(ledgerSnapshot.reservedBytes) +
                " committed=" + std::to_string(ledgerSnapshot.committedBytes) +
                " retiring=" + std::to_string(ledgerSnapshot.retiringBytes);
            return false;
        }
        return resolve_context_cuda_resources(
            deviceContextKey,
            contextEpoch,
            deviceLedger,
            outResourceOwner,
            outResources,
            outError);
    }

    bool Root::apply_grain_static_membership_and_copy_owner(
        const JuicerCuda::ResourceManager::DeviceContextKey& deviceContextKey,
        std::uint64_t contextEpoch,
        const std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& deviceLedger,
        std::uint64_t instanceToken,
        std::uint64_t snapshotId,
        bool active,
        detail::GrainStaticMembershipChange& outChange,
        CudaResourceOwner& outOwner,
        JuicerCuda::Resources*& outResources,
        JuicerCuda::Failure& outError) {
        outChange = detail::GrainStaticMembershipChange{};
        outOwner.reset();
        outResources = nullptr;
        outError = {};
        if (deviceContextKey.deviceId < 0 || !deviceContextKey.contextOpaque ||
            contextEpoch == 0 || instanceToken == 0 || snapshotId == 0 ||
            !deviceLedger) {
            outError.diagnostic =
                "ResourceDescriptorMismatch phase=grain_static field=membership_identity";
            return false;
        }

        const ContextCudaResourceKey contextKey{
            deviceContextKey,
            contextEpoch};
        CudaResourceOwner candidateOwner;
        CudaResourceOwner retiredOwner;
#if JUICER_DIAGNOSTICS_COMPILED
        bool ownerCopied = false;
        bool rootOwnerRetired = false;
#endif
        {
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            CudaContextResourceEntry& entry = _cudaContextResources[contextKey];
            detail::GrainStaticMembershipChange change;
            const detail::GrainStaticMembershipResult result =
                detail::apply_grain_static_membership(
                    entry.grainInstances,
                    instanceToken,
                    snapshotId,
                    active,
                    change);
            switch (result) {
                case detail::GrainStaticMembershipResult::Applied:
                case detail::GrainStaticMembershipResult::Idempotent:
                    break;
                case detail::GrainStaticMembershipResult::StaleSnapshot:
                    outError.diagnostic =
                        "ResourceDescriptorMismatch phase=grain_static field=stale_snapshot";
                    return false;
                case detail::GrainStaticMembershipResult::ConflictingEqualSnapshot:
                    outError.diagnostic =
                        "ResourceDescriptorMismatch phase=grain_static field=equal_snapshot_conflict";
                    return false;
                case detail::GrainStaticMembershipResult::InvalidInput:
                default:
                    outError.diagnostic =
                        "ResourceDescriptorMismatch phase=grain_static field=membership_update";
                    return false;
            }

            if (active) {
                if (!entry.grainOwner) {
                    candidateOwner = CudaResourceOwner(
                        JuicerCuda::create(
                            deviceContextKey,
                            contextEpoch,
                            deviceLedger,
                            outError),
                        CudaResourcesDeleter{});
                    if (!candidateOwner) {
                        if (change.applied) {
                            (void)detail::rollback_grain_static_membership(
                                entry.grainInstances,
                                change);
                        }
                        if (!entry.frameOwner && entry.grainInstances.empty()) {
                            _cudaContextResources.erase(contextKey);
                        }
                        outError.diagnostic = "failed to allocate grain-static CUDA resources";
                        return false;
                    }
                    if (!(candidateOwner->ownerContextKey == deviceContextKey) ||
                        candidateOwner->contextEpoch != contextEpoch ||
                        candidateOwner->deviceLedger != deviceLedger) {
                        if (change.applied) {
                            (void)detail::rollback_grain_static_membership(
                                entry.grainInstances,
                                change);
                        }
                        if (!entry.frameOwner && entry.grainInstances.empty()) {
                            _cudaContextResources.erase(contextKey);
                        }
                        outError.diagnostic =
                            "grain-static CUDA resources resolved for a different context";
                        return false;
                    }
                    entry.grainOwner = std::move(candidateOwner);
                }
                if (!(entry.grainOwner->ownerContextKey == deviceContextKey) ||
                    entry.grainOwner->contextEpoch != contextEpoch ||
                    entry.grainOwner->deviceLedger != deviceLedger) {
                    if (change.applied) {
                        (void)detail::rollback_grain_static_membership(
                            entry.grainInstances,
                            change);
                    }
                    outError.diagnostic =
                        "grain-static CUDA resources resolved for a different context";
                    return false;
                }
                outOwner = entry.grainOwner;
#if JUICER_DIAGNOSTICS_COMPILED
                ownerCopied = static_cast<bool>(outOwner);
#endif
            } else if (!detail::grain_static_has_active_instance(
                           entry.grainInstances)) {
                retiredOwner = std::move(entry.grainOwner);
#if JUICER_DIAGNOSTICS_COMPILED
                rootOwnerRetired = static_cast<bool>(retiredOwner);
#endif
            }
            outChange = change;
        }

#if JUICER_DIAGNOSTICS_COMPILED
        if (JTRACE_ENABLED(1)) {
            std::string message =
                "event=grain_static_membership active=";
            message += active ? "1" : "0";
            message += " owner_copied_before_unlock=";
            message += ownerCopied ? "1" : "0";
            message += " root_owner_retired_after_unlock=";
            message += rootOwnerRetired ? "1" : "0";
            message += " snapshot_id=";
            message += std::to_string(
                static_cast<unsigned long long>(snapshotId));
            JTRACE("GRAINLIFE", message);
        }
#endif

        outResources = outOwner.get();
        if (active && !outResources) {
            outError.diagnostic =
                "grain-static CUDA resources missing after owner copy";
            return false;
        }
        return true;
    }

    void Root::rollback_grain_static_membership(
        const ContextCudaResourceKey& contextKey,
        const detail::GrainStaticMembershipChange& change) noexcept {
        if (!change.applied) {
            return;
        }
        try {
            CudaResourceOwner retiredOwner;
#if JUICER_DIAGNOSTICS_COMPILED
            bool rolledBack = false;
#endif
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            const auto entryIt = _cudaContextResources.find(contextKey);
            if (entryIt == _cudaContextResources.end() ||
                !detail::rollback_grain_static_membership(
                    entryIt->second.grainInstances,
                    change)) {
                return;
            }
#if JUICER_DIAGNOSTICS_COMPILED
            rolledBack = true;
#endif
            if (!detail::grain_static_has_active_instance(
                    entryIt->second.grainInstances)) {
                retiredOwner = std::move(entryIt->second.grainOwner);
            }
            if (!entryIt->second.frameOwner &&
                !entryIt->second.grainOwner &&
                entryIt->second.grainInstances.empty()) {
                _cudaContextResources.erase(entryIt);
            }
#if JUICER_DIAGNOSTICS_COMPILED
            if (rolledBack && JTRACE_ENABLED(1)) {
                JTRACE(
                    "GRAINLIFE",
                    "event=grain_static_conditional_rollback applied=1");
            }
#endif
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }
    bool Root::begin_submission(
        JuicerCuda::ResourceManager::SubmissionTransaction& transaction,
        const JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        std::shared_ptr<JuicerCuda::DeviceAllocationLedger>& outDeviceLedger,
        JuicerCuda::Failure& outError) {
        outError.status = {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0};
        if (!resolve_cuda_device_ledger(
                snapshot.deviceContextKey,
                outDeviceLedger,
                outError)) {
            return false;
        }
        return JuicerCuda::ResourceManager::begin_submission(
            transaction,
            snapshot,
            outDeviceLedger->device_budget_bytes(),
            outError);
    }


    bool Root::retire_known_contexts(JuicerCuda::Failure& outError) {
        outError = {};
        std::vector<JuicerCuda::ResourceManager::DeviceContextKey> keys;
        JuicerCuda::ResourceManager::registry_snapshot_context_keys(keys);
        {
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            for (const auto& [resourceKey, entry] : _cudaContextResources) {
                (void)entry;
                if (std::find(
                        keys.begin(),
                        keys.end(),
                        resourceKey.deviceContextKey) == keys.end()) {
                    keys.push_back(resourceKey.deviceContextKey);
                }
            }
        }
        for (const auto& key : keys) {
            if (!retire_cuda_context(key, false, outError)) {
                return false;
            }
        }
        keys.clear();
        JuicerCuda::ResourceManager::registry_snapshot_context_keys(keys);
        if (!keys.empty()) {
            std::array<char, 128> diagnostic{};
            std::snprintf(diagnostic.data(), diagnostic.size(), "context retire incomplete; live_contexts=%zu", keys.size());
            JuicerCuda::set_failure(outError, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, diagnostic.data());
            return false;
        }
        std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
        for (const auto& [deviceId, ledger] : _cudaDeviceLedgers) {
            if (!ledger) {
                continue;
            }
            const JuicerCuda::DeviceLedgerSnapshot ledgerSnapshot =
                ledger->snapshot();
            if (ledgerSnapshot.chargedBytes != 0 ||
                ledgerSnapshot.recordCount != 0) {
                std::array<char, 192> diagnostic{};
                std::snprintf(diagnostic.data(), diagnostic.size(), "shutdown retained CUDA ledger records device=%d charged=%llu records=%llu", deviceId, static_cast<unsigned long long>(ledgerSnapshot.chargedBytes), static_cast<unsigned long long>(ledgerSnapshot.recordCount));
                JuicerCuda::set_failure(outError, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, diagnostic.data());
                return false;
            }
        }
        return true;
    }

    bool Root::release_cuda_context_resource_owners() noexcept {
        try {
            CudaContextResourceMap contextResources;
            std::lock_guard<std::mutex> lock(_cudaResourcesMutex);
            contextResources.swap(_cudaContextResources);
            _cudaDeviceLedgers.clear();
            return true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            return false;
        }
    }

    bool Root::stop_frame_preparation() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            _acceptFramePreparation = false;
            return true;
        } catch (...) {
            return false;
        }
    }

    void Root::finish_frame_preparation() noexcept {
        try {
            std::lock_guard<std::mutex> lock(_framePreparationMutex);
            if (_activeFramePreparations > 0) {
                --_activeFramePreparations;
            }
            if (_activeFramePreparations == 0) {
                _framePreparationCv.notify_all();
            }
        } catch (...) {
            JuicerLogging::discard_current_exception();
        }
    }

    bool Root::wait_for_frame_preparation() noexcept {
        try {
            std::unique_lock<std::mutex> lock(_framePreparationMutex);
            _framePreparationCv.wait(lock, [this]() {
                return _activeFramePreparations == 0;
            });
            return true;
        } catch (...) {
            JuicerLogging::discard_current_exception();
            return false;
        }
    }


} // namespace JuicerProcess
