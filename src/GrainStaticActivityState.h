#pragma once

#include <cstdint>
#include <unordered_map>

namespace JuicerProcess::detail {

    struct GrainStaticInstanceState final {
        std::uint64_t registryGeneration = 0;
        std::uint64_t latestSnapshotId = 0;
        bool active = false;
    };

    using GrainStaticInstanceMap =
        std::unordered_map<std::uint64_t, GrainStaticInstanceState>;

    struct GrainStaticMembershipChange final {
        std::uint64_t instanceToken = 0;
        std::uint64_t appliedSnapshotId = 0;
        GrainStaticInstanceState previous{};
        bool hadPrevious = false;
        bool applied = false;
    };

    enum class GrainStaticMembershipResult : std::uint8_t {
        Applied,
        Idempotent,
        StaleRegistryGeneration,
        StaleSnapshot,
        ConflictingEqualSnapshot,
        InvalidInput
    };

    inline bool grain_static_has_active_instance(
        const GrainStaticInstanceMap& instances) noexcept {
        for (const auto& item : instances) {
            if (item.second.active) {
                return true;
            }
        }
        return false;
    }

    inline GrainStaticMembershipResult apply_grain_static_membership(
        GrainStaticInstanceMap& instances,
        std::uint64_t instanceToken,
        std::uint64_t registryGeneration,
        std::uint64_t snapshotId,
        bool active,
        GrainStaticMembershipChange& outChange) {
        outChange = GrainStaticMembershipChange{};
        if (instanceToken == 0 || registryGeneration == 0 || snapshotId == 0) {
            return GrainStaticMembershipResult::InvalidInput;
        }

        const auto current = instances.find(instanceToken);
        if (current != instances.end()) {
            const GrainStaticInstanceState& state = current->second;
            if (registryGeneration < state.registryGeneration) {
                return GrainStaticMembershipResult::StaleRegistryGeneration;
            }
            if (registryGeneration == state.registryGeneration) {
                if (snapshotId < state.latestSnapshotId) {
                    return GrainStaticMembershipResult::StaleSnapshot;
                }
                if (snapshotId == state.latestSnapshotId) {
                    return active == state.active
                               ? GrainStaticMembershipResult::Idempotent
                               : GrainStaticMembershipResult::ConflictingEqualSnapshot;
                }
            }
            outChange.previous = state;
            outChange.hadPrevious = true;
        }

        instances[instanceToken] =
            GrainStaticInstanceState{registryGeneration, snapshotId, active};
        outChange.instanceToken = instanceToken;
        outChange.appliedSnapshotId = snapshotId;
        outChange.applied = true;
        return GrainStaticMembershipResult::Applied;
    }

    inline bool rollback_grain_static_membership(
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

} // namespace JuicerProcess::detail
