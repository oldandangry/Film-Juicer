#pragma once

// Internal forward declarations for single-TU split ownership files.
// This header is included from JuicerCudaResourceManager.cpp inside the
// ResourceManager anonymous namespace.

std::uint64_t estimate_graph_cache_active_bytes_for_context(const DeviceContextKey& key) noexcept;
std::uint64_t evict_noncritical_graph_entries_for_context(const DeviceContextKey& key) noexcept;
std::uint64_t tier_target_bytes(const ResourceManagerConfigEffective& cfg, ResourceTier tier) noexcept;
std::uint64_t pressure_total_bytes(const PressureInput& input) noexcept;
bool pressure_policy_enabled(const ResourceManagerConfigEffective& cfg) noexcept;
