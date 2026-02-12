// Cuda/ResourceManager/JuicerCudaResourceKeys.h
//
// Phase-0 key scaffolding.
#pragma once

#include <cstdint>

#include "Cuda/ResourceManager/JuicerCudaResourceTypes.h"

namespace JuicerCuda {
namespace ResourceManager {

constexpr std::uint32_t kLutKeySchemaVersion = 1u;
constexpr std::uint32_t kScanLutFormatVersion = 1u;
constexpr std::uint32_t kScanLutResolutionMin = 17u;
constexpr std::uint32_t kScanLutResolutionMax = 128u;

std::uint64_t normalize_key_u64(std::uint64_t value) noexcept;
std::uint64_t normalize_key_float(double value, double scale) noexcept;
std::uint32_t normalize_scan_lut_resolution(std::uint32_t value) noexcept;

std::uint64_t make_scan_lut_key_digest(
    std::uint32_t medium,
    std::uint64_t tablesHash,
    std::uint64_t densityRangeHash,
    std::uint32_t lutResolution,
    std::uint32_t lutFormatVersion = kScanLutFormatVersion,
    std::uint32_t keySchemaVersion = kLutKeySchemaVersion) noexcept;

KeyDigests make_key_digests(
    std::uint64_t uploadCoreHash,
    std::uint64_t dirHash,
    std::uint64_t scannerHash,
    std::uint64_t autoExposureHash) noexcept;

std::uint64_t key_digest_for_kind(const KeyDigests& digests, ResourceKind kind) noexcept;
void set_key_digest_for_kind(KeyDigests& digests, ResourceKind kind, std::uint64_t hashValue) noexcept;

KeyDigests normalize_key_digests(const KeyDigests& digests) noexcept;

} // namespace ResourceManager
} // namespace JuicerCuda
