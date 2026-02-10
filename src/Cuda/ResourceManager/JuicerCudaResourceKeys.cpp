// Cuda/ResourceManager/JuicerCudaResourceKeys.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceKeys.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "Hash.h"

namespace JuicerCuda {
namespace ResourceManager {

std::uint64_t normalize_key_u64(std::uint64_t value) noexcept {
    if (value == 0) {
        return 1;
    }
    return value;
}

std::uint64_t normalize_key_float(double value, double scale) noexcept {
    if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 0.0) {
        return 1;
    }
    const double scaled = value * scale;
    if (!std::isfinite(scaled)) {
        return 1;
    }
    const long long quantized = static_cast<long long>(std::llround(scaled));
    const std::uint64_t raw = static_cast<std::uint64_t>(quantized);
    return normalize_key_u64(raw);
}

std::uint32_t normalize_scan_lut_resolution(std::uint32_t value) noexcept {
    return std::clamp(value, kScanLutResolutionMin, kScanLutResolutionMax);
}

std::uint64_t make_scan_lut_key_digest(
    std::uint32_t medium,
    std::uint64_t tablesHash,
    std::uint64_t densityRangeHash,
    std::uint32_t lutResolution,
    std::uint32_t lutFormatVersion,
    std::uint32_t keySchemaVersion) noexcept {
    if (medium > 1u ||
        tablesHash == 0 ||
        densityRangeHash == 0 ||
        lutFormatVersion == 0 ||
        keySchemaVersion == 0) {
        return 0;
    }

    const std::uint64_t fields[] = {
        static_cast<std::uint64_t>(keySchemaVersion),
        static_cast<std::uint64_t>(medium),
        tablesHash,
        densityRangeHash,
        static_cast<std::uint64_t>(normalize_scan_lut_resolution(lutResolution)),
        static_cast<std::uint64_t>(lutFormatVersion)
    };
    return Hash::hash_bytes(fields, sizeof(fields));
}

KeyDigests make_key_digests(
    std::uint64_t uploadCoreHash,
    std::uint64_t dirHash,
    std::uint64_t scannerHash,
    std::uint64_t autoExposureHash) noexcept {
    KeyDigests digests{};
    digests.uploadCoreHash = normalize_key_u64(uploadCoreHash);
    digests.dirHash = normalize_key_u64(dirHash);
    digests.scannerHash = normalize_key_u64(scannerHash);
    digests.autoExposureHash = normalize_key_u64(autoExposureHash);
    return digests;
}

KeyDigests normalize_key_digests(const KeyDigests& digests) noexcept {
    return make_key_digests(
        digests.uploadCoreHash,
        digests.dirHash,
        digests.scannerHash,
        digests.autoExposureHash);
}

} // namespace ResourceManager
} // namespace JuicerCuda
