// Cuda/ResourceManager/JuicerCudaResourceKeys.h
//
// Phase-0 key scaffolding.
#pragma once

#include <cstdint>

#include "Cuda/ResourceManager/JuicerCudaResourceTypes.h"

namespace JuicerCuda {
namespace ResourceManager {

std::uint64_t normalize_key_u64(std::uint64_t value) noexcept;
std::uint64_t normalize_key_float(double value, double scale) noexcept;

KeyDigests make_key_digests(
    std::uint64_t uploadCoreHash,
    std::uint64_t dirHash,
    std::uint64_t scannerHash) noexcept;

KeyDigests normalize_key_digests(const KeyDigests& digests) noexcept;

} // namespace ResourceManager
} // namespace JuicerCuda
