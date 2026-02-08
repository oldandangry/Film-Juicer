// Cuda/ResourceManager/JuicerCudaResourceKeys.h
//
// Phase-0 key scaffolding.
#pragma once

#include <cstdint>

#include "Cuda/ResourceManager/JuicerCudaResourceTypes.h"

namespace JuicerCuda {
namespace ResourceManager {

KeyDigests make_key_digests(
    std::uint64_t uploadCoreHash,
    std::uint64_t dirHash,
    std::uint64_t scannerHash) noexcept;

} // namespace ResourceManager
} // namespace JuicerCuda

