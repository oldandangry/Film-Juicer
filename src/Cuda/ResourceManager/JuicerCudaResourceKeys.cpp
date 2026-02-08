// Cuda/ResourceManager/JuicerCudaResourceKeys.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceKeys.h"

namespace JuicerCuda {
namespace ResourceManager {

KeyDigests make_key_digests(
    std::uint64_t uploadCoreHash,
    std::uint64_t dirHash,
    std::uint64_t scannerHash) noexcept {
    KeyDigests digests{};
    digests.uploadCoreHash = uploadCoreHash;
    digests.dirHash = dirHash;
    digests.scannerHash = scannerHash;
    return digests;
}

} // namespace ResourceManager
} // namespace JuicerCuda

