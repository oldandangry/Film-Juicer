// Cuda/ResourceManager/JuicerCudaResourceState.cpp

#include "Cuda/ResourceManager/JuicerCudaResourceState.h"

namespace JuicerCuda {
namespace ResourceManager {

ResourceManagerState& global_state() noexcept {
    static ResourceManagerState state{};
    return state;
}

} // namespace ResourceManager
} // namespace JuicerCuda

