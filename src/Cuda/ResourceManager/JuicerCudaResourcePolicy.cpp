// Cuda/ResourceManager/JuicerCudaResourcePolicy.cpp

#include "Cuda/ResourceManager/JuicerCudaResourcePolicy.h"

namespace JuicerCuda {
namespace ResourceManager {

AcquireDecision default_acquire_decision() noexcept {
    return AcquireDecision{};
}

PressureDecision default_pressure_decision() noexcept {
    return PressureDecision{};
}

ReservationDecision default_reservation_decision() noexcept {
    return ReservationDecision{};
}

} // namespace ResourceManager
} // namespace JuicerCuda

