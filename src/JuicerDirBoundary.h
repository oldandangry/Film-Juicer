#pragma once

#include <array>
#include <cstdint>

#include "juicer_cuda_api.h"

namespace JuicerCuda {
    struct Failure;
}
namespace Spektrafilm {
    struct DirGaussianComponentPlan;
}

namespace JuicerDirBoundary {
    FjDirBoundaryInput project_input(const Spektrafilm::DirGaussianComponentPlan&, std::array<int, 2> dimensions);
    bool prepare(const FjHostPreparation*, const FjDirBoundaryInput&, FjDirBoundaryView&, JuicerCuda::Failure&);
} // namespace JuicerDirBoundary
