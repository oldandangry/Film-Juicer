// Cuda/JuicerCudaPhase3Gate.h
//
// Phase 3 scaffolding: centralized feature gating for the negative-only CUDA pipeline.
//
// Kept header-only and free of project type dependencies so it can be used from host code and
// CUDA TUs without dragging in large include graphs.
#pragma once

#include <string>

namespace JuicerCuda {

    struct Phase3GateInput {
        bool printBypass = true;
        bool scannerUseLut = true;

        // Scanner optics features (implemented in Phase 3 passes).
        bool glareActive = false;
        float lensBlurSigmaPx = 0.0f;
        float unsharpSigmaPx = 0.0f;
        float unsharpAmount = 0.0f;
    };

    inline bool phase3_negative_only_supported(const Phase3GateInput& in, std::string& outReason) {
        if (!in.printBypass) {
            outReason = "print pipeline (PrintBypass=false) is not supported yet";
            return false;
        }
        outReason.clear();
        return true;
    }

} // namespace JuicerCuda
