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

        // Optics features not planned for Phase 3 (Phase 4).
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
        if (in.scannerUseLut) {
            outReason = "scanner LUT path (ScannerUseLut=true) is not supported yet";
            return false;
        }
        if (in.glareActive) {
            outReason = "glare optics are not supported yet";
            return false;
        }
        if (in.lensBlurSigmaPx > 0.0f) {
            outReason = "scanner lens blur is not supported yet";
            return false;
        }
        if (in.unsharpSigmaPx > 0.0f || in.unsharpAmount > 0.0f) {
            outReason = "scanner unsharp mask is not supported yet";
            return false;
        }
        outReason.clear();
        return true;
    }

} // namespace JuicerCuda

