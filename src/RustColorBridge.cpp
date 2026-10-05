#include "RustColorBridge.h"

#include <utility>

#include "Cuda/JuicerCudaExecutor.h"
#include "SpectralProcessing.h"
#include "juicer_legacy_api.h"

namespace JuicerColor {
    std::array<float, 9> cat16_matrix(const Spectral::ChromaticAdaptationWhites& whites) {
        std::array<float, 9> result{};
        const FjStatus status = fj_legacy_cat16_matrix(whites.source, whites.destination, result.data());
        if (status.category != FJ_STATUS_SUCCESS) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, "CAT16 matrix preparation failed");
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        return result;
    }

    std::array<float, 3> adapt_cat16(const std::array<float, 3>& xyz, const Spectral::ChromaticAdaptationWhites& whites) {
        std::array<float, 3> result{};
        const FjStatus status = fj_legacy_adapt_cat16(xyz.data(), whites.source, whites.destination, result.data());
        if (status.category != FJ_STATUS_SUCCESS) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, "CAT16 scalar adaptation failed");
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        return result;
    }

    std::array<float, 9> cat02_matrix(const Spectral::ChromaticAdaptationWhites& whites) {
        std::array<float, 9> result{};
        const FjStatus status = fj_legacy_cat02_matrix(whites.source, whites.destination, result.data());
        if (status.category != FJ_STATUS_SUCCESS) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, "CAT02 matrix preparation failed");
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        return result;
    }

    std::array<float, 3> adapt_cat02(const std::array<float, 3>& xyz, const Spectral::ChromaticAdaptationWhites& whites) {
        std::array<float, 3> result{};
        const FjStatus status = fj_legacy_adapt_cat02(xyz.data(), whites.source, whites.destination, result.data());
        if (status.category != FJ_STATUS_SUCCESS) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, "CAT02 scalar adaptation failed");
            throw JuicerCuda::ExecutionFailure{std::move(failure)};
        }
        return result;
    }
} // namespace JuicerColor
