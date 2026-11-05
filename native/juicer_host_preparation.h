#pragma once

#include <cstdint>

#include "juicer_legacy_api.h"

struct FjHostPreparation;
namespace Scanner {
    struct MediumView;
}
namespace JuicerCuda {
    struct Failure;

    // Successful views borrow the immediate Rust call; native upload owns every
    // queued source byte before that call returns.
    bool prepare_scanner_lut(const FjHostPreparation&, const Scanner::MediumView&, std::uint32_t, FjScannerLutView&, Failure&);
    bool prepare_gaussian_kernel(const FjHostPreparation&, const FjGaussianKernelInput&, FjGaussianKernelView&, Failure&);
} // namespace JuicerCuda
