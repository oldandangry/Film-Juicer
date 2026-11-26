#pragma once


#include "juicer_cuda_api.h"

struct FjHostPreparation;
namespace JuicerCuda {
    struct Failure;

    // Successful views borrow the immediate Rust call; native upload owns every
    // queued source byte before that call returns.
    bool prepare_gaussian_kernel(const FjHostPreparation&, const FjGaussianKernelInput&, FjGaussianKernelView&, Failure&);
} // namespace JuicerCuda
