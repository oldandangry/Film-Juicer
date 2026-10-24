#pragma once

#include <memory>
#include <optional>
#include <span>

#include "juicer_legacy_api.h"

namespace JuicerCuda {
    struct Failure;
}

namespace JuicerGaussian {
    // SF_TEMP_BRIDGE_GAUSSIAN_KERNEL: miss-local foreign hold; remove S4.E.
    class Kernel final {
    public:
        static std::optional<Kernel> acquire(int radius, float sigma, JuicerCuda::Failure&);
        Kernel(Kernel&&) noexcept;
        Kernel& operator=(Kernel&&) noexcept;
        Kernel(const Kernel&) = delete;
        Kernel& operator=(const Kernel&) = delete;
        std::span<const float> weights() const noexcept;

    private:
        struct Release {
            void operator()(FjGaussianKernelOwner*) const noexcept;
        };
        Kernel(std::unique_ptr<FjGaussianKernelOwner, Release>, FjGaussianKernelView) noexcept;
        std::unique_ptr<FjGaussianKernelOwner, Release> _owner;
        FjGaussianKernelView _view;
    };
} // namespace JuicerGaussian
