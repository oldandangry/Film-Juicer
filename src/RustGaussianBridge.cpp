#include "RustGaussianBridge.h"

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <utility>

#include "Cuda/JuicerCudaFailure.h"

namespace JuicerGaussian {
    namespace {
        bool envelope(FjStatus status, const FjGaussianKernelFailure& failure) {
            constexpr std::array<std::uint32_t, 6> categories{
                FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE};
            return failure.reason < categories.size() && status.category == categories[failure.reason] &&
                   status.api == FJ_API_NONE && status.native_code == 0;
        }
        bool diagnostic(const FjErrorBuffer& error, const std::array<char, 256>& text, const FjGaussianKernelFailure& failure) {
            constexpr std::array messages{"", "invalid Gaussian kernel arguments", "Gaussian kernel radius outside supported range", "invalid Gaussian kernel sigma", "Gaussian kernel allocation failed", "Gaussian kernel preparation panicked"};
            return error.data == text.data() && error.capacity == text.size() && error.length < text.size() &&
                   text[error.length] == '\0' && failure.reason < messages.size() &&
                   std::string_view(text.data(), error.length) == messages[failure.reason];
        }
        void invalid(JuicerCuda::Failure& failure) {
            JuicerCuda::set_failure(failure, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "Gaussian kernel projection invalid");
        }
    } // namespace

    void Kernel::Release::operator()(FjGaussianKernelOwner* owner) const noexcept {
        (void)fj_legacy_gaussian_kernel_release(owner, nullptr);
    }
    Kernel::Kernel(std::unique_ptr<FjGaussianKernelOwner, Release> owner, FjGaussianKernelView view) noexcept
        : _owner(std::move(owner)), _view(view) {
    }
    Kernel::Kernel(Kernel&&) noexcept = default;
    Kernel& Kernel::operator=(Kernel&&) noexcept = default;
    std::span<const float> Kernel::weights() const noexcept {
        return _owner ? std::span<const float>(_view.weights.data, _view.weights.count) : std::span<const float>{};
    }
    std::optional<Kernel> Kernel::acquire(int radius, float sigma, JuicerCuda::Failure& rejected) {
        rejected = {};
        const FjGaussianKernelInput input{radius, sigma};
        FjGaussianKernelOwner* raw = nullptr;
        FjGaussianKernelFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        auto status = fj_legacy_gaussian_kernel_acquire(&input, &raw, &failure, &error);
        std::unique_ptr<FjGaussianKernelOwner, Release> owner(raw);
        if (!envelope(status, failure) || !diagnostic(error, text, failure) ||
            (status.category == FJ_STATUS_SUCCESS ? !raw : raw != nullptr)) {
            invalid(rejected);
            return std::nullopt;
        }
        if (status.category != FJ_STATUS_SUCCESS) {
            JuicerCuda::set_failure(rejected, status, std::string_view(text.data(), error.length));
            return std::nullopt;
        }
        FjGaussianKernelView view{};
        status = fj_legacy_gaussian_kernel_view(owner.get(), &view, &failure, &error);
        if (!envelope(status, failure) || !diagnostic(error, text, failure) ||
            (failure.reason != FJ_GAUSSIAN_KERNEL_NONE && failure.reason != FJ_GAUSSIAN_KERNEL_RAW && failure.reason != FJ_GAUSSIAN_KERNEL_PANIC)) {
            invalid(rejected);
            return std::nullopt;
        }
        if (status.category != FJ_STATUS_SUCCESS) {
            if (view.radius != 0 || std::bit_cast<std::uint32_t>(view.sigma_px) != 0 || view.weights.data || view.weights.count) {
                invalid(rejected);
            } else {
                JuicerCuda::set_failure(rejected, status, std::string_view(text.data(), error.length));
            }
            return std::nullopt;
        }
        const auto address = reinterpret_cast<std::uintptr_t>(view.weights.data);
        const auto count = radius >= 1 && radius <= 2048 ? static_cast<std::size_t>(radius) * 2u + 1u : 0u;
        if (!count || view.radius != radius || std::bit_cast<std::uint32_t>(view.sigma_px) != std::bit_cast<std::uint32_t>(sigma) ||
            view.weights.count != count || !address || address % alignof(float) != 0 ||
            address > std::numeric_limits<std::uintptr_t>::max() - count * sizeof(float)) {
            invalid(rejected);
            return std::nullopt;
        }
        return Kernel(std::move(owner), view);
    }
} // namespace JuicerGaussian
