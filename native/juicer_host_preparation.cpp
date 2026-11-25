#include "juicer_host_preparation.h"

#include <array>
#include <bit>
#include <limits>
#include <string_view>

#include "Cuda/JuicerCudaFailure.h"
#include "juicer_cuda_call_scope.h"

namespace JuicerCuda {
    namespace {
        bool diagnostic_valid(const FjErrorBuffer& error, const std::array<char, 256>& text) {
            return error.data == text.data() && error.capacity == text.size() &&
                   error.length < text.size() && text[error.length] == '\0';
        }
        bool span_valid(FjFloatSpan span, std::size_t count) {
            const auto address = reinterpret_cast<std::uintptr_t>(span.data);
            return count && span.count == count && address && address % alignof(float) == 0 &&
                   count <= std::numeric_limits<std::uintptr_t>::max() / sizeof(float) &&
                   address <= std::numeric_limits<std::uintptr_t>::max() - count * sizeof(float);
        }
    } // namespace

    bool prepare_gaussian_kernel(const FjHostPreparation& preparation, const FjGaussianKernelInput& input, FjGaussianKernelView& view, Failure& rejected) {
        view = {};
        const auto invalid = [&]() -> bool {
            set_failure(rejected, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "Gaussian kernel projection invalid");
            return false;
        };
        if (!preparation.user || !preparation.gaussian)
            return invalid();
        FjGaussianKernelFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = preparation.gaussian(preparation.user, &input, &view, &failure, &error);
        if (!diagnostic_valid(error, text))
            return invalid();
        constexpr std::array<std::uint32_t, 6> categories{FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_ALLOCATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE};
        if (status.api != FJ_API_NONE || status.native_code != 0 ||
            failure.reason >= categories.size() || status.category != categories[failure.reason])
            return invalid();
        if (status.category != FJ_STATUS_SUCCESS) {
            if (view.radius || std::bit_cast<std::uint32_t>(view.sigma_px) || view.weights.data || view.weights.count)
                return invalid();
            set_failure(rejected, status, std::string_view(text.data(), error.length));
            return false;
        }
        const auto count = input.radius >= 1 && input.radius <= 2048 ? static_cast<std::size_t>(input.radius) * 2u + 1u : 0u;
        if (error.length || view.radius != input.radius || std::bit_cast<std::uint32_t>(view.sigma_px) != std::bit_cast<std::uint32_t>(input.sigma_px) || !span_valid(view.weights, count))
            return invalid();
        rejected = {};
        return true;
    }
} // namespace JuicerCuda
