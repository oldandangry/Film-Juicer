#include "juicer_host_preparation.h"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <string_view>
#include <utility>

#include "Cuda/JuicerCudaFailure.h"
#include "Scanner.h"
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

    bool prepare_scanner_lut(const FjHostPreparation& preparation, const Scanner::MediumView& medium, std::uint32_t resolution, FjScannerLutView& view, Failure& rejected) {
        view = {};
        const auto invalid = [&]() -> bool {
            set_failure(rejected, {FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, "Scanner LUT preparation: invalid transport contract");
            throw ExecutionFailure{std::move(rejected)};
        };
        if (!preparation.user || !preparation.scanner_lut || !preparation.gaussian)
            return invalid();
        const auto span = [](std::span<const float> samples) -> FjFloatSpan {
            return {samples.empty() ? nullptr : samples.data(), samples.size()};
        };
        const auto& tables = medium.tables;
        FjScannerLutInput input{span(tables.epsC), span(tables.epsM), span(tables.epsY), span(tables.Ax), span(tables.Ay), span(tables.Az), span(tables.baseDensityMin), {}, {}, tables.K, tables.invYn, tables.hasBaseline ? 1u : 0u, static_cast<std::uint32_t>(medium.medium), resolution};
        std::copy_n(medium.range.min_cmy, 3, input.range_min_cmy);
        std::copy_n(medium.range.inv_max_cmy, 3, input.inverse_span_cmy);
        FjScannerLutFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = preparation.scanner_lut(preparation.user, &input, &view, &failure, &error);
        if (!diagnostic_valid(error, text) || status.api != FJ_API_NONE || status.native_code != 0)
            return invalid();
        const std::array arrays{view.log2_xyz, view.slope_c, view.slope_m, view.slope_y, view.cell_min, view.cell_max};
        if (status.category != FJ_STATUS_SUCCESS) {
            const bool backingAllocation = failure.reason == FJ_SCANNER_LUT_REASON_ALLOCATION && failure.buffer == 11;
            const auto category = failure.reason == FJ_SCANNER_LUT_REASON_RAW                                                          ? FJ_STATUS_UNSUPPORTED_INPUT
                                  : failure.reason == FJ_SCANNER_LUT_REASON_SIZE || failure.reason == FJ_SCANNER_LUT_REASON_ALLOCATION ? FJ_STATUS_ALLOCATION_FAILURE
                                  : failure.reason == FJ_SCANNER_LUT_REASON_PANIC                                                      ? FJ_STATUS_INTERNAL_FAILURE
                                                                                                                                       : FJ_STATUS_PREPARATION_FAILURE;
            if (failure.reason < 1 || failure.reason > 6 || status.category != category ||
                (failure.reason == FJ_SCANNER_LUT_REASON_NONFINITE ? failure.channel > 2 || failure.index == UINT64_MAX : failure.channel != UINT32_MAX || failure.index != UINT64_MAX) ||
                ((failure.reason == FJ_SCANNER_LUT_REASON_SIZE || failure.reason == FJ_SCANNER_LUT_REASON_ALLOCATION) ? !backingAllocation && (failure.buffer < 1 || failure.buffer > 10) : failure.buffer != 0) ||
                view.resolution || std::any_of(arrays.begin(), arrays.end(), [](FjFloatSpan samples) {
                    return samples.data || samples.count;
                }))
                return invalid();
            set_failure(rejected, status, std::string_view(text.data(), error.length));
            return false;
        }
        if (error.length || failure.reason || failure.channel != UINT32_MAX || failure.index != UINT64_MAX || failure.buffer || view.resolution != resolution || resolution < 2)
            return invalid();
        const auto side = static_cast<std::size_t>(resolution);
        if (side > std::numeric_limits<std::size_t>::max() / side / side / 3u)
            return invalid();
        const auto nodes = side * side * side * 3u;
        const auto cells = (side - 1u) * (side - 1u) * (side - 1u) * 3u;
        for (std::size_t i = 0; i < arrays.size(); ++i)
            if (!span_valid(arrays[i], i < 4 ? nodes : cells))
                return invalid();
        rejected = {};
        return true;
    }

    bool prepare_gaussian_kernel(const FjHostPreparation& preparation, const FjGaussianKernelInput& input, FjGaussianKernelView& view, Failure& rejected) {
        view = {};
        const auto invalid = [&]() -> bool {
            set_failure(rejected, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "Gaussian kernel projection invalid");
            return false;
        };
        if (!preparation.user || !preparation.scanner_lut || !preparation.gaussian)
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
