#include "RustDirBoundaryBridge.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

#include "Cuda/JuicerCudaFailure.h"
#include "RenderRecipe.h"

namespace JuicerDirBoundary {
    namespace {
        constexpr std::array<const char*, 14> kMessages{
            "",
            "spatial DIR reflected-boundary arguments are invalid",
            "spatial DIR reflected-boundary request is invalid",
            "spatial DIR reflected-boundary axis size is invalid",
            "spatial DIR reflected-boundary policy is invalid",
            "spatial DIR causal reflected-boundary solve is singular",
            "spatial DIR causal reflected-boundary weights are nonfinite",
            "spatial DIR causal reflected-boundary certification failed",
            "spatial DIR reflected terminal system escaped its tail",
            "spatial DIR reflected terminal solve is singular",
            "spatial DIR reflected terminal certification failed",
            "spatial DIR reflected-boundary storage size is invalid",
            "spatial DIR reflected-boundary allocation failed",
            "spatial DIR reflected-boundary preparation panicked"};

        void invalid(JuicerCuda::Failure& failure) {
            JuicerCuda::set_failure(failure, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "spatial DIR reflected-boundary view is invalid");
        }

        bool envelope(const FjDirBoundaryInput& input, FjStatus status, const FjDirBoundaryFailure& failure, const FjErrorBuffer& error, const std::array<char, 256>& text) {
            constexpr std::array<std::uint32_t, 14> categories{0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 8};
            if (failure.reason >= categories.size() || status.category != categories[failure.reason] || status.api != FJ_API_NONE || status.native_code != 0 ||
                error.data != text.data() || error.capacity != text.size() || error.length >= text.size() || text[error.length] != '\0' ||
                std::string_view(text.data(), error.length) != kMessages[failure.reason])
                return false;
            if (failure.axis > 2 || (failure.axis == 0 ? failure.length != 0 : failure.length != (failure.axis == 1 ? input.width : input.height)))
                return false;
            if (failure.reason == FJ_DIR_BOUNDARY_CAUSAL_CERTIFICATION)
                return failure.axis != 0 && failure.row < 3 &&
                       std::bit_cast<std::uint32_t>(failure.sigma_px) == std::bit_cast<std::uint32_t>(input.component.sigma_px) &&
                       std::bit_cast<std::uint64_t>(failure.tolerance) == std::bit_cast<std::uint64_t>(input.component.boundary_certification_tolerance);
            if (failure.row != 3 || std::bit_cast<std::uint32_t>(failure.sigma_px) || std::bit_cast<std::uint64_t>(failure.sum) || std::bit_cast<std::uint64_t>(failure.tolerance))
                return false;
            if (failure.reason <= 2 || failure.reason == 11 || failure.reason == 13)
                return failure.axis == 0;
            return failure.reason == 12 || failure.axis != 0;
        }

        void reject(FjStatus status, const FjDirBoundaryFailure& failure, JuicerCuda::Failure& out) {
            if (failure.reason == FJ_DIR_BOUNDARY_CAUSAL_CERTIFICATION) {
                std::ostringstream detail;
                detail << kMessages[failure.reason]
                       << " length=" << failure.length
                       << " sigma=" << failure.sigma_px
                       << " sum=" << failure.sum
                       << " tolerance=" << failure.tolerance;
                JuicerCuda::set_failure(out, status, detail.str());
            } else {
                JuicerCuda::set_failure(out, status, kMessages[failure.reason]);
            }
        }

        bool empty(const FjDirBoundaryView& view) {
            return !view.weights && !view.count && !view.identity && !view.width && !view.height && !view.horizontal_length && !view.vertical_length &&
                   !view.horizontal_terminal_size && !view.vertical_terminal_size &&
                   std::all_of(std::begin(view.horizontal_terminal), std::end(view.horizontal_terminal), [](double value) {
                       return std::bit_cast<std::uint64_t>(value) == 0;
                   }) &&
                   std::all_of(std::begin(view.vertical_terminal), std::end(view.vertical_terminal), [](double value) {
                       return std::bit_cast<std::uint64_t>(value) == 0;
                   });
        }

        bool terminal(const double (&matrix)[9], int size) {
            const auto extent = static_cast<std::size_t>(size);
            for (std::size_t index = 0; index < 9; ++index) {
                if (index < extent * extent) {
                    if (!std::isfinite(matrix[index]))
                        return false;
                } else if (std::bit_cast<std::uint64_t>(matrix[index]) != 0) {
                    return false;
                }
            }
            return true;
        }

        bool admit(const FjDirBoundaryInput& input, std::uint64_t key, const FjDirBoundaryView& view) {
            if (view.identity != key || view.width != input.width || view.height != input.height || input.width <= 0 || input.height <= 0 ||
                view.horizontal_length < std::min(3, input.width) || view.horizontal_length > input.width || view.horizontal_length > std::numeric_limits<int>::max() / 3 ||
                view.vertical_length < std::min(3, input.height) || view.vertical_length > input.height || view.vertical_length > std::numeric_limits<int>::max() / 3 ||
                view.horizontal_terminal_size != std::min(3, input.width) || view.vertical_terminal_size != std::min(3, input.height))
                return false;
            const auto horizontal = static_cast<std::size_t>(view.horizontal_length), vertical = static_cast<std::size_t>(view.vertical_length);
            if (horizontal > std::numeric_limits<std::size_t>::max() - vertical || horizontal + vertical > std::numeric_limits<std::size_t>::max() / 3)
                return false;
            const auto count = 3 * (horizontal + vertical);
            if (view.count != count || count > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(double))
                return false;
            const auto address = reinterpret_cast<std::uintptr_t>(view.weights);
            return address && address % alignof(double) == 0 && address <= std::numeric_limits<std::uintptr_t>::max() - count * sizeof(double) &&
                   terminal(view.horizontal_terminal, view.horizontal_terminal_size) && terminal(view.vertical_terminal, view.vertical_terminal_size);
        }
    } // namespace

    FjDirBoundaryInput project_input(const Spektrafilm::DirGaussianComponentPlan& component, std::array<int, 2> dimensions) {
        FjDirBoundaryInput input{};
        auto& c = input.component;
        c.sigma_px = component.sigmaPixels;
        c.weight = component.weight;
        c.radius = component.radius;
        c.reference_operator = static_cast<std::uint32_t>(component.referenceOperator);
        c.feedforward = component.iir.feedforward;
        std::copy(component.iir.feedback.begin(), component.iir.feedback.end(), c.feedback);
        c.normalization_denominator = component.iir.normalizationDenominator;
        c.feedforward_numerator = component.iir.feedforwardNumerator;
        std::copy(component.iir.feedbackNumerators.begin(), component.iir.feedbackNumerators.end(), c.feedback_numerators);
        c.boundary_truncation_accuracy = component.boundaryTruncationAccuracy;
        c.boundary_certification_tolerance = component.boundaryCertificationTolerance;
        c.boundary_derivation_version = component.boundaryDerivationVersion;
        input.width = dimensions[0];
        input.height = dimensions[1];
        return input;
    }

    bool describe(const FjDirBoundaryInput& input, std::uint64_t& key, JuicerCuda::Failure& out) {
        out = {};
        key = 0;
        FjDirBoundaryFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_legacy_dir_boundary_describe(&input, &key, &failure, &error);
        if (!envelope(input, status, failure, error, text) || (failure.reason != 0 && failure.reason != 1 && failure.reason != 2 && failure.reason != 13) ||
            (status.category != FJ_STATUS_SUCCESS && key != 0)) {
            key = 0;
            invalid(out);
            return false;
        }
        if (status.category != FJ_STATUS_SUCCESS) {
            reject(status, failure, out);
            return false;
        }
        return true;
    }

    void Boundary::Release::operator()(FjDirBoundaryOwner* owner) const noexcept {
        (void)fj_legacy_dir_boundary_release(owner, nullptr);
    }
    Boundary::Boundary(std::unique_ptr<FjDirBoundaryOwner, Release> owner, FjDirBoundaryView view) noexcept : _owner(std::move(owner)), _view(view) {}
    Boundary::Boundary(Boundary&&) noexcept = default;
    Boundary& Boundary::operator=(Boundary&&) noexcept = default;
    FjDirBoundaryView Boundary::view() const noexcept {
        return _owner ? _view : FjDirBoundaryView{};
    }

    std::optional<Boundary> Boundary::acquire(const FjDirBoundaryInput& input, std::uint64_t key, JuicerCuda::Failure& out) {
        out = {};
        FjDirBoundaryOwner* raw = nullptr;
        FjDirBoundaryFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        auto status = fj_legacy_dir_boundary_acquire(&input, &raw, &failure, &error);
        std::unique_ptr<FjDirBoundaryOwner, Release> owner(raw);
        if (!envelope(input, status, failure, error, text) || (status.category == FJ_STATUS_SUCCESS ? !raw : raw != nullptr)) {
            invalid(out);
            return std::nullopt;
        }
        if (status.category != FJ_STATUS_SUCCESS) {
            reject(status, failure, out);
            return std::nullopt;
        }
        FjDirBoundaryView view{};
        status = fj_legacy_dir_boundary_view(owner.get(), &view, &failure, &error);
        if (!envelope(input, status, failure, error, text) || (failure.reason != 0 && failure.reason != 1 && failure.reason != 13)) {
            invalid(out);
            return std::nullopt;
        }
        if (status.category != FJ_STATUS_SUCCESS) {
            if (empty(view))
                reject(status, failure, out);
            else
                invalid(out);
            return std::nullopt;
        }
        if (!admit(input, key, view)) {
            invalid(out);
            return std::nullopt;
        }
        return Boundary(std::move(owner), view);
    }
} // namespace JuicerDirBoundary
