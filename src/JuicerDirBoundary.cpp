#include "JuicerDirBoundary.h"

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
#include "RenderValues.h"

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
        c.boundary_identity = component.boundaryIdentity;
        input.width = dimensions[0];
        input.height = dimensions[1];
        return input;
    }

    bool prepare(const FjHostPreparation* host, const FjDirBoundaryInput& input, FjDirBoundaryView& view, JuicerCuda::Failure& out) {
        if (!host || !host->dir_boundary) {
            JuicerCuda::set_failure(out, {FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "missing spatial DIR boundary callback");
            return false;
        }
        FjDirBoundaryFailure detail{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = host->dir_boundary(host->user, &input, &view, &detail, &error);
        if (status.category != FJ_STATUS_SUCCESS) {
            if (detail.reason > FJ_DIR_BOUNDARY_NONE && detail.reason < kMessages.size() &&
                status.api == FJ_API_NONE && status.native_code == 0) {
                reject(status, detail, out);
            } else {
                invalid(out);
            }
            return false;
        }
        if (status.api != FJ_API_NONE || status.native_code != 0 ||
            detail.reason != FJ_DIR_BOUNDARY_NONE || !admit(input, input.component.boundary_identity, view)) {
            invalid(out);
            return false;
        }
        return true;
    }
} // namespace JuicerDirBoundary
