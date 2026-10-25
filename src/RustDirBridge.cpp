#include "RustDirBridge.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace JuicerDir {
    namespace {
        template <typename T>
        bool positive_zero(T value) {
            if constexpr (sizeof(T) == sizeof(std::uint32_t)) {
                return std::bit_cast<std::uint32_t>(value) == 0;
            } else {
                return std::bit_cast<std::uint64_t>(value) == 0;
            }
        }

        bool same(const FjExtent& a, const FjExtent& b) {
            return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
        }

        bool cleared(const FjExtent& v) {
            return v.x == 0 && v.y == 0 && v.width == 0 && v.height == 0;
        }

        bool empty_iir(const FjDirGaussian& v) {
            return positive_zero(v.feedforward) && positive_zero(v.normalization_denominator) &&
                   positive_zero(v.feedforward_numerator) &&
                   std::all_of(std::begin(v.feedback), std::end(v.feedback), positive_zero<double>) &&
                   std::all_of(std::begin(v.feedback_numerators), std::end(v.feedback_numerators), positive_zero<double>);
        }

        bool empty_component(const FjDirGaussian& v, bool nativeDefaults) {
            return positive_zero(v.sigma_px) && positive_zero(v.weight) && v.radius == 0 &&
                   v.reference_operator == FJ_DIR_OPERATOR_NONE && empty_iir(v) &&
                   (nativeDefaults ? v.boundary_truncation_accuracy == 1.0e-11 && v.boundary_certification_tolerance == 5.0e-7 && v.boundary_derivation_version == 1
                                   : positive_zero(v.boundary_truncation_accuracy) && positive_zero(v.boundary_certification_tolerance) && v.boundary_derivation_version == 0);
        }

        bool cleared(const FjDirFrameMath& v) {
            return v.present == 0 && v.component_count == 0 && v.recipe_hash == 0 &&
                   cleared(v.render_extent) && cleared(v.full_frame_extent) && cleared(v.filter_domain_extent) &&
                   std::all_of(std::begin(v.components), std::end(v.components), [](const auto& c) {
                       return empty_component(c, false);
                   });
        }

        bool boundary_defaults(const FjDirGaussian& v) {
            return v.boundary_truncation_accuracy == 1.0e-11 &&
                   v.boundary_certification_tolerance == 5.0e-7 && v.boundary_derivation_version == 1;
        }

        bool complete_component(const FjDirGaussian& v) {
            if (!boundary_defaults(v)) {
                return false;
            }
            switch (v.reference_operator) {
                case FJ_DIR_OPERATOR_IDENTITY:
                    return v.radius == 0 && empty_iir(v);
                case FJ_DIR_OPERATOR_FIR_REFLECT:
                    return v.radius >= 1 && v.radius <= 9 && empty_iir(v);
                case FJ_DIR_OPERATOR_YVV_REFLECT:
                    return v.radius >= 9;
                default:
                    return false;
            }
        }

        bool associated(const FjDirFrameInput& input, const FjDirFrameMath& v) {
            return v.present == 1 && v.recipe_hash == input.recipe_hash &&
                   same(v.render_extent, input.render_extent) && same(v.full_frame_extent, input.full_frame_extent) &&
                   same(v.filter_domain_extent, input.full_frame_extent);
        }

        bool active_shape(const FjDirFrameInput& input, const FjDirFrameMath& v) {
            if (!associated(input, v) || (v.component_count != 1 && v.component_count != 4) ||
                !same(v.render_extent, v.full_frame_extent)) {
                return false;
            }
            for (unsigned i = 0; i < 4; ++i) {
                if (i < v.component_count ? !complete_component(v.components[i]) : !empty_component(v.components[i], true)) {
                    return false;
                }
            }
            return true;
        }

        bool failure_shape(const FjDirFrameInput& input, const FjDirFrameFailure& failure) {
            const auto& v = failure.trace;
            const auto field = failure.field;
            const auto component = failure.component;
            if (field < 1 || field > 12) {
                return false;
            }
            if (field <= 3) {
                return component == 4 && cleared(v);
            }
            if (!associated(input, v) || v.component_count > 4) {
                return false;
            }
            switch (field) {
                case FJ_DIR_FRAME_FIELD_PARTIAL_RENDER_WINDOW:
                    if (component != 4 || v.component_count != 0 || same(v.render_extent, v.full_frame_extent)) {
                        return false;
                    }
                    break;
                case FJ_DIR_FRAME_FIELD_CORE:
                    if (component != 0 || v.component_count != 0) {
                        return false;
                    }
                    break;
                case FJ_DIR_FRAME_FIELD_TAIL:
                    if (component != 4 || v.component_count != 1) {
                        return false;
                    }
                    break;
                case FJ_DIR_FRAME_FIELD_COMPONENT_VALUES:
                    if (component < 1 || component > 3 || v.component_count != component) {
                        return false;
                    }
                    break;
                case FJ_DIR_FRAME_FIELD_COMPONENT_CAPACITY:
                    if (component != 4 || v.component_count != 4) {
                        return false;
                    }
                    break;
                case FJ_DIR_FRAME_FIELD_EMPTY_PLAN:
                    if (component != 4 || (v.component_count != 1 && v.component_count != 4)) {
                        return false;
                    }
                    break;
                default:
                    if (component >= 4 || v.component_count != component) {
                        return false;
                    }
                    break;
            }
            for (unsigned i = 0; i < 4; ++i) {
                const auto& c = v.components[i];
                if (i < v.component_count) {
                    if (!complete_component(c)) {
                        return false;
                    }
                } else if (i == component && field >= 9 && field <= 11) {
                    if (!boundary_defaults(c)) {
                        return false;
                    }
                    if (field == 9 ? c.radius != 0 || c.reference_operator != 0 || !empty_iir(c)
                                   : c.radius < 9 || c.reference_operator != FJ_DIR_OPERATOR_YVV_REFLECT) {
                        return false;
                    }
                    if (field == 10 && (!positive_zero(c.feedforward) ||
                                        !std::all_of(std::begin(c.feedback), std::end(c.feedback), positive_zero<double>))) {
                        return false;
                    }
                } else if (!empty_component(c, true)) {
                    return false;
                }
            }
            return true;
        }

        MathOutcome::BoundaryFailure boundary_failure(FjStatus status, std::string_view text) {
            JuicerCuda::Failure failure;
            JuicerCuda::set_failure(failure, status, std::string(text));
            return {std::move(failure)};
        }
    } // namespace

    MathOutcome::MathOutcome(const FjDirFrameInput& input, Result result) : _input(input), _result(std::move(result)) {}

    const MathOutcome::Result& MathOutcome::result() const noexcept {
        return _result;
    }

    const FjDirFrameInput& MathOutcome::input() const noexcept {
        return _input;
    }

    MathOutcome MathOutcome::admit(
        const FjDirFrameInput& input, FjStatus status, const FjDirFrameMath& math, const FjDirFrameFailure& failure, const FjErrorBuffer& error, const std::array<char, 256>& text) {
        const auto reject = [&] {
            return MathOutcome(input, boundary_failure({FJ_STATUS_PREPARATION_FAILURE, FJ_API_NONE, 0}, "spatial DIR frame projection invalid"));
        };
        constexpr std::array categories{FJ_STATUS_SUCCESS, FJ_STATUS_UNSUPPORTED_INPUT, FJ_STATUS_PREPARATION_FAILURE, FJ_STATUS_INTERNAL_FAILURE};
        constexpr std::array fields{"", "pixel_size_um", "render_extent", "full_frame_extent", "partial_render_window", "core", "tail", "component_values", "component_capacity", "radius", "iir_denominator", "iir_output", "empty_plan"};
        if (failure.reason >= categories.size() || status.category != categories[failure.reason] || status.api != FJ_API_NONE || status.native_code != 0 ||
            error.data != text.data() || error.capacity != text.size() || error.length >= text.size() || text[error.length] != '\0') {
            return reject();
        }
        const std::string_view message(text.data(), error.length);
        if (failure.reason == FJ_DIR_FRAME_PREPARATION) {
            if (!cleared(math) || !failure_shape(input, failure) ||
                message != std::string("SpatialDirFrameFailure field=") + fields[failure.field]) {
                return reject();
            }
            return MathOutcome(input, MathematicalFailure{failure});
        }
        if (failure.field != 0 || failure.component != 4 || !cleared(failure.trace)) {
            return reject();
        }
        if (failure.reason != 0) {
            const std::string_view expected = failure.reason == FJ_DIR_FRAME_RAW ? "invalid spatial DIR frame arguments" : "spatial DIR frame preparation panicked";
            return cleared(math) && message == expected ? MathOutcome(input, boundary_failure(status, message)) : reject();
        }
        if (!message.empty() || input.active > 1) {
            return reject();
        }
        // The input activity comparison deliberately masks NaN/nonpositive size.
        if (input.active == 0 || !(input.diffusion_size_um > 0.0f)) {
            return cleared(math) ? MathOutcome(input, Inactive{}) : reject();
        }
        return active_shape(input, math) ? MathOutcome(input, Active{math}) : reject();
    }

    MathOutcome prepare_math(
        const DirCouplersRecipe& recipe, float pixelSizeUm, Spektrafilm::DirFrameExtent renderExtent, Spektrafilm::DirFrameExtent fullFrameExtent) {
        const FjDirFrameInput input{
            recipe.active ? 1u : 0u, recipe.diffusionSizeUm, recipe.diffusionTailUm, recipe.diffusionTailWeight, pixelSizeUm, {renderExtent.x, renderExtent.y, renderExtent.width, renderExtent.height}, {fullFrameExtent.x, fullFrameExtent.y, fullFrameExtent.width, fullFrameExtent.height}, recipe.hash};
        FjDirFrameMath math{};
        FjDirFrameFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_legacy_dir_frame_prepare(&input, &math, &failure, &error);
        return MathOutcome::admit(input, status, math, failure, error, text);
    }
} // namespace JuicerDir

namespace {
    static_assert(sizeof(FjDirFrameInput) == 64 && alignof(FjDirFrameInput) == 8);
    static_assert(offsetof(FjDirFrameInput, active) == 0);
    static_assert(std::is_same_v<decltype(FjDirFrameInput::active), uint32_t>);
    static_assert(offsetof(FjDirFrameInput, diffusion_size_um) == 4);
    static_assert(std::is_same_v<decltype(FjDirFrameInput::diffusion_size_um), float>);
    static_assert(offsetof(FjDirFrameInput, diffusion_tail_um) == 8);
    static_assert(std::is_same_v<decltype(FjDirFrameInput::diffusion_tail_um), float>);
    static_assert(offsetof(FjDirFrameInput, diffusion_tail_weight) == 12);
    static_assert(std::is_same_v<decltype(FjDirFrameInput::diffusion_tail_weight), float>);
    static_assert(offsetof(FjDirFrameInput, pixel_size_um) == 16);
    static_assert(std::is_same_v<decltype(FjDirFrameInput::pixel_size_um), float>);
    static_assert(offsetof(FjDirFrameInput, render_extent) == 20);
    static_assert(std::is_same_v<decltype(FjDirFrameInput::render_extent), FjExtent>);
    static_assert(offsetof(FjDirFrameInput, full_frame_extent) == 36);
    static_assert(std::is_same_v<decltype(FjDirFrameInput::full_frame_extent), FjExtent>);
    static_assert(offsetof(FjDirFrameInput, recipe_hash) == 56);
    static_assert(std::is_same_v<decltype(FjDirFrameInput::recipe_hash), uint64_t>);
    static_assert(sizeof(FjDirFrameMath) == 512 && alignof(FjDirFrameMath) == 8);
    static_assert(offsetof(FjDirFrameMath, present) == 0);
    static_assert(std::is_same_v<decltype(FjDirFrameMath::present), uint32_t>);
    static_assert(offsetof(FjDirFrameMath, component_count) == 4);
    static_assert(std::is_same_v<decltype(FjDirFrameMath::component_count), uint32_t>);
    static_assert(offsetof(FjDirFrameMath, recipe_hash) == 8);
    static_assert(std::is_same_v<decltype(FjDirFrameMath::recipe_hash), uint64_t>);
    static_assert(offsetof(FjDirFrameMath, render_extent) == 16);
    static_assert(std::is_same_v<decltype(FjDirFrameMath::render_extent), FjExtent>);
    static_assert(offsetof(FjDirFrameMath, full_frame_extent) == 32);
    static_assert(std::is_same_v<decltype(FjDirFrameMath::full_frame_extent), FjExtent>);
    static_assert(offsetof(FjDirFrameMath, filter_domain_extent) == 48);
    static_assert(std::is_same_v<decltype(FjDirFrameMath::filter_domain_extent), FjExtent>);
    static_assert(offsetof(FjDirFrameMath, components) == 64);
    static_assert(std::is_same_v<decltype(FjDirFrameMath::components), FjDirGaussian[4]>);
    static_assert(sizeof(FjDirFrameFailure) == 528 && alignof(FjDirFrameFailure) == 8);
    static_assert(offsetof(FjDirFrameFailure, reason) == 0);
    static_assert(std::is_same_v<decltype(FjDirFrameFailure::reason), uint32_t>);
    static_assert(offsetof(FjDirFrameFailure, field) == 4);
    static_assert(std::is_same_v<decltype(FjDirFrameFailure::field), uint32_t>);
    static_assert(offsetof(FjDirFrameFailure, component) == 8);
    static_assert(std::is_same_v<decltype(FjDirFrameFailure::component), uint32_t>);
    static_assert(offsetof(FjDirFrameFailure, trace) == 16);
    static_assert(std::is_same_v<decltype(FjDirFrameFailure::trace), FjDirFrameMath>);
    static_assert(sizeof(FjExtent) == 16 && alignof(FjExtent) == 4);
    static_assert(offsetof(FjExtent, x) == 0);
    static_assert(std::is_same_v<decltype(FjExtent::x), int32_t>);
    static_assert(offsetof(FjExtent, y) == 4);
    static_assert(std::is_same_v<decltype(FjExtent::y), int32_t>);
    static_assert(offsetof(FjExtent, width) == 8);
    static_assert(std::is_same_v<decltype(FjExtent::width), int32_t>);
    static_assert(offsetof(FjExtent, height) == 12);
    static_assert(std::is_same_v<decltype(FjExtent::height), int32_t>);
    static_assert(sizeof(FjDirGaussian) == 112 && alignof(FjDirGaussian) == 8);
    static_assert(offsetof(FjDirGaussian, sigma_px) == 0);
    static_assert(std::is_same_v<decltype(FjDirGaussian::sigma_px), float>);
    static_assert(offsetof(FjDirGaussian, weight) == 4);
    static_assert(std::is_same_v<decltype(FjDirGaussian::weight), float>);
    static_assert(offsetof(FjDirGaussian, radius) == 8);
    static_assert(std::is_same_v<decltype(FjDirGaussian::radius), int32_t>);
    static_assert(offsetof(FjDirGaussian, reference_operator) == 12);
    static_assert(std::is_same_v<decltype(FjDirGaussian::reference_operator), uint32_t>);
    static_assert(offsetof(FjDirGaussian, feedforward) == 16);
    static_assert(std::is_same_v<decltype(FjDirGaussian::feedforward), double>);
    static_assert(offsetof(FjDirGaussian, feedback) == 24);
    static_assert(std::is_same_v<decltype(FjDirGaussian::feedback), double[3]>);
    static_assert(offsetof(FjDirGaussian, normalization_denominator) == 48);
    static_assert(std::is_same_v<decltype(FjDirGaussian::normalization_denominator), double>);
    static_assert(offsetof(FjDirGaussian, feedforward_numerator) == 56);
    static_assert(std::is_same_v<decltype(FjDirGaussian::feedforward_numerator), double>);
    static_assert(offsetof(FjDirGaussian, feedback_numerators) == 64);
    static_assert(std::is_same_v<decltype(FjDirGaussian::feedback_numerators), double[3]>);
    static_assert(offsetof(FjDirGaussian, boundary_truncation_accuracy) == 88);
    static_assert(std::is_same_v<decltype(FjDirGaussian::boundary_truncation_accuracy), double>);
    static_assert(offsetof(FjDirGaussian, boundary_certification_tolerance) == 96);
    static_assert(std::is_same_v<decltype(FjDirGaussian::boundary_certification_tolerance), double>);
    static_assert(offsetof(FjDirGaussian, boundary_derivation_version) == 104);
    static_assert(std::is_same_v<decltype(FjDirGaussian::boundary_derivation_version), uint32_t>);
    static_assert(sizeof(FjSpatialDir) == 520 && alignof(FjSpatialDir) == 8);
    static_assert(offsetof(FjSpatialDir, render_extent) == 0);
    static_assert(std::is_same_v<decltype(FjSpatialDir::render_extent), FjExtent>);
    static_assert(offsetof(FjSpatialDir, full_frame_extent) == 16);
    static_assert(std::is_same_v<decltype(FjSpatialDir::full_frame_extent), FjExtent>);
    static_assert(offsetof(FjSpatialDir, filter_domain_extent) == 32);
    static_assert(std::is_same_v<decltype(FjSpatialDir::filter_domain_extent), FjExtent>);
    static_assert(offsetof(FjSpatialDir, component_count) == 48);
    static_assert(std::is_same_v<decltype(FjSpatialDir::component_count), size_t>);
    static_assert(offsetof(FjSpatialDir, components) == 56);
    static_assert(std::is_same_v<decltype(FjSpatialDir::components), FjDirGaussian[4]>);
    static_assert(offsetof(FjSpatialDir, recipe_hash) == 504);
    static_assert(std::is_same_v<decltype(FjSpatialDir::recipe_hash), uint64_t>);
    static_assert(offsetof(FjSpatialDir, hash) == 512);
    static_assert(std::is_same_v<decltype(FjSpatialDir::hash), uint64_t>);
    static_assert(FJ_DIR_FRAME_NONE == 0U);
    static_assert(FJ_DIR_FRAME_RAW == 1U);
    static_assert(FJ_DIR_FRAME_PREPARATION == 2U);
    static_assert(FJ_DIR_FRAME_PANIC == 3U);
    static_assert(FJ_DIR_FRAME_FIELD_NONE == 0U);
    static_assert(FJ_DIR_FRAME_FIELD_PIXEL_SIZE_UM == 1U);
    static_assert(FJ_DIR_FRAME_FIELD_RENDER_EXTENT == 2U);
    static_assert(FJ_DIR_FRAME_FIELD_FULL_FRAME_EXTENT == 3U);
    static_assert(FJ_DIR_FRAME_FIELD_PARTIAL_RENDER_WINDOW == 4U);
    static_assert(FJ_DIR_FRAME_FIELD_CORE == 5U);
    static_assert(FJ_DIR_FRAME_FIELD_TAIL == 6U);
    static_assert(FJ_DIR_FRAME_FIELD_COMPONENT_VALUES == 7U);
    static_assert(FJ_DIR_FRAME_FIELD_COMPONENT_CAPACITY == 8U);
    static_assert(FJ_DIR_FRAME_FIELD_RADIUS == 9U);
    static_assert(FJ_DIR_FRAME_FIELD_IIR_DENOMINATOR == 10U);
    static_assert(FJ_DIR_FRAME_FIELD_IIR_OUTPUT == 11U);
    static_assert(FJ_DIR_FRAME_FIELD_EMPTY_PLAN == 12U);
    static_assert(FJ_DIR_FRAME_NO_COMPONENT == 4U);
    static_assert(FJ_DIR_OPERATOR_NONE == 0U);
    static_assert(FJ_DIR_OPERATOR_IDENTITY == 1U);
    static_assert(FJ_DIR_OPERATOR_FIR_REFLECT == 2U);
    static_assert(FJ_DIR_OPERATOR_YVV_REFLECT == 3U);
    static_assert(std::is_same_v<decltype(&fj_legacy_dir_frame_prepare), FjStatus (*)(const FjDirFrameInput*, FjDirFrameMath*, FjDirFrameFailure*, FjErrorBuffer*)>);
} // namespace
