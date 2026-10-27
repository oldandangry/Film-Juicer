#include "RustEffectsFrameBridge.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <type_traits>

static_assert(sizeof(FjEffectsFrameInput) == 776 && alignof(FjEffectsFrameInput) == 8);
static_assert(sizeof(FjEffectsFrameMath) == 208 && alignof(FjEffectsFrameMath) == 8);
static_assert(sizeof(FjEffectsFrameFailure) == 4 && alignof(FjEffectsFrameFailure) == 4);
static_assert(offsetof(FjEffectsFrameInput, active) == 0);
static_assert(offsetof(FjEffectsFrameInput, render_extent) == 4);
static_assert(offsetof(FjEffectsFrameInput, pixel_definition) == 20);
static_assert(offsetof(FjEffectsFrameInput, pixel_size_um) == 36);
static_assert(offsetof(FjEffectsFrameInput, film_format_long_edge_mm) == 40);
static_assert(offsetof(FjEffectsFrameInput, canonical_x) == 48);
static_assert(offsetof(FjEffectsFrameInput, canonical_y) == 56);
static_assert(offsetof(FjEffectsFrameInput, canonical_width) == 64);
static_assert(offsetof(FjEffectsFrameInput, canonical_height) == 72);
static_assert(offsetof(FjEffectsFrameInput, scale_x) == 80);
static_assert(offsetof(FjEffectsFrameInput, scale_y) == 88);
static_assert(offsetof(FjEffectsFrameInput, pixel_aspect_ratio) == 96);
static_assert(offsetof(FjEffectsFrameInput, frame_time) == 104);
static_assert(offsetof(FjEffectsFrameInput, frame_rate) == 112);
static_assert(offsetof(FjEffectsFrameInput, gate_weave_amount) == 120);
static_assert(offsetof(FjEffectsFrameInput, session_seed) == 128);
static_assert(offsetof(FjEffectsFrameInput, clip_token) == 136);
static_assert(offsetof(FjEffectsFrameInput, recipe_hash) == 144);
static_assert(offsetof(FjEffectsFrameInput, film_dust) == 152);
static_assert(offsetof(FjEffectsFrameInput, film_scratch) == 320);
static_assert(offsetof(FjEffectsFrameInput, gate_dust) == 464);
static_assert(offsetof(FjEffectsFrameInput, gate_scratch) == 632);
static_assert(offsetof(FjEffectsFrameMath, present) == 0);
static_assert(offsetof(FjEffectsFrameMath, flags) == 4);
static_assert(offsetof(FjEffectsFrameMath, render_extent) == 8);
static_assert(offsetof(FjEffectsFrameMath, full_frame_extent) == 24);
static_assert(offsetof(FjEffectsFrameMath, origins) == 40);
static_assert(offsetof(FjEffectsFrameMath, sample_step_x_mm) == 136);
static_assert(offsetof(FjEffectsFrameMath, sample_step_y_mm) == 140);
static_assert(offsetof(FjEffectsFrameMath, roi_offset_x) == 144);
static_assert(offsetof(FjEffectsFrameMath, roi_offset_y) == 148);
static_assert(offsetof(FjEffectsFrameMath, gate_width) == 152);
static_assert(offsetof(FjEffectsFrameMath, gate_height) == 156);
static_assert(offsetof(FjEffectsFrameMath, session_seed) == 160);
static_assert(offsetof(FjEffectsFrameMath, clip_token) == 168);
static_assert(offsetof(FjEffectsFrameMath, weave_dx_px) == 176);
static_assert(offsetof(FjEffectsFrameMath, weave_dy_px) == 180);
static_assert(offsetof(FjEffectsFrameMath, weave_cos_rot) == 184);
static_assert(offsetof(FjEffectsFrameMath, weave_sin_rot) == 188);
static_assert(offsetof(FjEffectsFrameMath, recipe_hash) == 192);
static_assert(offsetof(FjEffectsFrameMath, hash) == 200);
static_assert(offsetof(FjEffectsFrameFailure, reason) == 0);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::active), uint32_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::render_extent), FjExtent>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::pixel_definition), FjExtent>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::pixel_size_um), float>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::film_format_long_edge_mm), float>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::canonical_x), double>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::canonical_y), double>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::canonical_width), double>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::canonical_height), double>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::scale_x), double>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::scale_y), double>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::pixel_aspect_ratio), double>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::frame_time), double>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::frame_rate), double>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::gate_weave_amount), double>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::session_seed), uint64_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::clip_token), uint64_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::recipe_hash), uint64_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::film_dust), FjDust>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::film_scratch), FjScratch>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::gate_dust), FjDust>);
static_assert(std::is_same_v<decltype(FjEffectsFrameInput::gate_scratch), FjScratch>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::present), uint32_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::flags), uint32_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::render_extent), FjExtent>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::full_frame_extent), FjExtent>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::origins), FjDefectOrigin[4]>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::sample_step_x_mm), float>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::sample_step_y_mm), float>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::roi_offset_x), int32_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::roi_offset_y), int32_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::gate_width), int32_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::gate_height), int32_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::session_seed), uint64_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::clip_token), uint64_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::weave_dx_px), float>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::weave_dy_px), float>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::weave_cos_rot), float>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::weave_sin_rot), float>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::recipe_hash), uint64_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameMath::hash), uint64_t>);
static_assert(std::is_same_v<decltype(FjEffectsFrameFailure::reason), uint32_t>);

namespace JuicerEffects {
    namespace {
        constexpr const char* kInvalidResult = "ResourceDescriptorMismatch phase=effects_descriptor field=rust_result";
        bool zero(float v) {
            return std::bit_cast<std::uint32_t>(v) == 0;
        }
        bool empty(const FjExtent& v) {
            return v.x == 0 && v.y == 0 && v.width == 0 && v.height == 0;
        }
        bool empty(const FjDefectOrigin& v) {
            return v.cell_x == 0 && v.cell_y == 0 && zero(v.local_x_mm) && zero(v.local_y_mm);
        }
        bool empty(const FjEffectsFrameMath& v) {
            return v.present == 0 && v.flags == 0 && empty(v.render_extent) && empty(v.full_frame_extent) && std::all_of(std::begin(v.origins), std::end(v.origins), [](const auto& o) {
                       return empty(o);
                   }) &&
                   zero(v.sample_step_x_mm) && zero(v.sample_step_y_mm) && v.roi_offset_x == 0 && v.roi_offset_y == 0 && v.gate_width == 0 && v.gate_height == 0 && v.session_seed == 0 && v.clip_token == 0 && zero(v.weave_dx_px) && zero(v.weave_dy_px) && zero(v.weave_cos_rot) && zero(v.weave_sin_rot) && v.recipe_hash == 0 && v.hash == 0;
        }
        template <typename Extent>
        bool same_extent(const FjExtent& a, const Extent& b) {
            return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
        }
        FjDust project_dust(const DefectDustRecipe& v) {
            FjDust out{};
            out.cell_width_mm = v.cellWidthMm;
            out.cell_height_mm = v.cellHeightMm;
            out.slot_probability = v.slotProbability;
            out.softness_min_mm = v.softnessMinMm;
            out.softness_max_mm = v.softnessMaxMm;
            out.softness_size_cap_fraction = v.softnessSizeCapFraction;
            out.support_x_mm = v.supportXMm;
            out.support_y_mm = v.supportYMm;
            out.fiber_fraction = v.fiberFraction;
            out.fiber_drift_fraction = v.fiberDriftFraction;
            out.fiber_first_knot_min = v.fiberFirstKnotMin;
            out.fiber_first_knot_max = v.fiberFirstKnotMax;
            out.fiber_second_knot_min = v.fiberSecondKnotMin;
            out.fiber_second_knot_max = v.fiberSecondKnotMax;
            out.fiber_interior_width_min_fraction = v.fiberInteriorWidthMinFraction;
            out.fiber_interior_width_max_fraction = v.fiberInteriorWidthMaxFraction;
            out.diameter_min_mm = v.diameterMinMm;
            out.diameter_bulk_max_mm = v.diameterBulkMaxMm;
            out.diameter_max_mm = v.diameterMaxMm;
            out.diameter_tail_fraction = v.diameterTailFraction;
            out.fiber_length_min_mm = v.fiberLengthMinMm;
            out.fiber_length_max_mm = v.fiberLengthMaxMm;
            out.fiber_width_min_mm = v.fiberWidthMinMm;
            out.fiber_width_max_mm = v.fiberWidthMaxMm;
            out.opacity_faint_cumulative = v.opacityFaintCumulative;
            out.opacity_intermediate_cumulative = v.opacityIntermediateCumulative;
            out.compact_opacity_min = v.compactOpacityMin;
            out.compact_opacity_faint_end = v.compactOpacityFaintEnd;
            out.compact_opacity_intermediate_end = v.compactOpacityIntermediateEnd;
            out.compact_opacity_max = v.compactOpacityMax;
            out.fiber_opacity_min = v.fiberOpacityMin;
            out.fiber_opacity_faint_end = v.fiberOpacityFaintEnd;
            out.fiber_opacity_intermediate_end = v.fiberOpacityIntermediateEnd;
            out.fiber_opacity_max = v.fiberOpacityMax;
            out.compact_dominant_aspect_min = v.compactDominantAspectMin;
            out.compact_dominant_aspect_max = v.compactDominantAspectMax;
            out.compact_subsidiary_scale_min = v.compactSubsidiaryScaleMin;
            out.compact_subsidiary_scale_max = v.compactSubsidiaryScaleMax;
            out.compact_subsidiary_aspect_min = v.compactSubsidiaryAspectMin;
            out.compact_subsidiary_aspect_max = v.compactSubsidiaryAspectMax;
            out.compact_subsidiary_offset_max = v.compactSubsidiaryOffsetMax;
            out.compact_subsidiary_angle_max_radians = v.compactSubsidiaryAngleMaxRadians;
            return out;
        }
        FjScratch project_scratch(const DefectScratchRecipe& v) {
            FjScratch out{};
            out.cell_width_mm = v.cellWidthMm;
            out.cell_height_mm = v.cellHeightMm;
            out.slot_probability = v.slotProbability;
            out.softness_min_mm = v.softnessMinMm;
            out.softness_max_mm = v.softnessMaxMm;
            out.softness_size_cap_fraction = v.softnessSizeCapFraction;
            out.support_x_mm = v.supportXMm;
            out.support_y_mm = v.supportYMm;
            out.length_min_mm = v.lengthMinMm;
            out.length_bulk_max_mm = v.lengthBulkMaxMm;
            out.length_max_mm = v.lengthMaxMm;
            out.length_tail_fraction = v.lengthTailFraction;
            out.width_min_mm = v.widthMinMm;
            out.width_bulk_max_mm = v.widthBulkMaxMm;
            out.width_max_mm = v.widthMaxMm;
            out.width_tail_fraction = v.widthTailFraction;
            out.drift_fraction = v.driftFraction;
            out.first_knot_min = v.firstKnotMin;
            out.first_knot_max = v.firstKnotMax;
            out.second_knot_min = v.secondKnotMin;
            out.second_knot_max = v.secondKnotMax;
            out.interior_width_min_fraction = v.interiorWidthMinFraction;
            out.interior_width_max_fraction = v.interiorWidthMaxFraction;
            out.interior_depth_min_fraction = v.interiorDepthMinFraction;
            out.interior_depth_max_fraction = v.interiorDepthMaxFraction;
            out.endpoint_abrupt_probability = v.endpointAbruptProbability;
            out.interruption_probability = v.interruptionProbability;
            out.gap_center_min = v.gapCenterMin;
            out.gap_center_max = v.gapCenterMax;
            out.gap_span_min = v.gapSpanMin;
            out.gap_span_max = v.gapSpanMax;
            out.scuff_probability = v.scuffProbability;
            out.scuff_length_max_mm = v.scuffLengthMaxMm;
            out.scuff_angle_max_radians = v.scuffAngleMaxRadians;
            out.strength_min = v.strengthMin;
            out.strength_max = v.strengthMax;
            return out;
        }
        bool admit_math(const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput& input, const FjEffectsFrameMath& v) {
            if (!input.recipe)
                return false;
            if (!input.recipe->active)
                return empty(v);
            if (v.present != 1 || (v.flags & ~31u) != 0 || !same_extent(v.render_extent, input.renderExtent) || !same_extent(v.full_frame_extent, input.geometry.pixelDefinition) || v.render_extent.width <= 0 || v.render_extent.height <= 0 || v.full_frame_extent.width <= 0 || v.full_frame_extent.height <= 0 || v.recipe_hash != input.recipe->hash || v.session_seed != input.sessionSeed || v.clip_token != input.clipToken)
                return false;
            const bool weave = (v.flags & FJ_EFFECTS_WEAVE) != 0;
            const bool film = (v.flags & FJ_EFFECTS_FILM) != 0;
            const bool gate = (v.flags & FJ_EFFECTS_GATE_TRANSMITTANCE) != 0;
            const bool output = (v.flags & FJ_EFFECTS_GATE_OUTPUT) != 0;
            const bool full = (v.flags & FJ_EFFECTS_FULL_FRAME) != 0;
            if (output != (weave || gate) || full != weave || (weave && !same_extent(v.render_extent, v.full_frame_extent)))
                return false;
            if (film || gate) {
                if (!std::isfinite(v.sample_step_x_mm) || !(v.sample_step_x_mm > 0) || !std::isfinite(v.sample_step_y_mm) || !(v.sample_step_y_mm > 0))
                    return false;
            } else if (!zero(v.sample_step_x_mm) || !zero(v.sample_step_y_mm))
                return false;
            if (v.roi_offset_x < 0 || v.roi_offset_y < 0)
                return false;
            if (gate ? (v.gate_width <= 0 || v.gate_height <= 0 || v.gate_width > v.full_frame_extent.width || v.gate_height > v.full_frame_extent.height)
                     : (v.gate_width != 0 || v.gate_height != 0))
                return false;
            if (!std::isfinite(v.weave_dx_px) || !std::isfinite(v.weave_dy_px) || !std::isfinite(v.weave_cos_rot) || !std::isfinite(v.weave_sin_rot))
                return false;
            if (!weave && (!zero(v.weave_dx_px) || !zero(v.weave_dy_px) || v.weave_cos_rot != 1.0f || !zero(v.weave_sin_rot)))
                return false;
            for (const auto& o : v.origins) {
                if (std::abs(static_cast<double>(o.cell_x)) >= 4503599627370496.0 || std::abs(static_cast<double>(o.cell_y)) >= 4503599627370496.0 || !std::isfinite(o.local_x_mm) || o.local_x_mm < 0 || !std::isfinite(o.local_y_mm) || o.local_y_mm < 0)
                    return false;
            }
            return true;
        }
    } // namespace

    bool project_input(const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput& input, FjEffectsFrameInput& out, std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        if (!input.recipe) {
            diagnostic = "ResourceDescriptorMismatch phase=effects_descriptor field=geometry_or_policy";
            return false;
        }
        out.recipe_hash = input.recipe->hash;
        if (!input.recipe->active)
            return true;
        out.active = 1;
        const auto& g = input.geometry;
        out.render_extent = {input.renderExtent.x, input.renderExtent.y, input.renderExtent.width, input.renderExtent.height};
        out.pixel_definition = {g.pixelDefinition.x, g.pixelDefinition.y, g.pixelDefinition.width, g.pixelDefinition.height};
        out.pixel_size_um = input.pixelSizeUm;
        out.film_format_long_edge_mm = input.filmFormatLongEdgeMm;
        out.canonical_x = g.canonicalX;
        out.canonical_y = g.canonicalY;
        out.canonical_width = g.canonicalWidth;
        out.canonical_height = g.canonicalHeight;
        out.scale_x = g.scaleX;
        out.scale_y = g.scaleY;
        out.pixel_aspect_ratio = g.pixelAspectRatio;
        out.frame_time = input.frameTime;
        out.frame_rate = input.frameRate;
        out.gate_weave_amount = input.recipe->gateWeaveAmount;
        out.session_seed = input.sessionSeed;
        out.clip_token = input.clipToken;
        out.film_dust = project_dust(input.recipe->filmDust);
        out.film_scratch = project_scratch(input.recipe->filmScratch);
        out.gate_dust = project_dust(input.recipe->gateDust);
        out.gate_scratch = project_scratch(input.recipe->gateScratch);
        return true;
    }

    bool accept_result(const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput& input, FjStatus status, const FjEffectsFrameMath& math, const FjEffectsFrameFailure& failure, const FjErrorBuffer& error, const std::array<char, 256>& text, Spektrafilm::FilmJuicerEffectsFrameDescriptor& out, std::string& diagnostic) {
        out = {};
        diagnostic.clear();
        const bool failureShape = status.category == 0   ? failure.reason == 0
                                  : status.category == 2 ? failure.reason == 1
                                  : status.category == 3 ? (failure.reason == 2 || failure.reason == 3)
                                                         : status.category == 8 && failure.reason == 4;
        if (status.api != 0 || status.native_code != 0 || !failureShape || error.data != text.data() || error.capacity != text.size() || error.length >= text.size() || text[error.length] != '\0' || (status.category != 0 && !empty(math))) {
            diagnostic = kInvalidResult;
            return false;
        }
        if (status.category != 0) {
            diagnostic.assign(text.data(), error.length);
            return false;
        }
        if (error.length != 0 || !admit_math(input, math)) {
            diagnostic = kInvalidResult;
            return false;
        }
        if (!math.present)
            return true;
        out.renderExtent = {math.render_extent.x, math.render_extent.y, math.render_extent.width, math.render_extent.height};
        out.fullFrameExtent = {math.full_frame_extent.x, math.full_frame_extent.y, math.full_frame_extent.width, math.full_frame_extent.height};
        out.filmDust = input.recipe->filmDust;
        out.filmScratch = input.recipe->filmScratch;
        out.gateDust = input.recipe->gateDust;
        out.gateScratch = input.recipe->gateScratch;
        for (std::size_t i = 0; i < out.origins.size(); ++i) {
            const auto& o = math.origins[i];
            out.origins[i] = {o.cell_x, o.cell_y, o.local_x_mm, o.local_y_mm};
        }
        out.sampleStepXMm = math.sample_step_x_mm;
        out.sampleStepYMm = math.sample_step_y_mm;
        out.roiOffsetX = math.roi_offset_x;
        out.roiOffsetY = math.roi_offset_y;
        out.gateWidth = math.gate_width;
        out.gateHeight = math.gate_height;
        out.sessionSeed = math.session_seed;
        out.clipToken = math.clip_token;
        out.weaveActive = (math.flags & FJ_EFFECTS_WEAVE) != 0;
        out.filmActive = (math.flags & FJ_EFFECTS_FILM) != 0;
        out.gateTransmittanceActive = (math.flags & FJ_EFFECTS_GATE_TRANSMITTANCE) != 0;
        out.gateOutputActive = (math.flags & FJ_EFFECTS_GATE_OUTPUT) != 0;
        out.requiresFullFrame = (math.flags & FJ_EFFECTS_FULL_FRAME) != 0;
        out.weaveDxPx = math.weave_dx_px;
        out.weaveDyPx = math.weave_dy_px;
        out.weaveCosRot = math.weave_cos_rot;
        out.weaveSinRot = math.weave_sin_rot;
        out.recipeHash = math.recipe_hash;
        out.hash = math.hash;
        return true;
    }

    bool prepare_frame(const Spektrafilm::FilmJuicerEffectsFrameDescriptorInput& input, Spektrafilm::FilmJuicerEffectsFrameDescriptor& out, std::string& diagnostic) {
        out = {};
        FjEffectsFrameInput raw{};
        if (!project_input(input, raw, diagnostic))
            return false;
        FjEffectsFrameMath math{};
        FjEffectsFrameFailure failure{};
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        const auto status = fj_legacy_effects_frame_prepare(&raw, &math, &failure, &error);
        return accept_result(input, status, math, failure, error, text, out, diagnostic);
    }
} // namespace JuicerEffects
