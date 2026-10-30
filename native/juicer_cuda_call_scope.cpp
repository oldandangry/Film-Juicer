#include "juicer_cuda_call_scope.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <new>
#include <string_view>

#include "FilmEffectsFrameDescriptors.h"
#include "RenderRecipe.h"
#include "RustEffectsFrameBridge.h"
#include "Cuda/JuicerCudaFailure.h"

namespace {

    constexpr std::string_view kInvalidScope = "invalid CUDA call scope";
    constexpr std::string_view kMissingAdmission = "CUDA call scope is not admitted";
    constexpr std::string_view kDescriptorMismatch = "ResourceDescriptorMismatch phase=frame_descriptors field=rust_result";

    FjStatus write_failure(FjErrorBuffer* error, std::uint32_t category, std::string_view diagnostic) noexcept {
        return JuicerCuda::write_status({category, FJ_API_NONE, 0}, diagnostic, error);
    }

    FjStatus write_scope_error(JuicerCuda::CallScope* scope, FjErrorBuffer* error, FjStatus status, std::string_view diagnostic) noexcept {
        if (scope) {
            scope->set_diagnostic(diagnostic);
            return JuicerCuda::write_status(status, scope->diagnostic(), error);
        }
        return JuicerCuda::write_status(status, diagnostic, error);
    }

    FjStatus scope_error(JuicerCuda::CallScope* scope, FjErrorBuffer* error, std::uint32_t category, std::string_view diagnostic) noexcept {
        return write_scope_error(scope, error, {category, FJ_API_NONE, 0}, diagnostic);
    }

    bool same_extent(const FjExtent& a, const FjExtent& b) noexcept {
        return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
    }

    FjExtent frame_extent(const FjRect& rect) noexcept {
        return {rect.x1, rect.y1, rect.x2 - rect.x1, rect.y2 - rect.y1};
    }

    const char* route_label(std::uint32_t route) noexcept {
        switch (route) {
            case FJ_ROUTE_NEGATIVE_DIRECT:
            case FJ_ROUTE_POSITIVE_DIRECT:
                return "direct";
            case FJ_ROUTE_NEGATIVE_PRINT:
            case FJ_ROUTE_POSITIVE_PRINT:
                return "print";
            default:
                return nullptr;
        }
    }

    Spektrafilm::ProfilePolarity route_polarity(std::uint32_t route) noexcept {
        switch (route) {
            case FJ_ROUTE_NEGATIVE_DIRECT:
            case FJ_ROUTE_NEGATIVE_PRINT:
                return Spektrafilm::ProfilePolarity::Negative;
            case FJ_ROUTE_POSITIVE_DIRECT:
            case FJ_ROUTE_POSITIVE_PRINT:
                return Spektrafilm::ProfilePolarity::Positive;
            default:
                return Spektrafilm::ProfilePolarity::Unsupported;
        }
    }

    void encode_spatial_dir(const SpatialDirDescriptor& source, FjSpatialDir& out) noexcept {
        out = {};
        if (source.hash == 0) {
            return;
        }
        out.render_extent = {source.renderExtent.x, source.renderExtent.y, source.renderExtent.width, source.renderExtent.height};
        out.full_frame_extent = {source.fullFrameExtent.x, source.fullFrameExtent.y, source.fullFrameExtent.width, source.fullFrameExtent.height};
        out.filter_domain_extent = {source.filterDomainExtent.x, source.filterDomainExtent.y, source.filterDomainExtent.width, source.filterDomainExtent.height};
        out.component_count = static_cast<std::size_t>(source.filterPlan.componentCount);
        out.recipe_hash = source.dirRecipeHash;
        out.hash = source.hash;
        for (std::size_t index = 0; index < out.component_count; ++index) {
            const auto& input = source.filterPlan.components[index];
            FjDirGaussian& output = out.components[index];
            output.sigma_px = input.sigmaPixels;
            output.weight = input.weight;
            output.radius = input.radius;
            output.reference_operator = static_cast<std::uint32_t>(input.referenceOperator);
            output.feedforward = input.iir.feedforward;
            std::copy(input.iir.feedback.begin(), input.iir.feedback.end(), output.feedback);
            output.normalization_denominator = input.iir.normalizationDenominator;
            output.feedforward_numerator = input.iir.feedforwardNumerator;
            std::copy(input.iir.feedbackNumerators.begin(), input.iir.feedbackNumerators.end(), output.feedback_numerators);
            output.boundary_truncation_accuracy = input.boundaryTruncationAccuracy;
            output.boundary_certification_tolerance = input.boundaryCertificationTolerance;
            output.boundary_derivation_version = input.boundaryDerivationVersion;
        }
    }

    FjGrainGaussian encode_grain_gaussian(const Spektrafilm::VisualGrainGaussian& source) noexcept {
        return {source.sigmaPx, source.radius, source.hash};
    }

    void encode_grain(
        const FjGrainCoefficients& coefficients,
        const Spektrafilm::VisualGrainFrameDescriptor& descriptor,
        FjVisualGrain& out) noexcept {
        out = {};
        if (!descriptor.active) {
            return;
        }
        out.flags = coefficients.flags;
        if (descriptor.requiresFullFrame) {
            out.flags |= FJ_GRAIN_FULL_FRAME;
        }
        out.debug_view = coefficients.debug_view;
        out.render_extent = {descriptor.renderExtent.x, descriptor.renderExtent.y, descriptor.renderExtent.width, descriptor.renderExtent.height};
        out.full_frame_extent = {descriptor.fullFrameExtent.x, descriptor.fullFrameExtent.y, descriptor.fullFrameExtent.width, descriptor.fullFrameExtent.height};
        out.pixel_size_um = descriptor.pixelSizeUm;
        out.frame0 = descriptor.frame0;
        out.frame_alpha = descriptor.frameAlpha;
        out.seed_base = descriptor.seedBase;
        out.seed_base_next = descriptor.seedBaseNext;
        out.session_seed = descriptor.sessionSeed;
        out.clip_token = descriptor.clipToken;
        out.pitch_px = descriptor.pitchPx;
        out.breathing_period_frames = descriptor.breathingPeriodFrames;
        out.clump_morph_period_frames = descriptor.clumpMorphPeriodFrames;
        out.wang_cell_mm = descriptor.wangCellMm;
        out.breathing_amplitude = descriptor.breathingAmplitude;
        out.breathing_cell_um_small = descriptor.breathingCellUmSmall;
        out.breathing_cell_um_large = descriptor.breathingCellUmLarge;
        out.breathing_mix = descriptor.breathingMix;
        out.breathing_drift_um_per_frame = descriptor.breathingDriftUmPerFrame;
        out.debug_scale = descriptor.debugScale;
        out.size_mix_weights[0] = descriptor.effectiveFineWeight;
        out.size_mix_weights[1] = descriptor.effectiveMidWeight;
        out.size_mix_weights[2] = descriptor.effectiveCoarseWeight;
        out.size_mix_gain = descriptor.sizeMixGain;
        out.sublayer_count = coefficients.sublayer_count;
        std::copy_n(coefficients.micro_structure, 2, out.micro_structure);
        out.clump_temporal_mix = coefficients.clump_temporal_mix;
        out.size_mix_scale = coefficients.size_mix_scale;
        out.amplitude = coefficients.amplitude;
        out.chroma_mix = coefficients.chroma_mix;
        out.chroma_shared_weight = coefficients.chroma_shared_weight;
        out.chroma_independent_weight = coefficients.chroma_independent_weight;
        std::copy_n(coefficients.particle_density_min_cmy, 3, out.particle_density_min_cmy);
        std::copy_n(coefficients.uniformity_cmy, 3, out.uniformity_cmy);
        std::copy_n(coefficients.layer_axis_block_prefix_max[0], 16, out.layer_axis_block_prefix_max[0]);
        std::copy_n(coefficients.layer_axis_block_prefix_max[1], 16, out.layer_axis_block_prefix_max[1]);
        std::copy_n(coefficients.layer_axis_block_prefix_max[2], 16, out.layer_axis_block_prefix_max[2]);
        std::copy(descriptor.densityMaxCmy.begin(), descriptor.densityMaxCmy.end(), out.density_max_cmy);
        std::copy(descriptor.nParticlesCmy.begin(), descriptor.nParticlesCmy.end(), out.n_particles_cmy);
        std::copy(descriptor.odParticleCmy.begin(), descriptor.odParticleCmy.end(), out.od_particle_cmy);
        for (std::size_t layer = 0; layer < 3; ++layer) {
            out.correlation[layer] = encode_grain_gaussian(descriptor.correlation[layer]);
            std::copy(descriptor.densityMinLayers[layer].begin(), descriptor.densityMinLayers[layer].end(), out.density_min_layers[layer]);
            std::copy(descriptor.densityMaxLayers[layer].begin(), descriptor.densityMaxLayers[layer].end(), out.density_max_layers[layer]);
            std::copy(descriptor.nParticlesLayers[layer].begin(), descriptor.nParticlesLayers[layer].end(), out.n_particles_layers[layer]);
            std::copy(descriptor.odParticleLayers[layer].begin(), descriptor.odParticleLayers[layer].end(), out.od_particle_layers[layer]);
            for (std::size_t channel = 0; channel < 3; ++channel) {
                out.dye_cloud[layer][channel] = encode_grain_gaussian(descriptor.dyeCloud[layer][channel]);
            }
        }
        out.recipe_hash = descriptor.recipeHash;
        out.density_layers_hash = descriptor.densityCurvesLayersHash;
        out.hash = descriptor.hash;
    }

    void encode_effects(
        const FjFrameMathInput& input,
        const Spektrafilm::FilmJuicerEffectsFrameDescriptor& descriptor,
        FjFilmEffects& out) noexcept {
        out = {};
        if (descriptor.hash == 0) {
            return;
        }
        out.render_extent = {descriptor.renderExtent.x, descriptor.renderExtent.y, descriptor.renderExtent.width, descriptor.renderExtent.height};
        out.full_frame_extent = {descriptor.fullFrameExtent.x, descriptor.fullFrameExtent.y, descriptor.fullFrameExtent.width, descriptor.fullFrameExtent.height};
        out.film_dust = input.film_dust;
        out.film_scratch = input.film_scratch;
        out.gate_dust = input.gate_dust;
        out.gate_scratch = input.gate_scratch;
        for (std::size_t index = 0; index < descriptor.origins.size(); ++index) {
            const auto& origin = descriptor.origins[index];
            out.origins[index] = {origin.cellX, origin.cellY, origin.localXMm, origin.localYMm};
        }
        out.sample_step_x_mm = descriptor.sampleStepXMm;
        out.sample_step_y_mm = descriptor.sampleStepYMm;
        out.roi_offset_x = descriptor.roiOffsetX;
        out.roi_offset_y = descriptor.roiOffsetY;
        out.gate_width = descriptor.gateWidth;
        out.gate_height = descriptor.gateHeight;
        out.session_seed = descriptor.sessionSeed;
        out.clip_token = descriptor.clipToken;
        if (descriptor.weaveActive)
            out.flags |= FJ_EFFECTS_WEAVE;
        if (descriptor.filmActive)
            out.flags |= FJ_EFFECTS_FILM;
        if (descriptor.gateTransmittanceActive)
            out.flags |= FJ_EFFECTS_GATE_TRANSMITTANCE;
        if (descriptor.gateOutputActive)
            out.flags |= FJ_EFFECTS_GATE_OUTPUT;
        if (descriptor.requiresFullFrame)
            out.flags |= FJ_EFFECTS_FULL_FRAME;
        out.weave_dx_px = descriptor.weaveDxPx;
        out.weave_dy_px = descriptor.weaveDyPx;
        out.weave_cos_rot = descriptor.weaveCosRot;
        out.weave_sin_rot = descriptor.weaveSinRot;
        out.recipe_hash = descriptor.recipeHash;
        out.hash = descriptor.hash;
    }

} // namespace

namespace JuicerCuda {

    CallScope::CallScope(FjCuda* owner, std::optional<std::uint32_t> route, CallCompletionContinuation completion, void* completionUser) noexcept
        : _owner(owner), _route(route), _completion(completion), _completionUser(completionUser) {
    }

    FjCudaCallAccess CallScope::access() noexcept {
        return {this, &CallScope::precheck, &CallScope::admit, &CallScope::release, &CallScope::inspect, &CallScope::preview, &CallScope::frame_descriptors, &CallScope::render, &CallScope::complete};
    }

    PendingContextLossRecovery& CallScope::recovery() noexcept {
        return _recovery;
    }

    const std::string& CallScope::diagnostic() const noexcept {
        return _diagnostic;
    }

    void CallScope::set_diagnostic(std::string_view value) noexcept {
        try {
            _diagnostic.assign(value);
        } catch (...) {
            _diagnostic.clear();
        }
    }

    FjStatus CallScope::precheck(void* user, FjErrorBuffer* error) noexcept {
        auto* scope = static_cast<CallScope*>(user);
        if (!scope || !scope->_owner || scope->_completed || scope->_admissionAttempted) {
            return write_failure(error, FJ_STATUS_UNSUPPORTED_INPUT, kInvalidScope);
        }
        try {
            check_native_call_admission(scope->_owner);
            scope->_diagnostic.clear();
            return JuicerCuda::write_status({FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}, error);
        } catch (const ExecutionFailure& failure) {
            return write_scope_error(scope, error, failure.failure.status, failure.failure.diagnostic);
        } catch (const std::bad_alloc&) {
            return scope_error(scope, error, FJ_STATUS_ALLOCATION_FAILURE, "CUDA call precheck allocation failed");
        } catch (const std::exception& detail) {
            return scope_error(scope, error, FJ_STATUS_INTERNAL_FAILURE, detail.what());
        } catch (...) {
            return scope_error(scope, error, FJ_STATUS_INTERNAL_FAILURE, "CUDA call precheck failed");
        }
    }

    FjStatus CallScope::admit(void* user, FjErrorBuffer* error) noexcept {
        auto* scope = static_cast<CallScope*>(user);
        if (!scope || !scope->_owner || scope->_call ||
            scope->_admissionAttempted || scope->_completed) {
            return scope_error(scope, error, FJ_STATUS_UNSUPPORTED_INPUT, kInvalidScope);
        }
        scope->_admissionAttempted = true;
        try {
            scope->_call.emplace(scope->_owner);
            scope->_diagnostic.clear();
            return JuicerCuda::write_status({FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}, error);
        } catch (const ExecutionFailure& failure) {
            return write_scope_error(scope, error, failure.failure.status, failure.failure.diagnostic);
        } catch (const std::bad_alloc&) {
            return scope_error(scope, error, FJ_STATUS_ALLOCATION_FAILURE, "CUDA call admission allocation failed");
        } catch (const std::exception& detail) {
            return scope_error(scope, error, FJ_STATUS_INTERNAL_FAILURE, detail.what());
        } catch (...) {
            return scope_error(scope, error, FJ_STATUS_INTERNAL_FAILURE, "CUDA call admission failed");
        }
    }

    void CallScope::release(void* user) noexcept {
        auto* scope = static_cast<CallScope*>(user);
        if (scope && scope->_call && !scope->_released) {
            scope->_call.reset();
            scope->_released = true;
        }
    }

    FjStatus CallScope::inspect(void* user, const FjFrame* frame, FjCudaContext* context, FjErrorBuffer* error) noexcept {
        auto* scope = static_cast<CallScope*>(user);
        if (context) {
            *context = {};
        }
        return scope ? scope->inspect(frame, context, error)
                     : write_failure(error, FJ_STATUS_UNSUPPORTED_INPUT, kInvalidScope);
    }

    FjStatus CallScope::inspect(const FjFrame* frame, FjCudaContext* context, FjErrorBuffer* error) noexcept {
        if (context)
            *context = {};
        if (!_call) {
            return scope_error(this, error, FJ_STATUS_UNSUPPORTED_INPUT, kMissingAdmission);
        }
        try {
            _diagnostic.clear();
            _frame.reset();
            _hasInspectedContext = false;
            _pixelSizeUm = 0.0f;
            std::array<char, 2048> text{};
            FjErrorBuffer inspectError{text.data(), text.size(), 0};
            const FjStatus result = _call->inspect(frame, context, &inspectError);
            _diagnostic.assign(text.data(), inspectError.length);
            JuicerCuda::write_status(result, _diagnostic, error);
            if (result.category == FJ_STATUS_SUCCESS) {
                _frame = *frame;
                _inspectedContext = *context;
                _pixelSizeUm = frame->pixel_size_um;
                _hasInspectedContext = true;
            } else {
                if (context)
                    *context = {};
                _frame.reset();
            }
            return result;
        } catch (const std::bad_alloc&) {
            set_diagnostic("CUDA inspection scope allocation failed");
            if (context)
                *context = {};
            return JuicerCuda::write_status({FJ_STATUS_ALLOCATION_FAILURE, FJ_API_NONE, 0}, _diagnostic, error);
        } catch (const std::exception& detail) {
            set_diagnostic(detail.what());
            if (context)
                *context = {};
            return JuicerCuda::write_status({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, _diagnostic, error);
        } catch (...) {
            set_diagnostic("CUDA inspection scope failed");
            if (context)
                *context = {};
            return JuicerCuda::write_status({FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, _diagnostic, error);
        }
    }

    FjStatus CallScope::preview(void* user, const FjRect* source, const FjRect* meter, std::uint32_t method, FjAutoExposure* out, FjErrorBuffer* error) noexcept {
        auto* scope = static_cast<CallScope*>(user);
        if (out)
            *out = {};
        return scope ? scope->preview(source, meter, method, out, error)
                     : write_failure(error, FJ_STATUS_UNSUPPORTED_INPUT, kInvalidScope);
    }

    FjStatus CallScope::preview(const FjRect* source, const FjRect* meter, std::uint32_t method, FjAutoExposure* out, FjErrorBuffer* error) noexcept {
        if (out)
            *out = {};
        if (_completed || !out || !source || !meter || method > FJ_METER_HIGHLIGHT_WEIGHTED ||
            source->x2 < source->x1 || source->y2 < source->y1 ||
            meter->x2 < meter->x1 || meter->y2 < meter->y1) {
            return scope_error(this, error, FJ_STATUS_UNSUPPORTED_INPUT, "invalid auto-exposure preview arguments");
        }
        const auto width = static_cast<std::int64_t>(meter->x2) - meter->x1;
        const auto height = static_cast<std::int64_t>(meter->y2) - meter->y1;
        if (std::max(width, height) > std::numeric_limits<int>::max()) {
            return scope_error(this, error, FJ_STATUS_UNSUPPORTED_INPUT, "auto-exposure preview extent exceeds native limits");
        }
        try {
            const auto preview = make_auto_exposure_preview_descriptor(
                {source->x1, source->y1, source->x2, source->y2},
                {meter->x1, meter->y1, meter->x2, meter->y2},
                static_cast<Spektrafilm::AutoExposureMethod>(method));
            out->source_bounds = *source;
            out->meter_bounds = *meter;
            out->preview_width = preview.previewWidth;
            out->preview_height = preview.previewHeight;
            out->method = method;
            out->hash = preview.hash;
            _diagnostic.clear();
            return JuicerCuda::write_status({FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}, error);
        } catch (const std::bad_alloc&) {
            *out = {};
            return scope_error(this, error, FJ_STATUS_ALLOCATION_FAILURE, "auto-exposure preview allocation failed");
        } catch (const std::exception& detail) {
            *out = {};
            return scope_error(this, error, FJ_STATUS_INTERNAL_FAILURE, detail.what());
        } catch (...) {
            *out = {};
            return scope_error(this, error, FJ_STATUS_INTERNAL_FAILURE, "auto-exposure preview failed");
        }
    }

    FjStatus CallScope::frame_descriptors(void* user, const FjFrameMathInput* input, FjNativeFrameDescriptors* out, FjErrorBuffer* error) noexcept {
        auto* scope = static_cast<CallScope*>(user);
        if (out)
            *out = {};
        return scope ? scope->frame_descriptors(input, out, error)
                     : write_failure(error, FJ_STATUS_UNSUPPORTED_INPUT, kInvalidScope);
    }

    FjStatus CallScope::frame_descriptors(const FjFrameMathInput* input,
                                          FjNativeFrameDescriptors* out,
                                          FjErrorBuffer* error) noexcept {
        if (out)
            *out = {};
        if (_completed || !_call || !_hasInspectedContext || !_frame || !_route || !input || !out ||
            !route_label(*_route)) {
            return scope_error(this, error, FJ_STATUS_UNSUPPORTED_INPUT, kMissingAdmission);
        }
        try {
            const FjFrame& frame = *_frame;
            const FjExtent expectedRender = frame_extent(frame.render_window);
            const FjExtent expectedFull = frame_extent(frame.full_frame_extent);
            if ((input->dir.present != 0 &&
                 (!same_extent(input->dir.render_extent, expectedRender) ||
                  !same_extent(input->dir.full_frame_extent, expectedFull))) ||
                (input->grain.present != 0 &&
                 (!same_extent(input->grain.render_extent, expectedRender) ||
                  !same_extent(input->grain.full_frame_extent, expectedFull) ||
                  std::bit_cast<std::uint32_t>(input->grain.pixel_size_um) != std::bit_cast<std::uint32_t>(_pixelSizeUm))) ||
                (input->effects.present != 0 &&
                 (!same_extent(input->effects.render_extent, expectedRender) ||
                  !same_extent(input->effects.full_frame_extent, frame.effects_geometry.pixel_definition)))) {
                return scope_error(this, error, FJ_STATUS_PREPARATION_FAILURE, kDescriptorMismatch);
            }
            SpatialDirDescriptor dir{};
            if (!Spektrafilm::compose_spatial_dir_descriptor(input->dir, _pixelSizeUm, route_label(*_route), dir)) {
                return scope_error(this, error, FJ_STATUS_PREPARATION_FAILURE, "ResourceDescriptorMismatch phase=dir_descriptor field=rust_result");
            }
            Spektrafilm::VisualGrainFrameDescriptor grain{};
            const auto polarity = route_polarity(*_route);
            std::string diagnostic;
            if (!Spektrafilm::compose_visual_grain_frame_descriptor(
                    input->grain_coefficients, polarity, input->grain, grain, diagnostic)) {
                return scope_error(this, error, FJ_STATUS_PREPARATION_FAILURE, diagnostic.empty() ? kDescriptorMismatch : std::string_view(diagnostic));
            }
            if (grain.active && (grain.sessionSeed != frame.session_seed || grain.clipToken != frame.clip_token)) {
                return scope_error(this, error, FJ_STATUS_PREPARATION_FAILURE, kDescriptorMismatch);
            }
            Spektrafilm::FilmJuicerEffectsFrameDescriptor effects{};
            if (!JuicerEffects::compose_frame_descriptor(input->effects, input->film_dust, input->film_scratch, input->gate_dust, input->gate_scratch, effects, diagnostic)) {
                return scope_error(this, error, FJ_STATUS_PREPARATION_FAILURE, diagnostic.empty() ? kDescriptorMismatch : std::string_view(diagnostic));
            }
            if (effects.hash != 0 &&
                (!same_extent(input->effects.full_frame_extent, frame.effects_geometry.pixel_definition) ||
                 input->effects.session_seed != frame.session_seed || input->effects.clip_token != frame.clip_token)) {
                return scope_error(this, error, FJ_STATUS_PREPARATION_FAILURE, kDescriptorMismatch);
            }
            FjNativeFrameDescriptors descriptors{};
            encode_spatial_dir(dir, descriptors.spatial_dir);
            encode_grain(input->grain_coefficients, grain, descriptors.grain);
            encode_effects(*input, effects, descriptors.effects);
            *out = descriptors;
            _diagnostic.clear();
            return JuicerCuda::write_status({FJ_STATUS_SUCCESS, FJ_API_NONE, 0}, {}, error);
        } catch (const std::bad_alloc&) {
            *out = {};
            return scope_error(this, error, FJ_STATUS_ALLOCATION_FAILURE, "native frame descriptor allocation failed");
        } catch (const std::exception& detail) {
            *out = {};
            return scope_error(this, error, FJ_STATUS_INTERNAL_FAILURE, detail.what());
        } catch (...) {
            *out = {};
            return scope_error(this, error, FJ_STATUS_INTERNAL_FAILURE, "native frame descriptor composition failed");
        }
    }

    FjRenderOutcome CallScope::render(void* user, const FjCudaContext* context, const FjFrame* frame, const FjSubmission* submission, const FjPreparedHostData* prepared, const FjHostPreparation* hostPreparation, FjAbortCallback abort, FjErrorBuffer* error) noexcept {
        auto* scope = static_cast<CallScope*>(user);
        if (!scope) {
            return {write_failure(error, FJ_STATUS_UNSUPPORTED_INPUT, kInvalidScope), 0};
        }
        return scope->render(context, frame, submission, prepared, hostPreparation, abort, error);
    }

    FjRenderOutcome CallScope::render(const FjCudaContext* context, const FjFrame* frame, const FjSubmission* submission, const FjPreparedHostData* prepared, const FjHostPreparation* hostPreparation, FjAbortCallback abort, FjErrorBuffer* error) noexcept {
        if (!_call) {
            const FjStatus status = scope_error(this, error, FJ_STATUS_UNSUPPORTED_INPUT, kMissingAdmission);
            return {status, 0};
        }
        if (!_hasInspectedContext || !context ||
            context->device_id != _inspectedContext.device_id ||
            context->context != _inspectedContext.context) {
            const FjStatus status = scope_error(this, error, FJ_STATUS_UNSUPPORTED_INPUT, "CUDA render context differs from inspected scope");
            return {status, 0};
        }
        try {
            _diagnostic.clear();
            const FjRenderOutcome outcome = _call->render(context, frame, submission, prepared, abort, _recovery, _diagnostic, hostPreparation);
            JuicerCuda::write_status(outcome.status, _diagnostic, error);
            return outcome;
        } catch (const ExecutionFailure& failure) {
            const FjStatus status = write_scope_error(this, error, failure.failure.status, failure.failure.diagnostic);
            return {status, 0};
        } catch (const std::bad_alloc&) {
            const FjStatus status = scope_error(this, error, FJ_STATUS_ALLOCATION_FAILURE, "CUDA render allocation failed");
            return {status, 0};
        } catch (const std::exception& detail) {
            const FjStatus status = scope_error(this, error, FJ_STATUS_INTERNAL_FAILURE, detail.what());
            return {status, 0};
        } catch (...) {
            const FjStatus status = scope_error(this, error, FJ_STATUS_INTERNAL_FAILURE, "CUDA render failed");
            return {status, 0};
        }
    }

    FjCallCompletionResult CallScope::complete(void* user, const FjCallCompletionInput* input) noexcept {
        auto* scope = static_cast<CallScope*>(user);
        return scope ? scope->complete(input) : FjCallCompletionResult{{{FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0}, 0}, 1};
    }

    FjCallCompletionResult CallScope::complete(const FjCallCompletionInput* input) noexcept {
        if (_completed || !input || _call || !_completion) {
            return {{{FJ_STATUS_INTERNAL_FAILURE, FJ_API_NONE, 0}, 0}, 1};
        }
        _completed = true;
        try {
            FjCallCompletionInput completed = *input;
            if (!_diagnostic.empty()) {
                completed.diagnostic = {_diagnostic.data(), _diagnostic.size()};
            }
            return _completion(_completionUser, &completed);
        } catch (...) {
            return {input->outcome, 1};
        }
    }

} // namespace JuicerCuda
