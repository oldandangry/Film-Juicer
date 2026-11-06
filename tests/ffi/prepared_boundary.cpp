#include "prepared_boundary.h"
#include "prepared_raw_render_fixture.h"
#include "juicer_test_api.h"
#include "juicer_cuda_call_scope.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

#include "prepared_descriptors.h"
#include "prepared_frame_fixture.h"
#include "spatial_optics_fixture.h"
#include "juicer_cuda_owner.h"
#include "Cuda/JuicerCudaHostViews.h"

namespace {

    void bind_state(const FjPreparedStateView& state, FjPreparedHostData& prepared) {
        prepared = {};
        prepared.route = state.route;
        prepared.capture_polarity = state.polarity;
        prepared.film_profile_key = state.film_profile_key;
        prepared.print_profile_key = state.print_profile_key;
        prepared.film_profile_asset_version = state.film_profile_asset_version;
        prepared.print_profile_asset_version = state.print_profile_asset_version;
        prepared.recipe_hash = state.recipe_hash;
        prepared.film_exposure = state.film_exposure;
        prepared.film_development = state.film_development;
        const auto& dir = state.dir_recipe;
        auto& executionDir = prepared.dir_couplers;
        executionDir.mode = dir.mode;
        std::copy_n(dir.matrix_rgb, 9, executionDir.matrix_rgb);
        std::copy_n(dir.density_max_rgb, 3, executionDir.density_max_rgb);
        std::copy_n(dir.density_ref_rgb, 3, executionDir.density_ref_rgb);
        std::copy_n(dir.donor_k_rgb, 3, executionDir.donor_k_rgb);
        std::copy_n(dir.receiver_c_ref_rgb, 3, executionDir.receiver_c_ref_rgb);
        std::copy_n(dir.receiver_kr_rgb, 3, executionDir.receiver_kr_rgb);
        std::copy_n(dir.compensated_axes_rgb, 3, executionDir.compensated_axes_rgb);
        executionDir.compensated_axes_hash = dir.compensated_axes_hash;
        executionDir.hash = dir.hash;
        const auto bounds = [](const FjFilmDensityBounds& source) {
            FjDensityBounds out{};
            std::copy_n(source.minimum_cmy, 3, out.min_cmy);
            std::copy_n(source.inv_span_cmy, 3, out.inv_span_cmy);
            out.hash = source.hash;
            return out;
        };
        prepared.scanner_bounds = bounds(state.scanner_bounds);
        prepared.enlarger_film_bounds = bounds(state.enlarger_film_bounds);
        const auto& scanner = state.scanner_tables;
        prepared.scanner_spectra = {scanner.dyes_cmy[2], scanner.dyes_cmy[1], scanner.dyes_cmy[0], scanner.weighted_xyz[0], scanner.weighted_xyz[1], scanner.weighted_xyz[2], scanner.has_baseline ? scanner.baseline_min : FjFloatSpan{}, scanner.inv_yn};
        prepared.scanner_lut = {state.scanner_lut.density_bounds_hash, state.scanner_lut.resolution, state.scanner_lut.hash};
        const auto& correction = state.scanner_correction;
        prepared.scanner_correction = {correction.active ? FJ_CORRECTION_ACTIVE : 0u, correction.xyz_slope, correction.xyz_offset, correction.exposure_scale};
        const auto& post = state.scanner_post_effects;
        prepared.scanner_post = {post.glare_active ? FJ_POST_GLARE : 0u, post.glare_percent, post.glare_roughness, post.glare_blur_sigma_px, post.glare_blur_radius, post.lens_blur_sigma_px, post.lens_blur_radius, post.unsharp_sigma_px, post.unsharp_radius, post.unsharp_amount, post.hash};
        prepared.output_color = state.output_color;
        prepared.print = state.print;
    }

} // namespace

namespace JuicerCudaTest {

    namespace {
        bool render_held_state(const FjAssets* assets, FjCuda* cuda, const PreparedState& owner, const FjPreparedStateView& state, const FjFrame& frame, const FjSubmission& expected, std::uint64_t expectedRecipeHash, FjErrorBuffer& error, std::string& diagnostic) {
            const auto frameFloor = std::floor(frame.time_frames);
            if (state.recipe_hash != expectedRecipeHash || state.upload_core_hash != expected.upload_core_hash ||
                state.dir_recipe.hash != expected.dir_hash || state.scanner_hash != expected.scanner_hash || expected.instance_token == 0 ||
                !std::isfinite(frameFloor) || frameFloor < -0x1p63 || frameFloor >= 0x1p63 ||
                expected.frame_token != static_cast<std::uint64_t>(static_cast<std::int64_t>(frameFloor))) {
                diagnostic = "held-state render identity does not match its frame submission";
                return false;
            }
            error.length = 0;
            if (error.capacity != 0 && error.data != nullptr)
                error.data[0] = '\0';
            const FjExtent full{frame.full_frame_extent.x1, frame.full_frame_extent.y1, frame.full_frame_extent.x2 - frame.full_frame_extent.x1, frame.full_frame_extent.y2 - frame.full_frame_extent.y1};
            FjPreparedFramePolicy policy{};
            const auto policyStatus = owner.frame_policy(full, policy, error);
            if (policyStatus.category != FJ_STATUS_SUCCESS) {
                diagnostic.assign(error.data, error.length);
                return false;
            }
            struct Completion {
                unsigned calls = 0;
                bool ordered = false;
                FjCuda* cuda = nullptr;
            } completion;
            completion.cuda = cuda;
            JuicerCuda::CallScope scope(cuda, state.route, [](void* user, const FjCallCompletionInput* input) noexcept -> FjCallCompletionResult {
                auto& observed = *static_cast<Completion*>(user);
                ++observed.calls;
                try {
                    // The real provider must release its admission before completion.
                    JuicerCuda::check_native_call_admission(observed.cuda);
                    observed.ordered = input->render_outcome_valid == 1 &&
                                       input->has_inspected_context == 1 && input->recovery_scope_entered == 1 &&
                                       input->force_fatal == 0 && input->abort_failed == 0;
                } catch (...) {
                    observed.ordered = false;
                }
                return {input->outcome, observed.ordered && input->outcome.status.category == FJ_STATUS_SUCCESS ? 0u : 1u};
            },
                                        &completion);
            struct Selection {
                unsigned calls = 0;
                FjSubmission expected{};
                bool matched = true;
            } selection;
            selection.expected = expected;
            const FjSubmissionAccess latch{expected.instance_token,
                                           expected.frame_token,
                                           &selection,
                                           [](void* user, const FjCudaContext*, const FjSubmission* requested, FjSubmission* out, FjErrorBuffer*) -> FjStatus {
                                               auto& selected = *static_cast<Selection*>(user);
                                               ++selected.calls;
                                               selected.matched = requested &&
                                                                  requested->instance_token == selected.expected.instance_token &&
                                                                  requested->frame_token == selected.expected.frame_token &&
                                                                  requested->upload_core_hash == selected.expected.upload_core_hash &&
                                                                  requested->dir_hash == selected.expected.dir_hash &&
                                                                  requested->scanner_hash == selected.expected.scanner_hash &&
                                                                  requested->auto_exposure_hash == selected.expected.auto_exposure_hash;
                                               if (!selected.matched)
                                                   return {FJ_STATUS_UNSUPPORTED_INPUT, FJ_API_NONE, 0};
                                               *out = *requested;
                                               out->submission_id = selected.expected.submission_id;
                                               return {FJ_STATUS_SUCCESS, FJ_API_NONE, 0};
                                           }};
            const auto result = owner.render(assets, scope.access(), frame, policy, latch, {}, error);
            if (result.outcome.status.category != FJ_STATUS_SUCCESS || result.fatal != 0 ||
                completion.calls != 1 || !completion.ordered || selection.calls != 1 || !selection.matched || scope.recovery().pending) {
                diagnostic = scope.diagnostic().empty() ? std::string(error.data, error.length) : scope.diagnostic();
                if (diagnostic.empty())
                    diagnostic = "held rendering completion/selection order failed";
                return false;
            }
            return true;
        }
    } // namespace

    bool execute_prepared_state(FjRuntimeView runtime, const ParamSnapshot& params, const FjFrame& frame, std::uint64_t instanceToken, bool contract, bool lifetime, std::string& diagnostic, StateLifetimeCase lifetimeCase) {
        if (!runtime.assets || !runtime.cuda) {
            diagnostic = "prepared-state fixture requires a live Runtime view";
            return false;
        }
        FjStatus status{};
        auto owner = prepare_state(runtime.assets, params, status, diagnostic);
        if (status.category != FJ_STATUS_SUCCESS)
            return false;
        std::array<char, 2048> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        FjPreparedStateView state{};
        if (owner->view(state, error).category != FJ_STATUS_SUCCESS) {
            diagnostic.assign(message.data(), error.length);
            return false;
        }
        if (state.optics.hash != 0 || state.grain.hash != 0 || state.effects.hash != 0 || state.dir_recipe.hash == 0) {
            diagnostic = "held-state fixture requires active DIR and disabled frame effects";
            return false;
        }
        FjPreparedHostData prepared{};
        bind_state(state, prepared);
        const auto& dir = state.dir_recipe;
        const auto extent = [](FjRect rect) {
            return FjExtent{rect.x1, rect.y1, rect.x2 - rect.x1, rect.y2 - rect.y1};
        };
        const FjDirFrameInput input{1, dir.diffusion_size_um, dir.diffusion_tail_um, dir.diffusion_tail_weight, frame.pixel_size_um, extent(frame.render_window), extent(frame.full_frame_extent), dir.hash};
        Spektrafilm::SpatialDirDescriptor spatial;
        if (!JuicerTestSupport::prepare_dir_frame(input, "prepared-state", spatial)) {
            diagnostic = "held-state DIR frame completion failed";
            return false;
        }
        encode_spatial_dir_descriptor(spatial, prepared.spatial_dir);
        const JuicerCuda::FrameRect nativeBounds{frame.source.bounds.x1, frame.source.bounds.y1, frame.source.bounds.x2, frame.source.bounds.y2};
        const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(nativeBounds, nativeBounds, static_cast<Spektrafilm::AutoExposureMethod>(state.input_policy.meter));
        prepared.auto_exposure = {{meter.sourceX1, meter.sourceY1, meter.sourceX2, meter.sourceY2}, {meter.meterX1, meter.meterY1, meter.meterX2, meter.meterY2}, meter.previewWidth, meter.previewHeight, static_cast<std::uint32_t>(meter.method), meter.hash};
        FjCudaContext context{};
        if (fj_cuda_inspect(runtime.cuda, &frame, &context, &error).category != FJ_STATUS_SUCCESS) {
            diagnostic.assign(message.data(), error.length);
            return false;
        }
        const FjSubmission submission{instanceToken, static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(frame.time_frames))), 1, state.upload_core_hash, dir.hash, state.scanner_hash, meter.hash};
        const auto invocationContext = std::string("held-route-") + std::to_string(state.route) + "-instance-" + std::to_string(instanceToken) +
                                       "-frame-" + std::to_string(submission.frame_token) + "-contract-" + (contract ? "yes" : "no") +
                                       "-lifetime-" + (lifetime ? "yes" : "no") + "-case-" + std::to_string(static_cast<unsigned>(lifetimeCase));
        check_frame_bindings(runtime.cuda, context, frame, submission, prepared, &state, nullptr, invocationContext);
        const bool success = (contract || (lifetime && lifetimeCase == StateLifetimeCase::SuppliedStream))
                                 ? check_render_contract(runtime.cuda, context, frame, submission, prepared, &error, invocationContext).category == FJ_STATUS_SUCCESS
                                 : render_raw_prepared(runtime.cuda, frame, submission, prepared, {}, &error).status.category == FJ_STATUS_SUCCESS;
        if (!success) {
            diagnostic.assign(message.data(), error.length);
            return false;
        }
        if (lifetime)
            check_prepared_state_lifetime(owner, runtime.assets, runtime.cuda, context, frame, submission, prepared, &error, lifetimeCase);
        else if (!render_held_state(runtime.assets, runtime.cuda, *owner, state, frame, submission, state.recipe_hash, error, diagnostic))
            return false;
        return true;
    }

    bool execute_boundary(
        FjRuntimeView runtime,
        const PreparedState& owner,
        const FjFrame& frame,
        JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        std::string& diagnostic,
        const std::string& invocationPhase) {
        diagnostic.clear();
        if (!runtime.assets || !runtime.cuda) {
            diagnostic = "held-state boundary requires a live Runtime view";
            return false;
        }
        std::array<char, 2048> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        FjPreparedStateView state{};
        if (owner.view(state, error).category != FJ_STATUS_SUCCESS) {
            diagnostic.assign(message.data(), error.length);
            return false;
        }
        const auto extent = [](FjRect rect) {
            return FjExtent{rect.x1, rect.y1, rect.x2 - rect.x1, rect.y2 - rect.y1};
        };
        PreparedFrameFixture::FrameFacts facts;
        facts.renderExtent = extent(frame.render_window);
        facts.fullFrameExtent = extent(frame.full_frame_extent);
        facts.pixelDefinition = frame.effects_geometry.pixel_definition;
        facts.pixelSizeUm = frame.pixel_size_um;
        facts.filmFormatLongEdgeMm = state.input_policy.film_format_long_edge_mm;
        facts.canonicalX = frame.effects_geometry.canonical_x;
        facts.canonicalY = frame.effects_geometry.canonical_y;
        facts.canonicalWidth = frame.effects_geometry.canonical_width;
        facts.canonicalHeight = frame.effects_geometry.canonical_height;
        facts.scaleX = frame.effects_geometry.scale_x;
        facts.scaleY = frame.effects_geometry.scale_y;
        facts.pixelAspectRatio = frame.effects_geometry.pixel_aspect_ratio;
        facts.frameTime = frame.time_frames;
        facts.frameRate = frame.frame_rate;
        facts.sessionSeed = frame.session_seed;
        facts.clipToken = frame.clip_token;
        struct Invocation {
            const PreparedState& owner;
            FjRuntimeView runtime;
            const FjFrame& frame;
            JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot;
            const std::string& phase;
        } invocation{owner, runtime, frame, snapshot, invocationPhase};
        const auto baseline = fj_test_noise_acquisition_count();
        const bool success = PreparedFrameFixture::with_prepared_frame_inputs(
            runtime.assets, owner, facts, &invocation, [](void* user, const PreparedFrameFixture::Inputs& inputs, std::string& diagnostic) {
                const auto& invocation = *static_cast<Invocation*>(user);
                const FjRuntimeView runtime = invocation.runtime;
                const auto& frame = invocation.frame;
                const auto& snapshot = invocation.snapshot;
                const auto& state = inputs.state;
                const JuicerCuda::FrameRect bounds{frame.source.bounds.x1, frame.source.bounds.y1, frame.source.bounds.x2, frame.source.bounds.y2};
                const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(
                    bounds, bounds, static_cast<Spektrafilm::AutoExposureMethod>(state.input_policy.meter));
                if (snapshot.instanceToken.value == 0 || snapshot.snapshotId == 0 ||
                    snapshot.keyDigests.uploadCoreHash != state.upload_core_hash ||
                    snapshot.keyDigests.dirHash != state.dir_recipe.hash ||
                    snapshot.keyDigests.scannerHash != state.scanner_hash ||
                    snapshot.keyDigests.autoExposureHash != meter.hash ||
                    inputs.descriptors.route != static_cast<Spektrafilm::ScanRoute>(state.route)) {
                    diagnostic = "held-state boundary identity differs from its submission";
                    return false;
                }
                FjPreparedHostData prepared{};
                bind_state(state, prepared);
                encode_prepared_descriptors(inputs.descriptors, prepared);
                // The fixed-value writer preserves frame values; large spans borrow the held state.
                prepared.output_color = state.output_color;
                prepared.auto_exposure = {{meter.sourceX1, meter.sourceY1, meter.sourceX2, meter.sourceY2},
                                          {meter.meterX1, meter.meterY1, meter.meterX2, meter.meterY2},
                                          meter.previewWidth,
                                          meter.previewHeight,
                                          static_cast<std::uint32_t>(meter.method),
                                          meter.hash};
                if (state.grain.active != 0) {
                    const auto& noise = *inputs.preparation.noise;
                    prepared.noise = {{noise.stbn.data(), noise.stbn.size()},
                                      noise.stbnWidth,
                                      noise.stbnHeight,
                                      noise.stbnFrames,
                                      {noise.wangTiles.data(), noise.wangTiles.size()},
                                      {noise.wangLut.data(), noise.wangLut.size()},
                                      noise.wangWidth,
                                      noise.wangHeight,
                                      static_cast<std::size_t>(noise.wangCount),
                                      noise.wangColors};
                }
                const FjCudaContext context{snapshot.deviceContextKey.deviceId, reinterpret_cast<std::uintptr_t>(snapshot.deviceContextKey.contextOpaque)};
                const FjSubmission submission{snapshot.instanceToken.value, snapshot.frameToken.value, snapshot.snapshotId, snapshot.keyDigests.uploadCoreHash, snapshot.keyDigests.dirHash, snapshot.keyDigests.scannerHash, snapshot.keyDigests.autoExposureHash};
                std::array<char, 2048> message{};
                FjErrorBuffer error{message.data(), message.size(), 0};
                const auto acquisitions = fj_test_noise_acquisition_count();
                const auto invocationContext = std::string("boundary-route-") + std::to_string(state.route) + "-instance-" + std::to_string(snapshot.instanceToken.value) +
                                               "-frame-" + std::to_string(snapshot.frameToken.value) + "-snapshot-" + std::to_string(snapshot.snapshotId) +
                                               "-time-" + std::to_string(frame.time_frames) + "-phase-" + invocation.phase;
                check_frame_bindings(runtime.cuda, context, frame, submission, prepared, &state, inputs.hostPreparation, invocationContext);
                const auto result = render_raw_prepared(runtime.cuda, frame, submission, prepared, {}, &error, inputs.hostPreparation);
                if (result.status.category != FJ_STATUS_SUCCESS) {
                    diagnostic.assign(message.data(), error.length);
                    return false;
                }
                if (!render_held_state(runtime.assets, runtime.cuda, invocation.owner, state, frame, submission, state.recipe_hash, error, diagnostic))
                    return false;
                if (fj_test_noise_acquisition_count() != acquisitions) {
                    diagnostic = "held-state boundary performed a native noise fallback acquisition";
                    return false;
                }
                return true;
            },
            diagnostic);
        if (success && fj_test_noise_acquisition_count() != baseline + (state.grain.active ? 1u : 0u)) {
            diagnostic = "noise fixture activity acquisition count changed";
            return false;
        }
        return success;
    }

} // namespace JuicerCudaTest
