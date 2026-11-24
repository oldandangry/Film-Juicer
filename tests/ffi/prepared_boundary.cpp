#include "prepared_boundary.h"
#include "prepared_raw_render_fixture.h"
#include "juicer_test_api.h"
#include "juicer_cuda_call_scope.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "prepared_descriptors.h"
#include "prepared_frame_fixture.h"
#include "spatial_optics_fixture.h"
#include "juicer_cuda_owner.h"
#include "juicer_cuda_prepared.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "Hash.h"

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

    bool check_optional_scanner_base(FjCuda* cuda, const FjFrame& frame, const FjSubmission& originalSubmission, const FjPreparedHostData& original, const FjHostPreparation* host, std::string& diagnostic) {
        const auto& source = frame.source;
        const auto& destination = frame.destination;
        const auto& bounds = destination.bounds;
        const auto& window = frame.render_window;
        if (source.components != FJ_COMPONENTS_RGBA || destination.components != FJ_COMPONENTS_RGBA || source.row_bytes <= 0 || destination.row_bytes <= 0 ||
            bounds.x2 <= bounds.x1 || bounds.y2 <= bounds.y1 || bounds.x1 != window.x1 || bounds.x2 != window.x2 || bounds.y1 != window.y1 || bounds.y2 != window.y2 ||
            source.bounds.x1 != bounds.x1 || source.bounds.x2 != bounds.x2 || source.bounds.y1 != bounds.y1 || source.bounds.y2 != bounds.y2) {
            diagnostic = "optional scanner base fixture requires a matching pitched RGBA destination";
            return false;
        }
        const auto stride = static_cast<std::size_t>(destination.row_bytes);
        const auto sourceStride = static_cast<std::size_t>(source.row_bytes);
        const auto columns = static_cast<std::size_t>(bounds.x2 - bounds.x1);
        const auto rows = static_cast<std::size_t>(bounds.y2 - bounds.y1);
        const auto rgbaBytes = columns * 4 * sizeof(float);
        if (stride <= rgbaBytes || sourceStride < rgbaBytes) {
            diagnostic = "optional scanner base fixture is missing its pitch guard";
            return false;
        }
        auto* const stream = reinterpret_cast<cudaStream_t>(frame.stream);
        auto* const device = reinterpret_cast<void*>(destination.address);
        const auto* const sourceDevice = reinterpret_cast<const void*>(source.address);
        std::vector<std::byte> initial(stride * rows);
        std::vector<std::byte> sourceBytes(sourceStride * rows);
        if (cudaStreamSynchronize(stream) != cudaSuccess || cudaMemcpy(initial.data(), device, initial.size(), cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaMemcpy(sourceBytes.data(), sourceDevice, sourceBytes.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
            diagnostic = "could not read initial optional scanner base destination";
            return false;
        }

        const std::array<float, 81> zeros{};
        auto absent = original;
        absent.scanner_spectra.base_density = {};
        auto explicitZero = absent;
        explicitZero.scanner_spectra.base_density = {zeros.data(), zeros.size()};
        const auto select = [&originalSubmission](const FjPreparedHostData& value) {
            auto selected = originalSubmission;
            const auto& color = value.output_color;
            const std::uint64_t gamut = (color.flags & FJ_COLOR_GAMUT_COMPRESSION) != 0 ? color.gamut_recipe_hash : 0;
            const auto combined = Hash::hash_uint64_values({JuicerCuda::spectral_binding_digest(value), gamut});
            selected.scanner_hash = JuicerCuda::ResourceManager::make_key_digests(value.recipe_hash, value.dir_couplers.hash, combined, value.auto_exposure.hash).scannerHash;
            return selected;
        };
        const auto absentSubmission = select(absent);
        const auto zeroSubmission = select(explicitZero);
        if (absentSubmission.scanner_hash == zeroSubmission.scanner_hash) {
            diagnostic = "scanner base presence did not change the selected binding";
            return false;
        }
        std::array<char, 256> text{};
        FjErrorBuffer error{text.data(), text.size(), 0};
        unsigned abortQueries = 0;
        const FjAbortCallback abort{[](void* user) noexcept -> std::uint32_t {
                                        ++*static_cast<unsigned*>(user);
                                        return FJ_ABORT_CONTINUE;
                                    },
                                    &abortQueries};
        const auto rejected = JuicerCudaTest::render_raw_prepared(cuda, frame, absentSubmission, explicitZero, abort, &error, host);
        if (rejected.status.category != FJ_STATUS_UNSUPPORTED_INPUT || rejected.status.api != FJ_API_NONE || rejected.status.native_code != 0 ||
            abortQueries != 0 || std::string_view{text.data(), error.length} != "MalformedPreparedHostData field=submission_identity") {
            diagnostic = "explicit zero scanner base was accepted with the absent binding";
            return false;
        }

        std::vector<std::byte> withoutBase(initial.size());
        std::vector<std::byte> withBase(initial.size());
        const auto render = [&](const FjPreparedHostData& value, const FjSubmission& submission, std::vector<std::byte>& result) {
            if (cudaMemcpy(device, initial.data(), initial.size(), cudaMemcpyHostToDevice) != cudaSuccess)
                return false;
            const auto outcome = JuicerCudaTest::render_raw_prepared(cuda, frame, submission, value, {}, &error, host);
            return outcome.status.category == FJ_STATUS_SUCCESS && outcome.status.api == FJ_API_NONE && outcome.status.native_code == 0 &&
                   cudaStreamSynchronize(stream) == cudaSuccess && cudaMemcpy(result.data(), device, result.size(), cudaMemcpyDeviceToHost) == cudaSuccess;
        };
        if (!render(absent, absentSubmission, withoutBase) || !render(explicitZero, zeroSubmission, withBase) || withoutBase != withBase) {
            diagnostic = "coherent absent and explicit zero scanner bases did not produce identical pitched output";
            return false;
        }
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                const auto columnOffset = column * 4 * sizeof(float);
                const auto* output = withBase.data() + row * stride + columnOffset;
                std::array<float, 3> rgb{};
                std::memcpy(rgb.data(), output, sizeof(rgb));
                if (!std::ranges::all_of(rgb, [](float value) {
                        return std::isfinite(value);
                    }) ||
                    std::memcmp(output + sizeof(rgb), sourceBytes.data() + row * sourceStride + columnOffset + sizeof(rgb), sizeof(float)) != 0) {
                    diagnostic = "optional scanner base case changed alpha or produced nonfinite colour";
                    return false;
                }
            }
            const auto guard = row * stride + rgbaBytes;
            if (std::memcmp(initial.data() + guard, withBase.data() + guard, stride - rgbaBytes) != 0) {
                diagnostic = "optional scanner base case wrote the row guard";
                return false;
            }
        }
        if (cudaMemcpy(device, initial.data(), initial.size(), cudaMemcpyHostToDevice) != cudaSuccess) {
            diagnostic = "could not restore the original scanner base destination";
            return false;
        }
        std::cout << "CTEST_FULL_OUTPUT\noptional scanner base absent key=" << absentSubmission.scanner_hash << " present-zero key=" << zeroSubmission.scanner_hash << " matched pitched output\n";
        return true;
    }

} // namespace

namespace JuicerCudaTest {

    namespace {
        struct SubmissionObservation {
            unsigned calls = 0;
            FjSubmission submission{};
            std::uint32_t reused = 0;
        };

        thread_local SubmissionObservation* submissionObservation = nullptr;

        void trace_submission(void*, const FjSubmission* submission, std::uint32_t reused) noexcept {
            if (submissionObservation && submission) {
                ++submissionObservation->calls;
                submissionObservation->submission = *submission;
                submissionObservation->reused = reused;
            }
        }

        struct SubmissionOwner {
            FjInstance* instance = nullptr;
            FjCuda* cuda = nullptr;
            std::uint64_t token = 0;

            bool close() noexcept {
                FjErrorBuffer ignored{nullptr, 0, 0};
                FjCuda* retiredCuda = cuda;
                cuda = nullptr;
                const bool retired = !retiredCuda || fj_cuda_retire_instance(retiredCuda, token, &ignored).category == FJ_STATUS_SUCCESS;
                FjInstance* released = instance;
                instance = nullptr;
                const bool destroyed = !released || fj_test_instance_destroy(released, &ignored).category == FJ_STATUS_SUCCESS;
                return retired && destroyed;
            }

            ~SubmissionOwner() {
                (void)close();
            }
        };

        bool render_held_state(const FjAssets* assets, FjCuda* cuda, const JuicerTestSupport::PreparedStateHold& owner, const FjPreparedStateView& state, const FjFrame& frame, const FjSubmission& expected, std::uint64_t expectedRecipeHash, FjErrorBuffer& error, std::string& diagnostic) {
            const auto frameFloor = std::floor(frame.time_frames);
            const auto stateDigests = JuicerCuda::ResourceManager::make_key_digests(
                state.upload_core_hash, state.dir_recipe.hash, state.scanner_hash, expected.auto_exposure_hash);
            if (state.recipe_hash != expectedRecipeHash || stateDigests.uploadCoreHash != expected.upload_core_hash ||
                stateDigests.dirHash != expected.dir_hash || stateDigests.scannerHash != expected.scanner_hash || expected.instance_token == 0 ||
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
            // End raw-fixture membership before the new Rust owner's sequence starts.
            if (fj_cuda_retire_instance(cuda, expected.instance_token, &error).category != FJ_STATUS_SUCCESS) {
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
            SubmissionOwner submissionOwner;
            if (fj_test_instance_create(&submissionOwner.instance, &error).category != FJ_STATUS_SUCCESS || !submissionOwner.instance ||
                fj_test_instance_bind_identity(submissionOwner.instance, frame.session_seed, expected.instance_token, &error).category != FJ_STATUS_SUCCESS) {
                diagnostic = "held-state submission owner initialization failed";
                return false;
            }
            submissionOwner.cuda = cuda;
            submissionOwner.token = expected.instance_token;
            SubmissionObservation observation;
            submissionObservation = &observation;
            FjCudaCallAccess access = scope.access();
            access.trace_submission = trace_submission;
            const auto result = owner.render(assets, access, frame, policy, submissionOwner.instance, expected.frame_token, {}, error);
            submissionObservation = nullptr;
            const auto& selected = observation.submission;
            const bool matched = observation.calls == 1 && observation.reused == 0 && selected.submission_id == 1 &&
                                 selected.instance_token == expected.instance_token && selected.frame_token == expected.frame_token &&
                                 selected.upload_core_hash == expected.upload_core_hash && selected.dir_hash == expected.dir_hash &&
                                 selected.scanner_hash == expected.scanner_hash && selected.auto_exposure_hash == expected.auto_exposure_hash;
            if (result.outcome.status.category != FJ_STATUS_SUCCESS || result.fatal != 0 ||
                completion.calls != 1 || !completion.ordered || !matched || scope.recovery().pending) {
                diagnostic = scope.diagnostic().empty() ? std::string(error.data, error.length) : scope.diagnostic();
                if (diagnostic.empty())
                    diagnostic = "held rendering completion/submission observation failed";
                return false;
            }
            if (!submissionOwner.close()) {
                diagnostic = "held-state submission owner retirement failed";
                return false;
            }
            return true;
        }
    } // namespace

    bool execute_prepared_state(FjRuntimeView runtime, const JuicerTestSupport::PreparationInput& params, const FjFrame& frame, std::uint64_t instanceToken, bool contract, bool lifetime, std::string& diagnostic, StateLifetimeCase lifetimeCase) {
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
        const auto submissionDigests = JuicerCuda::ResourceManager::make_key_digests(state.upload_core_hash, dir.hash, state.scanner_hash, meter.hash);
        const FjSubmission submission{instanceToken, static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(frame.time_frames))), 1, submissionDigests.uploadCoreHash, submissionDigests.dirHash, submissionDigests.scannerHash, submissionDigests.autoExposureHash};
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
        const JuicerTestSupport::PreparedStateHold& owner,
        const FjFrame& frame,
        JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        std::string& diagnostic,
        const std::string& invocationPhase,
        bool grainDispatchContract) {
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
            const JuicerTestSupport::PreparedStateHold& owner;
            FjRuntimeView runtime;
            const FjFrame& frame;
            JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot;
            const std::string& phase;
            bool grainDispatchContract;
        } invocation{owner, runtime, frame, snapshot, invocationPhase, grainDispatchContract};
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
                const auto stateDigests = JuicerCuda::ResourceManager::make_key_digests(
                    state.upload_core_hash, state.dir_recipe.hash, state.scanner_hash, meter.hash);
                if (snapshot.instanceToken.value == 0 || snapshot.snapshotId == 0 ||
                    snapshot.keyDigests.uploadCoreHash != stateDigests.uploadCoreHash ||
                    snapshot.keyDigests.dirHash != stateDigests.dirHash ||
                    snapshot.keyDigests.scannerHash != stateDigests.scannerHash ||
                    snapshot.keyDigests.autoExposureHash != stateDigests.autoExposureHash ||
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
                if (invocation.phase == "negative-print/cold" &&
                    !check_optional_scanner_base(runtime.cuda, frame, submission, prepared, inputs.hostPreparation, diagnostic))
                    return false;
                if (invocation.grainDispatchContract)
                    check_grain_dispatch(runtime.cuda, frame, submission, prepared, inputs.hostPreparation);
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
