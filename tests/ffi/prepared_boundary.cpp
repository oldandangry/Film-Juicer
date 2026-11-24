#include "prepared_boundary.h"
#include "prepared_raw_render_fixture.h"
#include "../ofx/scanner_reference_capture.h"
#include "juicer_test_api.h"
#include "juicer_cuda_call_scope.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "prepared_descriptors.h"
#include "prepared_frame_fixture.h"
#include "spatial_optics_fixture.h"
#include "test_asset_workspace.h"
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
        const auto& scanner = state.scanner_tables;
        prepared.scanner_spectra = {scanner.dyes_cmy[2], scanner.dyes_cmy[1], scanner.dyes_cmy[0], scanner.weighted_xyz[0], scanner.weighted_xyz[1], scanner.weighted_xyz[2], scanner.has_baseline ? scanner.baseline_min : FjFloatSpan{}, scanner.inv_yn};
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
        if (contract)
            check_scan_readback(runtime.cuda, context, frame, submission, prepared);
        if (lifetime)
            check_prepared_state_lifetime(owner, runtime.assets, runtime.cuda, context, frame, submission, prepared, &error, lifetimeCase);
        else if (!render_held_state(runtime.assets, runtime.cuda, *owner, state, frame, submission, state.recipe_hash, error, diagnostic))
            return false;
        return true;
    }

    static bool execute_boundary(
        FjRuntimeView runtime,
        const JuicerTestSupport::PreparedStateHold& owner,
        const FjFrame& frame,
        JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        std::string& diagnostic,
        const std::string& invocationPhase,
        bool grainDispatchContract,
        const JuicerTestSupport::PreparationInput* sourceParams,
        JuicerCuda::RenderTest::PrintLaunchObservation* printObservation) {
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
            JuicerCuda::RenderTest::PrintLaunchObservation* printObservation;
        } invocation{owner, runtime, frame, snapshot, invocationPhase, grainDispatchContract, printObservation};
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
                if (!invocation.printObservation)
                    check_frame_bindings(runtime.cuda, context, frame, submission, prepared, &state, inputs.hostPreparation, invocationContext);
                if (invocation.phase == "negative-print/cold" &&
                    !check_optional_scanner_base(runtime.cuda, frame, submission, prepared, inputs.hostPreparation, diagnostic))
                    return false;
                if (invocation.grainDispatchContract)
                    check_grain_dispatch(runtime.cuda, frame, submission, prepared, inputs.hostPreparation);
                JuicerCuda::RenderTest::PrintLaunchObservation printLaunch;
                const bool observePrint = invocation.printObservation != nullptr;
                std::optional<ofx_test::ScanReferenceCapture> printCapture;
                if (observePrint)
                    printCapture.emplace(reinterpret_cast<cudaStream_t>(frame.stream), true);
                if (observePrint)
                    JuicerCuda::PinnedUploadTest::reset_upload_observation();
                if (observePrint)
                    JuicerCuda::RenderTest::set_print_launch_observer(&printLaunch);
                const auto result = render_raw_prepared(runtime.cuda, frame, submission, prepared, {}, &error, inputs.hostPreparation);
                if (observePrint) {
                    JuicerCuda::RenderTest::set_print_launch_observer(nullptr);
                    printLaunch.uploads = JuicerCuda::PinnedUploadTest::finish_upload_observation();
                }
                if (result.status.category != FJ_STATUS_SUCCESS) {
                    diagnostic.assign(message.data(), error.length);
                    return false;
                }
                if (observePrint) {
                    JuicerCuda::RenderTest::complete_print_launch_observation(printLaunch, frame.stream);
                    printLaunch.printCyanTap = printCapture->print_cyan();
                    const auto& print = state.print;
                    if (!printLaunch.valid || printLaunch.spectralSampleCount != 81 || printLaunch.filmDensityHash == 0 ||
                        printLaunch.profileTablesHash == 0 || printLaunch.mainIlluminantHash == 0 ||
                        (print.preflash_exposure > 0.0f && printLaunch.preflashIlluminantHash == 0)) {
                        diagnostic = "negative-print launch did not expose complete resident Print resources";
                        return false;
                    }
                    if (std::bit_cast<std::uint32_t>(printLaunch.printExposure) != std::bit_cast<std::uint32_t>(print.exposure) ||
                        std::bit_cast<std::uint32_t>(printLaunch.printMidgrayFactor) != std::bit_cast<std::uint32_t>(print.normalizer) ||
                        std::bit_cast<std::uint32_t>(printLaunch.printPreflashExposure) != std::bit_cast<std::uint32_t>(print.preflash_exposure)) {
                        diagnostic = "negative-print launch scalar words differ from held Print state";
                        return false;
                    }
                    for (std::size_t channel = 0; channel < 3; ++channel) {
                        if (std::bit_cast<std::uint32_t>(printLaunch.printPreflashRaw[channel]) !=
                            std::bit_cast<std::uint32_t>(print.preflash_raw_cmy[channel])) {
                            diagnostic = "negative-print launch preflash C/M/Y words differ from held Print state";
                            return false;
                        }
                    }
                    if (invocation.phase == "negative-print/shipping-balance-b" &&
                        (std::bit_cast<std::uint32_t>(printLaunch.printExposure) != 0x40000000u ||
                         std::bit_cast<std::uint32_t>(printLaunch.printMidgrayFactor) != 0x3f56cddfu ||
                         std::bit_cast<std::uint32_t>(printLaunch.printExposure * printLaunch.printMidgrayFactor) != 0x3fd6cddfu)) {
                        diagnostic = "shipping balance B launch words do not match the reviewed guarded product";
                        return false;
                    }
                    if ((invocation.phase == "negative-print/shipping-balance-a" ||
                         invocation.phase == "negative-print/shipping-balance-a-restored") &&
                        (std::bit_cast<std::uint32_t>(print.exposure) != 0x40000000u || state.print_normalization_mode != 0u)) {
                        diagnostic = "shipping balance A did not retain fixed exposure and normalization mode zero";
                        return false;
                    }
                    if (invocation.phase == "negative-print/shipping-balance-b" && state.print_normalization_mode != 3u) {
                        diagnostic = "shipping balance B did not select normalization mode three";
                        return false;
                    }
                    *invocation.printObservation = printLaunch;
                    return true;
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
        if (success && sourceParams && invocationPhase == "negative-print/cold") {
            auto* const stream = reinterpret_cast<cudaStream_t>(frame.stream);
            const auto& destination = frame.destination;
            const auto rows = static_cast<std::size_t>(destination.bounds.y2 - destination.bounds.y1);
            const auto stride = static_cast<std::size_t>(destination.row_bytes) / sizeof(float);
            const auto width = static_cast<std::size_t>(destination.bounds.x2 - destination.bounds.x1);
            constexpr std::size_t channels = FJ_COMPONENTS_RGBA;
            if (width != 7 || rows != 5 || destination.components != channels || stride <= width * channels) {
                diagnostic = "shipping Print balance requires the reviewed guarded 7 by 5 RGBA fixture";
                return false;
            }
            struct Destination {
                float* data = nullptr;
                ~Destination() {
                    if (data)
                        (void)cudaFree(data);
                }
            } variantDestination;
            const std::size_t bytes = rows * stride * sizeof(float);
            if (cudaMalloc(&variantDestination.data, bytes) != cudaSuccess) {
                diagnostic = "allocate shipping Print balance destination";
                return false;
            }
            constexpr float canary = -8.0f;
            const std::vector<float> initial(rows * stride, canary);
            std::array<std::vector<float>, 3> outputs;
            std::array<JuicerCuda::RenderTest::PrintLaunchObservation, 3> launches;
            SubmissionOwner variantOwner;
            variantOwner.cuda = runtime.cuda;
            variantOwner.token = snapshot.instanceToken.value ^ 0x5342414c414e4345ull;
            auto render_variant = [&](const JuicerTestSupport::PreparationInput& variantParams,
                                      std::size_t index,
                                      std::string_view phase) {
                FjStatus prepareStatus{};
                std::string prepareDiagnostic;
                auto variant = prepare_state(runtime.assets, variantParams, prepareStatus, prepareDiagnostic);
                if (prepareStatus.category != FJ_STATUS_SUCCESS || !variant) {
                    diagnostic = prepareDiagnostic.empty() ? "shipping Print variant preparation failed" : prepareDiagnostic;
                    return false;
                }
                FjPreparedStateView variantState{};
                std::array<char, 2048> stateText{};
                FjErrorBuffer stateError{stateText.data(), stateText.size(), 0};
                if (variant->view(variantState, stateError).category != FJ_STATUS_SUCCESS) {
                    diagnostic.assign(stateText.data(), stateError.length);
                    return false;
                }
                FjFrame variantFrame = frame;
                variantFrame.destination.address = reinterpret_cast<std::uintptr_t>(variantDestination.data);
                const JuicerCuda::FrameRect bounds{variantFrame.source.bounds.x1,
                                                   variantFrame.source.bounds.y1,
                                                   variantFrame.source.bounds.x2,
                                                   variantFrame.source.bounds.y2};
                const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(
                    bounds, bounds, static_cast<Spektrafilm::AutoExposureMethod>(variantState.input_policy.meter));
                auto variantSnapshot = snapshot;
                variantSnapshot.instanceToken.value = variantOwner.token;
                variantSnapshot.snapshotId += index + 1;
                variantSnapshot.frameToken.value = static_cast<std::uint64_t>(static_cast<std::int64_t>(std::floor(variantFrame.time_frames)));
                variantSnapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(
                    variantState.upload_core_hash, variantState.dir_recipe.hash, variantState.scanner_hash, meter.hash);
                if (cudaMemcpyAsync(variantDestination.data, initial.data(), bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
                    !execute_boundary(runtime, *variant, variantFrame, variantSnapshot, diagnostic, std::string(phase), false, nullptr, &launches[index])) {
                    if (diagnostic.empty())
                        diagnostic = "shipping Print guarded render failed";
                    return false;
                }
                auto& output = outputs[index];
                output.resize(rows * stride);
                if (cudaMemcpy(output.data(), variantDestination.data, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
                    diagnostic = "read shipping Print balance destination";
                    return false;
                }
                for (std::size_t row = 0; row < rows; ++row) {
                    const auto start = output.begin() + static_cast<std::ptrdiff_t>(row * stride);
                    const auto active = start + static_cast<std::ptrdiff_t>(width * channels);
                    if (!std::all_of(start, active, [](float value) {
                            return std::isfinite(value) && value != canary;
                        }) ||
                        !std::all_of(active, start + static_cast<std::ptrdiff_t>(stride), [](float value) {
                            return value == canary;
                        })) {
                        diagnostic = "shipping Print balance wrote invalid output or pitch guards";
                        return false;
                    }
                }
                return true;
            };
            JuicerTestSupport::PreparationInput balanceA = *sourceParams;
            balanceA.values.spectral_upsampling_mode = 0;
            balanceA.values.print_exposure = 2.0;
            balanceA.values.print_preflash_exposure = 0.125;
            balanceA.values.glare_active = 0;
            balanceA.values.normalize_print_exposure = 0;
            balanceA.values.print_exposure_compensation = 0;
            JuicerTestSupport::PreparationInput balanceB = balanceA;
            balanceB.values.normalize_print_exposure = 1;
            balanceB.values.print_exposure_compensation = 1;
            if (!render_variant(balanceA, 0, "negative-print/shipping-balance-a") ||
                !render_variant(balanceB, 1, "negative-print/shipping-balance-b") ||
                !render_variant(balanceA, 2, "negative-print/shipping-balance-a-restored")) {
                return false;
            }
            const auto same_pixels = [bytes](const auto& a, const auto& b) {
                return std::memcmp(a.data(), b.data(), bytes) == 0;
            };
            if (!same_pixels(outputs[0], outputs[2])) {
                diagnostic = "shipping Print balance did not restore A exactly";
                return false;
            }
            for (const auto& launch : std::span(launches).subspan(1)) {
                const auto& first = launches.front();
                if (launch.uploads.print != JuicerCuda::PinnedUploadTest::TransferCount{}) {
                    diagnostic = "shipping Print balance caused Print upload calls or bytes";
                    return false;
                }
                if (launch.mainIlluminant != first.mainIlluminant || launch.preflashIlluminant != first.preflashIlluminant ||
                    launch.mainIlluminantHash != first.mainIlluminantHash || launch.preflashIlluminantHash != first.preflashIlluminantHash ||
                    launch.mainLight != first.mainLight || launch.preflashLight != first.preflashLight ||
                    launch.profileTablesHash != first.profileTablesHash || launch.filmDensityHash != first.filmDensityHash) {
                    diagnostic = "shipping Print balance changed the resident light or film and paper table identities";
                    return false;
                }
            }
            constexpr double expectedContrast = 0.29777477608705993;
            const double a = launches[0].printCyanTap;
            const double b = launches[1].printCyanTap;
            const double k = launches[1].printMidgrayFactor;
            const double contrast = b - k * a;
            const double roundingLimit = 4.0 * std::numeric_limits<float>::epsilon() * (std::abs(b) + std::abs(k * a));
            if (std::abs(contrast - expectedContrast) > roundingLimit ||
                std::abs(contrast - expectedContrast / 2.0) <= roundingLimit ||
                std::bit_cast<std::uint32_t>(launches[2].printCyanTap) !=
                    std::bit_cast<std::uint32_t>(launches[0].printCyanTap)) {
                diagnostic = "shipping Print balance cyan contrast " + std::to_string(contrast) +
                             " failed the reviewed exposure and Float32 bounds";
                return false;
            }
            std::printf("shipping Print cyan (1,1) A=%.9g B=%.9g contrast=%.17g limit=%.9g\n", a, b, contrast, roundingLimit);
            if (!variantOwner.close()) {
                diagnostic = "retire shipping Print balance resource membership";
                return false;
            }
        }
        return success;
    }

    bool execute_boundary(
        FjRuntimeView runtime,
        const JuicerTestSupport::PreparedStateHold& owner,
        const FjFrame& frame,
        JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        std::string& diagnostic,
        const std::string& invocationPhase,
        bool grainDispatchContract,
        const JuicerTestSupport::PreparationInput* sourceParams) {
        return execute_boundary(runtime, owner, frame, snapshot, diagnostic, invocationPhase, grainDispatchContract, sourceParams, nullptr);
    }

    void check_shipping_print_preflash(const std::filesystem::path& resources, const JuicerTestSupport::PreparationInput& sourceParams) {
        using JuicerCuda::PinnedUploadTest::TransferCount;
        using nlohmann::json;
        const auto require = [](bool condition, std::string_view detail) {
            if (!condition)
                throw std::runtime_error(std::string(detail));
        };
        const char* const artifacts = std::getenv("JUICER_TEST_ARTIFACT_DIR");
        require(artifacts && *artifacts, "shipping Print preflash requires the native test artifact directory");
        const auto root = JuicerTestSupport::test_asset_workspaces.create(std::filesystem::path(artifacts) / "print-preflash");
        JuicerTestSupport::copy_print_preparation_assets(resources, root);
        const auto profile = root / "profiles/kodak_portra_400.json";
        std::ifstream input(profile, std::ios::binary);
        require(static_cast<bool>(input), "open disposable Portra film profile");
        const std::string original{std::istreambuf_iterator<char>(input), {}};
        input.close();
        auto edited = json::parse(original);
        auto& base = edited.at("data").at("base_density");
        require(base.is_array() && base.size() == 81, "disposable Portra base has the expected 81 wavelengths");
        for (auto& sample : base) {
            if (!sample.is_null())
                sample = sample.get<double>() + 0.125;
        }

        struct Buffers {
            cudaStream_t stream = nullptr;
            float* source = nullptr;
            float* destination = nullptr;
            ~Buffers() {
                if (stream)
                    (void)cudaStreamSynchronize(stream);
                (void)cudaFree(destination);
                (void)cudaFree(source);
                if (stream)
                    (void)cudaStreamDestroy(stream);
            }
        } buffers;
        constexpr int width = 7;
        constexpr int height = 5;
        constexpr std::size_t channels = 4;
        constexpr std::size_t stride = width * channels + 5;
        constexpr float guard = -8.0f;
        std::array<float, stride * height> source{};
        source.fill(guard);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const auto pixel = static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(x) * channels;
                for (int c = 0; c < 3; ++c)
                    source[pixel + static_cast<std::size_t>(c)] = static_cast<float>((x + 3 * y + 7 * c) % 19) / 19.0f;
                source[pixel + 3] = 0.125f + static_cast<float>((x + y) % 4) * 0.25f;
            }
        }
        require(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking) == cudaSuccess &&
                    cudaMalloc(&buffers.source, sizeof(source)) == cudaSuccess && cudaMalloc(&buffers.destination, sizeof(source)) == cudaSuccess &&
                    cudaMemcpyAsync(buffers.source, source.data(), sizeof(source), cudaMemcpyHostToDevice, buffers.stream) == cudaSuccess,
                "prepare guarded Print preflash input on the supplied stream");
        auto initial = source;
        initial.fill(guard);

        JuicerTestSupport::RuntimeFixture owner;
        owner.create(root);
        const auto runtime = owner.borrow();
        const auto reload = [&](const std::string& document) {
            std::ofstream output(profile, std::ios::binary | std::ios::trunc);
            output.write(document.data(), static_cast<std::streamsize>(document.size()));
            output.close();
            require(static_cast<bool>(output) && fj_test_assets_release_cached_payloads(runtime.assets, nullptr).category == FJ_STATUS_SUCCESS,
                    "refresh actual Assets after the disposable film edit");
        };
        auto controls = sourceParams;
        controls.values.spectral_upsampling_mode = 0;
        controls.values.glare_active = 0;
        controls.values.print_preflash_exposure = 0.125;
        const auto prepare = [&] {
            FjStatus status{};
            std::string detail;
            auto prepared = prepare_state(runtime.assets, controls, status, detail);
            require(status.category == FJ_STATUS_SUCCESS && prepared.has_value(), detail.empty() ? "prepare shipping preflash product" : detail);
            return std::move(*prepared);
        };

        constexpr FjRect bounds{0, 0, width, height};
        constexpr FjExtent full{0, 0, width, height};
        FjFrame frame{};
        frame.source = {reinterpret_cast<std::uintptr_t>(buffers.source), bounds, stride * sizeof(float), channels, FJ_DEPTH_FLOAT32};
        frame.destination = {reinterpret_cast<std::uintptr_t>(buffers.destination), bounds, stride * sizeof(float), channels, FJ_DEPTH_FLOAT32};
        frame.render_window = bounds;
        frame.full_frame_extent = bounds;
        frame.stream = reinterpret_cast<std::uintptr_t>(buffers.stream);
        frame.flags = FJ_FRAME_STREAM_PRESENT;
        frame.frame_rate = 24.0;
        frame.session_seed = 0x20260923;
        frame.clip_token = 0x5312;
        frame.effects_geometry = {full, 0.0, 0.0, width, height, 1.0, 1.0, 1.0};
        FjCudaContext context{};
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot;
        snapshot.instanceToken.value = 0x5348464c415348ull;
        snapshot.frameToken.value = 0;
        using Observation = JuicerCuda::RenderTest::PrintLaunchObservation;
        struct Sample {
            Observation launch;
            std::array<float, stride * height> output{};
        };
        const auto render = [&](const JuicerTestSupport::PreparedStateHold& product) {
            FjPreparedStateView state{};
            FjPreparedFramePolicy policy{};
            FjErrorBuffer error{};
            require(product.view(state, error).category == FJ_STATUS_SUCCESS && product.frame_policy(full, policy, error).category == FJ_STATUS_SUCCESS,
                    "read held shipping Print preflash frame values");
            frame.pixel_size_um = policy.geometry.pixel_size_um;
            require(fj_cuda_inspect(runtime.cuda, &frame, &context, nullptr).category == FJ_STATUS_SUCCESS, "inspect shipping preflash CUDA owner");
            snapshot.deviceContextKey = {context.device_id, reinterpret_cast<void*>(context.context)};
            ++snapshot.snapshotId;
            const JuicerCuda::FrameRect extent{0, 0, width, height};
            const auto meter = JuicerCuda::make_auto_exposure_preview_descriptor(extent, extent, static_cast<Spektrafilm::AutoExposureMethod>(state.input_policy.meter));
            snapshot.keyDigests = JuicerCuda::ResourceManager::make_key_digests(state.upload_core_hash, state.dir_recipe.hash, state.scanner_hash, meter.hash);
            require(cudaMemcpyAsync(buffers.destination, initial.data(), sizeof(initial), cudaMemcpyHostToDevice, buffers.stream) == cudaSuccess,
                    "reset shipping preflash output canaries");
            Sample sample;
            std::string detail;
            const bool rendered = execute_boundary(runtime, product, frame, snapshot, detail, "negative-print/shipping-preflash", false, nullptr, &sample.launch);
            require(rendered, detail);
            require(cudaMemcpy(sample.output.data(), buffers.destination, sizeof(sample.output), cudaMemcpyDeviceToHost) == cudaSuccess,
                    "read completed shipping preflash output");
            for (std::size_t word = 0; word < sample.output.size(); ++word) {
                const auto x = word % stride;
                const auto value = sample.output[word];
                if (x >= width * channels || x % channels == 3)
                    require(std::bit_cast<std::uint32_t>(value) == std::bit_cast<std::uint32_t>(source[word]), "shipping preflash modified alpha or pitch guards");
                else
                    require(std::isfinite(value) && value != guard, "shipping preflash left an invalid output word");
            }
            return sample;
        };
        const auto same_lights = [](const Observation& a, const Observation& b) {
            return a.mainIlluminant == b.mainIlluminant && a.mainIlluminantHash == b.mainIlluminantHash && a.mainLight == b.mainLight &&
                   a.preflashIlluminant == b.preflashIlluminant && a.preflashIlluminantHash == b.preflashIlluminantHash && a.preflashLight == b.preflashLight;
        };
        {
            const auto heldA = prepare();
            const auto a = render(heldA);
            reload(edited.dump());
            const auto b = render(prepare());
            reload(original);
            const auto restored = render(prepare());
            const auto raw_words = [](const Observation& launch) {
                std::array<std::uint32_t, 3> words{};
                std::transform(std::begin(launch.printPreflashRaw), std::end(launch.printPreflashRaw), words.begin(), [](float value) {
                    return std::bit_cast<std::uint32_t>(value);
                });
                return words;
            };
            const auto aRaw = raw_words(a.launch);
            const auto bRaw = raw_words(b.launch);
            require(aRaw[0] == 0x40ecdb2fu && aRaw == raw_words(restored.launch) &&
                        bRaw[0] != bRaw[1] && bRaw[1] != bRaw[2] && bRaw[0] != bRaw[2] &&
                        bRaw[0] != aRaw[0] && bRaw[1] != aRaw[1] && bRaw[2] != aRaw[2] && b.launch.printPreflashExposure == 0.125f,
                    "shipping preflash C/M/Y words or amount do not match the A/B/A controls");
            require(same_lights(a.launch, b.launch) && same_lights(a.launch, restored.launch), "shipping preflash changed the resident light slots");
            constexpr TransferCount oneLane{1, 81 * sizeof(float)};
            for (const auto* changed : {&b.launch, &restored.launch}) {
                require(changed->uploads.mainIlluminant == TransferCount{} && changed->uploads.preflashIlluminant == TransferCount{} &&
                            changed->uploads.print == TransferCount{4, std::size_t{4} * 81 * sizeof(float)} &&
                            std::all_of(changed->uploads.filmDensity.begin(), changed->uploads.filmDensity.end(), [oneLane](TransferCount lane) {
                                return lane == oneLane;
                            }),
                        "shipping preflash did not reuse both lights and upload exactly four film lanes");
            }
            const auto same_words = [](const auto& lhs, const auto& rhs) {
                return std::memcmp(lhs.data(), rhs.data(), sizeof(lhs)) == 0;
            };
            require(a.launch.filmDensityHash != b.launch.filmDensityHash && a.launch.filmDensityHash == restored.launch.filmDensityHash &&
                        !same_words(a.launch.filmBaseWords, b.launch.filmBaseWords) && same_words(a.launch.filmBaseWords, restored.launch.filmBaseWords) &&
                        same_words(a.launch.filmDyes, b.launch.filmDyes) && same_words(a.launch.filmDyes, restored.launch.filmDyes) &&
                        std::memcmp(a.output.data(), restored.output.data(), sizeof(a.output)) == 0,
                    "shipping preflash did not restore the exact film base and guarded A output");
            controls.values.print_preflash_exposure = 0.0;
            for (const auto* phase : {"disabled", "disabled warm"}) {
                const auto disabled = render(prepare()).launch;
                require(disabled.preflashIlluminant == 0 && disabled.preflashIlluminantHash == 0 &&
                            disabled.printPreflashExposure == 0.0f && raw_words(disabled) == std::array<std::uint32_t, 3>{} &&
                            disabled.uploads.print == TransferCount{},
                        phase);
            }
        }
        require(fj_cuda_retire_instance(runtime.cuda, snapshot.instanceToken.value, nullptr).category == FJ_STATUS_SUCCESS &&
                    owner.close().category == FJ_STATUS_SUCCESS,
                "retire the disposable shipping preflash owner");
        require(JuicerTestSupport::test_asset_workspaces.finish(0, true) == 0, "clean the drained disposable preflash workspace");
        std::puts("shipping Print preflash A/B/restored-A and disabled/warm passed with zero light upload calls and bytes");
    }

} // namespace JuicerCudaTest
