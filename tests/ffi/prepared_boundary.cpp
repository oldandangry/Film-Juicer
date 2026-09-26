#include "prepared_boundary.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

#include "prepared_descriptors.h"
#include "juicer_cuda_prepared.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "FocusedRenderPayload.h"
#include "ResourceAssetLibrary.h"
#include "SpectralProcessing.h"

namespace {

    FjFloatSpan project(std::span<const float> source) {
        return source.empty() ? FjFloatSpan{} : FjFloatSpan{source.data(), source.size()};
    }

    FjFloatSpan project(const JuicerCuda::ThreeChannelSamplesView& source) {
        // Transport the contiguous scalar representation without indexing it as
        // a float array. The native decoder reads fixed triplets through bytes.
        const auto bytes = source.object_bytes();
        return bytes.empty() ? FjFloatSpan{} : FjFloatSpan{reinterpret_cast<const float*>(bytes.data()), source.scalar_count()};
    }

    FjByteSpan project(std::span<const std::uint8_t> source) {
        return source.empty() ? FjByteSpan{} : FjByteSpan{source.data(), source.size()};
    }

    FjStringView project(std::string_view source) {
        return source.empty() ? FjStringView{} : FjStringView{source.data(), source.size()};
    }

    FjRect project(const JuicerCuda::FrameRect& source) {
        return {source.x1, source.y1, source.x2, source.y2};
    }

    JuicerCuda::FrameRect decode(const FjRect& source) {
        return {source.x1, source.y1, source.x2, source.y2};
    }

    void encode_bounds(const DensityBoundsRecipe& source, FjDensityBounds& out) {
        std::copy(source.dataMinCmy.begin(), source.dataMinCmy.end(), out.min_cmy);
        std::copy(source.invSpanCmy.begin(), source.invSpanCmy.end(), out.inv_span_cmy);
        out.hash = source.hash;
    }

    void encode_film(const RenderRecipe& recipe, const FocusedRenderPayload& payload, const JuicerCuda::FocusedRouteResourceInput& focused, FjPreparedHostData& out) {
        auto& exposure = out.film_exposure;
        exposure.input_color_space = static_cast<std::uint32_t>(recipe.filmRaw.inputColorSpace);
        exposure.method = static_cast<std::uint32_t>(recipe.filmRaw.rgbToRawMethod);
        exposure.flags = (recipe.filmRaw.inputCctfDecoding ? FJ_FILM_DECODE_CCTF : 0u) |
                         (payload.filmRawConfig.applyInputChromaticAdapt ? FJ_FILM_ADAPT_INPUT : 0u) |
                         (recipe.filmRaw.autoExposureEnabled ? FJ_FILM_AUTO_EXPOSURE : 0u);
        exposure.manual_exposure_ev = recipe.filmRaw.manualExposureCompensationEv;
        exposure.route_correction_scale = Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute)
                                              ? 1.0f
                                              : out.scanner_correction.exposure_scale;
        exposure.mallett_green_midgray_scale = recipe.filmRaw.mallettGreenMidgrayScale;
        std::copy_n(payload.filmRawConfig.inputRGBToXYZ.m, 9, exposure.input_rgb_to_xyz);
        std::copy_n(payload.filmRawConfig.inputXYZAdapt.m, 9, exposure.input_xyz_adapt);
        std::copy_n(payload.filmRawConfig.xyzToLinearSrgb.m, 9, exposure.xyz_to_linear_srgb);
        exposure.sensitivity_rgb = project(focused.filmRaw.finalSensitivityRgb);
        if (recipe.filmRaw.rgbToRawMethod == Spektrafilm::RgbToRawMethod::Mallett2019) {
            exposure.mallett_illuminant = project(focused.exposureIlluminant);
            exposure.mallett_basis = project(focused.mallettBasis);
        } else {
            exposure.tc_lut_rgba = project(focused.filmTcLut);
        }
        exposure.sensitivity_hash = recipe.filmRaw.finalSensitivityHash;
        exposure.tc_lut_hash = recipe.filmRaw.tcLutHash;
        exposure.hash = recipe.filmRaw.hash;
        auto& development = out.film_development;
        development.log_exposure = project(focused.filmDevelop.logExposure);
        development.density_rgb = project(focused.filmDevelop.normalizedDensityCurvesRgb);
        if (focused.wantDensityLayers) {
            for (std::size_t layer = 0; layer < 3; ++layer) {
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    development.density_layers[layer][channel] = project(focused.filmDevelop.densityCurvesLayers[layer][channel]);
                }
            }
            development.density_layers_hash = recipe.filmDevelop.densityCurvesLayersHash;
        }
        std::copy(recipe.filmDevelop.densityCurveGamma.begin(), recipe.filmDevelop.densityCurveGamma.end(), development.gamma_rgb);
        development.density_curves_hash = recipe.filmDevelop.normalizedDensityCurvesHash;
        development.hash = recipe.filmDevelop.hash;
        if (recipe.dirCouplers.active) {
            auto& dir = out.dir_couplers;
            dir.mode = static_cast<std::uint32_t>(recipe.dirCouplers.nonlinearMode);
            for (std::size_t channel = 0; channel < 3; ++channel) {
                std::copy(recipe.dirCouplers.matrixRgb[channel].begin(), recipe.dirCouplers.matrixRgb[channel].end(), dir.matrix_rgb + channel * 3u);
                dir.compensated_axes_rgb[channel] = project(focused.dirCouplers.compensatedDensityCurveAxesRgb[channel]);
            }
            std::copy(recipe.dirCouplers.densityMaxRgb.begin(), recipe.dirCouplers.densityMaxRgb.end(), dir.density_max_rgb);
            std::copy(recipe.dirCouplers.densityRefRgb.begin(), recipe.dirCouplers.densityRefRgb.end(), dir.density_ref_rgb);
            std::copy(recipe.dirCouplers.donorKRgb.begin(), recipe.dirCouplers.donorKRgb.end(), dir.donor_k_rgb);
            std::copy(recipe.dirCouplers.receiverCRefRgb.begin(), recipe.dirCouplers.receiverCRefRgb.end(), dir.receiver_c_ref_rgb);
            std::copy(recipe.dirCouplers.receiverKrRgb.begin(), recipe.dirCouplers.receiverKrRgb.end(), dir.receiver_kr_rgb);
            dir.compensated_axes_hash = recipe.dirCouplers.compensatedDensityCurveAxesHash;
            dir.hash = recipe.dirCouplers.hash;
        }
    }

    void encode_print(const JuicerCuda::PrintResourceInput& source, const PrintExposureRecipe& exposure, FjPrint& out) {
        out.film_density_cmy = project(source.filmChannelDensityCmy);
        out.film_base_density = project(source.filmBaseDensity);
        out.sensitivity_cmy = project(source.printSensitivityCmy);
        out.log_exposure = project(source.printLogExposure);
        out.density_cmy = project(source.printDensityCurvesCmy);
        out.main_illuminant = project(source.mainIlluminant);
        const auto& descriptors = source.descriptors;
        if (descriptors.preflashActive) {
            out.flags = FJ_PRINT_PREFLASH;
            out.preflash_illuminant = project(source.preflashIlluminant);
            std::copy(source.preflashRawCmy.begin(), source.preflashRawCmy.end(), out.preflash_raw_cmy);
        }
        out.main_cc_cmy[0] = descriptors.mainIlluminant.cmyCc.c;
        out.main_cc_cmy[1] = descriptors.mainIlluminant.cmyCc.m;
        out.main_cc_cmy[2] = descriptors.mainIlluminant.cmyCc.y;
        out.preflash_cc_cmy[0] = descriptors.preflashIlluminant.cmyCc.c;
        out.preflash_cc_cmy[1] = descriptors.preflashIlluminant.cmyCc.m;
        out.preflash_cc_cmy[2] = descriptors.preflashIlluminant.cmyCc.y;
        out.exposure = exposure.printExposure;
        out.preflash_exposure = exposure.preflashExposure;
        out.normalizer = source.normalizer;
        out.film_density_hash = descriptors.filmDensityTables.hash;
        out.profile_tables_hash = descriptors.profileTables.hash;
        out.main_illuminant_hash = descriptors.mainIlluminant.hash;
        out.preflash_illuminant_hash = descriptors.preflashIlluminant.hash;
        out.preflash_raw_hash = descriptors.preflashRaw.hash;
        out.balance_hash = descriptors.balance.hash;
        out.hash = descriptors.hash;
    }

    void write_error(FjErrorBuffer* error, std::string_view message) noexcept {
        if (!error) {
            return;
        }
        error->length = 0;
        if (error->capacity == 0 || !error->data) {
            return;
        }
        const std::size_t count = std::min(message.size(), error->capacity - 1u);
        std::copy_n(message.data(), count, error->data);
        error->data[count] = '\0';
        error->length = count;
    }

} // namespace

extern "C" int fj_test_execute_prepared_cpp(const FjPreparedHostData* prepared, const FjFrame* frame, const FjCudaContext* context, const FjSubmission* submission, FjErrorBuffer* error) {
    JuicerCuda::PendingContextLossRecovery recovery;
    try {
        if (!prepared || !frame || !context || !submission || context->device_id < 0 || context->context == 0 ||
            frame->source.address == 0 || frame->destination.address == 0 || frame->source.address == frame->destination.address ||
            frame->source.row_bytes <= 0 || frame->destination.row_bytes <= 0 ||
            (frame->source.components != FJ_COMPONENTS_RGB && frame->source.components != FJ_COMPONENTS_RGBA) ||
            frame->source.components != frame->destination.components ||
            frame->source.depth != FJ_DEPTH_FLOAT32 || frame->destination.depth != FJ_DEPTH_FLOAT32 ||
            (frame->flags & ~(FJ_FRAME_STREAM_PRESENT | FJ_FRAME_TRACE_INFO | FJ_FRAME_TRACE_VERBOSE)) != 0) {
            write_error(error, "unsupported prepared-boundary fixture input");
            return 0;
        }
        const auto& geometry = frame->effects_geometry;
        const Spektrafilm::FilmJuicerEffectsGeometry effectsGeometry{
            {geometry.pixel_definition.x, geometry.pixel_definition.y, geometry.pixel_definition.width, geometry.pixel_definition.height},
            geometry.canonical_x,
            geometry.canonical_y,
            geometry.canonical_width,
            geometry.canonical_height,
            geometry.scale_x,
            geometry.scale_y,
            geometry.pixel_aspect_ratio};
        const auto& meter = prepared->auto_exposure;
        const JuicerCuda::AutoExposurePreviewDescriptor metering{
            meter.source_bounds.x1, meter.source_bounds.y1, meter.source_bounds.x2, meter.source_bounds.y2, meter.meter_bounds.x1, meter.meter_bounds.y1, meter.meter_bounds.x2, meter.meter_bounds.y2, meter.preview_width, meter.preview_height, static_cast<Spektrafilm::AutoExposureMethod>(meter.method), meter.hash};
        const std::optional<Spektrafilm::DiffusionFrameSetDescriptor> diffusion;
        const std::optional<ScatterHalationFrameDescriptor> halation;
        const auto* sourceBase = reinterpret_cast<const unsigned char*>(frame->source.address);
        const std::ptrdiff_t sourceOffset =
            static_cast<std::ptrdiff_t>(frame->render_window.y1 - frame->source.bounds.y1) * frame->source.row_bytes +
            static_cast<std::ptrdiff_t>(frame->render_window.x1 - frame->source.bounds.x1) *
                static_cast<std::ptrdiff_t>(frame->source.components * sizeof(float));
        const JuicerCuda::ExecutionFrame execution{
            decode(frame->source.bounds), decode(frame->render_window), decode(frame->full_frame_extent), sourceBase, sourceBase + sourceOffset, reinterpret_cast<unsigned char*>(frame->destination.address), frame->source.row_bytes, frame->destination.row_bytes, static_cast<int>(frame->source.components), reinterpret_cast<void*>(frame->stream), diffusion, halation, effectsGeometry, frame->pixel_size_um, frame->time_frames, frame->frame_rate, frame->session_seed, static_cast<std::uintptr_t>(frame->clip_token), metering, (frame->flags & FJ_FRAME_TRACE_INFO) != 0, (frame->flags & FJ_FRAME_TRACE_VERBOSE) != 0};
        JuicerCuda::ResourceManager::SubmissionSnapshot snapshot{
            {submission->instance_token}, {submission->frame_token}, submission->submission_id, {context->device_id, reinterpret_cast<void*>(context->context)}, 1, {submission->upload_core_hash, submission->dir_hash, submission->scanner_hash, submission->auto_exposure_hash}};
        std::string diagnostic;
        if (!JuicerCuda::execute_prepared_host_data(*prepared, execution, snapshot, recovery, {}, diagnostic)) {
            write_error(error, diagnostic);
            return 0;
        }
        write_error(error, {});
        return 1;
    } catch (const JuicerCuda::ExecutionFailure&) {
        write_error(error, recovery.detail.empty() ? "prepared executor failed" : recovery.detail);
    } catch (const std::exception& exception) {
        write_error(error, exception.what());
    } catch (...) {
        write_error(error, "prepared boundary exception");
    }
    return 0;
}

namespace JuicerCudaTest {

    bool execute_boundary(
        const RenderRecipe& recipe,
        const FocusedRenderPayload& payload,
        const JuicerCuda::ExecutionFrame& frame,
        JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const JuicerCuda::PreparedDescriptors& descriptors,
        std::string& diagnostic) {
        diagnostic.clear();
        JuicerCuda::PreparedDescriptors preparedDescriptors = descriptors;
        JuicerCuda::FocusedRouteResourceInput focused;
        const JuicerCuda::FocusedRouteResourcePreparation request{
            &recipe, &payload.exposureTables, &payload.filmRawConfig, payload.filmTcLut ? &*payload.filmTcLut : nullptr, &payload.scannerTables, &payload.scannerColor, &preparedDescriptors.scanner, payload.outputBoundaryTable.get()};
        if (!JuicerCuda::build_focused_route_resource_input(request, focused, diagnostic)) {
            return false;
        }
        const bool print = Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute);
        std::array<float, 81> preflashIlluminant{};
        JuicerCuda::PrintResourceInput printInput;
        if (print) {
            if (!JuicerCuda::build_print_resource_input(
                    {&recipe, &JuicerProcess::root().assets(), payload.printMainIlluminant ? &*payload.printMainIlluminant : nullptr},
                    printInput,
                    preflashIlluminant,
                    diagnostic)) {
                return false;
            }
            const Scanner::PrintCorrectionDerivationInput correction{
                &recipe, &payload.scannerTables, printInput.mainIlluminant.data(), 81, printInput.preflashRawCmy.data(), printInput.normalizer};
            if (!Scanner::build_print_scanner_color_correction_descriptor(correction, preparedDescriptors.correction, diagnostic)) {
                return false;
            }
        }
        std::shared_ptr<const JuicerAssets::StaticNoisePayloadSet> noiseOwner;
        JuicerCuda::StaticNoiseInput noise;
        if (recipe.visualGrain.active) {
            noiseOwner = JuicerProcess::root().assets().static_noise_payloads();
            if (!noiseOwner || !JuicerCuda::build_static_noise_input(*noiseOwner, noise, diagnostic)) {
                return false;
            }
        }
        FjPreparedHostData prepared{};
        encode_prepared_descriptors(preparedDescriptors, prepared);
        prepared.film_profile_key = project(recipe.profileRoute.filmProfileKey);
        prepared.print_profile_key = print ? project(recipe.profileRoute.printProfileKey) : FjStringView{};
        prepared.film_profile_asset_version = recipe.profileRoute.filmProfileAssetVersionToken;
        prepared.print_profile_asset_version = print ? recipe.profileRoute.printProfileAssetVersionToken : 0;
        prepared.recipe_hash = recipe.hash;
        encode_film(recipe, payload, focused, prepared);
        encode_bounds(recipe.densityBounds, prepared.scanner_bounds);
        if (print) {
            encode_bounds(recipe.enlargerFilmBounds, prepared.enlarger_film_bounds);
            encode_print(printInput, recipe.print.exposure, prepared.print);
        }
        const auto& scanner = focused.scannerTables;
        prepared.scanner_spectra = {project(scanner.epsY), project(scanner.epsM), project(scanner.epsC), project(scanner.Ax), project(scanner.Ay), project(scanner.Az), scanner.hasBaseline ? project(scanner.baseDensityMin) : FjFloatSpan{}, scanner.invYn};
        if (focused.outputGamut.enabled) {
            prepared.output_color.gamut_cmax = project(focused.outputGamut.cmax);
            prepared.output_color.gamut_table_hash = focused.outputGamut.tableHash;
        }
        if (recipe.visualGrain.active) {
            prepared.noise = {project(noise.stbn), noise.stbnWidth, noise.stbnHeight, noise.stbnFrames, project(noise.wangTiles), project(noise.wangLut), noise.wangWidth, noise.wangHeight, static_cast<std::size_t>(noise.wangCount), noise.wangColors};
        }
        const auto& meter = frame.autoExposureDescriptor;
        prepared.auto_exposure = {{meter.sourceX1, meter.sourceY1, meter.sourceX2, meter.sourceY2},
                                  {meter.meterX1, meter.meterY1, meter.meterX2, meter.meterY2},
                                  meter.previewWidth,
                                  meter.previewHeight,
                                  static_cast<std::uint32_t>(meter.method),
                                  meter.hash};
        const auto& geometry = frame.effectsGeometry;
        const FjFrame projectedFrame{
            {reinterpret_cast<std::uintptr_t>(frame.sourceBase), project(frame.sourceBounds), frame.sourceRowBytes, static_cast<std::uint32_t>(frame.components), FJ_DEPTH_FLOAT32},
            {reinterpret_cast<std::uintptr_t>(frame.destination), project(frame.renderWindow), frame.destinationRowBytes, static_cast<std::uint32_t>(frame.components), FJ_DEPTH_FLOAT32},
            project(frame.renderWindow),
            project(frame.fullFrameExtent),
            reinterpret_cast<std::uintptr_t>(frame.stream),
            FJ_FRAME_STREAM_PRESENT | (frame.traceInfo ? FJ_FRAME_TRACE_INFO : 0u) | (frame.traceVerbose ? FJ_FRAME_TRACE_VERBOSE : 0u),
            frame.pixelSizeUm,
            frame.timeFrames,
            frame.frameRate,
            frame.sessionSeed,
            frame.clipToken,
            {{geometry.pixelDefinition.x, geometry.pixelDefinition.y, geometry.pixelDefinition.width, geometry.pixelDefinition.height},
             geometry.canonicalX,
             geometry.canonicalY,
             geometry.canonicalWidth,
             geometry.canonicalHeight,
             geometry.scaleX,
             geometry.scaleY,
             geometry.pixelAspectRatio}};
        const FjCudaContext context{snapshot.deviceContextKey.deviceId, reinterpret_cast<std::uintptr_t>(snapshot.deviceContextKey.contextOpaque)};
        const FjSubmission submission{snapshot.instanceToken.value, snapshot.frameToken.value, snapshot.snapshotId, snapshot.keyDigests.uploadCoreHash, snapshot.keyDigests.dirHash, snapshot.keyDigests.scannerHash, snapshot.keyDigests.autoExposureHash};
        std::array<char, 2048> message{};
        FjErrorBuffer error{message.data(), message.size(), 0};
        const int result = fj_test_execute_prepared_c(&prepared, &projectedFrame, &context, &submission, &error);
        if (result == 0) {
            diagnostic.assign(message.data(), error.length);
        }
        return result != 0;
    }

} // namespace JuicerCudaTest
