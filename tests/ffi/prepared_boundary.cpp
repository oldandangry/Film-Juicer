#include "prepared_boundary.h"
#include "juicer_test_api.h"

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
#include "juicer_cuda_owner.h"
#include "Cuda/JuicerCudaHostViews.h"
#include "FocusedRenderPayload.h"
#include "ResourceAssetLibrary.h"
#include "ColorTransforms.h"

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


    FjStringView project(std::string_view source) {
        return source.empty() ? FjStringView{} : FjStringView{source.data(), source.size()};
    }

    FjRect project(const JuicerCuda::FrameRect& source) {
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

} // namespace

namespace JuicerCudaTest {

    bool execute_boundary(
        const RenderRecipe& recipe,
        const FocusedRenderPayload& payload,
        const JuicerCuda::ExecutionFrame& frame,
        JuicerCuda::ResourceManager::SubmissionSnapshot& snapshot,
        const JuicerCuda::PreparedDescriptors& descriptors,
        std::string& diagnostic,
        bool checkContract) {
        diagnostic.clear();
        const auto sourceBaseline = fj_test_noise_acquisition_count();
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
        struct NoiseRelease {
            void operator()(FjNoise* owner) const noexcept {
                (void)fj_test_noise_release(owner, nullptr);
            }
        };
        std::unique_ptr<FjNoise, NoiseRelease> noiseOwner;
        FjStaticNoise noise{};
        if (recipe.visualGrain.active) {
            constexpr std::string_view root = JUICER_TEST_RESOURCE_DIR;
            std::array<char, 512> message{};
            FjErrorBuffer error{message.data(), message.size(), 0};
            FjNoise* acquired = nullptr;
            const auto result = fj_test_noise_acquire({root.data(), root.size()}, &acquired, &error);
            noiseOwner.reset(acquired);
            if (result.category != FJ_STATUS_SUCCESS ||
                fj_test_noise_view(noiseOwner.get(), &noise, &error).category != FJ_STATUS_SUCCESS) {
                diagnostic = message.data();
                return false;
            }
        }
        if (fj_test_noise_acquisition_count() != sourceBaseline + (recipe.visualGrain.active ? 1u : 0u)) {
            diagnostic = "noise fixture activity acquisition count changed";
            return false;
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
            prepared.noise = noise;
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
        const auto sourceAcquisitions = fj_test_noise_acquisition_count();
        check_frame_bindings(JuicerCuda::borrowed_owner(), context, projectedFrame, submission, prepared, recipe);
        const int result = checkContract
                               ? check_render_contract(JuicerCuda::borrowed_owner(), context, projectedFrame, submission, prepared, &error).category == FJ_STATUS_SUCCESS
                               : fj_test_execute_prepared_c(JuicerCuda::borrowed_owner(), &prepared, &projectedFrame, &context, &submission, &error);
        if (fj_test_noise_acquisition_count() != sourceAcquisitions) {
            diagnostic = "supplied noise performed a fallback acquisition";
            return false;
        }
        if (result == 0) {
            diagnostic.assign(message.data(), error.length);
        }
        return result != 0;
    }

} // namespace JuicerCudaTest
