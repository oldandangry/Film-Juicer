#include "juicer_cuda_prepared.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>

#include "ColorTransforms.h"
#include "SpectralProcessing.h"
#include "Cuda/JuicerCudaHostViews.h"

namespace {

    static_assert(static_cast<std::uint32_t>(Spektrafilm::RgbToRawMethod::Hanatos2025) == FJ_RAW_HANATOS_2025);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::RgbToRawMethod::Mallett2019) == FJ_RAW_MALLETT_2019);
    static_assert(static_cast<std::uint32_t>(Spektrafilm::RgbToRawMethod::Arctic2026beta04) == FJ_RAW_ARCTIC_2026_BETA04);
    static_assert(static_cast<std::uint32_t>(DirNonlinearMode::Inactive) == FJ_DIR_INACTIVE);
    static_assert(static_cast<std::uint32_t>(DirNonlinearMode::NegativeDonorLangmuir) == FJ_DIR_NEGATIVE_DONOR_LANGMUIR);
    static_assert(static_cast<std::uint32_t>(DirNonlinearMode::PositiveReceiverLangmuir) == FJ_DIR_POSITIVE_RECEIVER_LANGMUIR);

    bool malformed(std::string& diagnostic, const char* field) {
        diagnostic = "MalformedPreparedHostData field=";
        diagnostic += field;
        return false;
    }

    template <typename Span>
    bool exact_span(const Span& span, std::size_t count, const char* field, std::string& diagnostic) {
        return (span.count == count && ((count == 0) == (span.data == nullptr))) ||
               malformed(diagnostic, field);
    }

    std::span<const float> floats(const FjFloatSpan& source) {
        return {source.data, source.count};
    }

    JuicerCuda::ThreeChannelSamplesView channel_samples(const FjFloatSpan& source) {
        // Five interleaved families may originate in native typed rows. Their
        // counts are validated before this representation-only byte view is made.
        return JuicerCuda::ThreeChannelSamplesView(std::span<const std::byte>{
            reinterpret_cast<const std::byte*>(source.data), source.count * sizeof(float)});
    }

    std::span<const std::uint8_t> bytes(const FjByteSpan& source) {
        return {source.data, source.count};
    }

    bool product(std::size_t left, std::size_t right, std::size_t& out) {
        if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
            return false;
        }
        out = left * right;
        return true;
    }

    bool validate_spans(const FjPreparedHostData& source, std::string& diagnostic) {
        const bool print = source.route == FJ_ROUTE_NEGATIVE_PRINT || source.route == FJ_ROUTE_POSITIVE_PRINT;
        const bool layers = (source.grain.flags & FJ_GRAIN_SUBLAYERS) != 0;
        const bool grain = (source.grain.flags & FJ_GRAIN_ACTIVE) != 0;
        const bool gamut = (source.output_color.flags & FJ_COLOR_GAMUT_COMPRESSION) != 0;
        const auto& exposure = source.film_exposure;
        const auto& development = source.film_development;
        const auto& dir = source.dir_couplers;
        if (exposure.method > FJ_RAW_ARCTIC_2026_BETA04 || exposure.input_color_space > FJ_INPUT_SRGB_REC709 ||
            (exposure.flags & ~(FJ_FILM_DECODE_CCTF | FJ_FILM_ADAPT_INPUT | FJ_FILM_AUTO_EXPOSURE)) != 0 ||
            dir.mode > FJ_DIR_POSITIVE_RECEIVER_LANGMUIR ||
            source.auto_exposure.method > FJ_METER_HIGHLIGHT_WEIGHTED ||
            (source.print.flags & ~FJ_PRINT_PREFLASH) != 0) {
            return malformed(diagnostic, "film_or_print_tag");
        }
        if ((!print && source.print.flags != 0) ||
            (dir.mode == FJ_DIR_INACTIVE && (dir.hash != 0 || dir.compensated_axes_hash != 0)) ||
            (!layers && development.density_layers_hash != 0)) {
            return malformed(diagnostic, "inactive_family");
        }
        if (!exact_span(source.film_profile_key, source.film_profile_key.count, "film_profile_key", diagnostic) ||
            !exact_span(source.print_profile_key, source.print_profile_key.count, "print_profile_key", diagnostic)) {
            return false;
        }
        const bool mallett = exposure.method == FJ_RAW_MALLETT_2019;
        if (!exact_span(exposure.sensitivity_rgb, std::size_t{81} * 3u, "film_exposure.sensitivity_rgb", diagnostic) ||
            !exact_span(exposure.tc_lut_rgba, mallett ? 0u : std::size_t{192} * 192u * 4u, "film_exposure.tc_lut_rgba", diagnostic) ||
            !exact_span(exposure.mallett_illuminant, mallett ? 81u : 0u, "film_exposure.mallett_illuminant", diagnostic) ||
            !exact_span(exposure.mallett_basis, mallett ? std::size_t{81} * 3u : 0u, "film_exposure.mallett_basis", diagnostic)) {
            return false;
        }
        const std::size_t samples = development.log_exposure.count;
        if (samples == 0 || samples > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 3u) {
            return malformed(diagnostic, "film_development.log_exposure.count");
        }
        if (!exact_span(development.log_exposure, samples, "film_development.log_exposure", diagnostic) ||
            !exact_span(development.density_rgb, samples * 3u, "film_development.density_rgb", diagnostic)) {
            return false;
        }
        for (const auto& layer : development.density_layers) {
            for (const auto& channel : layer) {
                if (!exact_span(channel, layers ? samples : 0u, "film_development.density_layers", diagnostic)) {
                    return false;
                }
            }
        }
        for (const auto& axis : dir.compensated_axes_rgb) {
            if (!exact_span(axis, dir.mode == FJ_DIR_INACTIVE ? 0u : samples, "dir_couplers.compensated_axes_rgb", diagnostic)) {
                return false;
            }
        }
        const auto& scanner = source.scanner_spectra;
        for (const auto& spectrum : {scanner.dye_c, scanner.dye_m, scanner.dye_y, scanner.weighted_x, scanner.weighted_y, scanner.weighted_z}) {
            if (!exact_span(spectrum, 81u, "scanner_spectra", diagnostic)) {
                return false;
            }
        }
        if (!exact_span(scanner.base_density, scanner.base_density.count == 0 ? 0u : 81u, "scanner_spectra.base_density", diagnostic) ||
            !exact_span(source.output_color.gamut_cmax, gamut ? std::size_t{64} * 720u : 0u, "output_color.gamut_cmax", diagnostic)) {
            return false;
        }
        const auto& printInput = source.print;
        const std::size_t printSamples = print ? printInput.log_exposure.count : 0u;
        if (print && (printSamples == 0 || printSamples > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 3u)) {
            return malformed(diagnostic, "print.log_exposure.count");
        }
        const bool preflash = print && (printInput.flags & FJ_PRINT_PREFLASH) != 0;
        if (!exact_span(printInput.film_density_cmy, print ? std::size_t{81} * 3u : 0u, "print.film_density_cmy", diagnostic) ||
            !exact_span(printInput.film_base_density, print ? 81u : 0u, "print.film_base_density", diagnostic) ||
            !exact_span(printInput.sensitivity_cmy, print ? std::size_t{81} * 3u : 0u, "print.sensitivity_cmy", diagnostic) ||
            !exact_span(printInput.log_exposure, printSamples, "print.log_exposure", diagnostic) ||
            !exact_span(printInput.density_cmy, printSamples * 3u, "print.density_cmy", diagnostic) ||
            !exact_span(printInput.main_illuminant, print ? 81u : 0u, "print.main_illuminant", diagnostic) ||
            !exact_span(printInput.preflash_illuminant, preflash ? 81u : 0u, "print.preflash_illuminant", diagnostic)) {
            return false;
        }
        const auto& noise = source.noise;
        std::size_t stbnCount = 0;
        std::size_t wangCount = 0;
        std::size_t lutCount = 0;
        if (!grain && (noise.stbn_width != 0 || noise.stbn_height != 0 || noise.stbn_frames != 0 ||
                       noise.wang_width != 0 || noise.wang_height != 0 || noise.wang_tile_count != 0 || noise.wang_colors != 0)) {
            return malformed(diagnostic, "inactive_noise");
        }
        if (grain) {
            if (noise.stbn_width <= 0 || noise.stbn_height <= 0 || noise.stbn_frames <= 0 ||
                noise.wang_width <= 0 || noise.wang_height <= 0 || noise.wang_colors <= 0 ||
                noise.wang_tile_count == 0 || noise.wang_tile_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                !product(static_cast<std::size_t>(noise.stbn_width), static_cast<std::size_t>(noise.stbn_height), stbnCount) ||
                !product(stbnCount, static_cast<std::size_t>(noise.stbn_frames), stbnCount) ||
                !product(static_cast<std::size_t>(noise.wang_width), static_cast<std::size_t>(noise.wang_height), wangCount) ||
                !product(wangCount, noise.wang_tile_count, wangCount) ||
                !product(static_cast<std::size_t>(noise.wang_colors), static_cast<std::size_t>(noise.wang_colors), lutCount) ||
                !product(lutCount, lutCount, lutCount)) {
                return malformed(diagnostic, "noise.dimensions");
            }
        }
        return exact_span(noise.stbn, stbnCount, "noise.stbn", diagnostic) &&
               exact_span(noise.wang_tiles, wangCount, "noise.wang_tiles", diagnostic) &&
               exact_span(noise.wang_lut, lutCount, "noise.wang_lut", diagnostic);
    }

    JuicerCuda::PrintResourceInput decode_print(const FjPreparedHostData& source) {
        const FjPrint& print = source.print;
        JuicerCuda::PrintResourceInput out;
        out.filmChannelDensityCmy = channel_samples(print.film_density_cmy);
        out.filmBaseDensity = floats(print.film_base_density);
        out.printLogExposure = floats(print.log_exposure);
        out.printDensityCurvesCmy = channel_samples(print.density_cmy);
        out.printSensitivityCmy = channel_samples(print.sensitivity_cmy);
        out.mainIlluminant = floats(print.main_illuminant);
        out.preflashIlluminant = floats(print.preflash_illuminant);
        std::copy_n(print.preflash_raw_cmy, 3, out.preflashRawCmy.begin());
        out.normalizer = print.normalizer;
        auto& descriptors = out.descriptors;
        descriptors.filmDensityTables.filmProfileAssetVersionToken = source.film_profile_asset_version;
        descriptors.filmDensityTables.spectralSampleCount = 81;
        descriptors.filmDensityTables.hash = print.film_density_hash;
        descriptors.profileTables.printProfileAssetVersionToken = source.print_profile_asset_version;
        descriptors.profileTables.densitySampleCount = static_cast<std::uint32_t>(print.log_exposure.count);
        descriptors.profileTables.spectralSampleCount = 81;
        descriptors.profileTables.hash = print.profile_tables_hash;
        descriptors.mainIlluminant.hash = print.main_illuminant_hash;
        descriptors.mainIlluminant.cmyCc = {print.main_cc_cmy[0], print.main_cc_cmy[1], print.main_cc_cmy[2]};
        descriptors.preflashIlluminant.hash = print.preflash_illuminant_hash;
        descriptors.preflashIlluminant.cmyCc = {print.preflash_cc_cmy[0], print.preflash_cc_cmy[1], print.preflash_cc_cmy[2]};
        descriptors.preflashRaw.hash = print.preflash_raw_hash;
        descriptors.preflashRaw.filmProfileAssetVersionToken = source.film_profile_asset_version;
        descriptors.preflashRaw.printProfileAssetVersionToken = source.print_profile_asset_version;
        descriptors.preflashRaw.filteredPreflashIlluminantHash = print.preflash_illuminant_hash;
        descriptors.balance.filmProfileAssetVersionToken = source.film_profile_asset_version;
        descriptors.balance.printProfileAssetVersionToken = source.print_profile_asset_version;
        descriptors.balance.filmRawRecipeHash = source.film_exposure.hash;
        descriptors.balance.filmDevelopRecipeHash = source.film_development.hash;
        descriptors.balance.filteredMainIlluminantHash = print.main_illuminant_hash;
        descriptors.balance.hash = print.balance_hash;
        descriptors.preflashActive = (print.flags & FJ_PRINT_PREFLASH) != 0;
        descriptors.hash = print.hash;
        return out;
    }

} // namespace

namespace JuicerCuda {

    bool execute_prepared_host_data(
        const FjPreparedHostData& source,
        const ExecutionFrame& frame,
        ResourceManager::SubmissionSnapshot& snapshot,
        PendingContextLossRecovery& recovery,
        const DirFailureMessage& dirFailureMessage,
        std::string& diagnostic) {
        diagnostic.clear();
        PreparedDescriptors descriptors;
        if (!decode_prepared_descriptors(source, descriptors, diagnostic) || !validate_spans(source, diagnostic)) {
            return false;
        }
        const ExecutionFrame boundFrame{
            frame.sourceBounds, frame.renderWindow, frame.fullFrameExtent, frame.sourceBase, frame.source, frame.destination, frame.sourceRowBytes, frame.destinationRowBytes, frame.components, frame.stream, descriptors.diffusion, descriptors.halation, frame.effectsGeometry, frame.pixelSizeUm, frame.timeFrames, frame.frameRate, frame.sessionSeed, frame.clipToken, frame.autoExposureDescriptor, frame.traceInfo, frame.traceVerbose};
        const auto& exposure = source.film_exposure;
        const auto& development = source.film_development;
        const auto& dir = source.dir_couplers;
        const bool print = Spektrafilm::scan_route_is_print(descriptors.route);
        FocusedRouteResourceInput focused;
        focused.route = descriptors.route;
        focused.filmRaw = {static_cast<Spektrafilm::RgbToRawMethod>(exposure.method), channel_samples(exposure.sensitivity_rgb), exposure.sensitivity_hash, exposure.tc_lut_hash};
        focused.filmDevelop.logExposure = floats(development.log_exposure);
        focused.filmDevelop.normalizedDensityCurvesRgb = channel_samples(development.density_rgb);
        focused.filmDevelop.normalizedDensityCurvesHash = development.density_curves_hash;
        focused.filmDevelop.densityCurvesLayersHash = development.density_layers_hash;
        focused.wantDensityLayers = (source.grain.flags & FJ_GRAIN_SUBLAYERS) != 0;
        focused.filmDevelop.densityCurvesLayersRequired = focused.wantDensityLayers;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            focused.dirCouplers.compensatedDensityCurveAxesRgb[channel] = floats(dir.compensated_axes_rgb[channel]);
            for (std::size_t layer = 0; layer < 3; ++layer) {
                focused.filmDevelop.densityCurvesLayers[layer][channel] = floats(development.density_layers[layer][channel]);
            }
            focused.densityBounds.dataMinCmy[channel] = source.scanner_bounds.min_cmy[channel];
            focused.densityBounds.invSpanCmy[channel] = source.scanner_bounds.inv_span_cmy[channel];
        }
        focused.dirCouplers.active = dir.mode != FJ_DIR_INACTIVE;
        focused.dirCouplers.compensatedDensityCurveAxesHash = dir.compensated_axes_hash;
        focused.dirCouplers.hash = dir.hash;
        focused.densityBounds.hash = source.scanner_bounds.hash;
        focused.scannerLutDescriptor = descriptors.scanner;
        const auto& spectra = source.scanner_spectra;
        focused.scannerTables = {floats(spectra.dye_c), floats(spectra.dye_m), floats(spectra.dye_y), floats(spectra.weighted_x), floats(spectra.weighted_y), floats(spectra.weighted_z), floats(spectra.base_density), 81, spectra.inverse_y_normalization, spectra.base_density.count != 0};
        focused.exposureSampleCount = 81;
        focused.exposureIlluminant = floats(exposure.mallett_illuminant);
        focused.filmTcLut = floats(exposure.tc_lut_rgba);
        focused.mallettBasis = floats(exposure.mallett_basis);
        focused.mallettAvailable = exposure.method == FJ_RAW_MALLETT_2019;
        focused.mallettRows = focused.mallettAvailable ? 81 : 0;
        focused.mallettColumns = focused.mallettAvailable ? 3 : 0;
        focused.outputGamut = {floats(source.output_color.gamut_cmax), source.output_color.gamut_table_hash, source.output_color.gamut_recipe_hash, descriptors.outputGamut.enabled};

        Spectral::FilmRawConfig config;
        std::copy_n(exposure.input_rgb_to_xyz, 9, config.inputRGBToXYZ.m);
        std::copy_n(exposure.input_xyz_adapt, 9, config.inputXYZAdapt.m);
        std::copy_n(exposure.xyz_to_linear_srgb, 9, config.xyzToLinearSrgb.m);
        config.applyInputChromaticAdapt = (exposure.flags & FJ_FILM_ADAPT_INPUT) != 0;

        FilmPayloadInput film;
        film.inputColorSpace = static_cast<int>(exposure.input_color_space);
        film.inputCctfDecoding = (exposure.flags & FJ_FILM_DECODE_CCTF) != 0;
        film.method = focused.filmRaw.rgbToRawMethod;
        film.manualExposureEv = exposure.manual_exposure_ev;
        film.mallettGreenMidgrayScale = exposure.mallett_green_midgray_scale;
        film.sensitivityHash = exposure.sensitivity_hash;
        film.densityCurvesHash = development.density_curves_hash;
        film.densitySampleCount = development.log_exposure.count;
        std::copy_n(development.gamma_rgb, 3, film.gammaRgb.begin());
        film.dirMode = static_cast<DirNonlinearMode>(dir.mode);
        for (std::size_t channel = 0; channel < 3; ++channel) {
            std::copy_n(dir.matrix_rgb + channel * 3u, 3, film.dirMatrixRgb[channel].begin());
        }
        std::copy_n(dir.density_max_rgb, 3, film.densityMaxRgb.begin());
        std::copy_n(dir.density_ref_rgb, 3, film.densityRefRgb.begin());
        std::copy_n(dir.donor_k_rgb, 3, film.donorKRgb.begin());
        std::copy_n(dir.receiver_c_ref_rgb, 3, film.receiverCRefRgb.begin());
        std::copy_n(dir.receiver_kr_rgb, 3, film.receiverKrRgb.begin());
        film.dirAxesHash = dir.compensated_axes_hash;
        film.dirHash = dir.hash;
        film.densityBoundsHash = print ? source.enlarger_film_bounds.hash : source.scanner_bounds.hash;

        const auto& noise = source.noise;
        const StaticNoiseInput staticNoise{bytes(noise.stbn), bytes(noise.wang_tiles), bytes(noise.wang_lut), noise.stbn_width, noise.stbn_height, noise.stbn_frames, noise.wang_width, noise.wang_height, static_cast<int>(noise.wang_tile_count), noise.wang_colors};
        const std::string_view filmKey = source.film_profile_key.count == 0
                                             ? std::string_view{}
                                             : std::string_view(source.film_profile_key.data, source.film_profile_key.count);
        JuicerProcess::Root::PreparationIdentity identity;
        identity.scanRoute = descriptors.route;
        identity.capturePolarity = descriptors.capturePolarity;
        identity.filmProfileKey = filmKey;
        identity.filmProfileAssetVersionToken = source.film_profile_asset_version;
        identity.scatterHalationHash = source.optics.scatter_halation.recipe_hash;
        identity.grainHash = source.grain.recipe_hash;
        identity.grainDensityLayersHash = development.density_layers_hash;
        identity.grainActive = (source.grain.flags & FJ_GRAIN_ACTIVE) != 0;
        identity.effectsHash = source.effects.recipe_hash;
        identity.effectsActive = descriptors.effects.has_value();
        JuicerProcess::Root::PreparedFrameInput preparation{
            identity, focused, config, descriptors.color, descriptors.outputGamut, {}, &descriptors.post, &descriptors.spatialDir, descriptors.diffusion ? &*descriptors.diffusion : nullptr, descriptors.halation ? &*descriptors.halation : nullptr, descriptors.grain, descriptors.effects, &staticNoise, frame.renderWindow.x2 - frame.renderWindow.x1, frame.renderWindow.y2 - frame.renderWindow.y1};
        PrintExposureRecipe printExposure;
        if (print) {
            preparation.print = decode_print(source);
            printExposure.printExposure = source.print.exposure;
            printExposure.preflashExposure = source.print.preflash_exposure;
        }
        execute_prepared({preparation, descriptors, film, exposure.route_correction_scale, printExposure, source.recipe_hash, (exposure.flags & FJ_FILM_AUTO_EXPOSURE) != 0, boundFrame, snapshot}, recovery, dirFailureMessage);
        return true;
    }

} // namespace JuicerCuda
