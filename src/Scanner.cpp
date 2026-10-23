#include "Scanner.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "ColorTransforms.h"
#include "GaussianSciPy.h"
#include "Logging.h"
#include "RenderRecipe.h"
#include "RustColorBridge.h"
#include "RustScannerBridge.h"

namespace {

    constexpr float kScannerGaussianTruncate = 3.0f;
    constexpr int kMaxScannerGaussianRadius = 75;

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
    }

    void hash_string(std::uint64_t& hash, const std::string& value) {
        Hash::hash_bytes_update(hash, value.data(), value.size());
    }

    std::uint64_t hash_nan_preserving_floats(const float* values, std::size_t count) {
        const Hash::FloatSpanHash hashes = Hash::hash_float_span_with_nan_mask(values, count);
        return Hash::hash_uint64_values({hashes.valueHash, hashes.nanMaskHash});
    }

    std::uint64_t hash_color_runtime(
        Scanner::ScannerMedium medium,
        const Scanner::ScannerIlluminant& illuminant,
        const OutputEncoding::Params& encoding,
        std::uint64_t entryHash,
        std::uint64_t activeOutputGamutRecipeHash) {
        const std::uint64_t encHash = Hash::hash_uint64_values(
            {static_cast<std::uint64_t>(OutputEncoding::toIndex(encoding.colorSpace)),
             static_cast<std::uint64_t>(encoding.applyCctfEncoding),
             static_cast<std::uint64_t>(encoding.inputIsOutputSpace)});
        return Hash::hash_uint64_values(
            {entryHash,
             illuminant.hash,
             static_cast<std::uint64_t>(medium),
             encHash,
             activeOutputGamutRecipeHash});
    }

    int scanner_gaussian_radius_or_zero(float sigma) noexcept {
        if (!std::isfinite(sigma) || sigma <= 0.0f) {
            return 0;
        }
        return std::min(
            JuicerGaussian::scipy_gaussian_radius(
                sigma,
                kScannerGaussianTruncate),
            kMaxScannerGaussianRadius);
    }

    float scanner_gaussian_sigma_with_device_kernel_or_zero(float sigma) noexcept {
        if (!std::isfinite(sigma)) {
            return sigma;
        }
        if (sigma <= 0.0f) {
            return 0.0f;
        }
        return scanner_gaussian_radius_or_zero(sigma) > 0 ? sigma : 0.0f;
    }

    const char* bool_text(bool value) {
        return value ? "true" : "false";
    }

    void trace_scanner_correction_descriptor(
        const char* outcome,
        Spektrafilm::ScanRoute route,
        const ScannerOutputRecipe& output,
        float referenceBlackY,
        float referenceWhiteY,
        float targetBlack,
        float targetWhite,
        float slope,
        float offset,
        float correctedMidgray,
        float exposureScale,
        std::uint64_t descriptorHash,
        const char* diagnostic) {
        if (!JTRACE_ENABLED(2)) {
            return;
        }
        std::ostringstream oss;
        oss << "event=scanner_correction_descriptor"
            << " outcome=" << (outcome ? outcome : "unknown")
            << " route=" << Spektrafilm::scan_route_key(route)
            << " blackCorrection=" << bool_text(output.blackCorrection)
            << " whiteCorrection=" << bool_text(output.whiteCorrection)
            << " blackLevelEncoded=" << output.blackLevel
            << " whiteLevelEncoded=" << output.whiteLevel
            << " targetBlackLinear=" << targetBlack
            << " targetWhiteLinear=" << targetWhite
            << " referenceBlackY=" << referenceBlackY
            << " referenceWhiteY=" << referenceWhiteY
            << " referenceDeltaY=" << (referenceWhiteY - referenceBlackY)
            << " xyzSlope=" << slope
            << " xyzOffset=" << offset
            << " correctedMidgrayInput=" << correctedMidgray
            << " exposureScale=" << exposureScale
            << " descriptorHash=" << descriptorHash;
        if (diagnostic && diagnostic[0] != '\0') {
            oss << " diagnostic=\"" << diagnostic << "\"";
        }
        JTRACE_LEVEL(2, "SCAN", oss.str());
    }

    void trace_prepared_scanner_correction(
        const RenderRecipe& recipe,
        const FjScannerCorrectionTrace& trace,
        const std::string& diagnostic) {
        const char* outcome = nullptr;
        switch (trace.outcome) {
            case FJ_SCANNER_CORRECTION_TRACE_NONE:
                return;
            case FJ_SCANNER_CORRECTION_TRACE_ACCEPTED:
                outcome = "accepted";
                break;
            case FJ_SCANNER_CORRECTION_TRACE_MIDGRAY:
                outcome = "rejected_corrected_midgray";
                break;
            case FJ_SCANNER_CORRECTION_TRACE_MALFORMED:
                outcome = "rejected_malformed";
                break;
            case FJ_SCANNER_CORRECTION_TRACE_ZERO_HASH:
                outcome = "rejected_zero_hash";
                break;
            default:
                return; // The transport adapter rejects unknown outcomes.
        }
        trace_scanner_correction_descriptor(
            outcome, recipe.profileRoute.scanRoute, recipe.scannerOutput, trace.reference_black_y, trace.reference_white_y, trace.target_black_linear, trace.target_white_linear, trace.xyz_slope, trace.xyz_offset, trace.corrected_midgray, trace.exposure_scale, trace.hash, trace.outcome == FJ_SCANNER_CORRECTION_TRACE_ZERO_HASH ? "ResourceDescriptorMismatch phase=8B scanner correction hash" : diagnostic.c_str());
    }

} // namespace

namespace Scanner {

    SpectralTablesView spectral_tables_view(const Spectral::SpectralTables& tables) {
        return {tables.epsC, tables.epsM, tables.epsY, tables.Ax, tables.Ay, tables.Az, tables.baseDensityMin, tables.K, tables.invYn, tables.hasBaseline};
    }

    ColorRuntime build_color_runtime(
        ScannerMedium medium,
        const ScannerIlluminant& illuminant,
        const OutputEncoding::Params& outputEncoding,
        std::uint64_t activeOutputGamutRecipeHash) {
        ColorRuntime rt{};
        if (illuminant.hash == 0) {
            JTRACE("HASH", "FATAL: scanner illuminant hash invalid for color runtime");
            return rt;
        }
        const auto outSpace = JuicerColor::output_color(outputEncoding.colorSpace);
        if (outSpace.entry_hash == 0) {
            JTRACE("HASH", "FATAL: generated color space hash invalid");
            return rt;
        }
        Spectral::ChromaticAdaptationWhites whites{};
        whites.source = illuminant.whiteXYZ;
        whites.destination = outSpace.white_xyz;
        const auto adapt = JuicerColor::cat02_matrix(whites);
        std::copy(adapt.begin(), adapt.end(), rt.cat02);

        for (int i = 0; i < 9; ++i) {
            rt.xyzToRgb[i] = outSpace.xyz_to_rgb[i];
        }
        rt.preparedEncoding = outSpace.encoding;
        rt.encoding = outputEncoding;
        rt.encoding.inputIsOutputSpace = true;
        rt.illuminantXYZ[0] = illuminant.whiteXYZ[0];
        rt.illuminantXYZ[1] = illuminant.whiteXYZ[1];
        rt.illuminantXYZ[2] = illuminant.whiteXYZ[2];
        rt.outputGamutRecipeHash = activeOutputGamutRecipeHash;
        rt.hash = hash_color_runtime(
            medium,
            illuminant,
            rt.encoding,
            outSpace.entry_hash,
            activeOutputGamutRecipeHash);
        return rt;
    }

    std::uint64_t hash_scanner_spectral_lut_descriptor(
        const ScannerSpectralLutDescriptor& descriptor) {
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, descriptor.route);
        hash_value(hash, descriptor.medium);
        hash_value(hash, descriptor.polarity);
        hash_value(hash, descriptor.densityBoundsHash);
        hash_value(hash, descriptor.channelDensityHash);
        hash_value(hash, descriptor.baseDensityHash);
        hash_value(hash, descriptor.scanIlluminantHash);
        hash_value(hash, descriptor.observerHash);
        hash_value(hash, descriptor.lutResolution);
        hash_value(hash, kScannerSpectralLutSchemaVersion);
        return hash;
    }

    bool build_direct_scanner_spectral_lut_descriptor(
        const DirectScannerSpectralLutDescriptorInput& input,
        ScannerSpectralLutDescriptor& outDescriptor,
        std::string& outDiagnostic) {
        outDescriptor = ScannerSpectralLutDescriptor{};
        outDiagnostic.clear();
        if (!input.profileRoute || !input.densityBounds || !input.scannerOutput) {
            outDiagnostic = "ResourceDescriptorMismatch phase=3A scanner direct inputs unavailable";
            return false;
        }

        const ProfileRoute& profileRoute = *input.profileRoute;
        const DensityBoundsRecipe& densityBounds = *input.densityBounds;
        const ScannerOutputRecipe& scannerOutput = *input.scannerOutput;
        if (Spektrafilm::scan_route_is_print(profileRoute.scanRoute) ||
            !profileRoute.filmProfile) {
            outDiagnostic = "ResourceDescriptorMismatch phase=3A scanner direct recipe unavailable";
            return false;
        }
        if (densityBounds.hash == 0 ||
            densityBounds.medium != Spektrafilm::DensityMedium::Film) {
            outDiagnostic = "ResourceDescriptorMismatch phase=3A scanner density bounds unavailable";
            return false;
        }
        if (input.observerIdentity.empty()) {
            outDiagnostic = "ResourceDescriptorMismatch phase=3A scanner observer identity unavailable";
            return false;
        }
        if (scannerOutput.viewingIlluminant.empty()) {
            outDiagnostic = "ResourceDescriptorMismatch phase=3A scanner illuminant identity unavailable";
            return false;
        }

        const Profiles::FilmProfile& profile = *profileRoute.filmProfile;
        ScannerSpectralLutDescriptor descriptor{};
        descriptor.route = profileRoute.scanRoute;
        descriptor.medium = ScannedMediumKind::Film;
        descriptor.polarity = profileRoute.capturePolarity;
        descriptor.densityBoundsHash = densityBounds.hash;
        descriptor.channelDensityHash = hash_nan_preserving_floats(
            &profile.data.channelDensity[0][0],
            profile.data.channelDensity.size() * 3u);
        descriptor.baseDensityHash = hash_nan_preserving_floats(
            profile.data.baseDensity.data(),
            profile.data.baseDensity.size());
        descriptor.scanIlluminantHash = Hash::kFnvOffset;
        hash_string(descriptor.scanIlluminantHash, scannerOutput.viewingIlluminant);
        descriptor.observerHash = Hash::hash_bytes(
            input.observerIdentity.data(),
            input.observerIdentity.size());
        descriptor.lutResolution = scannerOutput.lutResolution;
        descriptor.hash = hash_scanner_spectral_lut_descriptor(descriptor);
        if (descriptor.channelDensityHash == 0 ||
            descriptor.baseDensityHash == 0 ||
            descriptor.scanIlluminantHash == 0 ||
            descriptor.observerHash == 0 ||
            descriptor.hash == 0) {
            outDiagnostic = "ResourceDescriptorMismatch phase=3A scanner descriptor hash invalid";
            return false;
        }
        outDescriptor = descriptor;
        return true;
    }

    bool build_print_scanner_spectral_lut_descriptor(
        const PrintScannerSpectralLutDescriptorInput& input,
        ScannerSpectralLutDescriptor& outDescriptor,
        std::string& outDiagnostic) {
        outDescriptor = ScannerSpectralLutDescriptor{};
        outDiagnostic.clear();
        if (!input.profileRoute || !input.densityBounds || !input.scannerOutput) {
            outDiagnostic = "ResourceDescriptorMismatch phase=4C scanner print inputs unavailable";
            return false;
        }
        const ProfileRoute& route = *input.profileRoute;
        const DensityBoundsRecipe& bounds = *input.densityBounds;
        const ScannerOutputRecipe& output = *input.scannerOutput;
        if (!Spektrafilm::scan_route_is_print(route.scanRoute) || !route.printProfile ||
            route.printProfileKey.empty() ||
            route.printProfileAssetVersionToken == 0 ||
            bounds.hash == 0 || bounds.medium != Spektrafilm::DensityMedium::Print ||
            bounds.source != Spektrafilm::DensityBoundsSource::PrintMediaAuthoredCurves ||
            output.medium != Spektrafilm::DensityMedium::Print ||
            output.viewingIlluminant != route.printProfile->info.viewingIlluminant.value ||
            input.observerIdentity.empty()) {
            outDiagnostic = "ResourceDescriptorMismatch phase=4C scanner print recipe";
            return false;
        }

        const Profiles::PrintProfile& profile = *route.printProfile;
        ScannerSpectralLutDescriptor descriptor{};
        descriptor.route = route.scanRoute;
        descriptor.medium = ScannedMediumKind::Print;
        descriptor.polarity = route.capturePolarity;
        descriptor.densityBoundsHash = bounds.hash;
        descriptor.channelDensityHash = hash_nan_preserving_floats(
            &profile.data.channelDensity[0][0],
            profile.data.channelDensity.size() * 3u);
        descriptor.baseDensityHash = hash_nan_preserving_floats(
            profile.data.baseDensity.data(),
            profile.data.baseDensity.size());
        descriptor.scanIlluminantHash = Hash::kFnvOffset;
        hash_string(descriptor.scanIlluminantHash, output.viewingIlluminant);
        descriptor.observerHash = Hash::hash_bytes(
            input.observerIdentity.data(),
            input.observerIdentity.size());
        descriptor.lutResolution = output.lutResolution;
        descriptor.hash = hash_scanner_spectral_lut_descriptor(descriptor);
        if (descriptor.channelDensityHash == 0 || descriptor.baseDensityHash == 0 ||
            descriptor.scanIlluminantHash == 0 || descriptor.observerHash == 0 ||
            descriptor.hash == 0) {
            outDiagnostic = "ResourceDescriptorMismatch phase=4C scanner print descriptor hash";
            return false;
        }
        outDescriptor = descriptor;
        return true;
    }

    bool build_direct_scanner_color_correction_descriptor(
        const RenderRecipe& recipe,
        const Spectral::SpectralTables& scannerTables,
        ScannerColorCorrectionDescriptor& outDescriptor,
        std::string& outDiagnostic) {
        FjScannerCorrectionTrace trace{};
        const bool success = JuicerScanner::prepare_direct(recipe, scannerTables, outDescriptor, trace, outDiagnostic);
        trace_prepared_scanner_correction(recipe, trace, outDiagnostic);
        return success;
    }

    bool build_print_scanner_color_correction_descriptor(
        const PrintCorrectionDerivationInput& input,
        ScannerColorCorrectionDescriptor& outDescriptor,
        std::string& outDiagnostic) {
        FjScannerCorrectionTrace trace{};
        const bool success = JuicerScanner::prepare_print(input, outDescriptor, trace, outDiagnostic);
        if (input.recipe)
            trace_prepared_scanner_correction(*input.recipe, trace, outDiagnostic);
        return success;
    }

    bool build_scanner_post_effects_descriptor(
        const ScannerOutputRecipe& recipe,
        ScannerPostEffectsDescriptor& outDescriptor,
        std::string& outDiagnostic) {
        outDescriptor = ScannerPostEffectsDescriptor{};
        outDescriptor.route = recipe.route;
        outDiagnostic.clear();
        const bool printRoute = Spektrafilm::scan_route_is_print(recipe.route);
        outDescriptor.glareActive = printRoute && recipe.glareActive && recipe.glarePercent > 0.0f;
        outDescriptor.glarePercent = outDescriptor.glareActive ? recipe.glarePercent : 0.0f;
        outDescriptor.glareRoughness = outDescriptor.glareActive ? recipe.glareRoughness : 0.0f;
        outDescriptor.glareBlurSigmaPx =
            outDescriptor.glareActive
                ? scanner_gaussian_sigma_with_device_kernel_or_zero(recipe.glareBlurSigmaPx)
                : 0.0f;
        outDescriptor.glareBlurRadius =
            scanner_gaussian_radius_or_zero(outDescriptor.glareBlurSigmaPx);
        outDescriptor.lensBlurSigmaPx =
            scanner_gaussian_sigma_with_device_kernel_or_zero(recipe.lensBlurSigmaPx);
        outDescriptor.lensBlurRadius =
            scanner_gaussian_radius_or_zero(outDescriptor.lensBlurSigmaPx);
        const float unsharpSigmaPx =
            scanner_gaussian_sigma_with_device_kernel_or_zero(recipe.unsharpSigmaPx);
        const bool unsharpActive = unsharpSigmaPx > 0.0f && recipe.unsharpAmount > 0.0f;
        outDescriptor.unsharpSigmaPx = unsharpActive ? unsharpSigmaPx : 0.0f;
        outDescriptor.unsharpRadius =
            scanner_gaussian_radius_or_zero(outDescriptor.unsharpSigmaPx);
        outDescriptor.unsharpAmount = unsharpActive ? recipe.unsharpAmount : 0.0f;
        const float values[] = {
            outDescriptor.glarePercent,
            outDescriptor.glareRoughness,
            outDescriptor.glareBlurSigmaPx,
            outDescriptor.lensBlurSigmaPx,
            outDescriptor.unsharpSigmaPx,
            outDescriptor.unsharpAmount};
        for (float value : values) {
            if (!std::isfinite(value) || value < 0.0f) {
                outDiagnostic = "MalformedRequiredProfileData phase=8C scanner post effects";
                return false;
            }
        }
        if (!outDescriptor.glareActive && outDescriptor.lensBlurSigmaPx <= 0.0f &&
            outDescriptor.unsharpSigmaPx <= 0.0f) {
            return true;
        }
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, outDescriptor.route);
        hash_value(hash, outDescriptor.glareActive);
        for (float value : values) {
            hash_value(hash, value);
        }
        hash_value(hash, outDescriptor.glareBlurRadius);
        hash_value(hash, outDescriptor.lensBlurRadius);
        hash_value(hash, outDescriptor.unsharpRadius);
        hash_value(hash, outDescriptor.schemaVersion);
        outDescriptor.hash = hash;
        return outDescriptor.hash != 0;
    }

} // namespace Scanner
