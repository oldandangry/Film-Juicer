#include "Scanner.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

#include "GaussianSciPy.h"
#include "Logging.h"
#include "RenderRecipe.h"
#include "RustScannerBridge.h"

namespace {

    constexpr float kScannerGaussianTruncate = 3.0f;
    constexpr int kMaxScannerGaussianRadius = 75;

    template <typename T>
    void hash_value(std::uint64_t& hash, const T& value) {
        Hash::hash_bytes_update(hash, &value, sizeof(value));
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
