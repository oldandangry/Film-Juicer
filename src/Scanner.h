// Scanner.h
#pragma once

#include <cstdint>
#include <string>
#include <string_view>

#include "Hash.h"
#include "OutputColor.h"
#include "ScanRoute.h"
#include "SpectralData.h"

struct DensityBoundsRecipe;
struct ProfileRoute;
struct ScannerOutputRecipe;
struct RenderRecipe;

namespace Scanner {

    enum class ScannerMedium : std::uint8_t {
        Negative = 0,
        Print = 1
    };

    struct ScannerIlluminant {
        Spectral::Curve curve;
        float normalization = 0.0f;
        float whiteXYZ[3]{0.0f, 0.0f, 0.0f};
        float whiteXY[2]{0.0f, 0.0f};
        std::uint64_t hash = 0;
    };

    struct ScannerDensityRange {
        float min_cmy[3]{0.0f, 0.0f, 0.0f};
        float max_cmy[3]{0.0f, 0.0f, 0.0f};
        float inv_max_cmy[3]{0.0f, 0.0f, 0.0f};
        std::uint64_t digest = 0;
    };

    struct ColorRuntime {
        float cat02[9]{0.0f};
        float xyzToRgb[9]{0.0f};
        float illuminantXYZ[3]{0.0f, 0.0f, 0.0f};
        OutputEncoding::Params encoding{};
        std::uint64_t hash = 0;
    };

    struct ScannerMediumRuntime {
        ScannerMedium medium = ScannerMedium::Negative;
        const Spectral::SpectralTables* tables = nullptr;
        ScannerDensityRange range;
    };

    enum class ScannedMediumKind : std::uint8_t {
        Film,
        Print
    };

    inline constexpr std::uint32_t kScannerSpectralLutSchemaVersion = 2;

    // ScannerSpectralLutDescriptor family contract:
    // Producer: focused direct or print-medium builder from selected recipe payloads.
    // Consumer: focused scanner LUT preparation behind PreparedCudaFrame.
    // Identity: route, medium/polarity, DensityBoundsRecipe hash, selected channel/base density,
    // viewing illuminant, observer, LUT resolution, and the fixed LUT representation schema.
    // Scanner correction, output color/CCTF, glare, blur, unsharp, frame bounds, auto exposure,
    // and downstream scanner effects are excluded.
    // Lifetime: plain host descriptor; no upload/admission/allocation occurs in Phase 3A.
    struct ScannerSpectralLutDescriptor {
        Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
        ScannedMediumKind medium = ScannedMediumKind::Film;
        Spektrafilm::ProfilePolarity polarity = Spektrafilm::ProfilePolarity::Unsupported;
        std::uint64_t densityBoundsHash = 0;
        std::uint64_t channelDensityHash = 0;
        std::uint64_t baseDensityHash = 0;
        std::uint64_t scanIlluminantHash = 0;
        std::uint64_t observerHash = 0;
        std::uint32_t lutResolution = 17;
        std::uint64_t hash = 0;
    };

    struct DirectScannerSpectralLutDescriptorInput {
        const ::ProfileRoute* profileRoute = nullptr;
        const ::DensityBoundsRecipe* densityBounds = nullptr;
        const ::ScannerOutputRecipe* scannerOutput = nullptr;
        std::string_view observerIdentity = "CIE1931_2deg_380_780_5nm";
    };

    struct PrintScannerSpectralLutDescriptorInput {
        const ::ProfileRoute* profileRoute = nullptr;
        const ::DensityBoundsRecipe* densityBounds = nullptr;
        const ::ScannerOutputRecipe* scannerOutput = nullptr;
        std::string_view observerIdentity = "CIE1931_2deg_380_780_5nm";
    };

    std::uint64_t hash_scanner_spectral_lut_descriptor(
        const ScannerSpectralLutDescriptor& descriptor);

    bool build_direct_scanner_spectral_lut_descriptor(
        const DirectScannerSpectralLutDescriptorInput& input,
        ScannerSpectralLutDescriptor& outDescriptor,
        std::string& outDiagnostic);

    bool build_print_scanner_spectral_lut_descriptor(
        const PrintScannerSpectralLutDescriptorInput& input,
        ScannerSpectralLutDescriptor& outDescriptor,
        std::string& outDiagnostic);

    struct ScannerColorCorrectionDescriptor {
        Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
        bool active = false;
        bool blackCorrection = false;
        bool whiteCorrection = false;
        float targetBlackLinear = 0.0f;
        float targetWhiteLinear = 1.0f;
        float referenceBlackY = 0.0f;
        float referenceWhiteY = 1.0f;
        float xyzSlope = 1.0f;
        float xyzOffset = 0.0f;
        float exposureScale = 1.0f;
        std::uint32_t schemaVersion = 1;
        std::uint64_t hash = 0;
    };

    struct PrintCorrectionDerivationInput {
        const ::RenderRecipe* recipe = nullptr;
        const Spectral::SpectralTables* scannerTables = nullptr;
        const float* mainIlluminant = nullptr;
        int spectralSampleCount = 0;
        const float* preflashRawCmy = nullptr;
        float normalizer = 1.0f;
    };

    struct ScannerPostEffectsDescriptor {
        Spektrafilm::ScanRoute route = Spektrafilm::kDefaultScanRoute;
        bool glareActive = false;
        float glarePercent = 0.0f;
        float glareRoughness = 0.0f;
        float glareBlurSigmaPx = 0.0f;
        float lensBlurSigmaPx = 0.0f;
        float unsharpSigmaPx = 0.0f;
        float unsharpAmount = 0.0f;
        std::uint32_t schemaVersion = 1;
        std::uint64_t hash = 0;

        bool active() const noexcept {
            return hash != 0;
        }
    };

    bool build_direct_scanner_color_correction_descriptor(
        const ::RenderRecipe& recipe,
        const Spectral::SpectralTables& scannerTables,
        ScannerColorCorrectionDescriptor& outDescriptor,
        std::string& outDiagnostic);

    bool build_print_scanner_color_correction_descriptor(
        const PrintCorrectionDerivationInput& input,
        ScannerColorCorrectionDescriptor& outDescriptor,
        std::string& outDiagnostic);

    bool build_scanner_post_effects_descriptor(
        const ::ScannerOutputRecipe& recipe,
        ScannerPostEffectsDescriptor& outDescriptor,
        std::string& outDiagnostic);

    // Canonical scanner spectral core (agx-emulsion parity):
    // - Accepts normalized density (0..1), denormalizes per medium range
    // - Converts dyes -> XYZ under the medium's spectral tables
    // - Applies log10(max(xyz, 0) + 1e-10)
    void spectral_to_log_xyz(const ScannerMediumRuntime& medium, const double D_norm[3], double logXYZ[3]);

    // Canonical normalization for LUT coordinates (mirrors agx _normalize_* semantics).
    void normalize_density(const ScannerMediumRuntime& medium, const float D_cmy[3], double D_norm[3]);

    ColorRuntime build_color_runtime(
        ScannerMedium medium,
        const ScannerIlluminant& illuminant,
        const OutputEncoding::Params& outputEncoding);

} // namespace Scanner
