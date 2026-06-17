// Scanner.h
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "Hash.h"
#include "ProfileJSONLoader.h"
#include "ScanRoute.h"
#include "SpectralData.h"
#include "OutputColor.h"

struct DensityBoundsRecipe;
struct ProfileRoute;
struct ScannerOutputRecipe;
struct PrintMediumHandoffRecipe;
struct RenderRecipe;

namespace Scanner {

    enum class ScannerMedium : int {
        Negative = 0,
        Print = 1
    };

    struct Options {
        float lensBlurSigmaPx = 0.0f;
        float unsharpSigmaPx = 0.7f;
        float unsharpAmount = 0.7f;
    };

    struct Settings {
        bool useLut = true;
        std::uint32_t lutResolution = 17;
    };

    struct DensityBuffer {
        // CMY dye densities in SoA form; `medium` tags whether the slab holds negative or print data.
        ScannerMedium medium = ScannerMedium::Negative;
        std::vector<float> c; // cyan dye density (from red layer)
        std::vector<float> m; // magenta dye density (from green layer)
        std::vector<float> y; // yellow dye density (from blue layer)
        int originX = 0;
        int originY = 0;
        int width = 0;
        int height = 0;
        std::ptrdiff_t stride = 0;
    };

    using ScannerDensityBuffer = DensityBuffer;

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

    struct ScannerStaticKey {
        ScannerMedium medium = ScannerMedium::Negative;
        std::uint64_t tablesHash = 0;
        std::uint64_t densityRangeHash = 0;
        std::uint32_t lutResolution = 0;
        std::uint64_t hash = 0;
    };

    struct ScannerRuntimeEffectsKey {
        std::uint64_t glareRuntimeHash = 0;
        std::uint64_t colorRuntimeHash = 0;
        std::uint64_t hash = 0;
    };

    struct ScannerRuntimeKey {
        std::uint64_t settingsHash = 0;
        std::uint32_t frameBoundsVersion = 0;
        std::uint64_t hash = 0;
    };

    struct ScannerKey {
        ScannerStaticKey staticKey;
        ScannerRuntimeEffectsKey effectsKey;
        ScannerRuntimeKey runtimeKey;
        std::uint64_t hash = 0;
    };

    struct SpectralLutBuffer {
        std::vector<double> cpu;
        std::uint64_t hash = 0;
        std::uint32_t res = 0;
        bool valid = false;
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
        ScannerIlluminant illuminant;
        Profiles::ProfileGlare glare;
        const ColorRuntime* color = nullptr;
        ScannerStaticKey staticKey;
        ScannerRuntimeEffectsKey effectsKey;
    };

    enum class ScannedMediumKind : std::uint8_t {
        Film,
        Print
    };

    enum class ScannerXyzNormalization : std::uint8_t {
        ScannerIlluminantY
    };

    enum class ScannerLutInterpolation : std::uint8_t {
        PchipClamped
    };

    enum class ScannerLutAxisOrder : std::uint8_t {
        Cmy
    };

    enum class ScannerLutStoredValueDomain : std::uint8_t {
        LogXyz
    };

    enum class ScannerLutLogBase : std::uint8_t {
        Base10
    };

    enum class ScannerLutNumericFormat : std::uint8_t {
        Float64
    };

    enum class ScannerLutOutputTripletOrder : std::uint8_t {
        Xyz
    };

    // ScannerSpectralLutDescriptor family contract:
    // Producer: focused direct or print-medium builder from selected recipe payloads.
    // Consumer: focused scanner LUT preparation behind PreparedCudaFrame.
    // Identity: route, medium/polarity, DensityBoundsRecipe hash, selected channel/base density,
    // viewing illuminant, observer/Y normalization, LUT resolution, interpolation, axes, stored
    // value domain/log base/format/output order, and schema version. Scanner correction, output
    // color/CCTF, glare, blur, unsharp, frame bounds, auto exposure, and downstream scanner effects
    // are excluded.
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
        ScannerXyzNormalization xyzNormalization = ScannerXyzNormalization::ScannerIlluminantY;
        std::uint32_t lutResolution = 17;
        ScannerLutInterpolation interpolation = ScannerLutInterpolation::PchipClamped;
        ScannerLutAxisOrder semanticInputAxisOrder = ScannerLutAxisOrder::Cmy;
        ScannerLutAxisOrder storageInputAxisOrder = ScannerLutAxisOrder::Cmy;
        ScannerLutStoredValueDomain storedValueDomain = ScannerLutStoredValueDomain::LogXyz;
        ScannerLutLogBase logBase = ScannerLutLogBase::Base10;
        ScannerLutNumericFormat numericFormat = ScannerLutNumericFormat::Float64;
        ScannerLutOutputTripletOrder storedOutputTripletOrder = ScannerLutOutputTripletOrder::Xyz;
        std::uint32_t schemaVersion = 2;
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
        const ::PrintMediumHandoffRecipe* mediumHandoff = nullptr;
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

    inline std::uint64_t hash_glare(const Profiles::ProfileGlare& glare) {
        const float floats[] = {
            glare.percent,
            glare.roughness,
            glare.blur,
            glare.printShadowCompensationFactor,
            glare.printShadowCompensationDensity,
            glare.printShadowCompensationTransition};
        for (float v : floats) {
            if (!std::isfinite(v)) {
                return 0;
            }
        }
        const std::uint64_t activeHash = Hash::hash_bytes(&glare.active, sizeof(glare.active));
        const std::uint64_t paramsHash = Hash::hash_float_span(
            floats, sizeof(floats) / sizeof(floats[0]));
        return Hash::hash_uint64_values({activeHash, paramsHash});
    }

    inline std::uint64_t compute_static_key_hash(const ScannerStaticKey& key) {
        return Hash::hash_uint64_values({static_cast<std::uint64_t>(key.medium),
                                         key.tablesHash,
                                         key.densityRangeHash,
                                         static_cast<std::uint64_t>(key.lutResolution)});
    }

    inline void finalize_static_key(ScannerStaticKey& key) {
        key.hash = compute_static_key_hash(key);
    }

    inline std::uint64_t identity_color_runtime_hash() {
        static const std::uint64_t h =
            Hash::hash_bytes("identity_color_runtime", sizeof("identity_color_runtime") - 1);
        return h;
    }

    inline void finalize_runtime_effects_key(ScannerRuntimeEffectsKey& key) {
        key.hash = Hash::hash_uint64_values({key.glareRuntimeHash, key.colorRuntimeHash});
    }

    inline void finalize_runtime_key(ScannerRuntimeKey& key) {
        key.hash = Hash::hash_uint64_values({key.settingsHash,
                                             static_cast<std::uint64_t>(key.frameBoundsVersion)});
    }

    inline void finalize_scanner_key(ScannerKey& key) {
        finalize_static_key(key.staticKey);
        finalize_runtime_effects_key(key.effectsKey);
        finalize_runtime_key(key.runtimeKey);
        key.hash =
            Hash::hash_uint64_values({key.staticKey.hash, key.effectsKey.hash, key.runtimeKey.hash});
    }

} // namespace Scanner
