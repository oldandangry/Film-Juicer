#include "Scanner.h"

#include <limits>
#include <sstream>

#include "GaussianSciPy.h"
#include "Logging.h"
#include "RenderRecipe.h"

namespace {

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

    float scanner_gaussian_sigma_with_device_kernel_or_zero(float sigma) noexcept {
        if (!std::isfinite(sigma)) {
            return sigma;
        }
        if (sigma <= 0.0f) {
            return 0.0f;
        }
        return JuicerGaussian::scipy_gaussian_radius(sigma, 4.0f) > 0 ? sigma : 0.0f;
    }

    float remove_srgb_cctf(float value) {
        const float clamped = std::clamp(value, 0.0f, 1.0f);
        return clamped <= 0.04045f
                   ? clamped / 12.92f
                   : std::pow((clamped + 0.055f) / 1.055f, 2.4f);
    }

    float density_to_light_sample_spektrafilm(float density, float illuminant) {
        const float transmitted = std::pow(10.0f, -density) * illuminant;
        return std::isnan(transmitted) ? 0.0f : transmitted;
    }

    Scanner::ScannerMediumRuntime make_scanner_medium(
        const DensityBoundsRecipe& bounds,
        const Spectral::SpectralTables& tables) {
        Scanner::ScannerMediumRuntime medium{};
        medium.medium = bounds.medium == Spektrafilm::DensityMedium::Print
                            ? Scanner::ScannerMedium::Print
                            : Scanner::ScannerMedium::Negative;
        medium.tables = &tables;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            medium.range.min_cmy[channel] =
                medium.medium == Scanner::ScannerMedium::Negative
                    ? -bounds.dataMinCmy[channel]
                    : bounds.dataMinCmy[channel];
            medium.range.max_cmy[channel] =
                bounds.invSpanCmy[channel] > 0.0f ? 1.0f / bounds.invSpanCmy[channel] : 0.0f;
            medium.range.inv_max_cmy[channel] = bounds.invSpanCmy[channel];
        }
        medium.range.digest = bounds.hash;
        return medium;
    }

    bool scan_reference_y(
        const Scanner::ScannerMediumRuntime& medium,
        const std::array<float, 3>& densityCmy,
        float& outY) {
        double normalized[3]{};
        double logXyz[3]{};
        Scanner::normalize_density(medium, densityCmy.data(), normalized);
        Scanner::spectral_to_log_xyz(medium, normalized, logXyz);
        const double y = std::pow(10.0, logXyz[1]);
        if (!std::isfinite(y)) {
            return false;
        }
        outY = static_cast<float>(y);
        return true;
    }

    float finite_average(const float* values, std::size_t count) {
        double sum = 0.0;
        std::size_t finiteCount = 0;
        for (std::size_t i = 0; i < count; ++i) {
            if (std::isfinite(values[i])) {
                sum += values[i];
                ++finiteCount;
            }
        }
        return finiteCount > 0 ? static_cast<float>(sum / static_cast<double>(finiteCount)) : 0.0f;
    }

    std::vector<float> average_density_curves(
        const std::vector<std::array<float, 3>>& curves) {
        std::vector<float> average;
        average.reserve(curves.size());
        for (const std::array<float, 3>& row : curves) {
            average.push_back(finite_average(row.data(), row.size()));
        }
        return average;
    }

    float interp_clamped_monotonic(
        float query,
        const std::vector<float>& x,
        const std::vector<float>& y) {
        if (x.empty() || x.size() != y.size()) {
            return 0.0f;
        }
        const bool ascending = x.size() < 2 || x.back() >= x.front();
        const auto at = [&](std::size_t index) {
            return ascending ? index : x.size() - 1u - index;
        };
        if (query <= x[at(0)]) {
            return y[at(0)];
        }
        if (query >= x[at(x.size() - 1u)]) {
            return y[at(y.size() - 1u)];
        }
        for (std::size_t i = 1; i < x.size(); ++i) {
            const std::size_t hi = at(i);
            const std::size_t lo = at(i - 1u);
            if (query <= x[hi]) {
                const float span = x[hi] - x[lo];
                const float t = span > 0.0f ? (query - x[lo]) / span : 0.0f;
                return y[lo] + t * (y[hi] - y[lo]);
            }
        }
        return y[at(y.size() - 1u)];
    }

    float sample_density_curve(
        float logExposure,
        const Profiles::SpektrafilmProfileSamples& data,
        std::size_t channel) {
        if (data.logExposure.empty() || data.logExposure.size() != data.densityCurves.size()) {
            return 0.0f;
        }
        std::vector<float> values;
        values.reserve(data.densityCurves.size());
        for (const std::array<float, 3>& row : data.densityCurves) {
            values.push_back(row[channel]);
        }
        return interp_clamped_monotonic(logExposure, data.logExposure, values);
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

    bool finish_correction_descriptor(
        const ScannerOutputRecipe& output,
        float referenceBlackY,
        float referenceWhiteY,
        float exposureScale,
        Scanner::ScannerColorCorrectionDescriptor& descriptor,
        std::string& outDiagnostic) {
        const float targetBlack = output.blackCorrection
                                      ? remove_srgb_cctf(output.blackLevel)
                                      : referenceBlackY;
        const float targetWhite = output.whiteCorrection
                                      ? remove_srgb_cctf(output.whiteLevel)
                                      : referenceWhiteY;
        const float slope =
            (targetWhite - targetBlack) / (referenceWhiteY - referenceBlackY + 1e-10f);
        const float offset = targetBlack - slope * referenceBlackY;
        const float correctedMidgray = slope != 0.0f
                                           ? (0.184f - offset) / slope
                                           : std::numeric_limits<float>::quiet_NaN();
        const float referenceDeltaY = referenceWhiteY - referenceBlackY;
        if (!std::isfinite(referenceBlackY) || !std::isfinite(referenceWhiteY) ||
            !(referenceDeltaY > 1.0e-6f) ||
            !std::isfinite(slope) || !std::isfinite(offset) ||
            !std::isfinite(exposureScale) || exposureScale <= 0.0f) {
            outDiagnostic = "MalformedRequiredProfileData phase=8B scanner correction";
            trace_scanner_correction_descriptor(
                "rejected_malformed",
                descriptor.route,
                output,
                referenceBlackY,
                referenceWhiteY,
                targetBlack,
                targetWhite,
                slope,
                offset,
                correctedMidgray,
                exposureScale,
                0,
                outDiagnostic.c_str());
            return false;
        }
        descriptor.active = true;
        descriptor.blackCorrection = output.blackCorrection;
        descriptor.whiteCorrection = output.whiteCorrection;
        descriptor.targetBlackLinear = targetBlack;
        descriptor.targetWhiteLinear = targetWhite;
        descriptor.referenceBlackY = referenceBlackY;
        descriptor.referenceWhiteY = referenceWhiteY;
        descriptor.xyzSlope = slope;
        descriptor.xyzOffset = offset;
        descriptor.exposureScale = exposureScale;
        std::uint64_t hash = Hash::kFnvOffset;
        hash_value(hash, descriptor.route);
        hash_value(hash, descriptor.blackCorrection);
        hash_value(hash, descriptor.whiteCorrection);
        hash_value(hash, descriptor.targetBlackLinear);
        hash_value(hash, descriptor.targetWhiteLinear);
        hash_value(hash, descriptor.referenceBlackY);
        hash_value(hash, descriptor.referenceWhiteY);
        hash_value(hash, descriptor.xyzSlope);
        hash_value(hash, descriptor.xyzOffset);
        hash_value(hash, descriptor.exposureScale);
        hash_value(hash, descriptor.schemaVersion);
        descriptor.hash = hash;
        const bool accepted = descriptor.hash != 0;
        trace_scanner_correction_descriptor(
            accepted ? "accepted" : "rejected_zero_hash",
            descriptor.route,
            output,
            referenceBlackY,
            referenceWhiteY,
            targetBlack,
            targetWhite,
            slope,
            offset,
            correctedMidgray,
            exposureScale,
            descriptor.hash,
            accepted ? "" : "ResourceDescriptorMismatch phase=8B scanner correction hash");
        return accepted;
    }

} // namespace

namespace Scanner {

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
        hash_value(hash, descriptor.xyzNormalization);
        hash_value(hash, descriptor.lutResolution);
        hash_value(hash, descriptor.interpolation);
        hash_value(hash, descriptor.semanticInputAxisOrder);
        hash_value(hash, descriptor.storageInputAxisOrder);
        hash_value(hash, descriptor.storedValueDomain);
        hash_value(hash, descriptor.logBase);
        hash_value(hash, descriptor.numericFormat);
        hash_value(hash, descriptor.storedOutputTripletOrder);
        hash_value(hash, descriptor.schemaVersion);
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

        const Profiles::ValidatedFilmProfile& profile = *profileRoute.filmProfile;
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
        descriptor.xyzNormalization = ScannerXyzNormalization::ScannerIlluminantY;
        descriptor.lutResolution = scannerOutput.lutResolution;
        descriptor.interpolation = ScannerLutInterpolation::PchipClamped;
        descriptor.semanticInputAxisOrder = ScannerLutAxisOrder::Cmy;
        descriptor.storageInputAxisOrder = ScannerLutAxisOrder::Cmy;
        descriptor.storedValueDomain = ScannerLutStoredValueDomain::LogXyz;
        descriptor.logBase = ScannerLutLogBase::Base10;
        descriptor.numericFormat = ScannerLutNumericFormat::Float64;
        descriptor.storedOutputTripletOrder = ScannerLutOutputTripletOrder::Xyz;
        descriptor.schemaVersion = 2;
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
        if (!input.profileRoute || !input.densityBounds || !input.scannerOutput ||
            !input.mediumHandoff) {
            outDiagnostic = "ResourceDescriptorMismatch phase=4C scanner print inputs unavailable";
            return false;
        }
        const ProfileRoute& route = *input.profileRoute;
        const DensityBoundsRecipe& bounds = *input.densityBounds;
        const ScannerOutputRecipe& output = *input.scannerOutput;
        const PrintMediumHandoffRecipe& handoff = *input.mediumHandoff;
        if (!Spektrafilm::scan_route_is_print(route.scanRoute) || !route.printProfile ||
            bounds.hash == 0 || bounds.medium != Spektrafilm::DensityMedium::Print ||
            bounds.source != Spektrafilm::DensityBoundsSource::PrintMediaAuthoredCurves ||
            output.medium != Spektrafilm::DensityMedium::Print ||
            handoff.medium != Spektrafilm::DensityMedium::Print ||
            handoff.printProfileKey != route.printProfileKey ||
            handoff.printProfileAssetVersionToken != route.printProfileAssetVersionToken ||
            handoff.viewingIlluminant != output.viewingIlluminant ||
            input.observerIdentity.empty()) {
            outDiagnostic = "ResourceDescriptorMismatch phase=4C scanner print handoff";
            return false;
        }

        const Profiles::ValidatedPrintProfile& profile = *route.printProfile;
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
        outDescriptor = ScannerColorCorrectionDescriptor{};
        outDescriptor.route = recipe.profileRoute.scanRoute;
        outDiagnostic.clear();
        const ScannerOutputRecipe& output = recipe.scannerOutput;
        if (!output.blackCorrection && !output.whiteCorrection) {
            return true;
        }
        if (recipe.profileRoute.scanRoute == Spektrafilm::ScanRoute::NegativeDirectScan) {
            return true;
        }
        if (recipe.profileRoute.scanRoute != Spektrafilm::ScanRoute::PositiveDirectScan ||
            !recipe.profileRoute.filmProfile || scannerTables.K <= 0) {
            outDiagnostic = "ResourceDescriptorMismatch phase=8B direct correction inputs";
            return false;
        }

        const Profiles::SpektrafilmFilmData& data = recipe.profileRoute.filmProfile->data;
        const ScannerMediumRuntime medium = make_scanner_medium(recipe.densityBounds, scannerTables);
        std::array<float, 3> black = recipe.densityBounds.dataMaxCmy;
        std::array<float, 3> white{};
        float referenceBlackY = 0.0f;
        float referenceWhiteY = 0.0f;
        if (!scan_reference_y(medium, black, referenceBlackY) ||
            !scan_reference_y(medium, white, referenceWhiteY)) {
            outDiagnostic = "MalformedRequiredProfileData phase=8B direct reference scan";
            return false;
        }

        const float targetBlack = output.blackCorrection
                                      ? remove_srgb_cctf(output.blackLevel)
                                      : referenceBlackY;
        const float targetWhite = output.whiteCorrection
                                      ? remove_srgb_cctf(output.whiteLevel)
                                      : referenceWhiteY;
        const float slope =
            (targetWhite - targetBlack) / (referenceWhiteY - referenceBlackY + 1e-10f);
        const float offset = targetBlack - slope * referenceBlackY;
        const float correctedMidgray = (0.184f - offset) / slope;
        if (!(std::isfinite(correctedMidgray) && correctedMidgray > 0.0f)) {
            outDiagnostic = "MalformedRequiredProfileData phase=8B direct corrected midgray";
            trace_scanner_correction_descriptor(
                "rejected_corrected_midgray",
                recipe.profileRoute.scanRoute,
                output,
                referenceBlackY,
                referenceWhiteY,
                targetBlack,
                targetWhite,
                slope,
                offset,
                correctedMidgray,
                std::numeric_limits<float>::quiet_NaN(),
                0,
                outDiagnostic.c_str());
            return false;
        }
        const float densityMidgray = -std::log10(0.184f);
        const float correctedDensityMidgray = -std::log10(correctedMidgray);
        std::vector<float> average = average_density_curves(data.densityCurves);
        std::vector<float> negativeAverage = average;
        for (float& value : negativeAverage) {
            value = -value;
        }
        const float baseAverage = finite_average(data.baseDensity.data(), data.baseDensity.size());
        const float correctedLogExposure = -interp_clamped_monotonic(
            -(correctedDensityMidgray - baseAverage),
            negativeAverage,
            data.logExposure);
        const float logExposure = -interp_clamped_monotonic(
            -(densityMidgray - baseAverage),
            negativeAverage,
            data.logExposure);
        const float exposureScale = std::pow(10.0f, logExposure - correctedLogExposure);
        return finish_correction_descriptor(
            output,
            referenceBlackY,
            referenceWhiteY,
            exposureScale,
            outDescriptor,
            outDiagnostic);
    }

    bool build_print_scanner_color_correction_descriptor(
        const PrintCorrectionDerivationInput& input,
        ScannerColorCorrectionDescriptor& outDescriptor,
        std::string& outDiagnostic) {
        outDescriptor = ScannerColorCorrectionDescriptor{};
        outDiagnostic.clear();
        if (!input.recipe) {
            outDiagnostic = "ResourceDescriptorMismatch phase=8B print correction recipe";
            return false;
        }
        const RenderRecipe& recipe = *input.recipe;
        outDescriptor.route = recipe.profileRoute.scanRoute;
        const ScannerOutputRecipe& output = recipe.scannerOutput;
        if (!output.blackCorrection && !output.whiteCorrection) {
            return true;
        }
        if (!Spektrafilm::scan_route_is_print(recipe.profileRoute.scanRoute) ||
            !recipe.profileRoute.filmProfile || !recipe.profileRoute.printProfile ||
            !input.scannerTables || !input.mainIlluminant || !input.preflashRawCmy ||
            input.spectralSampleCount != Spectral::kNumSamples ||
            input.scannerTables->K <= 0 || !std::isfinite(input.normalizer)) {
            outDiagnostic = "ResourceDescriptorMismatch phase=8B print correction inputs";
            return false;
        }

        const Profiles::SpektrafilmFilmData& film = recipe.profileRoute.filmProfile->data;
        const Profiles::SpektrafilmPrintData& print = recipe.profileRoute.printProfile->data;
        const auto film_to_print_density = [&](const std::array<float, 3>& filmDensity) {
            std::array<float, 3> raw{};
            for (int sample = 0; sample < input.spectralSampleCount; ++sample) {
                const std::size_t k = static_cast<std::size_t>(sample);
                const float spectralDensity =
                    film.baseDensity[k] +
                    filmDensity[0] * film.channelDensity[k][0] +
                    filmDensity[1] * film.channelDensity[k][1] +
                    filmDensity[2] * film.channelDensity[k][2];
                const float light =
                    density_to_light_sample_spektrafilm(spectralDensity, input.mainIlluminant[k]);
                for (std::size_t channel = 0; channel < 3; ++channel) {
                    const float sensitivity = print.linearSensitivity[k][channel];
                    if (std::isfinite(sensitivity)) {
                        raw[channel] += light * sensitivity;
                    }
                }
            }
            std::array<float, 3> density{};
            for (std::size_t channel = 0; channel < 3; ++channel) {
                raw[channel] =
                    raw[channel] * input.normalizer +
                    input.preflashRawCmy[channel] * recipe.print.exposure.preflashExposure;
                density[channel] = sample_density_curve(
                    std::log10(std::fmax(raw[channel], 0.0f) + 1e-10f),
                    print,
                    channel);
            }
            return density;
        };

        const std::array<float, 3> printBlack =
            film_to_print_density(recipe.enlargerFilmBounds.dataMinCmy);
        const std::array<float, 3> printWhite =
            film_to_print_density(recipe.enlargerFilmBounds.dataMaxCmy);
        const ScannerMediumRuntime medium = make_scanner_medium(
            recipe.densityBounds,
            *input.scannerTables);
        float referenceBlackY = 0.0f;
        float referenceWhiteY = 0.0f;
        if (!scan_reference_y(medium, printBlack, referenceBlackY) ||
            !scan_reference_y(medium, printWhite, referenceWhiteY)) {
            outDiagnostic = "MalformedRequiredProfileData phase=8B print reference scan";
            return false;
        }

        const float targetBlack = output.blackCorrection
                                      ? remove_srgb_cctf(output.blackLevel)
                                      : referenceBlackY;
        const float targetWhite = output.whiteCorrection
                                      ? remove_srgb_cctf(output.whiteLevel)
                                      : referenceWhiteY;
        const float slope =
            (targetWhite - targetBlack) / (referenceWhiteY - referenceBlackY + 1e-10f);
        const float offset = targetBlack - slope * referenceBlackY;
        const float correctedMidgray = (0.184f - offset) / slope;
        if (!(std::isfinite(correctedMidgray) && correctedMidgray > 0.0f)) {
            outDiagnostic = "MalformedRequiredProfileData phase=8B print corrected midgray";
            trace_scanner_correction_descriptor(
                "rejected_corrected_midgray",
                recipe.profileRoute.scanRoute,
                output,
                referenceBlackY,
                referenceWhiteY,
                targetBlack,
                targetWhite,
                slope,
                offset,
                correctedMidgray,
                std::numeric_limits<float>::quiet_NaN(),
                0,
                outDiagnostic.c_str());
            return false;
        }
        const float densityMidgray = -std::log10(0.184f);
        const float correctedDensityMidgray = -std::log10(correctedMidgray);
        const std::vector<float> average = average_density_curves(print.densityCurves);
        const float baseAverage = finite_average(print.baseDensity.data(), print.baseDensity.size());
        const float correctedLogExposure = interp_clamped_monotonic(
            correctedDensityMidgray - baseAverage,
            average,
            print.logExposure);
        const float logExposure = interp_clamped_monotonic(
            densityMidgray - baseAverage,
            average,
            print.logExposure);
        const float exposureScale = std::pow(10.0f, correctedLogExposure - logExposure);
        return finish_correction_descriptor(
            output,
            referenceBlackY,
            referenceWhiteY,
            exposureScale,
            outDescriptor,
            outDiagnostic);
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
        outDescriptor.lensBlurSigmaPx =
            scanner_gaussian_sigma_with_device_kernel_or_zero(recipe.lensBlurSigmaPx);
        const float unsharpSigmaPx =
            scanner_gaussian_sigma_with_device_kernel_or_zero(recipe.unsharpSigmaPx);
        const bool unsharpActive = unsharpSigmaPx > 0.0f && recipe.unsharpAmount > 0.0f;
        outDescriptor.unsharpSigmaPx = unsharpActive ? unsharpSigmaPx : 0.0f;
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
        hash_value(hash, outDescriptor.schemaVersion);
        outDescriptor.hash = hash;
        return outDescriptor.hash != 0;
    }

} // namespace Scanner
