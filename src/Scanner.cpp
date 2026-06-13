#include "Scanner.h"

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

} // namespace Scanner
