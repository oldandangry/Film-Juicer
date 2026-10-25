#include "DiffusionHostBehavior.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>

#include "Cuda/JuicerCudaFailure.h"
#include "RustDiffusionBridge.h"

namespace {

    constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;

    void hash_byte(std::uint64_t& hash, std::uint8_t value) {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= kFnvPrime;
    }

    void hash_u64_le(std::uint64_t& hash, std::uint64_t value) {
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            hash_byte(hash, static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffu));
        }
    }

    void hash_u32_le(std::uint64_t& hash, std::uint32_t value) {
        for (std::size_t byte = 0; byte < sizeof(value); ++byte) {
            hash_byte(hash, static_cast<std::uint8_t>((value >> (byte * 8)) & 0xffu));
        }
    }

    void hash_tag(std::uint64_t& hash, std::string_view tag) {
        for (const char value : tag) {
            hash_byte(hash, static_cast<std::uint8_t>(value));
        }
        hash_byte(hash, 0);
    }

    void hash_bool(std::uint64_t& hash, bool value) {
        hash_byte(hash, value ? 1u : 0u);
    }

    void hash_family(std::uint64_t& hash, Spektrafilm::DiffusionFilterFamily family) {
        hash_byte(hash, static_cast<std::uint8_t>(family));
    }

    void hash_schema(std::uint64_t& hash) {
        hash_u32_le(hash, Spektrafilm::kDiffusionHostBehaviorSchemaVersion);
    }

    void hash_double(std::uint64_t& hash, double value) {
        if (value == 0.0) {
            value = 0.0;
        }
        hash_u64_le(hash, std::bit_cast<std::uint64_t>(value));
    }

} // namespace

namespace Spektrafilm {

    std::uint64_t hash_diffusion_authored_controls(
        const DiffusionFilterAuthoredControls& controls) {
        std::uint64_t hash = kFnvOffset;
        hash_tag(hash, "diffusion-authored");
        hash_schema(hash);
        hash_bool(hash, controls.active);
        hash_family(hash, controls.family);
        hash_double(hash, controls.strength);
        hash_double(hash, controls.spatialScale);
        hash_double(hash, controls.haloWarmth);
        hash_double(hash, controls.coreIntensity);
        hash_double(hash, controls.coreSize);
        hash_double(hash, controls.haloIntensity);
        hash_double(hash, controls.haloSize);
        hash_double(hash, controls.bloomIntensity);
        hash_double(hash, controls.bloomSize);
        return hash;
    }

    // Retain the public sample builder's scalar argument order.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    bool build_diffusion_psf_sample_descriptor(
        const DiffusionFilterResolvedParameters& resolved,
        double pixelSizeUm,
        int fullFrameWidth,
        int fullFrameHeight,
        DiffusionPsfSampleDescriptor& out,
        std::string& diagnostic) {
        JuicerCuda::Failure failure;
        const bool success = JuicerDiffusion::prepare_sample(
            resolved, pixelSizeUm, fullFrameWidth, fullFrameHeight, out, failure);
        diagnostic = std::move(failure.diagnostic);
        return success;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    bool expand_diffusion_psf_components(
        const DiffusionPsfSampleDescriptor& descriptor,
        DiffusionPsfComponents& out,
        std::string& diagnostic) {
        JuicerCuda::Failure failure;
        const bool success = JuicerDiffusion::prepare_components(descriptor, out, failure);
        diagnostic = std::move(failure.diagnostic);
        return success;
    }

} // namespace Spektrafilm
